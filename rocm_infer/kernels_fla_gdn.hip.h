// Copyright (c) 2026 Sergey Subbotin
//
// kernels_fla_gdn.hip.h — FLA-style fused gated_delta_net_step rewrite for
// CDNA3 (gfx942). Two optimizations vs the scalar baseline in
// kernels.hip.h::gated_delta_net_step:
//
//   1. Passes A (decay+kv_mem), B (state outer update), and C (output dot
//      product) are restructured so the state row is read+written only
//      twice per token instead of three times (was: 3 reads + 2 writes).
//      Pass A reads state to compute kv_mem (no write-back). After delta
//      is known, the fused B+C pass reads state again, computes the new
//      state, writes it, and accumulates the output dot product in the
//      same loop.
//
//   2. State loads/stores are float4 vectorized, reducing the load/store
//      instruction count 4x. The compiler does not auto-vectorize the
//      scalar loop on gfx942 with -O2.
//
// LDS staging of k/q was tried and turned out to be neutral on this shape
// (128 threads of a single block all read the same 128-element vectors —
// the L1 cache absorbs the redundant reads cheaply, and the syncthreads
// barrier costs as much as it saves).
//
// The per-thread layout, kh_mode semantics, and global state buffer are
// unchanged. Gated behind FLASH_MOE_FLA_GDN env var.

#ifndef KERNELS_FLA_GDN_HIP_H
#define KERNELS_FLA_GDN_HIP_H

#include <hip/hip_runtime.h>

__global__ void gated_delta_net_step_fla(
    float* __restrict__ state,         // [64 * 128 * 128]
    const float* __restrict__ q,       // [2048] = [16 K-heads, 128]
    const float* __restrict__ k,       // [2048]
    const float* __restrict__ v,       // [8192] = [64 V-heads, 128]
    const float* __restrict__ g_decay, // [64]
    const float* __restrict__ beta_gate, // [64]
    float* __restrict__ output,        // [8192]
    uint32_t k_heads_per_v,            // = 4
    uint32_t kh_mode                   // 0 = MLX (div), 1 = GGUF (mod)
) {
    const uint32_t head_id = blockIdx.x;
    const uint32_t vi = threadIdx.x;
    const uint32_t n_kh = gridDim.x / k_heads_per_v;
    const uint32_t kh = (kh_mode == 0) ? (head_id / k_heads_per_v)
                                       : (head_id % n_kh);
    const float g = g_decay[head_id];
    const float beta = beta_gate[head_id];

    const uint32_t state_base = head_id * 128 * 128 + vi * 128;
    const uint32_t k_base = kh * 128;
    const uint32_t v_base = head_id * 128;

    float       *state_v4_base = state + state_base;
    const float *k_base_p = k + k_base;
    const float *q_base_p = q + k_base;
    float4       *state_v4 = reinterpret_cast<float4 *>(state_v4_base);
    const float4 *k_v4 = reinterpret_cast<const float4 *>(k_base_p);
    const float4 *q_v4 = reinterpret_cast<const float4 *>(q_base_p);

    // Pass A: kv_mem = sum_ki ((state[ki] * g) * k[ki])  — no write-back
    float kv_mem = 0.0f;
    for (uint32_t i = 0; i < 32; i++) {  // 32 float4 = 128 floats
        float4 s  = state_v4[i];
        float4 kv = k_v4[i];
        kv_mem += (s.x * kv.x + s.y * kv.y + s.z * kv.z + s.w * kv.w) * g;
    }

    const float delta = (v[v_base + vi] - kv_mem) * beta;

    // Fused pass B+C: state update + output dot product
    float out_val = 0.0f;
    for (uint32_t i = 0; i < 32; i++) {
        float4 s  = state_v4[i];
        float4 kv = k_v4[i];
        float4 qv = q_v4[i];
        s.x = s.x * g + kv.x * delta;
        s.y = s.y * g + kv.y * delta;
        s.z = s.z * g + kv.z * delta;
        s.w = s.w * g + kv.w * delta;
        state_v4[i] = s;
        out_val += s.x * qv.x + s.y * qv.y + s.z * qv.z + s.w * qv.w;
    }

    output[v_base + vi] = out_val;
}

#endif // KERNELS_FLA_GDN_HIP_H
