// Copyright (c) 2026 Sergey Subbotin
//
// kernels_fused_moe.hip.h — fused K-expert MoE dispatch for MLX 4-bit experts.
// Replaces the per-expert loop of K=4 × {gate, up, swiglu, down} kernel
// launches (16 launches per layer per token) with two kernel launches per
// layer per token: one for fused gate+up+SwiGLU and one for down.
//
// Per-block work is unchanged from dequant_matvec_4bit_fma_vec4 — same FMA
// kernel inlined. The win is dispatch consolidation and LDS reuse of the
// input vector across the gate and up rows of the same expert (currently
// each launch reloads x into its own LDS copy).
//
// MLX format only (g_quant_format != 1). The GGUF path keeps the existing
// launch_dequant_matvec_gguf calls because the per-block layout differs.

#ifndef KERNELS_FUSED_MOE_HIP_H
#define KERNELS_FUSED_MOE_HIP_H

#include <hip/hip_runtime.h>
#include "kernels.hip.h"  // for ROWS_PER_BLOCK, WARP_SIZE, bf16_to_f32, warp_reduce_sum

// One block per (expert, row_tile). Each warp computes one (gate, up) row
// plus the SwiGLU fusion, writing to gate_intermediate[expert * inter + row].
//
// expert_ptrs[k] points to the start of expert k's MLX-packed weight blob:
//   [gate_w][gate_s][gate_b][up_w][up_s][up_b][down_w][down_s][down_b]
// at the offsets EXP_GATE_W / EXP_GATE_S / ... defined in infer.hip.
__global__ void fused_moe_gate_up_swiglu_mlx(
    const void* const* __restrict__ expert_ptrs, // [K] device array
    uint32_t            gate_w_off,              // EXP_GATE_W (= 0)
    uint32_t            gate_s_off,
    uint32_t            gate_b_off,
    uint32_t            up_w_off,
    uint32_t            up_s_off,
    uint32_t            up_b_off,
    const float*        __restrict__ x,          // [HIDDEN_DIM]
    float*              __restrict__ inter,      // [K * MOE_INTERMEDIATE]
    uint32_t            hidden_dim,
    uint32_t            inter_dim
) {
    extern __shared__ float x_shared[];

    const uint32_t expert_id = blockIdx.x;
    const uint32_t row_tile  = blockIdx.y;
    const uint32_t lane      = threadIdx.x;
    const uint32_t warp_id   = threadIdx.y;
    const uint32_t row       = row_tile * ROWS_PER_BLOCK + warp_id;

    // Cooperative load of x into LDS — reused for both gate and up rows.
    const uint32_t tid = warp_id * WARP_SIZE + lane;
    for (uint32_t i = tid; i < hidden_dim; i += WARP_SIZE * ROWS_PER_BLOCK)
        x_shared[i] = x[i];
    __syncthreads();

    if (row >= inter_dim) return;

    const char* base = (const char*)expert_ptrs[expert_id];
    const uint32_t packed_cols = hidden_dim >> 3;
    const uint32_t num_groups  = hidden_dim >> 6;
    const uint32_t vec4_cols   = packed_cols >> 2;

    const uint32_t* gate_w = (const uint32_t*)(base + gate_w_off) + row * packed_cols;
    const uint16_t* gate_s = (const uint16_t*)(base + gate_s_off) + row * num_groups;
    const uint16_t* gate_b = (const uint16_t*)(base + gate_b_off) + row * num_groups;
    const uint32_t* up_w   = (const uint32_t*)(base + up_w_off)   + row * packed_cols;
    const uint16_t* up_s   = (const uint16_t*)(base + up_s_off)   + row * num_groups;
    const uint16_t* up_b   = (const uint16_t*)(base + up_b_off)   + row * num_groups;

    float gate_acc = 0.0f;
    float up_acc   = 0.0f;

    for (uint32_t vi = lane; vi < vec4_cols; vi += WARP_SIZE) {
        uint32_t base_col = vi << 2;
        uint32_t x_base   = base_col << 3;

        #pragma unroll
        for (uint32_t w = 0; w < 4; w++) {
            uint32_t g_idx = (base_col + w) >> 3;
            float gs = bf16_to_f32(gate_s[g_idx]);
            float gb = bf16_to_f32(gate_b[g_idx]);
            float us = bf16_to_f32(up_s[g_idx]);
            float ub = bf16_to_f32(up_b[g_idx]);
            uint32_t gp = gate_w[base_col + w];
            uint32_t up = up_w[base_col + w];
            uint32_t xb = x_base + (w << 3);

            #pragma unroll
            for (int n = 0; n < 8; n++) {
                float xi = x_shared[xb + n];
                float gn = (float)((gp >> (n * 4)) & 0xF);
                float un = (float)((up >> (n * 4)) & 0xF);
                gate_acc = fmaf(gn, gs * xi, fmaf(gb, xi, gate_acc));
                up_acc   = fmaf(un, us * xi, fmaf(ub, xi, up_acc));
            }
        }
    }

    gate_acc = warp_reduce_sum(gate_acc);
    up_acc   = warp_reduce_sum(up_acc);
    if (lane == 0) {
        float silu_g = gate_acc / (1.0f + expf(-gate_acc));
        inter[expert_id * inter_dim + row] = silu_g * up_acc;
    }
}

// One block per (expert, row_tile). Reads inter[expert * inter_dim ..] into
// LDS once, then each warp computes one down output row.
__global__ void fused_moe_down_mlx(
    const void* const* __restrict__ expert_ptrs,
    uint32_t            down_w_off,
    uint32_t            down_s_off,
    uint32_t            down_b_off,
    const float*        __restrict__ inter,    // [K * inter_dim]
    float*              __restrict__ out,      // [K * hidden_dim]
    uint32_t            hidden_dim,
    uint32_t            inter_dim
) {
    extern __shared__ float inter_shared[];

    const uint32_t expert_id = blockIdx.x;
    const uint32_t row_tile  = blockIdx.y;
    const uint32_t lane      = threadIdx.x;
    const uint32_t warp_id   = threadIdx.y;
    const uint32_t row       = row_tile * ROWS_PER_BLOCK + warp_id;

    // Cooperative load of this expert's inter slice into LDS.
    const uint32_t tid = warp_id * WARP_SIZE + lane;
    const float* inter_in = inter + expert_id * inter_dim;
    for (uint32_t i = tid; i < inter_dim; i += WARP_SIZE * ROWS_PER_BLOCK)
        inter_shared[i] = inter_in[i];
    __syncthreads();

    if (row >= hidden_dim) return;

    const char* base = (const char*)expert_ptrs[expert_id];
    const uint32_t packed_cols = inter_dim >> 3;
    const uint32_t num_groups  = inter_dim >> 6;
    const uint32_t vec4_cols   = packed_cols >> 2;

    const uint32_t* down_w = (const uint32_t*)(base + down_w_off) + row * packed_cols;
    const uint16_t* down_s = (const uint16_t*)(base + down_s_off) + row * num_groups;
    const uint16_t* down_b = (const uint16_t*)(base + down_b_off) + row * num_groups;

    float acc = 0.0f;

    for (uint32_t vi = lane; vi < vec4_cols; vi += WARP_SIZE) {
        uint32_t base_col = vi << 2;
        uint32_t x_base   = base_col << 3;

        #pragma unroll
        for (uint32_t w = 0; w < 4; w++) {
            uint32_t g_idx = (base_col + w) >> 3;
            float ds = bf16_to_f32(down_s[g_idx]);
            float db = bf16_to_f32(down_b[g_idx]);
            uint32_t dp = down_w[base_col + w];
            uint32_t xb = x_base + (w << 3);

            #pragma unroll
            for (int n = 0; n < 8; n++) {
                float xi = inter_shared[xb + n];
                float dn = (float)((dp >> (n * 4)) & 0xF);
                acc = fmaf(dn, ds * xi, fmaf(db, xi, acc));
            }
        }
    }

    acc = warp_reduce_sum(acc);
    if (lane == 0) out[expert_id * hidden_dim + row] = acc;
}

// Launcher: replaces K iterations of {launch_dequant_matvec(gate),
// launch_dequant_matvec(up), launch_swiglu, launch_dequant_matvec(down)}
// with two batched launches.
//
// inter scratch buffer is the caller's responsibility, sized [K * inter_dim].
// out is [K * hidden_dim].
static inline void launch_fused_moe_mlx(
    const void* const* d_expert_ptrs,
    uint32_t           gate_w_off, uint32_t gate_s_off, uint32_t gate_b_off,
    uint32_t           up_w_off,   uint32_t up_s_off,   uint32_t up_b_off,
    uint32_t           down_w_off, uint32_t down_s_off, uint32_t down_b_off,
    const float*       d_x,
    float*             d_inter,
    float*             d_out,
    uint32_t           hidden_dim,
    uint32_t           inter_dim,
    uint32_t           K,
    hipStream_t        stream = 0
) {
    dim3 block(WARP_SIZE, ROWS_PER_BLOCK);

    {
        dim3 grid(K, (inter_dim + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
        size_t smem = hidden_dim * sizeof(float);
        fused_moe_gate_up_swiglu_mlx<<<grid, block, smem, stream>>>(
            d_expert_ptrs, gate_w_off, gate_s_off, gate_b_off,
            up_w_off, up_s_off, up_b_off,
            d_x, d_inter, hidden_dim, inter_dim);
    }
    {
        dim3 grid(K, (hidden_dim + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
        size_t smem = inter_dim * sizeof(float);
        fused_moe_down_mlx<<<grid, block, smem, stream>>>(
            d_expert_ptrs, down_w_off, down_s_off, down_b_off,
            d_inter, d_out, hidden_dim, inter_dim);
    }
}

#endif // KERNELS_FUSED_MOE_HIP_H
