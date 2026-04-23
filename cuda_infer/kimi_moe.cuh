/*
 * kimi_moe.cuh — Kimi K2.6 / DeepSeek-V3 MoE forward path.
 *
 * Routing (noaux_tc, sigmoid scoring):
 *   scores            = sigmoid(x @ W_gate.T)                  [num_experts]
 *   scores_for_choice = scores + e_score_correction_bias        (for top-K only)
 *   topk_idx          = top-K indices of scores_for_choice
 *   topk_w            = scores[topk_idx]                        (unbiased)
 *   if norm_topk_prob: topk_w /= topk_w.sum()
 *   topk_w           *= routed_scaling_factor
 *
 * Per-expert forward (compressed-tensors sym-int4, group_size=32):
 *   gate = sym4_matvec(W_gate, x)                               [2048]
 *   up   = sym4_matvec(W_up,   x)                               [2048]
 *   glu  = silu(gate) * up                                      [2048]
 *   ef   = sym4_matvec(W_down, glu)                             [H=7168]
 *
 * Combine:
 *   h_out = h_in + shared(h_in_normed) + scaling * Σ_k w_k · expert_k
 *         ^residual                    ^routed-only scale
 *
 * On-disk expert layout (from repack_experts_kimi.py): for each expert, 6
 * tensors back-to-back, sizes match Kimi's [MOE_INT=2048, H=7168] / group_size=32:
 *     gate_packed  int32[2048, 896]   7,340,032 B
 *     gate_scale   bf16 [2048, 224]     917,504 B
 *     up_packed    int32[2048, 896]   7,340,032 B
 *     up_scale     bf16 [2048, 224]     917,504 B
 *     down_packed  int32[7168, 256]   7,340,032 B
 *     down_scale   bf16 [7168,  64]     917,504 B
 *   total                             24,772,608 B per expert.
 */
#pragma once

#include "kernels.cuh"
#include <cmath>
#include <cstdint>

// ---------------------------------------------------------------------------
// Constants / offsets for one expert's on-disk block.
// ---------------------------------------------------------------------------
#define KIMI_H                     7168
#define KIMI_MOE_INT               2048
#define KIMI_SYM4_GROUP_SIZE       32
#define KIMI_GATE_PACKED_BYTES     (KIMI_MOE_INT * (KIMI_H / 8) * 4)            // 7340032
#define KIMI_GATE_SCALE_BYTES      (KIMI_MOE_INT * (KIMI_H / KIMI_SYM4_GROUP_SIZE) * 2)  // 917504
#define KIMI_UP_PACKED_BYTES       KIMI_GATE_PACKED_BYTES
#define KIMI_UP_SCALE_BYTES        KIMI_GATE_SCALE_BYTES
#define KIMI_DOWN_PACKED_BYTES     (KIMI_H * (KIMI_MOE_INT / 8) * 4)            // 7340032
#define KIMI_DOWN_SCALE_BYTES      (KIMI_H * (KIMI_MOE_INT / KIMI_SYM4_GROUP_SIZE) * 2)  // 917504
#define KIMI_EXPERT_BLOCK_BYTES    (3 * 7340032 + 3 * 917504)                   // 24772608

// Pointer convenience: given a host or device base of the block, sub-pointers:
#define KIMI_GATE_PACKED_OFF 0
#define KIMI_GATE_SCALE_OFF  7340032
#define KIMI_UP_PACKED_OFF   8257536
#define KIMI_UP_SCALE_OFF    15597568
#define KIMI_DOWN_PACKED_OFF 16515072
#define KIMI_DOWN_SCALE_OFF  23855104

// ---------------------------------------------------------------------------
// Routing kernel: sigmoid + bias-corrected top-K + (optional) renormalize + scale.
// Single block; num_experts ≤ 1024 recommended. For Kimi (num_experts=384, K=8)
// a serial top-K on thread 0 is already microseconds.
//
// Shared memory: 2 * num_experts floats.
// ---------------------------------------------------------------------------
__global__ void kimi_moe_routing_noaux_tc(
    const float* __restrict__ logits,        // [num_experts]
    const float* __restrict__ bias,          // [num_experts]
    int*         __restrict__ out_indices,   // [K]
    float*       __restrict__ out_weights,   // [K]
    uint32_t num_experts,
    uint32_t K,
    float scaling_factor,
    int   renormalize                        // 1 = divide by sum before scaling
) {
    extern __shared__ float smem[];
    float* scores = smem;                    // [num_experts]  (unbiased)
    float* biased = smem + num_experts;      // [num_experts]  (for top-K only)

    uint32_t tid = threadIdx.x;
    uint32_t n   = blockDim.x;

    // 1. Compute sigmoid(logits), biased = scores + bias  — in parallel
    for (uint32_t i = tid; i < num_experts; i += n) {
        float s   = 1.0f / (1.0f + __expf(-logits[i]));
        scores[i] = s;
        biased[i] = s + bias[i];
    }
    __syncthreads();

    // 2. Top-K (serial on thread 0 — K is tiny)
    if (tid == 0) {
        for (uint32_t k = 0; k < K; k++) {
            int   best = -1;
            float best_val = -INFINITY;
            for (uint32_t i = 0; i < num_experts; i++) {
                bool taken = false;
                for (uint32_t j = 0; j < k; j++) {
                    if ((uint32_t)out_indices[j] == i) { taken = true; break; }
                }
                if (taken) continue;
                if (biased[i] > best_val) { best_val = biased[i]; best = (int)i; }
            }
            out_indices[k] = best;
            out_weights[k] = scores[best];     // use UNBIASED score
        }
        if (renormalize) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < K; k++) sum += out_weights[k];
            float inv = (sum > 0.0f) ? (1.0f / sum) : 1.0f;
            for (uint32_t k = 0; k < K; k++) out_weights[k] *= inv;
        }
        for (uint32_t k = 0; k < K; k++) out_weights[k] *= scaling_factor;
    }
}

static inline void launch_kimi_moe_routing_noaux_tc(
    const float* logits, const float* bias,
    int* out_indices, float* out_weights,
    uint32_t num_experts, uint32_t K,
    float scaling_factor, int renormalize,
    cudaStream_t stream = 0)
{
    dim3 block(256);
    dim3 grid(1);
    size_t smem = (size_t)num_experts * 2 * sizeof(float);
    kimi_moe_routing_noaux_tc<<<grid, block, smem, stream>>>(
        logits, bias, out_indices, out_weights,
        num_experts, K, scaling_factor, renormalize);
}

// ---------------------------------------------------------------------------
// Weighted accumulate: acc[i] += w * src[i]
// Used to fold each expert's output into a single [H] accumulator.
// ---------------------------------------------------------------------------
__global__ void kimi_weighted_accum(
    float*       __restrict__ acc,     // [dim]
    const float* __restrict__ src,     // [dim]
    float        w,                    // host-passed scalar weight
    uint32_t dim)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) acc[i] = acc[i] + w * src[i];
}

static inline void launch_kimi_weighted_accum(
    float* acc, const float* src, float w, uint32_t dim, cudaStream_t stream = 0)
{
    dim3 block(256);
    dim3 grid((dim + 255) / 256);
    kimi_weighted_accum<<<grid, block, 0, stream>>>(acc, src, w, dim);
}

// Device version — reads the weight from a device pointer (e.g. d_topk_weights[k]).
// Avoids a host→device copy when dispatching one expert at a time.
__global__ void kimi_weighted_accum_dw(
    float*       __restrict__ acc,
    const float* __restrict__ src,
    const float* __restrict__ w_ptr,   // [1]
    uint32_t dim)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) acc[i] = acc[i] + (*w_ptr) * src[i];
}

static inline void launch_kimi_weighted_accum_dw(
    float* acc, const float* src, const float* w_ptr, uint32_t dim,
    cudaStream_t stream = 0)
{
    dim3 block(256);
    dim3 grid((dim + 255) / 256);
    kimi_weighted_accum_dw<<<grid, block, 0, stream>>>(acc, src, w_ptr, dim);
}

// ---------------------------------------------------------------------------
// Combine: h_out[i] = h_in[i] + shared[i] + scaling * moe_accum[i]
// ---------------------------------------------------------------------------
__global__ void kimi_moe_combine(
    const float* __restrict__ h_in,       // residual
    const float* __restrict__ shared,     // shared expert output
    const float* __restrict__ moe_accum,  // Σ_k w_k · expert_k(x)
    float*       __restrict__ h_out,
    float        routed_scaling_factor,
    uint32_t dim)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    h_out[i] = h_in[i] + shared[i] + routed_scaling_factor * moe_accum[i];
}

static inline void launch_kimi_moe_combine(
    const float* h_in, const float* shared, const float* moe_accum,
    float* h_out, float scaling, uint32_t dim, cudaStream_t stream = 0)
{
    dim3 block(256);
    dim3 grid((dim + 255) / 256);
    kimi_moe_combine<<<grid, block, 0, stream>>>(
        h_in, shared, moe_accum, h_out, scaling, dim);
}

// ---------------------------------------------------------------------------
// Expert forward path (all-GPU, inputs + outputs on device).
//
//   gate_tmp = sym4_matvec(W_gate, x)                     [MOE_INT=2048]
//   up_tmp   = sym4_matvec(W_up,   x)                     [MOE_INT]
//   glu_tmp  = silu(gate_tmp) * up_tmp                    [MOE_INT]
//   out      = sym4_matvec(W_down, glu_tmp)               [H=7168]
//
// The three weight pointers (plus their scale pointers) come from one
// contiguous expert block in VRAM/RAM (cf. the KIMI_*_OFF constants).
// Caller owns all buffers.
// ---------------------------------------------------------------------------
static inline void kimi_expert_forward_sym4(
    const uint32_t* d_gate_packed, const uint16_t* d_gate_scale,
    const uint32_t* d_up_packed,   const uint16_t* d_up_scale,
    const uint32_t* d_down_packed, const uint16_t* d_down_scale,
    const float* d_x,                 // [H] activation
    float* d_gate_tmp,                // [MOE_INT] scratch
    float* d_up_tmp,                  // [MOE_INT] scratch
    float* d_glu_tmp,                 // [MOE_INT] scratch (can alias d_gate_tmp)
    float* d_out,                     // [H] expert output
    uint32_t H, uint32_t moe_int,
    cudaStream_t stream = 0)
{
    // gate = sym4_matvec(W_gate[moe_int, H], x[H])
    dequant_matvec_sym4_g32<<<
        dim3((moe_int + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK),
        dim3(32, ROWS_PER_BLOCK),
        H * sizeof(float), stream>>>(
        d_gate_packed, d_gate_scale, d_x, d_gate_tmp, moe_int, H);

    // up = sym4_matvec(W_up[moe_int, H], x[H])
    dequant_matvec_sym4_g32<<<
        dim3((moe_int + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK),
        dim3(32, ROWS_PER_BLOCK),
        H * sizeof(float), stream>>>(
        d_up_packed, d_up_scale, d_x, d_up_tmp, moe_int, H);

    // SwiGLU: silu(gate) * up  (existing kernel in kernels.cuh)
    swiglu_fused<<<(moe_int + 255) / 256, 256, 0, stream>>>(
        d_gate_tmp, d_up_tmp, d_glu_tmp, moe_int);

    // out = sym4_matvec(W_down[H, moe_int], glu[moe_int])
    dequant_matvec_sym4_g32<<<
        dim3((H + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK),
        dim3(32, ROWS_PER_BLOCK),
        moe_int * sizeof(float), stream>>>(
        d_down_packed, d_down_scale, d_glu_tmp, d_out, H, moe_int);
}

// Convenience: given the base device pointer of one expert's contiguous block,
// dispatch the forward path using the known sub-offsets.
static inline void kimi_expert_forward_from_block(
    const uint8_t* d_expert_block,     // base of one expert's 24,772,608 bytes
    const float* d_x,
    float* d_gate_tmp, float* d_up_tmp, float* d_glu_tmp, float* d_out,
    cudaStream_t stream = 0)
{
    kimi_expert_forward_sym4(
        (const uint32_t*)(d_expert_block + KIMI_GATE_PACKED_OFF),
        (const uint16_t*)(d_expert_block + KIMI_GATE_SCALE_OFF),
        (const uint32_t*)(d_expert_block + KIMI_UP_PACKED_OFF),
        (const uint16_t*)(d_expert_block + KIMI_UP_SCALE_OFF),
        (const uint32_t*)(d_expert_block + KIMI_DOWN_PACKED_OFF),
        (const uint16_t*)(d_expert_block + KIMI_DOWN_SCALE_OFF),
        d_x, d_gate_tmp, d_up_tmp, d_glu_tmp, d_out, KIMI_H, KIMI_MOE_INT, stream);
}
