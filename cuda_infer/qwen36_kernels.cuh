/*
 * qwen36_kernels.cuh — kernels specific to the Qwen3.6 35B-A3B-FP8 port.
 *
 * Adds FP8 e4m3 dequant + matvec with vLLM-style 128x128 block scales:
 *   weight       : float8_e4m3fn,  shape [N, K], row-major
 *   scale_inv    : bf16,           shape [ceil(N/128), ceil(K/128)]
 *   dequant      : w_f32[r,c] = float(w_fp8[r,c]) * scale_inv[r/128, c/128]
 *
 * Compute:   y[N] = W . x  (x is f32)
 *
 * Constraints assumed by this kernel:
 *   - K is a multiple of 128 (always true for Qwen3.6: K is hidden=2048 or
 *     a fan-out of 128).
 *   - N is a multiple of 128 (true for all 35B-A3B-FP8 weights).
 *   - blockDim.x = 128 threads/block (so each warp handles 32 cols at a time
 *     and 4 warps cover one 128-wide column block).
 */
#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace qwen36 {

// Convert one fp8 e4m3 byte to float using the device built-in cast.
__device__ __forceinline__ float fp8_e4m3_to_f32(uint8_t raw) {
    __nv_fp8_e4m3 v;
    v.__x = raw;
    return (float)v;
}

__device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 b) {
    return __bfloat162float(b);
}

// One block = one output row.  blockDim.x = 128.
//
// Each thread reads weight[row, cb*128 + tid] for each column block cb,
// multiplies by the column-block's scale, and accumulates  acc += scaled_w * x.
// We then warp- and block-reduce acc into y[row].
__global__ void dequant_matvec_fp8_block128(
    const uint8_t*       __restrict__ W,        // [N, K] fp8 e4m3 raw bytes
    const __nv_bfloat16* __restrict__ Sinv,     // [N/128, K/128]
    const float*         __restrict__ x,        // [K]
    float*               __restrict__ y,        // [N]
    uint32_t N, uint32_t K)
{
    const uint32_t row    = blockIdx.x;
    if (row >= N) return;
    const uint32_t Kblks  = K >> 7;          // K / 128
    const uint32_t rowblk = row >> 7;
    const uint32_t tid    = threadIdx.x;     // 0..127

    float acc = 0.0f;
    const uint8_t* row_w = W + (size_t)row * (size_t)K;
    const __nv_bfloat16* row_s = Sinv + (size_t)rowblk * (size_t)Kblks;

    #pragma unroll 1
    for (uint32_t cb = 0; cb < Kblks; cb++) {
        const float scale = bf16_to_f32(row_s[cb]);
        const uint32_t col = (cb << 7) + tid;
        const float w = fp8_e4m3_to_f32(row_w[col]);
        acc = fmaf(w * scale, x[col], acc);
    }

    // 4-warp reduction inside one block.
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffffu, acc, off);

    __shared__ float warp_sum[4];
    const int wid  = tid >> 5;
    const int lane = tid & 31;
    if (lane == 0) warp_sum[wid] = acc;
    __syncthreads();
    if (tid == 0) {
        y[row] = warp_sum[0] + warp_sum[1] + warp_sum[2] + warp_sum[3];
    }
}

// Convenience launcher.  Caller is responsible for ensuring N, K are multiples
// of 128 and that the device has the buffers correctly populated.
inline void launch_dequant_matvec_fp8_block128(
    const uint8_t* d_W, const __nv_bfloat16* d_Sinv,
    const float* d_x, float* d_y,
    uint32_t N, uint32_t K, cudaStream_t stream = 0)
{
    dim3 grid(N), block(128);
    dequant_matvec_fp8_block128<<<grid, block, 0, stream>>>(d_W, d_Sinv, d_x, d_y, N, K);
}

// ---------------------------------------------------------------------------
// RMSNorm with the Qwen3.6 "delta-from-1" weight parameterization:
//     y[i] = (1 + weight[i]) * x[i] * rsqrt(mean(x²) + eps)
//
// In Qwen3_5MoeRMSNorm the weight is stored as the offset from 1 (initialized
// to zero) so the +1 is part of the formula.  This is NOT used by the gated
// norm (Qwen3_5MoeRMSNormGated) which keeps the standard `weight * x * rsqrt`.
// ---------------------------------------------------------------------------
__global__ void rms_norm_bf16_plus_one(
    const float*    __restrict__ x,
    const uint16_t* __restrict__ weight,    // bf16, stored as (real_w - 1)
    float*          __restrict__ out,
    uint32_t dim, float eps)
{
    __shared__ float shared[32];
    float acc = 0.0f;
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x)
        acc += x[i] * x[i];

    // warp reduce
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
    uint32_t wid = threadIdx.x / 32;
    uint32_t lane = threadIdx.x & 31;
    if (lane == 0) shared[wid] = acc;
    __syncthreads();
    if (wid == 0) {
        uint32_t nw = (blockDim.x + 31) / 32;
        acc = (lane < nw) ? shared[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
        if (lane == 0) shared[0] = acc;
    }
    __syncthreads();

    float rms = rsqrtf(shared[0] / (float)dim + eps);
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        // (1 + w) * x * rms — keep the bf16 -> f32 in f32 throughout
        uint16_t w_bits = weight[i];
        float w = __uint_as_float((uint32_t)w_bits << 16);
        out[i] = (1.0f + w) * x[i] * rms;
    }
}

inline void launch_rms_norm_bf16_plus_one(
    const float* d_x, const uint16_t* d_w, float* d_y,
    uint32_t dim, float eps, cudaStream_t stream = 0)
{
    rms_norm_bf16_plus_one<<<1, 256, 0, stream>>>(d_x, d_w, d_y, dim, eps);
}

// ---------------------------------------------------------------------------
// Per-head RMSNorm with the (1+weight) form, applied to N independent rows of
// length `head_dim` (e.g. Q-norm/K-norm in the full-attention block: each
// q-head and k-head gets its own normalized projection over head_dim=256).
//
//   Grid: N rows (= num_heads * batch_for_one_layer)
//   Block: 256 threads (one per element of head_dim)
// ---------------------------------------------------------------------------
__global__ void rms_norm_per_row_plus_one(
    const float*    __restrict__ x,           // [N, head_dim]
    const uint16_t* __restrict__ weight,      // [head_dim] bf16, stored as delta from 1
    float*          __restrict__ out,         // [N, head_dim]
    uint32_t head_dim, float eps)
{
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const float* xrow = x   + (size_t)row * head_dim;
    float*       orow = out + (size_t)row * head_dim;

    extern __shared__ float smem[];
    float* sq = smem;                  // [head_dim] for x*x

    float v = (tid < head_dim) ? xrow[tid] : 0.0f;
    if (tid < head_dim) sq[tid] = v * v;
    __syncthreads();

    // single-thread reduction (head_dim ≤ 256, cheap on f64)
    __shared__ float rms;
    if (tid == 0) {
        double s = 0.0;
        for (uint32_t i = 0; i < head_dim; i++) s += (double)sq[i];
        rms = rsqrtf((float)(s / (double)head_dim) + eps);
    }
    __syncthreads();

    if (tid < head_dim) {
        float w_delta = __uint_as_float((uint32_t)weight[tid] << 16);
        orow[tid] = (1.0f + w_delta) * v * rms;
    }
}

inline void launch_rms_norm_per_row_plus_one(
    const float* d_x, const uint16_t* d_w, float* d_y,
    uint32_t num_rows, uint32_t head_dim, float eps, cudaStream_t stream = 0)
{
    dim3 grid(num_rows), block(head_dim);
    size_t smem = head_dim * sizeof(float);
    rms_norm_per_row_plus_one<<<grid, block, smem, stream>>>(d_x, d_w, d_y, head_dim, eps);
}

// ---------------------------------------------------------------------------
// Partial rotary-position embedding (rotate_half form), applied in place to
// a packed [num_heads, head_dim] tensor for a single token at position `pos`.
//   - Only the first `rotary_dim` dimensions of each head are rotated.
//   - cos / sin tables of length rotary_dim/2 (one entry per pair (i, i+rd/2))
//     are looked up at row `pos` of a precomputed [max_seq, rotary_dim/2] table.
//
// rotate_half formulation:
//     for i in [0, rd/2):
//        x_new[i]      = x[i]      * cos[i] - x[i + rd/2] * sin[i]
//        x_new[i+rd/2] = x[i+rd/2] * cos[i] + x[i]        * sin[i]
//   (cos[i+rd/2] == cos[i], sin[i+rd/2] == sin[i] in the cat(freqs,freqs) form)
//
// Block: rotary_dim/2 threads. Each thread handles one pair.
// Grid:  num_heads.
// ---------------------------------------------------------------------------
__global__ void rope_partial_inplace(
    float*       __restrict__ x,              // [num_heads, head_dim]
    const float* __restrict__ cos_pos,        // [rotary_dim/2]
    const float* __restrict__ sin_pos,        // [rotary_dim/2]
    uint32_t head_dim, uint32_t rotary_dim)
{
    const uint32_t h = blockIdx.x;
    const uint32_t i = threadIdx.x;
    const uint32_t half = rotary_dim >> 1;
    if (i >= half) return;

    float* xh = x + (size_t)h * head_dim;
    float a = xh[i];
    float b = xh[i + half];
    float c = cos_pos[i];
    float s = sin_pos[i];
    xh[i]        = a * c - b * s;
    xh[i + half] = b * c + a * s;
}

inline void launch_rope_partial_inplace(
    float* d_x, const float* d_cos_pos, const float* d_sin_pos,
    uint32_t num_heads, uint32_t head_dim, uint32_t rotary_dim,
    cudaStream_t stream = 0)
{
    rope_partial_inplace<<<num_heads, rotary_dim / 2, 0, stream>>>(
        d_x, d_cos_pos, d_sin_pos, head_dim, rotary_dim);
}

// Host: precompute the cos/sin table for positions [0, seq_len) using standard
// RoPE inv_freq = 1 / theta^(2i / rotary_dim) for i in [0, rotary_dim/2).
// Output buffers must be size `seq_len * rotary_dim/2` floats.
inline void rope_precompute_table(float* cos_out, float* sin_out,
                                  uint32_t seq_len, uint32_t rotary_dim,
                                  float theta)
{
    uint32_t half = rotary_dim / 2;
    for (uint32_t i = 0; i < half; i++) {
        float inv_freq = 1.0f / std::pow(theta, (float)(2 * i) / (float)rotary_dim);
        for (uint32_t t = 0; t < seq_len; t++) {
            float ang = (float)t * inv_freq;
            cos_out[t * half + i] = std::cos(ang);
            sin_out[t * half + i] = std::sin(ang);
        }
    }
}

} // namespace qwen36
