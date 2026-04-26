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

} // namespace qwen36
