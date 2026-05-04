// Copyright (c) 2026 Sergey Subbotin
//
// kernels_aotriton_attn.hip.h — fp32<->bf16 staging kernels used by the
// aotriton flash attention path in the full-attention branch of layer_forward.
// aotriton::v2::flash::attn_fwd consumes bf16/fp16 tensors only, while the
// rest of the engine keeps Q/K/V/Out in fp32. These kernels do the in-place
// staging copy.

#ifndef KERNELS_AOTRITON_ATTN_HIP_H
#define KERNELS_AOTRITON_ATTN_HIP_H

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

__device__ __forceinline__ uint16_t f32_to_bf16_bits(float f) {
    uint32_t u;
    __builtin_memcpy(&u, &f, sizeof(u));
    // round-to-nearest-even
    uint32_t lsb = (u >> 16) & 1;
    uint32_t bias = 0x7fff + lsb;
    u += bias;
    return (uint16_t)(u >> 16);
}

__device__ __forceinline__ float bf16_to_f32_dev(uint16_t b) {
    uint32_t u = ((uint32_t)b) << 16;
    float f;
    __builtin_memcpy(&f, &u, sizeof(f));
    return f;
}

__global__ void f32_to_bf16_kernel(const float *__restrict__ src,
                                   uint16_t *__restrict__ dst,
                                   int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = f32_to_bf16_bits(src[i]);
}

__global__ void bf16_to_f32_kernel(const uint16_t *__restrict__ src,
                                   float *__restrict__ dst,
                                   int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = bf16_to_f32_dev(src[i]);
}

#endif // KERNELS_AOTRITON_ATTN_HIP_H
