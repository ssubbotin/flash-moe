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
#include <hip/hip_fp16.h>
#include "kernels.hip.h"  // for ROWS_PER_BLOCK, WARP_SIZE, bf16_to_f32, warp_reduce_sum

// CDNA3 MFMA vector types
typedef _Float16 half4_t  __attribute__((ext_vector_type(4)));
typedef float    float4v_t __attribute__((ext_vector_type(4)));

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

// ----------------------------------------------------------------------------
// MFMA path (Task 4.1) — uses v_mfma_f32_16x16x16f16 for the inner loop.
//
// One wavefront (64 threads) per (expert, 16-row tile). Produces 16 output
// rows of gate*up*SwiGLU. The MFMA instruction layout for f32_16x16x16f16:
//
//   A[16,16] fp16: lane t holds A[t%16, (t/16)*4 + n] for n in [0,3]
//   B[16,16] fp16: lane t holds B[(t/16)*4 + n, t%16] for n in [0,3]
//   C[16,16] fp32: lane t holds C[(t/16)*4 + n, t%16] for n in [0,3]
//
// For matvec we broadcast x into all 16 N-cols of B, then extract one column
// of C as the per-row outputs. Wasteful in raw FLOPs (15/16 of MFMA work is
// redundant) but the win comes from amortizing the 4-bit -> fp16 dequant
// across 16 column outputs of one MFMA call.
//
// Each MFMA call covers K=16 hidden_dim elements. HIDDEN_DIM=4096 -> 256
// MFMA calls per gate accumulator + 256 for up = 512 MFMAs per wavefront.

__global__ void fused_moe_gate_up_swiglu_mfma_mlx(
    const void* const* __restrict__ expert_ptrs,
    uint32_t            gate_w_off,
    uint32_t            gate_s_off,
    uint32_t            gate_b_off,
    uint32_t            up_w_off,
    uint32_t            up_s_off,
    uint32_t            up_b_off,
    const float*        __restrict__ x,
    float*              __restrict__ inter,
    uint32_t            hidden_dim,
    uint32_t            inter_dim
) {
    const uint32_t expert_id = blockIdx.x;
    const uint32_t row_tile  = blockIdx.y;
    const uint32_t lane      = threadIdx.x;          // 0..63
    const uint32_t row_in_tile = lane & 15;          // 0..15 — A row this lane owns
    const uint32_t col_block   = lane >> 4;          // 0..3  — A col block this lane owns
    const uint32_t base_row    = row_tile * 16;

    // LDS: 16 fp16 broadcast x values per col tile (16 cols at a time)
    __shared__ _Float16 x_lds[16];

    const char* base = (const char*)expert_ptrs[expert_id];
    const uint32_t packed_cols = hidden_dim >> 3;    // hidden/8 uint32 per row
    const uint32_t num_groups  = hidden_dim >> 6;    // hidden/64 groups per row

    // Stage 1: dequantize this lane's A entries. Each lane owns:
    //   gate_w[base_row + row_in_tile, col_block*4 + n] for n in [0,3] (per col tile)
    // For one col tile (cols 0..15), each lane reads 4 nibbles from one row.
    //
    // 4 nibbles = 16 bits = half of a uint32. Two lanes share one uint32:
    //   lane (row_in_tile, 0) reads low half of gate_w[row_in_tile, packed_idx=0]
    //   lane (row_in_tile, 1) reads high half of gate_w[row_in_tile, packed_idx=0]
    //   lane (row_in_tile, 2) reads low half of gate_w[row_in_tile, packed_idx=1]
    //   lane (row_in_tile, 3) reads high half of gate_w[row_in_tile, packed_idx=1]

    const uint32_t* gate_w_p = (const uint32_t*)(base + gate_w_off);
    const uint16_t* gate_s_p = (const uint16_t*)(base + gate_s_off);
    const uint16_t* gate_b_p = (const uint16_t*)(base + gate_b_off);
    const uint32_t* up_w_p   = (const uint32_t*)(base + up_w_off);
    const uint16_t* up_s_p   = (const uint16_t*)(base + up_s_off);
    const uint16_t* up_b_p   = (const uint16_t*)(base + up_b_off);

    const uint32_t row = base_row + row_in_tile;
    if (row >= inter_dim) return;

    const uint32_t* gate_row_w = gate_w_p + row * packed_cols;
    const uint16_t* gate_row_s = gate_s_p + row * num_groups;
    const uint16_t* gate_row_b = gate_b_p + row * num_groups;
    const uint32_t* up_row_w   = up_w_p   + row * packed_cols;
    const uint16_t* up_row_s   = up_s_p   + row * num_groups;
    const uint16_t* up_row_b   = up_b_p   + row * num_groups;

    float4v_t gate_acc = {0.0f, 0.0f, 0.0f, 0.0f};
    float4v_t up_acc   = {0.0f, 0.0f, 0.0f, 0.0f};

    // Iterate col tiles of 16 elements
    for (uint32_t col_tile = 0; col_tile < hidden_dim; col_tile += 16) {
        // Load x[col_tile..col_tile+16] into LDS as fp16 once per tile.
        // 16 values, one per lane (lanes 0..15).
        if (lane < 16) {
            x_lds[lane] = (_Float16) x[col_tile + lane];
        }
        // Note: x_lds shared across all 64 lanes; need a syncthreads.
        // Single-wavefront block, so __builtin_amdgcn_s_barrier suffices,
        // but __syncthreads is portable.
        __syncthreads();

        // Each lane dequants 4 nibbles for its (row_in_tile, col_block*4..+3) slot of A.
        // For col_tile=c, A col index for this lane = col_block*4 + n in [0,3].
        // Hidden dim col = c + col_block*4 + n.
        // packed_idx = (c + col_block*4 + n)/8; nibble in packed = (c + col_block*4 + n)%8.
        // Group idx = (c + col_block*4 + n)/64.
        //
        // For c on a 16-boundary, col_block in [0..3] picks one of 4 cols of 4 elements,
        // specifically: lane reads nibbles [col_block*4 .. col_block*4+3] within the 16-col tile.
        // packed_idx: (c + col_block*4 + 0..3)/8.
        //   For col_block=0,1: c/8 (all 4 nibbles in one uint32 since col_block*4 in [0,7])
        //   For col_block=2,3: c/8 + 1 (col_block*4 in [8,15])

        uint32_t pidx0 = (col_tile >> 3) + (col_block >> 1);
        uint32_t nbase = (col_block & 1) * 4;
        uint32_t gp = gate_row_w[pidx0];
        uint32_t up = up_row_w[pidx0];

        // Group index for the 4 cols this lane owns. For 16-col tile aligned on 16,
        // and groups of 64, all 4 lane-cols sit in the same group (since 16 << 64).
        uint32_t g_idx = (col_tile + col_block * 4) >> 6;
        float gs = bf16_to_f32(gate_row_s[g_idx]);
        float gb = bf16_to_f32(gate_row_b[g_idx]);
        float us = bf16_to_f32(up_row_s[g_idx]);
        float ub = bf16_to_f32(up_row_b[g_idx]);

        half4_t a_gate, a_up;
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            uint32_t gn = (gp >> ((nbase + n) * 4)) & 0xF;
            uint32_t un = (up >> ((nbase + n) * 4)) & 0xF;
            a_gate[n] = (_Float16)((float)gn * gs + gb);
            a_up[n]   = (_Float16)((float)un * us + ub);
        }

        // B operand: x broadcast. Lane t holds B[(t/16)*4 + n, t%16] for n in [0,3].
        // For broadcast (same value across all N cols), we need lane t to hold
        // x[(t/16)*4 + n]   in slot n of half4. (t%16 is the N col, irrelevant for broadcast.)
        half4_t b;
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            b[n] = x_lds[col_block * 4 + n];
        }

        gate_acc = __builtin_amdgcn_mfma_f32_16x16x16f16(a_gate, b, gate_acc, 0, 0, 0);
        up_acc   = __builtin_amdgcn_mfma_f32_16x16x16f16(a_up,   b, up_acc,   0, 0, 0);
    }

    // C accumulator layout: lane t holds C[(t/16)*4 + n, t%16] for n in [0,3].
    // We want output[row] = C[row, 0] (any N col since all N cols carry the same scalar).
    // Lane (row_in_tile=r, col_block=cb) holds C[cb*4 + n, r]. So C[r, 0] is held by
    // lane (row_in_tile=0, col_block=0)'s element n=r/4? No — it's the OPPOSITE indexing.
    // Let me redo: lane t = (row_in_tile, col_block). t/16=col_block, t%16=row_in_tile.
    //   Lane (rit, cb) holds C[cb*4 + 0..3, rit].
    // For a given output row R, we need C[R, *]. R = cb*4 + n for some (cb, n).
    //   cb = R/4, n = R%4.
    // Then column = rit (any rit in [0,15]). Pick rit=0 by reading lane (0, R/4)'s elem n=R%4.
    //
    // Equivalently: lane (row_in_tile=0, col_block=cb) holds output rows cb*4 .. cb*4+3.
    //
    // To write 16 outputs from 4 contributing lanes (col_block=0..3, row_in_tile=0):
    //   Lane (0, 0) writes outputs [0,1,2,3] from gate_acc/up_acc[0..3]
    //   Lane (0, 1) writes outputs [4,5,6,7]
    //   Lane (0, 2) writes outputs [8,9,10,11]
    //   Lane (0, 3) writes outputs [12,13,14,15]

    if (row_in_tile == 0) {
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            uint32_t out_row = base_row + col_block * 4 + n;
            if (out_row < inter_dim) {
                float g = gate_acc[n];
                float u = up_acc[n];
                float silu_g = g / (1.0f + expf(-g));
                inter[expert_id * inter_dim + out_row] = silu_g * u;
            }
        }
    }
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
    int                use_mfma,         // FLASH_MOE_MOE_MFMA
    hipStream_t        stream = 0
) {
    dim3 block(WARP_SIZE, ROWS_PER_BLOCK);

    if (use_mfma) {
        dim3 mfma_block(WARP_SIZE, 1);
        dim3 mfma_grid(K, (inter_dim + 15) / 16);
        fused_moe_gate_up_swiglu_mfma_mlx<<<mfma_grid, mfma_block, 0, stream>>>(
            d_expert_ptrs, gate_w_off, gate_s_off, gate_b_off,
            up_w_off, up_s_off, up_b_off,
            d_x, d_inter, hidden_dim, inter_dim);
    } else {
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
