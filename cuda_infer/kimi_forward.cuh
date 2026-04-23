/*
 * kimi_forward.cuh — full forward path for Kimi K2.6.
 *
 * Includes:
 *   kimi_dense_mlp_forward   — layer 0 dense MLP (gate/up/down, bf16)
 *   kimi_shared_expert_forward — MoE layer shared expert (bf16)
 *   kimi_moe_layer_forward   — routing + K streamed experts + shared + combine
 *   kimi_layer_forward       — one transformer layer: attention + MLP/MoE with residuals and norms
 *   kimi_forward             — embed → N layers → final_norm → lm_head → argmax
 *
 * This header owns the per-step GPU state; it uses the scratch buffers that
 * kimi_alloc_scratch() put on KimiModel.
 *
 * The expert streaming path does pread → cudaMemcpy per selected expert. One
 * host buffer is kept in a file-level global (single-threaded driver).
 */
#pragma once

#include "kimi_loader.cuh"
#include "kimi_moe.cuh"
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>

// ---------------------------------------------------------------------------
// Small kernels: embed row copy, argmax.
// residual_add is already in kernels.cuh.
// ---------------------------------------------------------------------------

__global__ void kimi_embed_copy_bf16_to_f32(
    const uint16_t* __restrict__ embed,   // [vocab, H] bf16
    float*          __restrict__ out,     // [H] f32
    int token, uint32_t H)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H) return;
    uint32_t u = (uint32_t)embed[(size_t)token * H + i] << 16;
    float v; std::memcpy(&v, &u, 4);
    out[i] = v;
}

__global__ void kimi_argmax_logits(
    const float* __restrict__ logits,
    int*         __restrict__ out_idx,
    uint32_t vocab)
{
    __shared__ float smax[32];
    __shared__ int   simax[32];
    uint32_t tid = threadIdx.x;
    uint32_t nth = blockDim.x;

    float best_val = -INFINITY; int best_idx = 0;
    for (uint32_t i = tid; i < vocab; i += nth) {
        float v = logits[i];
        if (v > best_val) { best_val = v; best_idx = (int)i; }
    }
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_down_sync(0xFFFFFFFF, best_val, off);
        int   oi = __shfl_down_sync(0xFFFFFFFF, best_idx, off);
        if (ov > best_val) { best_val = ov; best_idx = oi; }
    }
    uint32_t lane = tid & 31, wid = tid >> 5;
    if (lane == 0) { smax[wid] = best_val; simax[wid] = best_idx; }
    __syncthreads();
    if (wid == 0) {
        best_val = (lane < (nth + 31) / 32) ? smax[lane]  : -INFINITY;
        best_idx = (lane < (nth + 31) / 32) ? simax[lane] : 0;
        for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_down_sync(0xFFFFFFFF, best_val, off);
            int   oi = __shfl_down_sync(0xFFFFFFFF, best_idx, off);
            if (ov > best_val) { best_val = ov; best_idx = oi; }
        }
        if (lane == 0) out_idx[0] = best_idx;
    }
}

// Host buffer for expert pread (single-threaded). File-scope static via inline.
static inline std::vector<uint8_t>& kimi_host_block() {
    static std::vector<uint8_t> b(KIMI_EXPERT_BLOCK_BYTES);
    return b;
}

// ---------------------------------------------------------------------------
// Debug helper: copy d_hidden[0..6] to host, print with L2 norm.
// ---------------------------------------------------------------------------
static int g_kimi_debug = 0;  // set via env KIMI_DEBUG=1

static inline void kimi_debug_print_hidden(const float* d_hidden, int H,
                                           const char* tag, int pos, int layer_idx)
{
    std::vector<float> h(H);
    cudaMemcpy(h.data(), d_hidden, (size_t)H * 4, cudaMemcpyDeviceToHost);
    double l2 = 0.0;
    int nan_count = 0, inf_count = 0;
    for (int i = 0; i < H; i++) {
        if (std::isnan(h[i])) nan_count++;
        else if (std::isinf(h[i])) inf_count++;
        else l2 += (double)h[i] * h[i];
    }
    l2 = std::sqrt(l2);
    printf("[pos=%d L%02d %s] L2=%.4g nan=%d inf=%d first6=[%.4g %.4g %.4g %.4g %.4g %.4g]\n",
        pos, layer_idx, tag, l2, nan_count, inf_count,
        h[0], h[1], h[2], h[3], h[4], h[5]);
    fflush(stdout);
}

// ---------------------------------------------------------------------------
// Dense MLP (layer 0)
// ---------------------------------------------------------------------------
static inline void kimi_dense_mlp_forward(
    KimiModel* K, const KimiLayer& L,
    const float* d_in, float* d_out)
{
    int H = K->cfg.mla.H;
    int I = K->cfg.dense_intermediate;
    launch_matvec_bf16(L.d_mlp_gate, d_in, K->d_dgate_tmp, (uint32_t)I, (uint32_t)H);
    launch_matvec_bf16(L.d_mlp_up,   d_in, K->d_dup_tmp,   (uint32_t)I, (uint32_t)H);
    launch_swiglu(K->d_dgate_tmp, K->d_dup_tmp, K->d_dglu_tmp, (uint32_t)I);
    launch_matvec_bf16(L.d_mlp_down, K->d_dglu_tmp, d_out, (uint32_t)H, (uint32_t)I);
}

// ---------------------------------------------------------------------------
// Shared expert (MoE layers)
// ---------------------------------------------------------------------------
static inline void kimi_shared_expert_forward(
    KimiModel* K, const KimiLayer& L,
    const float* d_in, float* d_out)
{
    int H = K->cfg.mla.H;
    int M = K->cfg.moe_intermediate;
    launch_matvec_bf16(L.d_shared_gate, d_in, K->d_sgate_tmp, (uint32_t)M, (uint32_t)H);
    launch_matvec_bf16(L.d_shared_up,   d_in, K->d_sup_tmp,   (uint32_t)M, (uint32_t)H);
    launch_swiglu(K->d_sgate_tmp, K->d_sup_tmp, K->d_sglu_tmp, (uint32_t)M);
    launch_matvec_bf16(L.d_shared_down, K->d_sglu_tmp, d_out, (uint32_t)H, (uint32_t)M);
}

// ---------------------------------------------------------------------------
// MoE layer forward (routing + K experts + shared + combine)
// ---------------------------------------------------------------------------
static inline void kimi_moe_layer_forward(
    KimiModel* K, const KimiLayer& L, int layer_idx,
    const float* d_norm_in, const float* d_residual, float* d_out)
{
    int H = K->cfg.mla.H;
    int ne = K->cfg.num_routed_experts;
    int Kexp = K->cfg.experts_per_tok;

    launch_matvec_bf16(L.d_router_gate, d_norm_in, K->d_router_logits,
                       (uint32_t)ne, (uint32_t)H);

    launch_kimi_moe_routing_noaux_tc(
        K->d_router_logits, L.d_router_bias,
        K->d_topk_idx, K->d_topk_w,
        (uint32_t)ne, (uint32_t)Kexp,
        K->cfg.routed_scaling_factor, K->cfg.norm_topk_prob);

    kimi_shared_expert_forward(K, L, d_norm_in, K->d_shared_out);

    std::vector<int> topk_idx(Kexp);
    cudaMemcpy(topk_idx.data(), K->d_topk_idx, (size_t)Kexp * 4, cudaMemcpyDeviceToHost);

    cudaMemset(K->d_moe_accum, 0, (size_t)H * 4);
    int fd = K->expert_fds[layer_idx];
    auto& host_block = kimi_host_block();
    for (int k = 0; k < Kexp; k++) {
        off_t off = (off_t)topk_idx[k] * KIMI_EXPERT_BLOCK_BYTES;
        ssize_t got = ::pread(fd, host_block.data(), KIMI_EXPERT_BLOCK_BYTES, off);
        if (got != (ssize_t)KIMI_EXPERT_BLOCK_BYTES) {
            fprintf(stderr, "kimi_forward: pread L%d E%d got %zd\n",
                    layer_idx, topk_idx[k], got);
            std::exit(1);
        }
        cudaMemcpy(K->d_expert_block, host_block.data(),
                   KIMI_EXPERT_BLOCK_BYTES, cudaMemcpyHostToDevice);
        kimi_expert_forward_from_block(
            K->d_expert_block, d_norm_in,
            K->d_gate_tmp, K->d_up_tmp, K->d_glu_tmp, K->d_expert_out);
        launch_kimi_weighted_accum_dw(
            K->d_moe_accum, K->d_expert_out, K->d_topk_w + k, (uint32_t)H);
    }

    // scaling_factor is baked into d_topk_w by the routing kernel, so pass 1.0
    launch_kimi_moe_combine(d_residual, K->d_shared_out, K->d_moe_accum,
                            d_out, 1.0f, (uint32_t)H);
}

// ---------------------------------------------------------------------------
// One layer forward: attention block + MLP/MoE block
// Input:  K->d_hidden (f32 [H])   — modified in place
// ---------------------------------------------------------------------------
static inline void kimi_layer_forward(KimiModel* K, int layer_idx, int pos) {
    KimiLayer& L = K->layers[layer_idx];
    int H = K->cfg.mla.H;
    int rope_half = K->cfg.mla.qk_rope_head_dim / 2;

    // Attention block
    cudaMemcpy(K->d_residual, K->d_hidden, (size_t)H * 4, cudaMemcpyDeviceToDevice);
    launch_rms_norm_bf16(K->d_hidden, L.d_input_layernorm, K->d_hidden_norm,
                         (uint32_t)H, K->cfg.rms_norm_eps);

    const float* d_cos_pos = K->d_cos_table + (size_t)pos * rope_half;
    const float* d_sin_pos = K->d_sin_table + (size_t)pos * rope_half;

    mla_attention_step(K->cfg.mla, &L.mla,
                       d_cos_pos, d_sin_pos,
                       K->d_hidden_norm, pos,
                       K->d_attn_out);

    dim3 block_add(256), grid_add((H + 255) / 256);
    residual_add<<<grid_add, block_add>>>(K->d_residual, K->d_attn_out, K->d_hidden, (uint32_t)H);

    // MLP / MoE block
    cudaMemcpy(K->d_residual, K->d_hidden, (size_t)H * 4, cudaMemcpyDeviceToDevice);
    launch_rms_norm_bf16(K->d_hidden, L.d_post_attn_layernorm, K->d_hidden_norm,
                         (uint32_t)H, K->cfg.rms_norm_eps);

    if (g_kimi_debug && (layer_idx >= 1 && layer_idx <= 6)) {
        char tag1[32], tag2[32];
        std::snprintf(tag1, sizeof tag1, "L%d post-attn",      layer_idx);
        std::snprintf(tag2, sizeof tag2, "L%d post-attn-norm", layer_idx);
        kimi_debug_print_hidden(K->d_hidden,      H, tag1, 0, layer_idx);
        kimi_debug_print_hidden(K->d_hidden_norm, H, tag2, 0, layer_idx);

        // Dump post-attn-norm to disk for cross-check with python
        if (layer_idx >= 2 && layer_idx <= 6) {
            std::vector<float> buf(H);
            cudaMemcpy(buf.data(), K->d_hidden_norm, (size_t)H*4, cudaMemcpyDeviceToHost);
            char p[64]; std::snprintf(p, sizeof p, "/tmp/kimi_L%d_hnorm.bin", layer_idx);
            FILE* f = std::fopen(p, "wb");
            std::fwrite(buf.data(), 4, H, f);
            std::fclose(f);
            printf("[debug] dumped L%d post-attn-norm to %s\n", layer_idx, p);
            fflush(stdout);
        }
    }

    if (!L.is_moe) {
        kimi_dense_mlp_forward(K, L, K->d_hidden_norm, K->d_mlp_out);
        residual_add<<<grid_add, block_add>>>(K->d_residual, K->d_mlp_out,
                                              K->d_hidden, (uint32_t)H);
    } else {
        // Debug breakdown for layers 1..3
        if (g_kimi_debug && (layer_idx >= 1 && layer_idx <= 6)) {
            int ne = K->cfg.num_routed_experts;
            int Kexp = K->cfg.experts_per_tok;
            launch_matvec_bf16(L.d_router_gate, K->d_hidden_norm, K->d_router_logits,
                               (uint32_t)ne, (uint32_t)H);
            launch_kimi_moe_routing_noaux_tc(
                K->d_router_logits, L.d_router_bias,
                K->d_topk_idx, K->d_topk_w,
                (uint32_t)ne, (uint32_t)Kexp,
                K->cfg.routed_scaling_factor, K->cfg.norm_topk_prob);
            cudaDeviceSynchronize();
            std::vector<int> tidx(Kexp); std::vector<float> tw(Kexp);
            cudaMemcpy(tidx.data(), K->d_topk_idx, (size_t)Kexp*4, cudaMemcpyDeviceToHost);
            cudaMemcpy(tw.data(),   K->d_topk_w,   (size_t)Kexp*4, cudaMemcpyDeviceToHost);
            std::vector<float> rlogits(ne);
            cudaMemcpy(rlogits.data(), K->d_router_logits, (size_t)ne*4, cudaMemcpyDeviceToHost);
            double rsum = 0; for (float v : rlogits) rsum += v*v;
            printf("[L%d routing] logits_L2=%.4g  topk_idx=", layer_idx, std::sqrt(rsum));
            for (int k = 0; k < Kexp; k++) printf("%d%s", tidx[k], k+1<Kexp?",":"");
            printf("  topk_w=");
            for (int k = 0; k < Kexp; k++) printf("%.4g%s", tw[k], k+1<Kexp?",":"");
            printf("\n"); fflush(stdout);

            kimi_shared_expert_forward(K, L, K->d_hidden_norm, K->d_shared_out);
            char stag[32]; std::snprintf(stag, sizeof stag, "L%d shared_out", layer_idx);
            kimi_debug_print_hidden(K->d_shared_out, H, stag, 0, layer_idx);

            cudaMemset(K->d_moe_accum, 0, (size_t)H * 4);
            int fd = K->expert_fds[layer_idx];
            auto& host_block = kimi_host_block();
            for (int k = 0; k < Kexp; k++) {
                off_t off = (off_t)tidx[k] * KIMI_EXPERT_BLOCK_BYTES;
                ssize_t got = ::pread(fd, host_block.data(), KIMI_EXPERT_BLOCK_BYTES, off);
                if (got != (ssize_t)KIMI_EXPERT_BLOCK_BYTES) { printf("pread short\n"); break; }
                cudaMemcpy(K->d_expert_block, host_block.data(),
                           KIMI_EXPERT_BLOCK_BYTES, cudaMemcpyHostToDevice);
                kimi_expert_forward_from_block(
                    K->d_expert_block, K->d_hidden_norm,
                    K->d_gate_tmp, K->d_up_tmp, K->d_glu_tmp, K->d_expert_out);
                if (k == 0) {
                    char etag[32]; std::snprintf(etag, sizeof etag, "L%d expert0_out", layer_idx);
                    kimi_debug_print_hidden(K->d_expert_out, H, etag, 0, layer_idx);
                }
                launch_kimi_weighted_accum_dw(
                    K->d_moe_accum, K->d_expert_out, K->d_topk_w + k, (uint32_t)H);
            }
            char mtag[32], rtag[48];
            std::snprintf(mtag, sizeof mtag, "L%d moe_accum", layer_idx);
            std::snprintf(rtag, sizeof rtag, "L%d residual(pre-combine)", layer_idx);
            kimi_debug_print_hidden(K->d_moe_accum, H, mtag, 0, layer_idx);
            kimi_debug_print_hidden(K->d_residual,  H, rtag, 0, layer_idx);

            launch_kimi_moe_combine(K->d_residual, K->d_shared_out, K->d_moe_accum,
                                    K->d_hidden, 1.0f, (uint32_t)H);
        } else {
            kimi_moe_layer_forward(K, L, layer_idx,
                                   K->d_hidden_norm, K->d_residual, K->d_hidden);
        }
    }
}

// ---------------------------------------------------------------------------
// Full single-token forward: embed → layers → final_norm → lm_head → argmax.
// ---------------------------------------------------------------------------
static inline int kimi_forward(KimiModel* K, int token, int pos) {
    int H = K->cfg.mla.H;

    dim3 block(256), grid((H + 255) / 256);
    kimi_embed_copy_bf16_to_f32<<<grid, block>>>(
        K->d_embed, K->d_hidden, token, (uint32_t)H);

    if (g_kimi_debug) kimi_debug_print_hidden(K->d_hidden, H, "embed", pos, -1);

    for (int li = 0; li < K->cfg.num_layers; li++) {
        kimi_layer_forward(K, li, pos);
        if (g_kimi_debug && (li < 8 || li % 10 == 0 || li == K->cfg.num_layers - 1))
            kimi_debug_print_hidden(K->d_hidden, H, "post-layer", pos, li);
    }

    launch_rms_norm_bf16(K->d_hidden, K->d_final_norm, K->d_hidden_norm,
                         (uint32_t)H, K->cfg.rms_norm_eps);
    launch_matvec_bf16(K->d_lm_head, K->d_hidden_norm, K->d_logits,
                       (uint32_t)K->cfg.vocab_size, (uint32_t)H);

    if (g_kimi_debug) {
        std::vector<float> logits(K->cfg.vocab_size);
        cudaMemcpy(logits.data(), K->d_logits, (size_t)K->cfg.vocab_size * 4, cudaMemcpyDeviceToHost);
        // top-5
        std::vector<int> idx(K->cfg.vocab_size);
        for (int i = 0; i < K->cfg.vocab_size; i++) idx[i] = i;
        std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
            [&](int a, int b){ return logits[a] > logits[b]; });
        printf("[pos=%d logits top5]", pos);
        for (int i = 0; i < 5; i++)
            printf(" %d(%.3f)", idx[i], logits[idx[i]]);
        printf("\n"); fflush(stdout);
    }

    int* d_next;
    cudaMalloc(&d_next, sizeof(int));
    kimi_argmax_logits<<<1, 1024>>>(K->d_logits, d_next, (uint32_t)K->cfg.vocab_size);
    int h_next = 0;
    cudaMemcpy(&h_next, d_next, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_next);
    return h_next;
}
