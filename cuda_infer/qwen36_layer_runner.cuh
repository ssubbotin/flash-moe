/*
 * qwen36_layer_runner.cuh — per-layer load/run/free helpers shared between
 * test_qwen36_layer.cu (single-layer validation) and test_qwen36_chain.cu
 * (40-layer end-to-end run).
 *
 * Three weight bundles per decoder layer:
 *   LinearAttnW : the GatedDeltaNet path  (input_layernorm + linear_attn.*)
 *   FullAttnW   : the full-attention path (input_layernorm + self_attn.*)
 *   MoEW        : the post-attn MoE block (post_attn_layernorm + mlp.*)
 *
 * Each "run_*" function consumes a sequence of tokens, advances the per-token
 * state appropriately (conv1d + delta state for linear; KV cache for full),
 * and writes the residual stream into d_resid_out / d_layer_out.
 *
 * For chain tests the caller can free expert weights after each layer to keep
 * total VRAM usage bounded; the small (non-expert) weights for one layer are
 * tiny (~50MB) so they can either be kept across layers or freed too.
 */
#pragma once

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "safetensors_io.cuh"
#include "qwen36_kernels.cuh"
#include "kernels.cuh"

#ifndef CUDA_OK
#define CUDA_OK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA: %s @ %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    std::exit(1); } } while (0)
#endif

namespace q36 {

// ---------- model constants ----------
constexpr uint32_t H            = 2048;
constexpr uint32_t QKV_DIM      = 8192;
constexpr uint32_t Z_DIM        = 4096;
constexpr uint32_t LIN_NV       = 32;
constexpr uint32_t LIN_NK       = 16;
constexpr uint32_t LIN_HEAD     = 128;
constexpr uint32_t LIN_KEY_TOT  = LIN_NK * LIN_HEAD;     // 2048
constexpr uint32_t LIN_VAL_TOT  = LIN_NV * LIN_HEAD;     // 4096
constexpr uint32_t FA_NQ        = 16;
constexpr uint32_t FA_NKV       = 2;
constexpr uint32_t FA_HEAD_D    = 256;
constexpr uint32_t FA_Q_DIM     = FA_NQ * FA_HEAD_D;     // 4096
constexpr uint32_t FA_Q_RAW_DIM = FA_NQ * FA_HEAD_D * 2; // 8192
constexpr uint32_t FA_KV_DIM    = FA_NKV * FA_HEAD_D;    // 512
constexpr uint32_t FA_HPK       = FA_NQ / FA_NKV;        // 8
constexpr uint32_t ROPE_DIM     = FA_HEAD_D / 4;         // 64
constexpr uint32_t ROPE_HALF    = ROPE_DIM / 2;          // 32
constexpr float    ROPE_THETA   = 10000000.0f;
constexpr float    RMS_EPS      = 1e-6f;
constexpr uint32_t INTER        = 512;
constexpr uint32_t N_EXPERTS    = 256;
constexpr uint32_t TOP_K        = 8;
constexpr uint32_t NUM_LAYERS   = 40;
constexpr uint32_t VOCAB        = 248320;

// ---------- helpers ----------
inline uint8_t* upload_bytes(st::ModelDir* M, const std::string& name) {
    std::vector<uint8_t> buf;
    if (!st::read_bytes(M, name, buf)) std::exit(1);
    uint8_t* d; CUDA_OK(cudaMalloc(&d, buf.size()));
    CUDA_OK(cudaMemcpy(d, buf.data(), buf.size(), cudaMemcpyHostToDevice));
    return d;
}

inline std::string layer_prefix(int li) {
    char buf[64]; snprintf(buf, sizeof(buf), "model.language_model.layers.%d", li);
    return buf;
}

inline bool is_full_attn(int li) {
    // Pattern from config.layer_types: linear×3, full×1 repeating → full at indices 3,7,11,...,39.
    return (li % 4) == 3;
}

// ---------- linear-attention bundle ----------
struct LinearAttnW {
    uint16_t *in_ln_w = nullptr;
    uint8_t  *qkv_w   = nullptr; __nv_bfloat16 *qkv_s = nullptr;
    uint8_t  *z_w     = nullptr; __nv_bfloat16 *z_s   = nullptr;
    uint16_t *a_w     = nullptr;
    uint16_t *b_w     = nullptr;
    uint16_t *conv_w  = nullptr;
    float    *A_log   = nullptr;
    uint16_t *dt_bias = nullptr;
    uint16_t *norm_w  = nullptr;
    uint8_t  *out_w   = nullptr; __nv_bfloat16 *out_s = nullptr;
};

inline void load_linear_attn(st::ModelDir* M, const std::string& prefix, LinearAttnW* L) {
    auto T = [&](const std::string& nm){ return prefix + "." + nm; };
    L->in_ln_w  = (uint16_t*)upload_bytes(M, T("input_layernorm.weight"));
    L->qkv_w    = upload_bytes(M, T("linear_attn.in_proj_qkv.weight"));
    L->qkv_s    = (__nv_bfloat16*)upload_bytes(M, T("linear_attn.in_proj_qkv.weight_scale_inv"));
    L->z_w      = upload_bytes(M, T("linear_attn.in_proj_z.weight"));
    L->z_s      = (__nv_bfloat16*)upload_bytes(M, T("linear_attn.in_proj_z.weight_scale_inv"));
    L->a_w      = (uint16_t*)upload_bytes(M, T("linear_attn.in_proj_a.weight"));
    L->b_w      = (uint16_t*)upload_bytes(M, T("linear_attn.in_proj_b.weight"));
    L->conv_w   = (uint16_t*)upload_bytes(M, T("linear_attn.conv1d.weight"));
    {
        uint16_t* bf = (uint16_t*)upload_bytes(M, T("linear_attn.A_log"));
        std::vector<uint16_t> h(LIN_NV);
        CUDA_OK(cudaMemcpy(h.data(), bf, LIN_NV * 2, cudaMemcpyDeviceToHost));
        std::vector<float> hf(LIN_NV);
        for (uint32_t i = 0; i < LIN_NV; i++) hf[i] = st::bf16_to_f32(h[i]);
        CUDA_OK(cudaMalloc(&L->A_log, LIN_NV * 4));
        CUDA_OK(cudaMemcpy(L->A_log, hf.data(), LIN_NV * 4, cudaMemcpyHostToDevice));
        cudaFree(bf);
    }
    L->dt_bias  = (uint16_t*)upload_bytes(M, T("linear_attn.dt_bias"));
    L->norm_w   = (uint16_t*)upload_bytes(M, T("linear_attn.norm.weight"));
    L->out_w    = upload_bytes(M, T("linear_attn.out_proj.weight"));
    L->out_s    = (__nv_bfloat16*)upload_bytes(M, T("linear_attn.out_proj.weight_scale_inv"));
}

inline void free_linear_attn(LinearAttnW* L) {
    cudaFree(L->in_ln_w);  cudaFree(L->qkv_w); cudaFree(L->qkv_s);
    cudaFree(L->z_w);      cudaFree(L->z_s);
    cudaFree(L->a_w);      cudaFree(L->b_w);
    cudaFree(L->conv_w);   cudaFree(L->A_log); cudaFree(L->dt_bias);
    cudaFree(L->norm_w);   cudaFree(L->out_w); cudaFree(L->out_s);
    *L = {};
}

// Persistent per-call scratch (kept in caller).
struct LinearAttnScratch {
    float *d_h, *d_qkv_pre, *d_qkv_post, *d_z, *d_a, *d_b, *d_g, *d_beta;
    float *d_q_rep, *d_k_rep, *d_v_full, *d_core, *d_norm_out, *d_attn;
    float *d_conv_state;     // [3, QKV_DIM]
    float *d_delta_state;    // [LIN_NV, LIN_HEAD, LIN_HEAD]
};

inline void alloc_linear_attn_scratch(LinearAttnScratch* S) {
    CUDA_OK(cudaMalloc(&S->d_h,            H * 4));
    CUDA_OK(cudaMalloc(&S->d_qkv_pre,      QKV_DIM * 4));
    CUDA_OK(cudaMalloc(&S->d_qkv_post,     QKV_DIM * 4));
    CUDA_OK(cudaMalloc(&S->d_z,            Z_DIM * 4));
    CUDA_OK(cudaMalloc(&S->d_a,            LIN_NV * 4));
    CUDA_OK(cudaMalloc(&S->d_b,            LIN_NV * 4));
    CUDA_OK(cudaMalloc(&S->d_g,            LIN_NV * 4));
    CUDA_OK(cudaMalloc(&S->d_beta,         LIN_NV * 4));
    CUDA_OK(cudaMalloc(&S->d_q_rep,        LIN_VAL_TOT * 4));
    CUDA_OK(cudaMalloc(&S->d_k_rep,        LIN_VAL_TOT * 4));
    CUDA_OK(cudaMalloc(&S->d_v_full,       LIN_VAL_TOT * 4));
    CUDA_OK(cudaMalloc(&S->d_core,         LIN_VAL_TOT * 4));
    CUDA_OK(cudaMalloc(&S->d_norm_out,     LIN_VAL_TOT * 4));
    CUDA_OK(cudaMalloc(&S->d_attn,         H * 4));
    CUDA_OK(cudaMalloc(&S->d_conv_state,   3 * QKV_DIM * 4));
    CUDA_OK(cudaMalloc(&S->d_delta_state,  LIN_NV * LIN_HEAD * LIN_HEAD * 4));
}

inline void run_linear_attn(const LinearAttnW& L, uint32_t SEQ,
                            float* d_input, float* d_resid_out,
                            LinearAttnScratch* S)
{
    CUDA_OK(cudaMemset(S->d_conv_state,  0, 3 * QKV_DIM * 4));
    CUDA_OK(cudaMemset(S->d_delta_state, 0, LIN_NV * LIN_HEAD * LIN_HEAD * 4));
    const float Q_SCALE = 1.0f / std::sqrt((float)LIN_HEAD);

    for (uint32_t t = 0; t < SEQ; t++) {
        const float* x_t = d_input + (size_t)t * H;

        qwen36::launch_rms_norm_bf16_plus_one(x_t, L.in_ln_w, S->d_h, H, RMS_EPS);
        qwen36::launch_dequant_matvec_fp8_block128(L.qkv_w, L.qkv_s, S->d_h, S->d_qkv_pre, QKV_DIM, H);

        dim3 cb(256), cg((QKV_DIM + 255) / 256);
        conv1d_step<<<cg, cb>>>(S->d_conv_state, S->d_qkv_pre, L.conv_w, S->d_qkv_post, QKV_DIM);

        qwen36::launch_dequant_matvec_fp8_block128(L.z_w, L.z_s, S->d_h, S->d_z, Z_DIM, H);
        launch_matvec_bf16(L.a_w, S->d_h, S->d_a, LIN_NV, H);
        launch_matvec_bf16(L.b_w, S->d_h, S->d_b, LIN_NV, H);
        compute_decay_beta<<<1, LIN_NV>>>(S->d_a, S->d_b, L.A_log, L.dt_bias, S->d_g, S->d_beta);

        // q/k/v split + repeat-interleave
        CUDA_OK(cudaMemcpy2D(S->d_q_rep,                 2 * LIN_HEAD * 4,
                             S->d_qkv_post,                  LIN_HEAD * 4,
                             LIN_HEAD * 4, LIN_NK, cudaMemcpyDeviceToDevice));
        CUDA_OK(cudaMemcpy2D(S->d_q_rep + LIN_HEAD,      2 * LIN_HEAD * 4,
                             S->d_qkv_post,                  LIN_HEAD * 4,
                             LIN_HEAD * 4, LIN_NK, cudaMemcpyDeviceToDevice));
        CUDA_OK(cudaMemcpy2D(S->d_k_rep,                 2 * LIN_HEAD * 4,
                             S->d_qkv_post + LIN_KEY_TOT, LIN_HEAD * 4,
                             LIN_HEAD * 4, LIN_NK, cudaMemcpyDeviceToDevice));
        CUDA_OK(cudaMemcpy2D(S->d_k_rep + LIN_HEAD,      2 * LIN_HEAD * 4,
                             S->d_qkv_post + LIN_KEY_TOT, LIN_HEAD * 4,
                             LIN_HEAD * 4, LIN_NK, cudaMemcpyDeviceToDevice));
        CUDA_OK(cudaMemcpyAsync(S->d_v_full, S->d_qkv_post + 2 * LIN_KEY_TOT,
                                LIN_VAL_TOT * 4, cudaMemcpyDeviceToDevice));

        l2_norm_qk<<<LIN_NV, LIN_HEAD>>>(S->d_q_rep, S->d_k_rep, LIN_HEAD);
        vec_scale<<<(LIN_VAL_TOT + 255) / 256, 256>>>(S->d_q_rep, Q_SCALE, LIN_VAL_TOT);

        gated_delta_net_step<<<LIN_NV, LIN_HEAD>>>(
            S->d_delta_state, S->d_q_rep, S->d_k_rep, S->d_v_full,
            S->d_g, S->d_beta, S->d_core, /*kpv=*/1);
        gated_rms_norm<<<LIN_NV, LIN_HEAD>>>(S->d_core, S->d_z, L.norm_w, S->d_norm_out, LIN_HEAD, RMS_EPS);
        qwen36::launch_dequant_matvec_fp8_block128(L.out_w, L.out_s, S->d_norm_out, S->d_attn, H, LIN_VAL_TOT);

        launch_residual_add(x_t, S->d_attn, d_resid_out + (size_t)t * H, H);
    }
}

// ---------- full-attention bundle ----------
struct FullAttnW {
    uint16_t *in_ln_w  = nullptr;
    uint8_t  *q_w = nullptr;  __nv_bfloat16 *q_s = nullptr;
    uint8_t  *k_w = nullptr;  __nv_bfloat16 *k_s = nullptr;
    uint8_t  *v_w = nullptr;  __nv_bfloat16 *v_s = nullptr;
    uint8_t  *o_w = nullptr;  __nv_bfloat16 *o_s = nullptr;
    uint16_t *q_norm_w = nullptr;
    uint16_t *k_norm_w = nullptr;
};

inline void load_full_attn(st::ModelDir* M, const std::string& prefix, FullAttnW* F) {
    auto T = [&](const std::string& nm){ return prefix + "." + nm; };
    F->in_ln_w  = (uint16_t*)upload_bytes(M, T("input_layernorm.weight"));
    F->q_w      = upload_bytes(M, T("self_attn.q_proj.weight"));
    F->q_s      = (__nv_bfloat16*)upload_bytes(M, T("self_attn.q_proj.weight_scale_inv"));
    F->k_w      = upload_bytes(M, T("self_attn.k_proj.weight"));
    F->k_s      = (__nv_bfloat16*)upload_bytes(M, T("self_attn.k_proj.weight_scale_inv"));
    F->v_w      = upload_bytes(M, T("self_attn.v_proj.weight"));
    F->v_s      = (__nv_bfloat16*)upload_bytes(M, T("self_attn.v_proj.weight_scale_inv"));
    F->o_w      = upload_bytes(M, T("self_attn.o_proj.weight"));
    F->o_s      = (__nv_bfloat16*)upload_bytes(M, T("self_attn.o_proj.weight_scale_inv"));
    F->q_norm_w = (uint16_t*)upload_bytes(M, T("self_attn.q_norm.weight"));
    F->k_norm_w = (uint16_t*)upload_bytes(M, T("self_attn.k_norm.weight"));
}

inline void free_full_attn(FullAttnW* F) {
    cudaFree(F->in_ln_w);
    cudaFree(F->q_w); cudaFree(F->q_s);
    cudaFree(F->k_w); cudaFree(F->k_s);
    cudaFree(F->v_w); cudaFree(F->v_s);
    cudaFree(F->o_w); cudaFree(F->o_s);
    cudaFree(F->q_norm_w); cudaFree(F->k_norm_w);
    *F = {};
}

struct FullAttnScratch {
    float *d_h, *d_q_full, *d_k_full, *d_v_full, *d_Q, *d_GATE, *d_K_norm;
    float *d_K_cache, *d_V_cache, *d_Q_cache, *d_GATE_cache;
    float *d_scores, *d_attn_per_head, *d_attn_out;
    float *d_cos, *d_sin;
    uint32_t cos_seq;
};

inline void alloc_full_attn_scratch(FullAttnScratch* S, uint32_t SEQ) {
    CUDA_OK(cudaMalloc(&S->d_h,             H * 4));
    CUDA_OK(cudaMalloc(&S->d_q_full,        FA_Q_RAW_DIM * 4));
    CUDA_OK(cudaMalloc(&S->d_k_full,        FA_KV_DIM    * 4));
    CUDA_OK(cudaMalloc(&S->d_v_full,        FA_KV_DIM    * 4));
    CUDA_OK(cudaMalloc(&S->d_Q,             FA_Q_DIM     * 4));
    CUDA_OK(cudaMalloc(&S->d_GATE,          FA_Q_DIM     * 4));
    CUDA_OK(cudaMalloc(&S->d_K_norm,        FA_KV_DIM    * 4));
    CUDA_OK(cudaMalloc(&S->d_K_cache,       SEQ * FA_KV_DIM * 4));
    CUDA_OK(cudaMalloc(&S->d_V_cache,       SEQ * FA_KV_DIM * 4));
    CUDA_OK(cudaMalloc(&S->d_Q_cache,       SEQ * FA_Q_DIM  * 4));
    CUDA_OK(cudaMalloc(&S->d_GATE_cache,    SEQ * FA_Q_DIM  * 4));
    CUDA_OK(cudaMalloc(&S->d_scores,        FA_NQ * SEQ * 4));
    CUDA_OK(cudaMalloc(&S->d_attn_per_head, FA_Q_DIM * 4));
    CUDA_OK(cudaMalloc(&S->d_attn_out,      H * 4));
    CUDA_OK(cudaMalloc(&S->d_cos,           SEQ * ROPE_HALF * 4));
    CUDA_OK(cudaMalloc(&S->d_sin,           SEQ * ROPE_HALF * 4));
    std::vector<float> h_cos(SEQ * ROPE_HALF), h_sin(SEQ * ROPE_HALF);
    qwen36::rope_precompute_table(h_cos.data(), h_sin.data(), SEQ, ROPE_DIM, ROPE_THETA);
    CUDA_OK(cudaMemcpy(S->d_cos, h_cos.data(), h_cos.size() * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(S->d_sin, h_sin.data(), h_sin.size() * 4, cudaMemcpyHostToDevice));
    S->cos_seq = SEQ;
}

inline void run_full_attn(const FullAttnW& F, uint32_t SEQ,
                          float* d_input, float* d_resid_out,
                          FullAttnScratch* S)
{
    const float SCALE = 1.0f / std::sqrt((float)FA_HEAD_D);

    for (uint32_t t = 0; t < SEQ; t++) {
        const float* x_t = d_input + (size_t)t * H;
        qwen36::launch_rms_norm_bf16_plus_one(x_t, F.in_ln_w, S->d_h, H, RMS_EPS);

        qwen36::launch_dequant_matvec_fp8_block128(F.q_w, F.q_s, S->d_h, S->d_q_full, FA_Q_RAW_DIM, H);
        qwen36::launch_dequant_matvec_fp8_block128(F.k_w, F.k_s, S->d_h, S->d_k_full, FA_KV_DIM, H);
        qwen36::launch_dequant_matvec_fp8_block128(F.v_w, F.v_s, S->d_h, S->d_v_full, FA_KV_DIM, H);

        CUDA_OK(cudaMemcpy2D(S->d_Q,                  FA_HEAD_D * 4,
                             S->d_q_full,             FA_HEAD_D * 2 * 4,
                             FA_HEAD_D * 4, FA_NQ, cudaMemcpyDeviceToDevice));
        CUDA_OK(cudaMemcpy2D(S->d_GATE,               FA_HEAD_D * 4,
                             S->d_q_full + FA_HEAD_D, FA_HEAD_D * 2 * 4,
                             FA_HEAD_D * 4, FA_NQ, cudaMemcpyDeviceToDevice));

        qwen36::launch_rms_norm_per_row_plus_one(S->d_Q,      F.q_norm_w, S->d_Q,      FA_NQ,  FA_HEAD_D, RMS_EPS);
        qwen36::launch_rms_norm_per_row_plus_one(S->d_k_full, F.k_norm_w, S->d_K_norm, FA_NKV, FA_HEAD_D, RMS_EPS);

        const float* d_cos_t = S->d_cos + (size_t)t * ROPE_HALF;
        const float* d_sin_t = S->d_sin + (size_t)t * ROPE_HALF;
        qwen36::launch_rope_partial_inplace(S->d_Q,      d_cos_t, d_sin_t, FA_NQ,  FA_HEAD_D, ROPE_DIM);
        qwen36::launch_rope_partial_inplace(S->d_K_norm, d_cos_t, d_sin_t, FA_NKV, FA_HEAD_D, ROPE_DIM);

        CUDA_OK(cudaMemcpyAsync(S->d_K_cache    + (size_t)t * FA_KV_DIM, S->d_K_norm,  FA_KV_DIM * 4, cudaMemcpyDeviceToDevice));
        CUDA_OK(cudaMemcpyAsync(S->d_V_cache    + (size_t)t * FA_KV_DIM, S->d_v_full,  FA_KV_DIM * 4, cudaMemcpyDeviceToDevice));
        CUDA_OK(cudaMemcpyAsync(S->d_Q_cache    + (size_t)t * FA_Q_DIM,  S->d_Q,       FA_Q_DIM  * 4, cudaMemcpyDeviceToDevice));
        CUDA_OK(cudaMemcpyAsync(S->d_GATE_cache + (size_t)t * FA_Q_DIM,  S->d_GATE,    FA_Q_DIM  * 4, cudaMemcpyDeviceToDevice));
    }

    for (uint32_t t = 0; t < SEQ; t++) {
        const uint32_t L = t + 1;
        const float* d_Q_t    = S->d_Q_cache    + (size_t)t * FA_Q_DIM;
        const float* d_GATE_t = S->d_GATE_cache + (size_t)t * FA_Q_DIM;
        const float* x_t      = d_input         + (size_t)t * H;

        {
            dim3 grid(FA_NQ * L), block(256);
            attn_scores<<<grid, block>>>(d_Q_t, S->d_K_cache, S->d_scores,
                                         FA_HEAD_D, FA_KV_DIM, L, SEQ,
                                         SCALE, FA_HPK, L);
        }
        attn_softmax<<<FA_NQ, 256>>>(S->d_scores, L, SEQ);
        {
            dim3 block(256), grid((FA_Q_DIM + 255) / 256);
            attn_values<<<grid, block>>>(S->d_scores, S->d_V_cache, S->d_attn_per_head,
                                         FA_HEAD_D, FA_KV_DIM, L, SEQ, FA_HPK);
        }
        {
            dim3 block(256), grid((FA_Q_DIM + 255) / 256);
            sigmoid_gate<<<grid, block>>>(S->d_attn_per_head, d_GATE_t, FA_Q_DIM);
        }
        qwen36::launch_dequant_matvec_fp8_block128(F.o_w, F.o_s, S->d_attn_per_head, S->d_attn_out, H, FA_Q_DIM);
        launch_residual_add(x_t, S->d_attn_out, d_resid_out + (size_t)t * H, H);
    }
}

// ---------- MoE bundle ----------
struct ExpertW {
    uint8_t* gate_w = nullptr; __nv_bfloat16* gate_s = nullptr;
    uint8_t* up_w   = nullptr; __nv_bfloat16* up_s   = nullptr;
    uint8_t* down_w = nullptr; __nv_bfloat16* down_s = nullptr;
};

struct MoEW {
    uint16_t *post_ln_w     = nullptr;
    uint16_t *router_w      = nullptr;
    uint16_t *shared_gate_w = nullptr;
    std::vector<ExpertW> experts;
    ExpertW shared;
};

inline void load_moe(st::ModelDir* M, const std::string& prefix, MoEW* X) {
    auto T = [&](const std::string& nm){ return prefix + "." + nm; };
    X->post_ln_w     = (uint16_t*)upload_bytes(M, T("post_attention_layernorm.weight"));
    X->router_w      = (uint16_t*)upload_bytes(M, T("mlp.gate.weight"));
    X->shared_gate_w = (uint16_t*)upload_bytes(M, T("mlp.shared_expert_gate.weight"));
    X->experts.resize(N_EXPERTS);
    for (uint32_t e = 0; e < N_EXPERTS; e++) {
        char buf[256];
        auto exT = [&](const char* nm){ snprintf(buf, sizeof(buf), "mlp.experts.%u.%s", e, nm); return T(buf); };
        X->experts[e].gate_w = upload_bytes(M, exT("gate_proj.weight"));
        X->experts[e].gate_s = (__nv_bfloat16*)upload_bytes(M, exT("gate_proj.weight_scale_inv"));
        X->experts[e].up_w   = upload_bytes(M, exT("up_proj.weight"));
        X->experts[e].up_s   = (__nv_bfloat16*)upload_bytes(M, exT("up_proj.weight_scale_inv"));
        X->experts[e].down_w = upload_bytes(M, exT("down_proj.weight"));
        X->experts[e].down_s = (__nv_bfloat16*)upload_bytes(M, exT("down_proj.weight_scale_inv"));
    }
    X->shared.gate_w = upload_bytes(M, T("mlp.shared_expert.gate_proj.weight"));
    X->shared.gate_s = (__nv_bfloat16*)upload_bytes(M, T("mlp.shared_expert.gate_proj.weight_scale_inv"));
    X->shared.up_w   = upload_bytes(M, T("mlp.shared_expert.up_proj.weight"));
    X->shared.up_s   = (__nv_bfloat16*)upload_bytes(M, T("mlp.shared_expert.up_proj.weight_scale_inv"));
    X->shared.down_w = upload_bytes(M, T("mlp.shared_expert.down_proj.weight"));
    X->shared.down_s = (__nv_bfloat16*)upload_bytes(M, T("mlp.shared_expert.down_proj.weight_scale_inv"));
}

inline void free_moe(MoEW* X) {
    cudaFree(X->post_ln_w); cudaFree(X->router_w); cudaFree(X->shared_gate_w);
    for (auto& e : X->experts) {
        cudaFree(e.gate_w); cudaFree(e.gate_s);
        cudaFree(e.up_w);   cudaFree(e.up_s);
        cudaFree(e.down_w); cudaFree(e.down_s);
    }
    cudaFree(X->shared.gate_w); cudaFree(X->shared.gate_s);
    cudaFree(X->shared.up_w);   cudaFree(X->shared.up_s);
    cudaFree(X->shared.down_w); cudaFree(X->shared.down_s);
    *X = {};
}

struct MoEScratch {
    float *d_h, *d_router_logits, *d_gate_pre, *d_up_pre, *d_hidden_e, *d_down_e;
    float *d_expert_sum, *d_shared_out, *d_shared_gate_logit, *d_mlp_out;
};

inline void alloc_moe_scratch(MoEScratch* S) {
    CUDA_OK(cudaMalloc(&S->d_h,                 H * 4));
    CUDA_OK(cudaMalloc(&S->d_router_logits,     N_EXPERTS * 4));
    CUDA_OK(cudaMalloc(&S->d_gate_pre,          INTER * 4));
    CUDA_OK(cudaMalloc(&S->d_up_pre,            INTER * 4));
    CUDA_OK(cudaMalloc(&S->d_hidden_e,          INTER * 4));
    CUDA_OK(cudaMalloc(&S->d_down_e,            H * 4));
    CUDA_OK(cudaMalloc(&S->d_expert_sum,        H * 4));
    CUDA_OK(cudaMalloc(&S->d_shared_out,        H * 4));
    CUDA_OK(cudaMalloc(&S->d_shared_gate_logit, 4));
    CUDA_OK(cudaMalloc(&S->d_mlp_out,           H * 4));
}

inline void run_moe(const MoEW& X, uint32_t SEQ,
                    float* d_resid1, float* d_layer_out,
                    MoEScratch* S)
{
    for (uint32_t t = 0; t < SEQ; t++) {
        const float* x_t = d_resid1 + (size_t)t * H;
        qwen36::launch_rms_norm_bf16_plus_one(x_t, X.post_ln_w, S->d_h, H, RMS_EPS);
        launch_matvec_bf16(X.router_w, S->d_h, S->d_router_logits, N_EXPERTS, H);

        std::vector<float> logits(N_EXPERTS);
        CUDA_OK(cudaMemcpy(logits.data(), S->d_router_logits, N_EXPERTS * 4, cudaMemcpyDeviceToHost));
        float lmax = *std::max_element(logits.begin(), logits.end());
        std::vector<double> probs(N_EXPERTS); double psum = 0;
        for (uint32_t i = 0; i < N_EXPERTS; i++) { probs[i] = std::exp((double)logits[i] - lmax); psum += probs[i]; }
        for (uint32_t i = 0; i < N_EXPERTS; i++) probs[i] /= psum;
        std::vector<int> idx(N_EXPERTS);
        for (uint32_t i = 0; i < N_EXPERTS; i++) idx[i] = i;
        std::partial_sort(idx.begin(), idx.begin() + TOP_K, idx.end(),
                          [&](int a, int b){ return probs[a] > probs[b]; });
        std::vector<int>   topk_idx(TOP_K);
        std::vector<float> topk_w(TOP_K);
        double wsum = 0;
        for (uint32_t k = 0; k < TOP_K; k++) { topk_idx[k] = idx[k]; topk_w[k] = (float)probs[idx[k]]; wsum += topk_w[k]; }
        for (uint32_t k = 0; k < TOP_K; k++) topk_w[k] = (float)((double)topk_w[k] / wsum);

        CUDA_OK(cudaMemset(S->d_expert_sum, 0, H * 4));
        for (uint32_t k = 0; k < TOP_K; k++) {
            int e = topk_idx[k];
            qwen36::launch_dequant_matvec_fp8_block128(X.experts[e].gate_w, X.experts[e].gate_s, S->d_h, S->d_gate_pre, INTER, H);
            qwen36::launch_dequant_matvec_fp8_block128(X.experts[e].up_w,   X.experts[e].up_s,   S->d_h, S->d_up_pre,   INTER, H);
            launch_swiglu(S->d_gate_pre, S->d_up_pre, S->d_hidden_e, INTER);
            qwen36::launch_dequant_matvec_fp8_block128(X.experts[e].down_w, X.experts[e].down_s, S->d_hidden_e, S->d_down_e, H, INTER);
            int B = 256, G = (H + B - 1) / B;
            vec_scale<<<G, B>>>(S->d_down_e, topk_w[k], H);
            launch_residual_add(S->d_expert_sum, S->d_down_e, S->d_expert_sum, H);
        }

        qwen36::launch_dequant_matvec_fp8_block128(X.shared.gate_w, X.shared.gate_s, S->d_h, S->d_gate_pre, INTER, H);
        qwen36::launch_dequant_matvec_fp8_block128(X.shared.up_w,   X.shared.up_s,   S->d_h, S->d_up_pre,   INTER, H);
        launch_swiglu(S->d_gate_pre, S->d_up_pre, S->d_hidden_e, INTER);
        qwen36::launch_dequant_matvec_fp8_block128(X.shared.down_w, X.shared.down_s, S->d_hidden_e, S->d_shared_out, H, INTER);

        launch_matvec_bf16(X.shared_gate_w, S->d_h, S->d_shared_gate_logit, 1, H);
        float h_logit = 0;
        CUDA_OK(cudaMemcpy(&h_logit, S->d_shared_gate_logit, 4, cudaMemcpyDeviceToHost));
        float h_sig = 1.0f / (1.0f + std::exp(-h_logit));
        int B = 256, G = (H + B - 1) / B;
        vec_scale<<<G, B>>>(S->d_shared_out, h_sig, H);

        launch_residual_add(S->d_expert_sum, S->d_shared_out, S->d_mlp_out, H);
        launch_residual_add(x_t, S->d_mlp_out, d_layer_out + (size_t)t * H, H);
    }
}

} // namespace q36
