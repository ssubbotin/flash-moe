/*
 * test_qwen36_layer.cu — single-layer end-to-end validation.
 *
 * For one decoder layer (linear-attention or full-attention), runs:
 *
 *      h        = L{N}_input
 *      h1n      = rms_norm_plus_one(h, input_layernorm.w)
 *      attn_out = (linear_attn or self_attn)(h1n)
 *      resid1   = h + attn_out                                   ← vs L{N}_post_attn_resid
 *      h2n      = rms_norm_plus_one(resid1, post_attention_layernorm.w)
 *      mlp_out  = MoE(h2n)                                       ← vs L{N}_mlp_out
 *      out      = resid1 + mlp_out                               ← vs L{N}_post_layer
 *
 * Reuses kernels we've already validated independently:
 *   qwen36::rms_norm_bf16_plus_one, rms_norm_per_row_plus_one,
 *   rope_partial_inplace, dequant_matvec_fp8_block128
 *   kernels.cuh: matvec_bf16, conv1d_step, compute_decay_beta,
 *                l2_norm_qk, gated_delta_net_step, gated_rms_norm,
 *                attn_scores, attn_softmax, attn_values, sigmoid_gate,
 *                swiglu_fused, vec_scale, residual_add
 *
 * Build:  make test_qwen36_layer
 * Run:    ./test_qwen36_layer --model-dir /path/to/qwen36-fp8 \
 *                              --oracle    /path/to/qwen36-oracle/france_bin \
 *                              --layer 0     (or 3 for full-attn)
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "safetensors_io.cuh"
#include "qwen36_kernels.cuh"
#include "kernels.cuh"

#define CUDA_OK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA: %s @ %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    std::exit(1); } } while (0)

// ---------------------------------------------------------------------------
// Oracle bin loader (shared boilerplate)
// ---------------------------------------------------------------------------
struct OracleEntry { std::vector<uint64_t> shape; std::string dtype; uint64_t nbytes; };
struct Oracle      { std::string dir; std::unordered_map<std::string, OracleEntry> manifest; };

static bool oracle_open(Oracle* O, const std::string& dir) {
    O->dir = dir;
    std::string mpath = dir + "/manifest.json";
    FILE* f = std::fopen(mpath.c_str(), "rb");
    if (!f) { fprintf(stderr, "oracle: cannot open %s\n", mpath.c_str()); return false; }
    std::fseek(f, 0, SEEK_END); long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    std::string body(n, 0);
    if ((long)std::fread(&body[0], 1, n, f) != n) { std::fclose(f); return false; }
    std::fclose(f);
    size_t i = 0;
    while (i < body.size()) {
        size_t kq = body.find('"', i); if (kq == std::string::npos) break;
        size_t kqe = body.find('"', kq + 1);
        std::string key = body.substr(kq + 1, kqe - kq - 1);
        size_t obj_open = body.find('{', kqe); if (obj_open == std::string::npos) break;
        int depth = 1; size_t j = obj_open + 1;
        for (; j < body.size() && depth > 0; j++) {
            if (body[j] == '{') depth++; else if (body[j] == '}') depth--;
        }
        std::string obj = body.substr(obj_open, j - obj_open); i = j;
        OracleEntry e{};
        size_t dt = obj.find("\"dtype\":");
        if (dt != std::string::npos) {
            size_t lq = obj.find('"', dt + 8); size_t rq = obj.find('"', lq + 1);
            e.dtype = obj.substr(lq + 1, rq - lq - 1);
        }
        size_t sh = obj.find("\"shape\":");
        if (sh != std::string::npos) {
            size_t lb = obj.find('[', sh), rb = obj.find(']', lb);
            std::string b = obj.substr(lb + 1, rb - lb - 1);
            size_t k = 0;
            while (k < b.size()) {
                while (k < b.size() && (b[k] == ' ' || b[k] == ',')) k++;
                if (k >= b.size()) break;
                e.shape.push_back(std::strtoull(b.c_str() + k, nullptr, 10));
                while (k < b.size() && b[k] != ',') k++;
            }
        }
        size_t nb = obj.find("\"nbytes\":");
        if (nb != std::string::npos) e.nbytes = std::strtoull(obj.c_str() + nb + 9, nullptr, 10);
        if (!e.dtype.empty()) O->manifest.emplace(key, std::move(e));
    }
    fprintf(stderr, "oracle: %zu tensors loaded from %s\n", O->manifest.size(), dir.c_str());
    return true;
}

static bool oracle_read_f32(Oracle* O, const std::string& key, std::vector<float>& out) {
    auto it = O->manifest.find(key);
    if (it == O->manifest.end()) { fprintf(stderr, "oracle: missing %s\n", key.c_str()); return false; }
    if (it->second.dtype != "f32") return false;
    std::string p = O->dir + "/" + key + ".bin";
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) return false;
    out.resize(it->second.nbytes / 4);
    if (std::fread(out.data(), 1, it->second.nbytes, f) != it->second.nbytes) { std::fclose(f); return false; }
    std::fclose(f);
    return true;
}

static void compare(const char* tag, const float* d_actual, const float* h_ref,
                    size_t n, double l2rel_thresh = 0.05)
{
    std::vector<float> h_actual(n);
    CUDA_OK(cudaMemcpy(h_actual.data(), d_actual, n * sizeof(float), cudaMemcpyDeviceToHost));
    double max_abs = 0; double l2_a = 0, l2_r = 0, l2_d = 0; int worst = 0;
    for (size_t i = 0; i < n; i++) {
        double a = h_actual[i], r = h_ref[i], d = std::fabs(a - r);
        if (d > max_abs) { max_abs = d; worst = (int)i; }
        l2_a += a*a; l2_r += r*r; l2_d += d*d;
    }
    double l2rel = std::sqrt(l2_d) / (std::sqrt(l2_r) + 1e-9);
    const char* mark = (l2rel < l2rel_thresh ? "OK " : "DIFF");
    printf("  [%s] %-22s n=%zu  max_abs=%.4e L2rel=%.4e  L2(act)=%.3f L2(ref)=%.3f\n",
           mark, tag, n, max_abs, l2rel, std::sqrt(l2_a), std::sqrt(l2_r));
}

static uint8_t* upload_bytes(st::ModelDir* M, const std::string& name) {
    std::vector<uint8_t> buf;
    if (!st::read_bytes(M, name, buf)) std::exit(1);
    uint8_t* d; CUDA_OK(cudaMalloc(&d, buf.size()));
    CUDA_OK(cudaMemcpy(d, buf.data(), buf.size(), cudaMemcpyHostToDevice));
    return d;
}

// ---------------------------------------------------------------------------
// Constants for Qwen3.6-35B-A3B
// ---------------------------------------------------------------------------
static constexpr uint32_t H            = 2048;
static constexpr uint32_t QKV_DIM      = 8192;
static constexpr uint32_t Z_DIM        = 4096;
static constexpr uint32_t LIN_NV       = 32;
static constexpr uint32_t LIN_NK       = 16;
static constexpr uint32_t LIN_HEAD     = 128;
static constexpr uint32_t LIN_KEY_TOT  = LIN_NK * LIN_HEAD;     // 2048
static constexpr uint32_t LIN_VAL_TOT  = LIN_NV * LIN_HEAD;     // 4096
static constexpr uint32_t CONV_K       = 4;
static constexpr uint32_t FA_NQ        = 16;
static constexpr uint32_t FA_NKV       = 2;
static constexpr uint32_t FA_HEAD_D    = 256;
static constexpr uint32_t FA_Q_DIM     = FA_NQ * FA_HEAD_D;     // 4096
static constexpr uint32_t FA_Q_RAW_DIM = FA_NQ * FA_HEAD_D * 2; // 8192
static constexpr uint32_t FA_KV_DIM    = FA_NKV * FA_HEAD_D;    // 512
static constexpr uint32_t FA_HPK       = FA_NQ / FA_NKV;        // 8
static constexpr uint32_t ROPE_DIM     = FA_HEAD_D / 4;         // 64
static constexpr uint32_t ROPE_HALF    = ROPE_DIM / 2;          // 32
static constexpr float    ROPE_THETA   = 10000000.0f;
static constexpr float    RMS_EPS      = 1e-6f;
static constexpr uint32_t INTER        = 512;
static constexpr uint32_t N_EXPERTS    = 256;
static constexpr uint32_t TOP_K        = 8;

// ---------------------------------------------------------------------------
// Run linear-attention block for SEQ tokens, output residual (input + attn).
// Caller provides ora_input (host f32, [SEQ, H]).
// d_resid_out (device, [SEQ, H]) is filled.  Compares optional substep tags
// for token 0 only to keep the log small.
// ---------------------------------------------------------------------------
struct LinearAttnW {
    uint16_t *in_ln_w;
    uint8_t  *qkv_w; __nv_bfloat16 *qkv_s;
    uint8_t  *z_w;   __nv_bfloat16 *z_s;
    uint16_t *a_w;
    uint16_t *b_w;
    uint16_t *conv_w;
    float    *A_log;
    uint16_t *dt_bias;
    uint16_t *norm_w;
    uint8_t  *out_w; __nv_bfloat16 *out_s;
};

static void load_linear_attn(st::ModelDir* M, const std::string& prefix, LinearAttnW* L) {
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

static void run_linear_attn(const LinearAttnW& L, uint32_t SEQ,
                            const std::vector<float>& ora_input,
                            float* d_input,           // [SEQ, H] resident on device
                            float* d_resid_out,       // [SEQ, H]
                            // scratch
                            float* d_h, float* d_qkv_pre, float* d_qkv_post,
                            float* d_z, float* d_a, float* d_b,
                            float* d_g, float* d_beta,
                            float* d_q_rep, float* d_k_rep, float* d_v_full,
                            float* d_core, float* d_norm_out, float* d_attn,
                            float* d_conv_state, float* d_delta_state)
{
    CUDA_OK(cudaMemset(d_conv_state, 0, 3 * QKV_DIM * 4));
    CUDA_OK(cudaMemset(d_delta_state, 0, LIN_NV * LIN_HEAD * LIN_HEAD * 4));
    const float Q_SCALE = 1.0f / std::sqrt((float)LIN_HEAD);

    for (uint32_t t = 0; t < SEQ; t++) {
        const float* x_t = d_input + (size_t)t * H;
        qwen36::launch_rms_norm_bf16_plus_one(x_t, L.in_ln_w, d_h, H, RMS_EPS);

        qwen36::launch_dequant_matvec_fp8_block128(L.qkv_w, L.qkv_s, d_h, d_qkv_pre, QKV_DIM, H);

        dim3 cb(256), cg((QKV_DIM + 255) / 256);
        conv1d_step<<<cg, cb>>>(d_conv_state, d_qkv_pre, L.conv_w, d_qkv_post, QKV_DIM);

        qwen36::launch_dequant_matvec_fp8_block128(L.z_w,  L.z_s,  d_h, d_z, Z_DIM, H);
        launch_matvec_bf16(L.a_w, d_h, d_a, LIN_NV, H);
        launch_matvec_bf16(L.b_w, d_h, d_b, LIN_NV, H);
        compute_decay_beta<<<1, LIN_NV>>>(d_a, d_b, L.A_log, L.dt_bias, d_g, d_beta);

        // q/k/v split + repeat-interleave (chunked: heads 2h and 2h+1 share head h)
        CUDA_OK(cudaMemcpy2D(d_q_rep,                 2 * LIN_HEAD * 4,
                             d_qkv_post,                  LIN_HEAD * 4,
                             LIN_HEAD * 4, LIN_NK, cudaMemcpyDeviceToDevice));
        CUDA_OK(cudaMemcpy2D(d_q_rep + LIN_HEAD,      2 * LIN_HEAD * 4,
                             d_qkv_post,                  LIN_HEAD * 4,
                             LIN_HEAD * 4, LIN_NK, cudaMemcpyDeviceToDevice));
        CUDA_OK(cudaMemcpy2D(d_k_rep,                 2 * LIN_HEAD * 4,
                             d_qkv_post + LIN_KEY_TOT, LIN_HEAD * 4,
                             LIN_HEAD * 4, LIN_NK, cudaMemcpyDeviceToDevice));
        CUDA_OK(cudaMemcpy2D(d_k_rep + LIN_HEAD,      2 * LIN_HEAD * 4,
                             d_qkv_post + LIN_KEY_TOT, LIN_HEAD * 4,
                             LIN_HEAD * 4, LIN_NK, cudaMemcpyDeviceToDevice));
        CUDA_OK(cudaMemcpyAsync(d_v_full, d_qkv_post + 2 * LIN_KEY_TOT,
                                LIN_VAL_TOT * 4, cudaMemcpyDeviceToDevice));

        l2_norm_qk<<<LIN_NV, LIN_HEAD>>>(d_q_rep, d_k_rep, LIN_HEAD);
        vec_scale<<<(LIN_VAL_TOT + 255) / 256, 256>>>(d_q_rep, Q_SCALE, LIN_VAL_TOT);

        gated_delta_net_step<<<LIN_NV, LIN_HEAD>>>(
            d_delta_state, d_q_rep, d_k_rep, d_v_full, d_g, d_beta, d_core, /*kpv=*/1);
        gated_rms_norm<<<LIN_NV, LIN_HEAD>>>(d_core, d_z, L.norm_w, d_norm_out, LIN_HEAD, RMS_EPS);
        qwen36::launch_dequant_matvec_fp8_block128(L.out_w, L.out_s, d_norm_out, d_attn, H, LIN_VAL_TOT);

        launch_residual_add(x_t, d_attn, d_resid_out + (size_t)t * H, H);
    }
}

// ---------------------------------------------------------------------------
// Full-attention block: runs SEQ tokens (prefill), output residual.
// ---------------------------------------------------------------------------
struct FullAttnW {
    uint16_t *in_ln_w;
    uint8_t  *q_w;  __nv_bfloat16 *q_s;
    uint8_t  *k_w;  __nv_bfloat16 *k_s;
    uint8_t  *v_w;  __nv_bfloat16 *v_s;
    uint8_t  *o_w;  __nv_bfloat16 *o_s;
    uint16_t *q_norm_w;
    uint16_t *k_norm_w;
};

static void load_full_attn(st::ModelDir* M, const std::string& prefix, FullAttnW* F) {
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

static void run_full_attn(const FullAttnW& F, uint32_t SEQ,
                          float* d_input, float* d_resid_out,
                          float* d_h, float* d_q_full, float* d_k_full, float* d_v_full,
                          float* d_Q, float* d_GATE, float* d_K_norm,
                          float* d_K_cache, float* d_V_cache,
                          float* d_Q_cache, float* d_GATE_cache,
                          float* d_scores, float* d_attn_per_head, float* d_attn_out,
                          float* d_cos, float* d_sin)
{
    const float SCALE = 1.0f / std::sqrt((float)FA_HEAD_D);

    // Pass 1: per-token projections, q/k norm, RoPE, fill caches
    for (uint32_t t = 0; t < SEQ; t++) {
        const float* x_t = d_input + (size_t)t * H;
        qwen36::launch_rms_norm_bf16_plus_one(x_t, F.in_ln_w, d_h, H, RMS_EPS);

        qwen36::launch_dequant_matvec_fp8_block128(F.q_w, F.q_s, d_h, d_q_full, FA_Q_RAW_DIM, H);
        qwen36::launch_dequant_matvec_fp8_block128(F.k_w, F.k_s, d_h, d_k_full, FA_KV_DIM, H);
        qwen36::launch_dequant_matvec_fp8_block128(F.v_w, F.v_s, d_h, d_v_full, FA_KV_DIM, H);

        CUDA_OK(cudaMemcpy2D(d_Q,                  FA_HEAD_D * 4,
                             d_q_full,             FA_HEAD_D * 2 * 4,
                             FA_HEAD_D * 4, FA_NQ, cudaMemcpyDeviceToDevice));
        CUDA_OK(cudaMemcpy2D(d_GATE,               FA_HEAD_D * 4,
                             d_q_full + FA_HEAD_D, FA_HEAD_D * 2 * 4,
                             FA_HEAD_D * 4, FA_NQ, cudaMemcpyDeviceToDevice));

        qwen36::launch_rms_norm_per_row_plus_one(d_Q,      F.q_norm_w, d_Q,      FA_NQ,  FA_HEAD_D, RMS_EPS);
        qwen36::launch_rms_norm_per_row_plus_one(d_k_full, F.k_norm_w, d_K_norm, FA_NKV, FA_HEAD_D, RMS_EPS);

        const float* d_cos_t = d_cos + (size_t)t * ROPE_HALF;
        const float* d_sin_t = d_sin + (size_t)t * ROPE_HALF;
        qwen36::launch_rope_partial_inplace(d_Q,      d_cos_t, d_sin_t, FA_NQ,  FA_HEAD_D, ROPE_DIM);
        qwen36::launch_rope_partial_inplace(d_K_norm, d_cos_t, d_sin_t, FA_NKV, FA_HEAD_D, ROPE_DIM);

        CUDA_OK(cudaMemcpyAsync(d_K_cache    + (size_t)t * FA_KV_DIM, d_K_norm, FA_KV_DIM * 4, cudaMemcpyDeviceToDevice));
        CUDA_OK(cudaMemcpyAsync(d_V_cache    + (size_t)t * FA_KV_DIM, d_v_full, FA_KV_DIM * 4, cudaMemcpyDeviceToDevice));
        CUDA_OK(cudaMemcpyAsync(d_Q_cache    + (size_t)t * FA_Q_DIM,  d_Q,      FA_Q_DIM  * 4, cudaMemcpyDeviceToDevice));
        CUDA_OK(cudaMemcpyAsync(d_GATE_cache + (size_t)t * FA_Q_DIM,  d_GATE,   FA_Q_DIM  * 4, cudaMemcpyDeviceToDevice));
    }

    // Pass 2: per-query attention + gate + o_proj + residual
    for (uint32_t t = 0; t < SEQ; t++) {
        const uint32_t L = t + 1;
        const float* d_Q_t    = d_Q_cache    + (size_t)t * FA_Q_DIM;
        const float* d_GATE_t = d_GATE_cache + (size_t)t * FA_Q_DIM;
        const float* x_t      = d_input      + (size_t)t * H;

        {
            dim3 grid(FA_NQ * L), block(256);
            attn_scores<<<grid, block>>>(d_Q_t, d_K_cache, d_scores,
                                         FA_HEAD_D, FA_KV_DIM, L, /*stride=*/SEQ,
                                         SCALE, FA_HPK, /*num_seq_tgs=*/L);
        }
        attn_softmax<<<FA_NQ, 256>>>(d_scores, L, /*stride=*/SEQ);
        {
            dim3 block(256), grid((FA_Q_DIM + 255) / 256);
            attn_values<<<grid, block>>>(d_scores, d_V_cache, d_attn_per_head,
                                         FA_HEAD_D, FA_KV_DIM, L, /*stride=*/SEQ, FA_HPK);
        }
        {
            dim3 block(256), grid((FA_Q_DIM + 255) / 256);
            sigmoid_gate<<<grid, block>>>(d_attn_per_head, d_GATE_t, FA_Q_DIM);
        }
        qwen36::launch_dequant_matvec_fp8_block128(F.o_w, F.o_s, d_attn_per_head, d_attn_out, H, FA_Q_DIM);
        launch_residual_add(x_t, d_attn_out, d_resid_out + (size_t)t * H, H);
    }
}

// ---------------------------------------------------------------------------
// MoE block (same as test_qwen36_moe.cu) — produces post-MoE residual.
// ---------------------------------------------------------------------------
struct ExpertW { uint8_t* gate_w; __nv_bfloat16* gate_s;
                 uint8_t* up_w;   __nv_bfloat16* up_s;
                 uint8_t* down_w; __nv_bfloat16* down_s; };
struct MoEW {
    uint16_t *post_ln_w;
    uint16_t *router_w;
    uint16_t *shared_gate_w;
    std::vector<ExpertW> experts;
    ExpertW shared;
};

static void load_moe(st::ModelDir* M, const std::string& prefix, MoEW* X) {
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

static void run_moe(const MoEW& X, uint32_t SEQ,
                    float* d_resid1, float* d_layer_out,
                    float* d_h, float* d_router_logits,
                    float* d_gate_pre, float* d_up_pre, float* d_hidden_e, float* d_down_e,
                    float* d_expert_sum, float* d_shared_out, float* d_shared_gate_logit,
                    float* d_mlp_out)
{
    for (uint32_t t = 0; t < SEQ; t++) {
        const float* x_t = d_resid1 + (size_t)t * H;
        qwen36::launch_rms_norm_bf16_plus_one(x_t, X.post_ln_w, d_h, H, RMS_EPS);

        launch_matvec_bf16(X.router_w, d_h, d_router_logits, N_EXPERTS, H);

        std::vector<float> logits(N_EXPERTS);
        CUDA_OK(cudaMemcpy(logits.data(), d_router_logits, N_EXPERTS * 4, cudaMemcpyDeviceToHost));
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

        CUDA_OK(cudaMemset(d_expert_sum, 0, H * 4));
        for (uint32_t k = 0; k < TOP_K; k++) {
            int e = topk_idx[k];
            qwen36::launch_dequant_matvec_fp8_block128(X.experts[e].gate_w, X.experts[e].gate_s, d_h, d_gate_pre, INTER, H);
            qwen36::launch_dequant_matvec_fp8_block128(X.experts[e].up_w,   X.experts[e].up_s,   d_h, d_up_pre,   INTER, H);
            launch_swiglu(d_gate_pre, d_up_pre, d_hidden_e, INTER);
            qwen36::launch_dequant_matvec_fp8_block128(X.experts[e].down_w, X.experts[e].down_s, d_hidden_e, d_down_e, H, INTER);
            int B = 256, G = (H + B - 1) / B;
            vec_scale<<<G, B>>>(d_down_e, topk_w[k], H);
            launch_residual_add(d_expert_sum, d_down_e, d_expert_sum, H);
        }

        qwen36::launch_dequant_matvec_fp8_block128(X.shared.gate_w, X.shared.gate_s, d_h, d_gate_pre, INTER, H);
        qwen36::launch_dequant_matvec_fp8_block128(X.shared.up_w,   X.shared.up_s,   d_h, d_up_pre,   INTER, H);
        launch_swiglu(d_gate_pre, d_up_pre, d_hidden_e, INTER);
        qwen36::launch_dequant_matvec_fp8_block128(X.shared.down_w, X.shared.down_s, d_hidden_e, d_shared_out, H, INTER);

        launch_matvec_bf16(X.shared_gate_w, d_h, d_shared_gate_logit, 1, H);
        float h_logit = 0;
        CUDA_OK(cudaMemcpy(&h_logit, d_shared_gate_logit, 4, cudaMemcpyDeviceToHost));
        float h_sig = 1.0f / (1.0f + std::exp(-h_logit));
        int B = 256, G = (H + B - 1) / B;
        vec_scale<<<G, B>>>(d_shared_out, h_sig, H);

        launch_residual_add(d_expert_sum, d_shared_out, d_mlp_out, H);
        launch_residual_add(x_t, d_mlp_out, d_layer_out + (size_t)t * H, H);
    }
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    std::string model_dir, oracle_dir;
    int layer_idx = 0;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--model-dir" && i + 1 < argc) model_dir = argv[++i];
        else if (a == "--oracle"    && i + 1 < argc) oracle_dir = argv[++i];
        else if (a == "--layer"     && i + 1 < argc) layer_idx = std::atoi(argv[++i]);
        else { fprintf(stderr, "usage: %s --model-dir DIR --oracle DIR [--layer N]\n", argv[0]); return 1; }
    }
    if (model_dir.empty() || oracle_dir.empty()) return 1;

    cudaDeviceProp p{}; cudaGetDeviceProperties(&p, 0);
    bool full_attn = (layer_idx % 4 == 3);
    printf("device: %s sm_%d%d  layer=%d (%s)\n", p.name, p.major, p.minor,
           layer_idx, full_attn ? "full_attention" : "linear_attention");

    st::ModelDir M{}; if (!st::open(&M, model_dir)) return 1;
    Oracle O{}; if (!oracle_open(&O, oracle_dir)) return 1;

    char prefix[128];
    snprintf(prefix, sizeof(prefix), "model.language_model.layers.%d", layer_idx);

    char tag[64];
    auto K = [&](const char* sub){ snprintf(tag, sizeof(tag), "L%02d_%s", layer_idx, sub); return std::string(tag); };

    std::vector<float> ora_input, ora_resid, ora_layer_out;
    if (!oracle_read_f32(&O, K("input"),           ora_input))     return 1;
    if (!oracle_read_f32(&O, K("post_attn_resid"), ora_resid))     return 1;
    if (!oracle_read_f32(&O, K("post_layer"),      ora_layer_out)) return 1;

    const auto& sh = O.manifest[K("input")].shape;
    const uint32_t SEQ = (uint32_t)sh[1];
    printf("seq_len = %u\n", SEQ);

    // ---- upload input to device ----
    float *d_input, *d_resid1, *d_layer_out;
    CUDA_OK(cudaMalloc(&d_input,     SEQ * H * 4));
    CUDA_OK(cudaMalloc(&d_resid1,    SEQ * H * 4));
    CUDA_OK(cudaMalloc(&d_layer_out, SEQ * H * 4));
    CUDA_OK(cudaMemcpy(d_input, ora_input.data(), SEQ * H * 4, cudaMemcpyHostToDevice));

    // ---- shared scratch ----
    float *d_h, *d_router_logits, *d_gate_pre, *d_up_pre, *d_hidden_e, *d_down_e;
    float *d_expert_sum, *d_shared_out, *d_shared_gate_logit, *d_mlp_out;
    CUDA_OK(cudaMalloc(&d_h,                H * 4));
    CUDA_OK(cudaMalloc(&d_router_logits,    N_EXPERTS * 4));
    CUDA_OK(cudaMalloc(&d_gate_pre,         INTER * 4));
    CUDA_OK(cudaMalloc(&d_up_pre,           INTER * 4));
    CUDA_OK(cudaMalloc(&d_hidden_e,         INTER * 4));
    CUDA_OK(cudaMalloc(&d_down_e,           H * 4));
    CUDA_OK(cudaMalloc(&d_expert_sum,       H * 4));
    CUDA_OK(cudaMalloc(&d_shared_out,       H * 4));
    CUDA_OK(cudaMalloc(&d_shared_gate_logit,4));
    CUDA_OK(cudaMalloc(&d_mlp_out,          H * 4));

    if (full_attn) {
        printf("loading full-attn weights...\n");
        FullAttnW F{}; load_full_attn(&M, prefix, &F);

        // Per-attn scratch
        float *d_q_full, *d_k_full, *d_v_full, *d_Q, *d_GATE, *d_K_norm;
        float *d_K_cache, *d_V_cache, *d_Q_cache, *d_GATE_cache;
        float *d_scores, *d_attn_per_head, *d_attn_out;
        float *d_cos, *d_sin;
        CUDA_OK(cudaMalloc(&d_q_full,         FA_Q_RAW_DIM * 4));
        CUDA_OK(cudaMalloc(&d_k_full,         FA_KV_DIM    * 4));
        CUDA_OK(cudaMalloc(&d_v_full,         FA_KV_DIM    * 4));
        CUDA_OK(cudaMalloc(&d_Q,              FA_Q_DIM     * 4));
        CUDA_OK(cudaMalloc(&d_GATE,           FA_Q_DIM     * 4));
        CUDA_OK(cudaMalloc(&d_K_norm,         FA_KV_DIM    * 4));
        CUDA_OK(cudaMalloc(&d_K_cache,        SEQ * FA_KV_DIM * 4));
        CUDA_OK(cudaMalloc(&d_V_cache,        SEQ * FA_KV_DIM * 4));
        CUDA_OK(cudaMalloc(&d_Q_cache,        SEQ * FA_Q_DIM  * 4));
        CUDA_OK(cudaMalloc(&d_GATE_cache,     SEQ * FA_Q_DIM  * 4));
        CUDA_OK(cudaMalloc(&d_scores,         FA_NQ * SEQ * 4));
        CUDA_OK(cudaMalloc(&d_attn_per_head,  FA_Q_DIM * 4));
        CUDA_OK(cudaMalloc(&d_attn_out,       H * 4));
        CUDA_OK(cudaMalloc(&d_cos,            SEQ * ROPE_HALF * 4));
        CUDA_OK(cudaMalloc(&d_sin,            SEQ * ROPE_HALF * 4));
        std::vector<float> h_cos(SEQ * ROPE_HALF), h_sin(SEQ * ROPE_HALF);
        qwen36::rope_precompute_table(h_cos.data(), h_sin.data(), SEQ, ROPE_DIM, ROPE_THETA);
        CUDA_OK(cudaMemcpy(d_cos, h_cos.data(), h_cos.size() * 4, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_sin, h_sin.data(), h_sin.size() * 4, cudaMemcpyHostToDevice));

        run_full_attn(F, SEQ, d_input, d_resid1,
                      d_h, d_q_full, d_k_full, d_v_full,
                      d_Q, d_GATE, d_K_norm,
                      d_K_cache, d_V_cache, d_Q_cache, d_GATE_cache,
                      d_scores, d_attn_per_head, d_attn_out,
                      d_cos, d_sin);
    } else {
        printf("loading linear-attn weights...\n");
        LinearAttnW Lw{}; load_linear_attn(&M, prefix, &Lw);

        float *d_qkv_pre, *d_qkv_post, *d_z, *d_a, *d_b, *d_g, *d_beta;
        float *d_q_rep, *d_k_rep, *d_v_full, *d_core, *d_norm_out, *d_attn;
        float *d_conv_state, *d_delta_state;
        CUDA_OK(cudaMalloc(&d_qkv_pre,    QKV_DIM * 4));
        CUDA_OK(cudaMalloc(&d_qkv_post,   QKV_DIM * 4));
        CUDA_OK(cudaMalloc(&d_z,          Z_DIM * 4));
        CUDA_OK(cudaMalloc(&d_a,          LIN_NV * 4));
        CUDA_OK(cudaMalloc(&d_b,          LIN_NV * 4));
        CUDA_OK(cudaMalloc(&d_g,          LIN_NV * 4));
        CUDA_OK(cudaMalloc(&d_beta,       LIN_NV * 4));
        CUDA_OK(cudaMalloc(&d_q_rep,      LIN_VAL_TOT * 4));
        CUDA_OK(cudaMalloc(&d_k_rep,      LIN_VAL_TOT * 4));
        CUDA_OK(cudaMalloc(&d_v_full,     LIN_VAL_TOT * 4));
        CUDA_OK(cudaMalloc(&d_core,       LIN_VAL_TOT * 4));
        CUDA_OK(cudaMalloc(&d_norm_out,   LIN_VAL_TOT * 4));
        CUDA_OK(cudaMalloc(&d_attn,       H * 4));
        CUDA_OK(cudaMalloc(&d_conv_state, 3 * QKV_DIM * 4));
        CUDA_OK(cudaMalloc(&d_delta_state,LIN_NV * LIN_HEAD * LIN_HEAD * 4));

        run_linear_attn(Lw, SEQ, ora_input, d_input, d_resid1,
                        d_h, d_qkv_pre, d_qkv_post, d_z, d_a, d_b, d_g, d_beta,
                        d_q_rep, d_k_rep, d_v_full, d_core, d_norm_out, d_attn,
                        d_conv_state, d_delta_state);
    }

    // Compare residual after attention against oracle
    for (uint32_t t = 0; t < SEQ; t++) {
        char tg[32]; snprintf(tg, sizeof(tg), "post_attn_resid t=%u", t);
        compare(tg, d_resid1 + (size_t)t * H, ora_resid.data() + t * H, H);
    }

    printf("\nloading MoE block (256 experts + shared)...\n");
    MoEW X{}; load_moe(&M, prefix, &X);

    run_moe(X, SEQ, d_resid1, d_layer_out,
            d_h, d_router_logits, d_gate_pre, d_up_pre, d_hidden_e, d_down_e,
            d_expert_sum, d_shared_out, d_shared_gate_logit, d_mlp_out);

    // Final layer output vs oracle
    for (uint32_t t = 0; t < SEQ; t++) {
        char tg[32]; snprintf(tg, sizeof(tg), "post_layer t=%u", t);
        compare(tg, d_layer_out + (size_t)t * H, ora_layer_out.data() + t * H, H);
    }

    st::close(&M);
    printf("\ndone.\n");
    return 0;
}
