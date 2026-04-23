/*
 * mla_forward.cuh — Multi-head Latent Attention forward path (decode-only)
 *
 * Usage:
 *
 *   MLAConfig cfg = {...};                               // from config.json
 *   MLALayer  L   = mla_layer_alloc(cfg, MAX_SEQ_LEN);
 *   mla_layer_upload_weights(&L, cfg, host_weights...);   // absorbs W_UK into Q,
 *                                                         // absorbs W_UV into O.
 *   mla_yarn_tables_precompute(cfg, MAX_SEQ_LEN, &h_cos, &h_sin);
 *   cudaMemcpy(d_cos_table, h_cos, ...);
 *
 *   // per decode step:
 *   mla_attention_step(cfg, &L, d_cos_at_pos, d_sin_at_pos,
 *                      d_hidden_in, pos, d_layer_out, stream);
 *
 * All device buffers are bf16 for weights and KV cache, f32 for activations and
 * accumulators. Matches the reference that test_mla.cu verified to ≤0.5% rel err.
 */
#pragma once

#include "kernels.cuh"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifndef MLA_RMS_EPS
#define MLA_RMS_EPS 1e-5f
#endif

struct MLAConfig {
    int H;               // hidden_size       (e.g. 7168)
    int num_heads;       // (e.g. 64)
    int q_lora_rank;     // (e.g. 1536)
    int kv_lora_rank;    // (e.g. 512)
    int qk_nope_head_dim;// (e.g. 128)
    int qk_rope_head_dim;// (e.g. 64)
    int v_head_dim;      // (e.g. 128)
    float rope_theta;    // (e.g. 50000)
    float yarn_factor;   // (e.g. 64)
    float yarn_beta_fast;// (e.g. 32)
    float yarn_beta_slow;// (e.g. 1)
    int   yarn_orig_max; // (e.g. 4096)
    float yarn_mscale;   // (e.g. 1.0)
    float yarn_mscale_all;// (e.g. 1.0)
    float softmax_scale; // precomputed: 1/sqrt(qk_nope+qk_rope) * get_mscale(f, mscale_all)^2
};

struct MLALayer {
    // Raw weights (bf16, device pointers)
    uint16_t* d_Wqa;          // [q_lora_rank, H]
    uint16_t* d_qa_norm;      // [q_lora_rank]
    uint16_t* d_Wqb_rope;     // [num_heads * qk_rope_head_dim, q_lora_rank]
    uint16_t* d_Wkva;         // [kv_lora_rank + qk_rope_head_dim, H]
    uint16_t* d_kva_norm;     // [kv_lora_rank]

    // Absorbed weights (bf16, device pointers) — computed at load time
    uint16_t* d_W_Q_abs;      // [num_heads * kv_lora_rank, q_lora_rank]
                              //   row (h*kvlora + k) = Wqb_nope[h,:,:].T @ Wkvb_k[h,:,k]
    uint16_t* d_W_O_abs;      // [H, num_heads * kv_lora_rank]
                              //   row o = Σ_h Wkvb_v[h,:,:].T @ Wo_row_of_h_block[o]

    // KV cache (bf16)
    uint16_t* d_kv_cache;     // [max_seq, kv_lora_rank]
    uint16_t* d_kr_cache;     // [max_seq, qk_rope_head_dim]
    int       cache_len;      // number of filled slots

    // Per-step scratch (f32)
    float* d_q_lora;          // [q_lora_rank]
    float* d_q_lora_n;        // [q_lora_rank]  (post RMS norm)
    float* d_q_abs;           // [num_heads, kv_lora_rank]
    float* d_q_rope;          // [num_heads, qk_rope_head_dim]
    float* d_kv_lat;          // [kv_lora_rank + qk_rope_head_dim]
    float* d_kv_comp;         // [kv_lora_rank]  (kv_lat[:kvlora] post RMS norm)
    float* d_attn_out;        // [num_heads, kv_lora_rank]
};

// Convenience: allocate all per-step scratch buffers in an MLALayer.
// Weights and caches are caller's responsibility (sizes depend on shard layout).
static inline void mla_layer_alloc_scratch(MLALayer* L, const MLAConfig& cfg) {
    int qlora  = cfg.q_lora_rank;
    int nh     = cfg.num_heads;
    int kvlora = cfg.kv_lora_rank;
    int d_rope = cfg.qk_rope_head_dim;
    cudaMalloc(&L->d_q_lora,   (size_t)qlora * 4);
    cudaMalloc(&L->d_q_lora_n, (size_t)qlora * 4);
    cudaMalloc(&L->d_q_abs,    (size_t)nh * kvlora * 4);
    cudaMalloc(&L->d_q_rope,   (size_t)nh * d_rope * 4);
    cudaMalloc(&L->d_kv_lat,   (size_t)(kvlora + d_rope) * 4);
    cudaMalloc(&L->d_kv_comp,  (size_t)kvlora * 4);
    cudaMalloc(&L->d_attn_out, (size_t)nh * kvlora * 4);
}

static inline void mla_layer_free_scratch(MLALayer* L) {
    cudaFree(L->d_q_lora);    cudaFree(L->d_q_lora_n);
    cudaFree(L->d_q_abs);     cudaFree(L->d_q_rope);
    cudaFree(L->d_kv_lat);    cudaFree(L->d_kv_comp);
    cudaFree(L->d_attn_out);
}

// ------------------------------------------------------------------------------
// Yarn RoPE helpers (host) — produce per-position cos/sin tables
// ------------------------------------------------------------------------------

static inline double mla_get_mscale(double factor, double m) {
    return factor <= 1 ? 1.0 : 0.1 * m * std::log(factor) + 1.0;
}

static inline double mla_find_corr_dim(double num_rot, int dim, double base, int orig) {
    return dim * std::log((double)orig / (num_rot * 2.0 * M_PI)) / (2.0 * std::log(base));
}

// Fill cos_out/sin_out with shape [max_seq, rope_dim/2].
// Caller uploads to GPU.
static inline void mla_yarn_tables_precompute(
    const MLAConfig& c, int max_seq,
    std::vector<float>& cos_out, std::vector<float>& sin_out)
{
    int dim   = c.qk_rope_head_dim;
    int half  = dim / 2;
    double base   = c.rope_theta;
    double factor = c.yarn_factor;

    std::vector<double> freqs(half);
    for (int i = 0; i < half; i++)
        freqs[i] = 1.0 / std::pow(base, (2.0 * i) / dim);

    double lo_d = mla_find_corr_dim(c.yarn_beta_fast, dim, base, c.yarn_orig_max);
    double hi_d = mla_find_corr_dim(c.yarn_beta_slow, dim, base, c.yarn_orig_max);
    int lo = std::max(0, (int)std::floor(lo_d));
    int hi = std::min(dim - 1, (int)std::ceil(hi_d));
    double lo_h = lo / 2.0, hi_h = hi / 2.0;
    double denom = std::max(hi_h - lo_h, 0.001);

    // Effective inv_freq: ramp between interpolation (factor-scaled) and extrapolation (original)
    std::vector<double> inv_freq(half);
    for (int i = 0; i < half; i++) {
        double r = (i - lo_h) / denom;
        r = std::max(0.0, std::min(1.0, r));
        inv_freq[i] = freqs[i] * (1.0 - r) / factor + freqs[i] * r;
    }

    double mscale_embed = mla_get_mscale(factor, c.yarn_mscale)
                        * mla_get_mscale(factor, c.yarn_mscale_all);

    cos_out.resize((size_t)max_seq * half);
    sin_out.resize((size_t)max_seq * half);
    for (int p = 0; p < max_seq; p++) {
        for (int i = 0; i < half; i++) {
            double angle = p * inv_freq[i];
            cos_out[(size_t)p * half + i] = (float)(std::cos(angle) * mscale_embed);
            sin_out[(size_t)p * half + i] = (float)(std::sin(angle) * mscale_embed);
        }
    }
}

static inline float mla_effective_softmax_scale(const MLAConfig& c) {
    float base = 1.0f / std::sqrt((float)(c.qk_nope_head_dim + c.qk_rope_head_dim));
    if (c.yarn_mscale_all != 0.0f) {
        double m = mla_get_mscale(c.yarn_factor, c.yarn_mscale_all);
        base *= (float)(m * m);
    }
    return base;
}

// ------------------------------------------------------------------------------
// Host-side weight absorption: compute W_Q_abs and W_O_abs from raw Wqb + Wkvb + Wo
// All inputs are f32 (caller decoded bf16 to f32 before calling). Outputs are bf16.
//
// W_Q_abs[h*kvlora + k, q] = Σ_d Wqb_nope[h, d, q] * Wkvb_k[h, d, k]
// W_O_abs[o, h*kvlora + k] = Σ_d Wkvb_v[h, d, k] * Wo[o, h*v_head + d]
//
// Layouts (caller-provided, all f32):
//   Wqb_full    : [num_heads, qk_nope + qk_rope, q_lora] — from q_b_proj
//   Wkvb_full   : [num_heads, qk_nope + v_head, kv_lora] — from kv_b_proj
//   Wo          : [H, num_heads * v_head]                — from o_proj
// Outputs (caller-allocated, f32 round-trippable to bf16 later):
//   W_Q_abs_out : [num_heads * kv_lora, q_lora]
//   W_O_abs_out : [H, num_heads * kv_lora]
// ------------------------------------------------------------------------------
static inline void mla_absorb_weights(
    const MLAConfig& c,
    const float* Wqb_full,    // [nh, qk_nope+qk_rope, qlora]
    const float* Wkvb_full,   // [nh, qk_nope+v_head, kvlora]
    const float* Wo,          // [H, nh*v_head]
    float*       W_Q_abs_out, // [nh*kvlora, qlora]
    float*       W_O_abs_out) // [H, nh*kvlora]
{
    int nh     = c.num_heads;
    int qlora  = c.q_lora_rank;
    int kvlora = c.kv_lora_rank;
    int d_nope = c.qk_nope_head_dim;
    int d_rope = c.qk_rope_head_dim;
    int d_v    = c.v_head_dim;
    int qkb_row = d_nope + d_rope;
    int kvb_row = d_nope + d_v;

    // W_Q_abs[h*kvlora + k, q] = Σ_{d in 0..d_nope} Wqb[h,d,q] * Wkvb_k[h,d,k]
    // Parallelised over (h, k); inner d loop broadcasts bk and streams over q
    // so AVX-512 FMAs hit the q-dimension contiguously.
    #pragma omp parallel for collapse(2)
    for (int h = 0; h < nh; h++) {
        for (int k = 0; k < kvlora; k++) {
            const float* Wqb_h_nope = Wqb_full  + (size_t)h * qkb_row * qlora;
            const float* Wkvb_h_k   = Wkvb_full + (size_t)h * kvb_row * kvlora;
            float* row = W_Q_abs_out + (size_t)(h*kvlora + k) * qlora;
            for (int q = 0; q < qlora; q++) row[q] = 0.0f;
            for (int d = 0; d < d_nope; d++) {
                float bk = Wkvb_h_k[(size_t)d * kvlora + k];
                const float* a = Wqb_h_nope + (size_t)d * qlora;
                for (int q = 0; q < qlora; q++) row[q] += a[q] * bk;
            }
        }
    }

    // W_O_abs[o, h*kvlora + k] = Σ_{d in 0..d_v} Wkvb_v[h,d,k] * Wo[o, h*d_v + d]
    std::memset(W_O_abs_out, 0, (size_t)c.H * nh * kvlora * sizeof(float));
    #pragma omp parallel for collapse(2) schedule(static)
    for (int h = 0; h < nh; h++) {
        for (int o = 0; o < c.H; o++) {
            const float* Wkvb_h_v = Wkvb_full + (size_t)h * kvb_row * kvlora + (size_t)d_nope * kvlora;
            const float* Wo_row_h = Wo + (size_t)o * nh * d_v + (size_t)h * d_v;
            float* out_row = W_O_abs_out + (size_t)o * nh * kvlora + (size_t)h * kvlora;
            for (int d = 0; d < d_v; d++) {
                float wd = Wo_row_h[d];
                const float* akv = Wkvb_h_v + (size_t)d * kvlora;
                for (int k = 0; k < kvlora; k++) out_row[k] += akv[k] * wd;
            }
        }
    }
}

// ------------------------------------------------------------------------------
// Small helpers: f32 RMS norm (host) — used for intermediate activations that
// would require a dedicated kernel otherwise. For hot-path use in the real
// inference driver, replace with rms_norm_bf16 GPU kernel from kernels.cuh.
// ------------------------------------------------------------------------------
static inline void mla_rms_norm_host_bf16w(
    const float* in, const uint16_t* w_bf16, float* out, int n, float eps)
{
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)in[i] * in[i];
    float inv = 1.0f / std::sqrt((float)(ss / n) + eps);
    for (int i = 0; i < n; i++) {
        uint32_t u = (uint32_t)w_bf16[i] << 16;
        float wf; std::memcpy(&wf, &u, 4);
        out[i] = in[i] * inv * wf;
    }
}

// ------------------------------------------------------------------------------
// One MLA decode step.
//
// Inputs:
//   cfg          : config (static)
//   L            : layer state (weights + cache + scratch)
//   d_cos_pos    : [qk_rope/2] on device — cos table at current position
//   d_sin_pos    : [qk_rope/2] on device — sin table at current position
//   d_hidden_in  : [H] on device, f32
//   pos          : token position (0-indexed)
//   d_layer_out  : [H] on device, f32   — output (before residual add)
//
// After the call, L.cache_len is incremented by 1 and d_kv_cache[pos],
// d_kr_cache[pos] are populated.
//
// NOTE: the two RMS norms (on q_lora and on kv_lat[:kvlora]) are currently run
// on the host via a device→host→device round-trip. This matches what
// test_mla.cu verified. Replace with GPU kernels for hot-path use.
// ------------------------------------------------------------------------------
static inline void mla_attention_step(
    const MLAConfig& cfg,
    MLALayer*        L,
    const float*     d_cos_pos,
    const float*     d_sin_pos,
    const float*     d_hidden_in,
    int              pos,
    float*           d_layer_out,
    cudaStream_t     stream = 0)
{
    const int H      = cfg.H;
    const int nh     = cfg.num_heads;
    const int qlora  = cfg.q_lora_rank;
    const int kvlora = cfg.kv_lora_rank;
    const int d_rope = cfg.qk_rope_head_dim;
    const int kv_full = kvlora + d_rope;

    // 1) q_lora = Wqa @ hidden
    launch_matvec_bf16(L->d_Wqa, d_hidden_in, L->d_q_lora, (uint32_t)qlora, (uint32_t)H);

    // 2) q_lora_n = rms_norm(q_lora, qa_norm)
    launch_rms_norm_bf16(L->d_q_lora, L->d_qa_norm, L->d_q_lora_n,
                         (uint32_t)qlora, MLA_RMS_EPS, stream);

    // 3) q_abs = W_Q_abs @ q_lora_n
    launch_matvec_bf16(L->d_W_Q_abs, L->d_q_lora_n, L->d_q_abs,
                       (uint32_t)(nh * kvlora), (uint32_t)qlora);

    // 4) q_rope_raw = Wqb_rope @ q_lora_n
    launch_matvec_bf16(L->d_Wqb_rope, L->d_q_lora_n, L->d_q_rope,
                       (uint32_t)(nh * d_rope), (uint32_t)qlora);

    // 5) RoPE q_rope in-place (per head)
    launch_mla_yarn_rope(L->d_q_rope, d_cos_pos, d_sin_pos,
                         (uint32_t)nh, (uint32_t)d_rope);

    // 6) kv_lat = Wkva @ hidden  (produces [kvlora|rope] concatenated)
    launch_matvec_bf16(L->d_Wkva, d_hidden_in, L->d_kv_lat,
                       (uint32_t)kv_full, (uint32_t)H);

    // 7a) kv_comp = rms_norm(kv_lat[:kvlora], kva_norm)
    //     rms_norm_bf16 walks x[0..dim) so we pass the base pointer with dim=kvlora.
    launch_rms_norm_bf16(L->d_kv_lat, L->d_kva_norm, L->d_kv_comp,
                         (uint32_t)kvlora, MLA_RMS_EPS, stream);
    // 7b) append kv_comp to cache slot pos as bf16
    launch_cast_f32_to_bf16(L->d_kv_comp,
                            L->d_kv_cache + (size_t)pos * kvlora,
                            (uint32_t)kvlora, stream);

    // 8a) RoPE on kv_lat[kvlora:] in-place (MQA: single shared K head, num_rows=1)
    launch_mla_yarn_rope(L->d_kv_lat + kvlora, d_cos_pos, d_sin_pos,
                         1, (uint32_t)d_rope);
    // 8b) append k_rope to cache slot pos as bf16
    launch_cast_f32_to_bf16(L->d_kv_lat + kvlora,
                            L->d_kr_cache + (size_t)pos * d_rope,
                            (uint32_t)d_rope, stream);

    L->cache_len = pos + 1;

    // 9) MLA attention
    launch_mla_attn_decode(L->d_q_abs, L->d_q_rope,
                           L->d_kv_cache, L->d_kr_cache, L->d_attn_out,
                           (uint32_t)nh, (uint32_t)L->cache_len,
                           (uint32_t)kvlora, (uint32_t)d_rope,
                           cfg.softmax_scale, stream);

    // 10) layer_out = W_O_abs @ attn_out.flatten()
    launch_matvec_bf16(L->d_W_O_abs, L->d_attn_out, d_layer_out,
                       (uint32_t)H, (uint32_t)(nh * kvlora));
}
