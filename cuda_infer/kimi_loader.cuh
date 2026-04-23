/*
 * kimi_loader.cuh — Minimal loader for Kimi K2.6 safetensors shards.
 *
 * The Kimi model is split across 64 safetensors files plus a
 * model.safetensors.index.json mapping each tensor name to its shard file.
 *
 * This loader:
 *   - Reads the index once, builds a tensor→(shard, offset, dtype, shape) map.
 *   - Caches one FILE* per shard (opens lazily).
 *   - Exposes kimi_read_tensor_bytes() / kimi_tensor_info() for arbitrary tensors.
 *   - Exposes kimi_load_mla_layer() which pulls the 7 MLA tensors for a given
 *     transformer layer, computes absorbed W_Q/W_O, and populates an MLALayer
 *     with everything needed by mla_attention_step().
 *
 * NOT handled here: routed experts (sym-int4 compressed-tensors format — that's
 * the repack pipeline's job), MoE gate bias, shared experts, dense layer 0 MLP,
 * embeddings, lm_head. Those get loaded by the inference driver separately.
 */
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cinttypes>
#include <string>
#include <unordered_map>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include "mla_forward.cuh"
#include "kimi_moe.cuh"

struct KimiTensor {
    std::string shard;    // shard filename (relative to model_dir)
    uint64_t    offset;   // absolute byte offset into shard
    uint64_t    nbytes;
    std::string dtype;    // "BF16" / "F32" / "I32"
    std::vector<uint64_t> shape;
};

struct KimiModelDir {
    std::string model_dir;
    std::unordered_map<std::string, KimiTensor> tensors;
    // FILE* cache keyed by shard filename
    std::unordered_map<std::string, FILE*> shard_fp;
};

// ---------------------------------------------------------------------------
// Tiny JSON helpers — scoped to what the index/header formats need.
// ---------------------------------------------------------------------------

// Strip whitespace and consume a JSON string starting at *p (must point at `"`).
// Returns the string value; advances *p past the closing quote.
static inline std::string kimi_json_read_string(const char*& p, const char* end) {
    while (p < end && (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r')) p++;
    if (p >= end || *p != '"') return "";
    p++;
    const char* start = p;
    while (p < end && *p != '"') {
        if (*p == '\\' && p + 1 < end) p += 2;
        else p++;
    }
    std::string s(start, p - start);
    if (p < end) p++;
    return s;
}

// Find the matching closing brace for the `{` pointed at *p. Advances *p past it.
static inline std::string kimi_json_read_object(const char*& p, const char* end) {
    if (p >= end || *p != '{') return "";
    int depth = 0;
    const char* start = p;
    while (p < end) {
        if (*p == '"') {
            p++;
            while (p < end && *p != '"') { if (*p == '\\' && p + 1 < end) p += 2; else p++; }
            if (p < end) p++;
            continue;
        }
        if (*p == '{') depth++;
        else if (*p == '}') { depth--; if (depth == 0) { p++; break; } }
        p++;
    }
    return std::string(start, p - start);
}

// ---------------------------------------------------------------------------
// Load the index file and build the tensor map.
// ---------------------------------------------------------------------------

static inline bool kimi_open(KimiModelDir* M, const std::string& model_dir) {
    M->model_dir = model_dir;
    std::string idx_path = model_dir + "/model.safetensors.index.json";
    FILE* f = std::fopen(idx_path.c_str(), "rb");
    if (!f) { fprintf(stderr, "kimi_open: cannot open %s\n", idx_path.c_str()); return false; }
    std::fseek(f, 0, SEEK_END); long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    std::vector<char> body(n);
    if ((long)std::fread(body.data(), 1, n, f) != n) { std::fclose(f); return false; }
    std::fclose(f);

    // Parse {"weight_map":{"tensor_name":"shard", ...}, ...}
    std::string src(body.data(), body.size());
    size_t wm = src.find("\"weight_map\"");
    if (wm == std::string::npos) { fprintf(stderr, "kimi_open: no weight_map\n"); return false; }
    size_t open_brace = src.find('{', wm);
    size_t close_brace = open_brace;
    int depth = 0;
    for (size_t i = open_brace; i < src.size(); i++) {
        if (src[i] == '{') depth++;
        else if (src[i] == '}') { depth--; if (depth == 0) { close_brace = i; break; } }
    }

    // Build shard map first — maps tensor name → shard file name
    std::unordered_map<std::string, std::string> name_to_shard;
    const char* p = src.data() + open_brace + 1;
    const char* e = src.data() + close_brace;
    while (p < e) {
        while (p < e && (*p == ' ' || *p == '\n' || *p == ',' || *p == '\t' || *p == '\r')) p++;
        if (p >= e || *p != '"') break;
        std::string name = kimi_json_read_string(p, e);
        while (p < e && (*p == ' ' || *p == ':' || *p == '\t')) p++;
        std::string shard = kimi_json_read_string(p, e);
        if (!name.empty() && !shard.empty()) name_to_shard.emplace(std::move(name), std::move(shard));
    }

    // Group tensors by shard to read each shard header only once
    std::unordered_map<std::string, std::vector<std::string>> shard_to_names;
    for (auto& kv : name_to_shard) shard_to_names[kv.second].push_back(kv.first);

    for (auto& kv : shard_to_names) {
        const std::string& shard = kv.first;
        std::string path = model_dir + "/" + shard;
        FILE* sf = std::fopen(path.c_str(), "rb");
        if (!sf) { fprintf(stderr, "kimi_open: cannot open shard %s\n", path.c_str()); return false; }

        uint64_t hdr_len = 0;
        if (std::fread(&hdr_len, 1, 8, sf) != 8) { std::fclose(sf); return false; }
        std::vector<char> hdr(hdr_len);
        if (std::fread(hdr.data(), 1, hdr_len, sf) != hdr_len) { std::fclose(sf); return false; }
        uint64_t data_base = 8 + hdr_len;
        std::fclose(sf);

        std::string H(hdr.begin(), hdr.end());
        for (auto& tn : kv.second) {
            std::string pat = "\"" + tn + "\":";
            size_t px = H.find(pat);
            if (px == std::string::npos) {
                fprintf(stderr, "kimi_open: missing tensor %s in %s\n", tn.c_str(), shard.c_str());
                return false;
            }
            const char* q = H.data() + px + pat.size();
            const char* qe = H.data() + H.size();
            // Expect object {"dtype":"...","shape":[...],"data_offsets":[..,..]}
            std::string obj = kimi_json_read_object(q, qe);
            KimiTensor t;
            t.shard = shard;

            // dtype
            size_t dt = obj.find("\"dtype\":");
            if (dt != std::string::npos) {
                const char* qp = obj.data() + dt + 8;
                const char* qe2 = obj.data() + obj.size();
                t.dtype = kimi_json_read_string(qp, qe2);
            }
            // shape
            size_t sh = obj.find("\"shape\":");
            if (sh != std::string::npos) {
                size_t lb = obj.find('[', sh), rb = obj.find(']', lb);
                std::string body2 = obj.substr(lb + 1, rb - lb - 1);
                size_t i = 0;
                while (i < body2.size()) {
                    while (i < body2.size() && (body2[i] == ' ' || body2[i] == ',')) i++;
                    if (i >= body2.size()) break;
                    t.shape.push_back(std::strtoull(body2.c_str() + i, nullptr, 10));
                    while (i < body2.size() && body2[i] != ',') i++;
                }
            }
            // data_offsets [beg, end]
            size_t dof = obj.find("\"data_offsets\":");
            if (dof != std::string::npos) {
                size_t lb = obj.find('[', dof), rb = obj.find(']', lb);
                std::string body2 = obj.substr(lb + 1, rb - lb - 1);
                uint64_t beg = std::strtoull(body2.c_str(), nullptr, 10);
                size_t comma = body2.find(',');
                uint64_t fin = std::strtoull(body2.c_str() + comma + 1, nullptr, 10);
                t.offset = data_base + beg;
                t.nbytes = fin - beg;
            }
            M->tensors.emplace(tn, std::move(t));
        }
    }
    return true;
}

static inline void kimi_close(KimiModelDir* M) {
    for (auto& kv : M->shard_fp) if (kv.second) std::fclose(kv.second);
    M->shard_fp.clear();
    M->tensors.clear();
}

static inline const KimiTensor* kimi_tensor_info(const KimiModelDir* M, const std::string& name) {
    auto it = M->tensors.find(name);
    return (it == M->tensors.end()) ? nullptr : &it->second;
}

static inline bool kimi_read_tensor_bytes(KimiModelDir* M, const std::string& name,
                                          std::vector<uint8_t>& out)
{
    auto it = M->tensors.find(name);
    if (it == M->tensors.end()) {
        fprintf(stderr, "kimi_read: missing tensor %s\n", name.c_str());
        return false;
    }
    const KimiTensor& t = it->second;
    FILE*& fp = M->shard_fp[t.shard];
    if (!fp) {
        std::string path = M->model_dir + "/" + t.shard;
        fp = std::fopen(path.c_str(), "rb");
        if (!fp) { fprintf(stderr, "kimi_read: open %s\n", path.c_str()); return false; }
    }
    // Use pread on the fileno — handles 64-bit offsets unambiguously and loops
    // internally for large reads (single fread of >2 GB has historically had
    // corner cases where it silently truncates).
    out.resize(t.nbytes);
    int fd = fileno(fp);
    size_t total = 0;
    while (total < t.nbytes) {
        size_t want = t.nbytes - total;
        // cap each call below INT_MAX to avoid any 32-bit truncation in the
        // syscall wrapper on any unusual libc
        if (want > (size_t)0x40000000) want = (size_t)0x40000000;  // 1 GiB per call
        ssize_t got = ::pread(fd, out.data() + total, want, (off_t)(t.offset + total));
        if (got <= 0) {
            fprintf(stderr, "kimi_read: pread %s got=%zd after %zu/%" PRIu64 " bytes\n",
                    name.c_str(), got, total, (uint64_t)t.nbytes);
            return false;
        }
        total += (size_t)got;
    }
    return true;
}

// ---------------------------------------------------------------------------
// BF16 conversion helpers
// ---------------------------------------------------------------------------

static inline float kimi_bf16_to_f32(uint16_t u) {
    uint32_t w = (uint32_t)u << 16;
    float v; std::memcpy(&v, &w, 4); return v;
}

static inline uint16_t kimi_f32_to_bf16(float x) {
    uint32_t u; std::memcpy(&u, &x, 4);
    uint32_t rb = 0x00007FFFu + ((u >> 16) & 1u);
    return (uint16_t)((u + rb) >> 16);
}

static inline void kimi_bf16_span_to_f32(const uint16_t* in, float* out, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = kimi_bf16_to_f32(in[i]);
}

static inline void kimi_f32_span_to_bf16(const float* in, uint16_t* out, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = kimi_f32_to_bf16(in[i]);
}

// Upload a bf16 buffer to a freshly allocated device pointer.
static inline uint16_t* kimi_upload_bf16(const uint16_t* host, size_t n) {
    uint16_t* d;
    cudaMalloc(&d, n * 2);
    cudaMemcpy(d, host, n * 2, cudaMemcpyHostToDevice);
    return d;
}

// Upload f32 → bf16 (round-to-nearest-even) to a freshly allocated device pointer.
static inline uint16_t* kimi_upload_f32_as_bf16(const float* host, size_t n) {
    std::vector<uint16_t> buf(n);
    kimi_f32_span_to_bf16(host, buf.data(), n);
    return kimi_upload_bf16(buf.data(), n);
}

// ---------------------------------------------------------------------------
// Load one layer's MLA weights and populate an MLALayer.
//
// Allocates (and the caller owns):
//   L->d_Wqa         [qlora, H]
//   L->d_qa_norm     [qlora]
//   L->d_Wqb_rope    [nh*d_rope, qlora]    — rope rows extracted from q_b_proj
//   L->d_Wkva        [kvlora+d_rope, H]
//   L->d_kva_norm    [kvlora]
//   L->d_W_Q_abs     [nh*kvlora, qlora]    — absorbed at load
//   L->d_W_O_abs     [H, nh*kvlora]        — absorbed at load
//   L->d_kv_cache    [max_seq, kvlora]
//   L->d_kr_cache    [max_seq, d_rope]
//   per-step scratch via mla_layer_alloc_scratch()
//
// Absorption math:
//   W_Q_abs[h*kvlora + k, q] = Σ_{d in 0..d_nope} Wqb[h, d, q] * Wkvb_k[h, d, k]
//   W_O_abs[o, h*kvlora + k] = Σ_{d in 0..d_v}   Wkvb_v[h, d, k] * Wo[o, h*d_v + d]
//
// Source tensor layouts (Kimi / DeepSeek-V3 convention):
//   q_b_proj.weight     : [nh*(d_nope+d_rope), qlora]       row-major
//   kv_b_proj.weight    : [nh*(d_nope+d_v),    kvlora]      row-major
//   o_proj.weight       : [H, nh*d_v]                        row-major
// ---------------------------------------------------------------------------
static inline bool kimi_load_mla_layer(
    KimiModelDir* M, const MLAConfig& cfg, int layer_idx, int max_seq_len,
    MLALayer* L)
{
    int H      = cfg.H;
    int nh     = cfg.num_heads;
    int qlora  = cfg.q_lora_rank;
    int kvlora = cfg.kv_lora_rank;
    int d_nope = cfg.qk_nope_head_dim;
    int d_rope = cfg.qk_rope_head_dim;
    int d_v    = cfg.v_head_dim;
    int qkb_row = d_nope + d_rope;
    int kvb_row = d_nope + d_v;

    char prefix[128];
    std::snprintf(prefix, sizeof prefix, "language_model.model.layers.%d.self_attn.", layer_idx);

    auto name = [&](const char* suf) { return std::string(prefix) + suf; };

    std::vector<uint8_t> qa_raw, qan_raw, qb_raw, kva_raw, kvan_raw, kvb_raw, o_raw;
    if (!kimi_read_tensor_bytes(M, name("q_a_proj.weight"),            qa_raw))  return false;
    if (!kimi_read_tensor_bytes(M, name("q_a_layernorm.weight"),       qan_raw)) return false;
    if (!kimi_read_tensor_bytes(M, name("q_b_proj.weight"),            qb_raw))  return false;
    if (!kimi_read_tensor_bytes(M, name("kv_a_proj_with_mqa.weight"),  kva_raw)) return false;
    if (!kimi_read_tensor_bytes(M, name("kv_a_layernorm.weight"),      kvan_raw))return false;
    if (!kimi_read_tensor_bytes(M, name("kv_b_proj.weight"),           kvb_raw)) return false;
    if (!kimi_read_tensor_bytes(M, name("o_proj.weight"),              o_raw))   return false;

    // Raw bf16 views
    const uint16_t* Wqa_bf      = (const uint16_t*)qa_raw.data();     // [qlora, H]
    const uint16_t* qa_norm_bf  = (const uint16_t*)qan_raw.data();    // [qlora]
    const uint16_t* Wqb_bf      = (const uint16_t*)qb_raw.data();     // [nh*(d_nope+d_rope), qlora]
    const uint16_t* Wkva_bf     = (const uint16_t*)kva_raw.data();    // [kvlora+d_rope, H]
    const uint16_t* kva_norm_bf = (const uint16_t*)kvan_raw.data();   // [kvlora]
    const uint16_t* Wkvb_bf     = (const uint16_t*)kvb_raw.data();    // [nh*(d_nope+d_v), kvlora]
    const uint16_t* Wo_bf       = (const uint16_t*)o_raw.data();      // [H, nh*d_v]

    // Upload raw bf16 weights directly (no conversion needed)
    L->d_Wqa     = kimi_upload_bf16(Wqa_bf,     (size_t)qlora * H);
    L->d_qa_norm = kimi_upload_bf16(qa_norm_bf, (size_t)qlora);
    L->d_Wkva    = kimi_upload_bf16(Wkva_bf,    (size_t)(kvlora + d_rope) * H);
    L->d_kva_norm= kimi_upload_bf16(kva_norm_bf,(size_t)kvlora);

    // Extract rope-rows from q_b_proj: for each head h, rows [h*qkb_row + d_nope, h*qkb_row + d_nope+d_rope)
    {
        std::vector<uint16_t> Wqb_rope_packed((size_t)nh * d_rope * qlora);
        for (int h = 0; h < nh; h++) {
            const uint16_t* src = Wqb_bf + ((size_t)h * qkb_row + d_nope) * qlora;
            uint16_t* dst = Wqb_rope_packed.data() + (size_t)h * d_rope * qlora;
            std::memcpy(dst, src, (size_t)d_rope * qlora * 2);
        }
        L->d_Wqb_rope = kimi_upload_bf16(Wqb_rope_packed.data(), Wqb_rope_packed.size());
    }

    // Decode the matmul inputs to f32 for absorption
    std::vector<float> Wqb_full((size_t)nh * qkb_row * qlora);
    kimi_bf16_span_to_f32(Wqb_bf, Wqb_full.data(), Wqb_full.size());

    std::vector<float> Wkvb_full((size_t)nh * kvb_row * kvlora);
    kimi_bf16_span_to_f32(Wkvb_bf, Wkvb_full.data(), Wkvb_full.size());

    std::vector<float> Wo_full((size_t)H * nh * d_v);
    kimi_bf16_span_to_f32(Wo_bf, Wo_full.data(), Wo_full.size());

    // Absorb
    std::vector<float> W_Q_abs((size_t)nh * kvlora * qlora);
    std::vector<float> W_O_abs((size_t)H * nh * kvlora);
    mla_absorb_weights(cfg, Wqb_full.data(), Wkvb_full.data(), Wo_full.data(),
                       W_Q_abs.data(), W_O_abs.data());

    // Upload absorbed weights as bf16 (round-trip matches kernel expectation)
    L->d_W_Q_abs = kimi_upload_f32_as_bf16(W_Q_abs.data(), W_Q_abs.size());
    L->d_W_O_abs = kimi_upload_f32_as_bf16(W_O_abs.data(), W_O_abs.size());

    // KV cache (zero-initialized)
    cudaMalloc(&L->d_kv_cache, (size_t)max_seq_len * kvlora * 2);
    cudaMalloc(&L->d_kr_cache, (size_t)max_seq_len * d_rope * 2);
    cudaMemset(L->d_kv_cache, 0, (size_t)max_seq_len * kvlora * 2);
    cudaMemset(L->d_kr_cache, 0, (size_t)max_seq_len * d_rope * 2);
    L->cache_len = 0;

    // Per-step scratch
    mla_layer_alloc_scratch(L, cfg);
    return true;
}

// Free all device buffers owned by an MLALayer populated by kimi_load_mla_layer.
static inline void kimi_free_mla_layer(MLALayer* L) {
    cudaFree(L->d_Wqa);      cudaFree(L->d_qa_norm);
    cudaFree(L->d_Wqb_rope); cudaFree(L->d_W_Q_abs);
    cudaFree(L->d_Wkva);     cudaFree(L->d_kva_norm);
    cudaFree(L->d_W_O_abs);
    cudaFree(L->d_kv_cache); cudaFree(L->d_kr_cache);
    mla_layer_free_scratch(L);
}

// ---------------------------------------------------------------------------
// Extended Kimi config (MLA + MoE + dense layer 0) — everything the forward
// driver needs. MLAConfig is a subset.
// ---------------------------------------------------------------------------
struct KimiConfig {
    MLAConfig mla;
    int num_layers;            // num_hidden_layers (61)
    int first_k_dense;         // first_k_dense_replace (1)
    int num_routed_experts;    // n_routed_experts (384)
    int num_shared_experts;    // n_shared_experts (1)
    int experts_per_tok;       // num_experts_per_tok (8)
    int moe_intermediate;      // moe_intermediate_size (2048)
    int dense_intermediate;    // intermediate_size (18432)  — used only by layer 0
    int vocab_size;            // 163840
    float routed_scaling_factor;  // 2.827
    int   norm_topk_prob;         // 1
    float rms_norm_eps;        // 1e-5
};

// Per-layer device-side weight handles. Empty pointers for weights that don't
// exist in that layer (e.g. routed_experts are streamed, not in here).
struct KimiLayer {
    // MLA (always present)
    MLALayer mla;

    // Post-attention RMS norm (always present)
    uint16_t* d_input_layernorm;      // [H]
    uint16_t* d_post_attn_layernorm;  // [H]

    // MLP / MoE — one of two paths populated:
    // (a) Dense (layer 0 only): mlp.gate_proj/up_proj/down_proj, intermediate=18432
    //     d_mlp_gate / d_mlp_up / d_mlp_down non-null
    // (b) MoE (layers 1..N-1): mlp.gate.weight + e_score_correction_bias + shared_experts.*
    //     d_router_gate / d_router_bias / d_shared_* non-null
    //     Routed experts are streamed from <packed_experts>/layer_N.bin (not here).
    int   is_moe;

    // Dense path
    uint16_t* d_mlp_gate;    // [dense_intermediate, H]
    uint16_t* d_mlp_up;      // [dense_intermediate, H]
    uint16_t* d_mlp_down;    // [H, dense_intermediate]

    // MoE path
    uint16_t* d_router_gate; // [num_routed_experts, H]  bf16
    float*    d_router_bias; // [num_routed_experts]     f32
    uint16_t* d_shared_gate; // [moe_intermediate, H]
    uint16_t* d_shared_up;   // [moe_intermediate, H]
    uint16_t* d_shared_down; // [H, moe_intermediate]
};

struct KimiModel {
    KimiConfig cfg;

    // Global weights
    uint16_t* d_embed;        // [vocab, H] bf16
    uint16_t* d_final_norm;   // [H] bf16
    uint16_t* d_lm_head;      // [vocab, H] bf16

    // Per-layer
    std::vector<KimiLayer> layers;

    // Yarn tables (precomputed on host, uploaded)
    float*   d_cos_table;     // [max_seq, rope/2]
    float*   d_sin_table;     // [max_seq, rope/2]
    int      max_seq_len;

    // Per-step scratch used across layers (f32)
    float*   d_hidden;        // [H]
    float*   d_hidden_norm;   // [H]
    float*   d_residual;      // [H]
    float*   d_attn_out;      // [H]
    float*   d_mlp_out;       // [H] (for dense path)
    // MoE scratch
    float*   d_router_logits; // [num_routed_experts]
    int*     d_topk_idx;      // [K]
    float*   d_topk_w;        // [K]
    float*   d_moe_accum;     // [H]
    // Expert scratch (reused across K experts within a layer)
    uint8_t* d_expert_block;  // [KIMI_EXPERT_BLOCK_BYTES]
    float*   d_gate_tmp;      // [moe_intermediate]
    float*   d_up_tmp;        // [moe_intermediate]
    float*   d_glu_tmp;       // [moe_intermediate]
    float*   d_expert_out;    // [H]
    // Shared expert scratch
    float*   d_sgate_tmp;     // [moe_intermediate]
    float*   d_sup_tmp;       // [moe_intermediate]
    float*   d_sglu_tmp;      // [moe_intermediate]
    float*   d_shared_out;    // [H]
    // Dense MLP scratch (only used if layer 0 present)
    float*   d_dgate_tmp;     // [dense_intermediate]
    float*   d_dup_tmp;       // [dense_intermediate]
    float*   d_dglu_tmp;      // [dense_intermediate]
    // Final logits
    float*   d_logits;        // [vocab_size]

    // Expert file descriptors (one per MoE layer)
    std::vector<int> expert_fds;
};

// Read config.json and populate MLAConfig. Returns true on success.
// ---------------------------------------------------------------------------
static inline bool kimi_load_mla_config(const std::string& model_dir, MLAConfig* cfg) {
    std::string path = model_dir + "/config.json";
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END); long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    std::vector<char> body(n);
    if ((long)std::fread(body.data(), 1, n, f) != n) { std::fclose(f); return false; }
    std::fclose(f);
    std::string s(body.data(), body.size());

    auto num = [&](const char* key, double def) -> double {
        std::string pat = "\"" + std::string(key) + "\"";
        size_t p = s.find(pat);
        if (p == std::string::npos) return def;
        p = s.find(':', p);
        while (p < s.size() && !(std::isdigit((unsigned char)s[p]) || s[p]=='-' || s[p]=='.')) p++;
        if (p >= s.size()) return def;
        return std::strtod(s.c_str() + p, nullptr);
    };

    cfg->H                = (int)num("hidden_size", 7168);
    cfg->num_heads        = (int)num("num_attention_heads", 64);
    cfg->q_lora_rank      = (int)num("q_lora_rank", 1536);
    cfg->kv_lora_rank     = (int)num("kv_lora_rank", 512);
    cfg->qk_nope_head_dim = (int)num("qk_nope_head_dim", 128);
    cfg->qk_rope_head_dim = (int)num("qk_rope_head_dim", 64);
    cfg->v_head_dim       = (int)num("v_head_dim", 128);
    cfg->rope_theta       = (float)num("rope_theta", 50000.0);
    cfg->yarn_factor      = (float)num("factor", 64.0);
    cfg->yarn_beta_fast   = (float)num("beta_fast", 32.0);
    cfg->yarn_beta_slow   = (float)num("beta_slow", 1.0);
    cfg->yarn_orig_max    = (int)num("original_max_position_embeddings", 4096);
    cfg->yarn_mscale      = (float)num("mscale", 1.0);
    cfg->yarn_mscale_all  = (float)num("mscale_all_dim", 1.0);
    cfg->softmax_scale    = mla_effective_softmax_scale(*cfg);
    return true;
}

// ---------------------------------------------------------------------------
// Full KimiConfig loader — extends the MLA-only reader with MoE + vocab knobs.
// ---------------------------------------------------------------------------
static inline bool kimi_load_full_config(const std::string& model_dir, KimiConfig* cfg) {
    if (!kimi_load_mla_config(model_dir, &cfg->mla)) return false;

    std::string path = model_dir + "/config.json";
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END); long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    std::vector<char> body(n);
    if ((long)std::fread(body.data(), 1, n, f) != n) { std::fclose(f); return false; }
    std::fclose(f);
    std::string s(body.data(), body.size());

    auto num = [&](const char* key, double def) -> double {
        std::string pat = "\"" + std::string(key) + "\"";
        size_t p = s.find(pat);
        if (p == std::string::npos) return def;
        p = s.find(':', p);
        while (p < s.size() && !(std::isdigit((unsigned char)s[p]) || s[p]=='-' || s[p]=='.')) p++;
        if (p >= s.size()) return def;
        return std::strtod(s.c_str() + p, nullptr);
    };
    auto boolnum = [&](const char* key, int def) -> int {
        std::string pat = "\"" + std::string(key) + "\"";
        size_t p = s.find(pat);
        if (p == std::string::npos) return def;
        p = s.find(':', p) + 1;
        while (p < s.size() && (s[p]==' '||s[p]=='\t'||s[p]=='\n')) p++;
        if (s.compare(p, 4, "true") == 0) return 1;
        if (s.compare(p, 5, "false") == 0) return 0;
        return def;
    };

    cfg->num_layers            = (int)num("num_hidden_layers",      61);
    cfg->first_k_dense         = (int)num("first_k_dense_replace",   1);
    cfg->num_routed_experts    = (int)num("n_routed_experts",      384);
    cfg->num_shared_experts    = (int)num("n_shared_experts",        1);
    cfg->experts_per_tok       = (int)num("num_experts_per_tok",     8);
    cfg->moe_intermediate      = (int)num("moe_intermediate_size", 2048);
    cfg->dense_intermediate    = (int)num("intermediate_size",    18432);
    cfg->vocab_size            = (int)num("vocab_size",          163840);
    cfg->routed_scaling_factor = (float)num("routed_scaling_factor", 2.827);
    cfg->norm_topk_prob        = boolnum("norm_topk_prob",            1);
    cfg->rms_norm_eps          = (float)num("rms_norm_eps",       1e-5);
    return true;
}

// ---------------------------------------------------------------------------
// Per-layer non-expert weight loader.
//
// Called AFTER kimi_load_mla_layer() has populated L->mla for this layer.
// Loads:
//   - input_layernorm.weight   (bf16, [H])
//   - post_attention_layernorm (bf16, [H])
//   - Either the dense MLP (layer 0) or the MoE gate + shared expert (layer >=1)
//
// Does NOT load routed experts — those are streamed from packed layer files.
// ---------------------------------------------------------------------------
static inline bool kimi_load_layer_non_expert(
    KimiModelDir* M, const KimiConfig& cfg, int layer_idx, KimiLayer* L)
{
    char prefix[128];
    std::snprintf(prefix, sizeof prefix, "language_model.model.layers.%d.", layer_idx);
    auto name = [&](const char* suf) { return std::string(prefix) + suf; };

    std::vector<uint8_t> buf;

    // Two RMS norms
    if (!kimi_read_tensor_bytes(M, name("input_layernorm.weight"), buf)) return false;
    L->d_input_layernorm = kimi_upload_bf16((const uint16_t*)buf.data(), cfg.mla.H);

    if (!kimi_read_tensor_bytes(M, name("post_attention_layernorm.weight"), buf)) return false;
    L->d_post_attn_layernorm = kimi_upload_bf16((const uint16_t*)buf.data(), cfg.mla.H);

    // Dense vs MoE
    int is_moe = (layer_idx >= cfg.first_k_dense);
    L->is_moe = is_moe;
    // Null out the unused path
    L->d_mlp_gate = L->d_mlp_up = L->d_mlp_down = nullptr;
    L->d_router_gate = L->d_shared_gate = L->d_shared_up = L->d_shared_down = nullptr;
    L->d_router_bias = nullptr;

    if (!is_moe) {
        // Dense path (layer 0)
        if (!kimi_read_tensor_bytes(M, name("mlp.gate_proj.weight"), buf)) return false;
        L->d_mlp_gate = kimi_upload_bf16((const uint16_t*)buf.data(),
                                         (size_t)cfg.dense_intermediate * cfg.mla.H);
        if (!kimi_read_tensor_bytes(M, name("mlp.up_proj.weight"), buf)) return false;
        L->d_mlp_up   = kimi_upload_bf16((const uint16_t*)buf.data(),
                                         (size_t)cfg.dense_intermediate * cfg.mla.H);
        if (!kimi_read_tensor_bytes(M, name("mlp.down_proj.weight"), buf)) return false;
        L->d_mlp_down = kimi_upload_bf16((const uint16_t*)buf.data(),
                                         (size_t)cfg.mla.H * cfg.dense_intermediate);
    } else {
        // MoE path
        if (!kimi_read_tensor_bytes(M, name("mlp.gate.weight"), buf)) return false;
        L->d_router_gate = kimi_upload_bf16((const uint16_t*)buf.data(),
                                            (size_t)cfg.num_routed_experts * cfg.mla.H);
        if (!kimi_read_tensor_bytes(M, name("mlp.gate.e_score_correction_bias"), buf)) return false;
        // e_score_correction_bias is F32 — store as f32 on device
        cudaMalloc(&L->d_router_bias, (size_t)cfg.num_routed_experts * 4);
        cudaMemcpy(L->d_router_bias, buf.data(),
                   (size_t)cfg.num_routed_experts * 4, cudaMemcpyHostToDevice);

        if (!kimi_read_tensor_bytes(M, name("mlp.shared_experts.gate_proj.weight"), buf)) return false;
        L->d_shared_gate = kimi_upload_bf16((const uint16_t*)buf.data(),
                                            (size_t)cfg.moe_intermediate * cfg.mla.H);
        if (!kimi_read_tensor_bytes(M, name("mlp.shared_experts.up_proj.weight"),   buf)) return false;
        L->d_shared_up   = kimi_upload_bf16((const uint16_t*)buf.data(),
                                            (size_t)cfg.moe_intermediate * cfg.mla.H);
        if (!kimi_read_tensor_bytes(M, name("mlp.shared_experts.down_proj.weight"), buf)) return false;
        L->d_shared_down = kimi_upload_bf16((const uint16_t*)buf.data(),
                                            (size_t)cfg.mla.H * cfg.moe_intermediate);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Load all global (non-layer) weights: embed_tokens, norm, lm_head.
// ---------------------------------------------------------------------------
static inline bool kimi_load_globals(KimiModelDir* M, const KimiConfig& cfg, KimiModel* K) {
    std::vector<uint8_t> buf;
    if (!kimi_read_tensor_bytes(M, "language_model.model.embed_tokens.weight", buf)) return false;
    K->d_embed = kimi_upload_bf16((const uint16_t*)buf.data(),
                                  (size_t)cfg.vocab_size * cfg.mla.H);

    if (!kimi_read_tensor_bytes(M, "language_model.model.norm.weight", buf)) return false;
    K->d_final_norm = kimi_upload_bf16((const uint16_t*)buf.data(), cfg.mla.H);

    if (!kimi_read_tensor_bytes(M, "language_model.lm_head.weight", buf)) return false;
    K->d_lm_head = kimi_upload_bf16((const uint16_t*)buf.data(),
                                    (size_t)cfg.vocab_size * cfg.mla.H);
    return true;
}

// ---------------------------------------------------------------------------
// Allocate per-step shared scratch on the model (sized from cfg).
// ---------------------------------------------------------------------------
static inline void kimi_alloc_scratch(KimiModel* K) {
    const KimiConfig& c = K->cfg;
    int H = c.mla.H;
    int moe_int = c.moe_intermediate;
    int dense_int = c.dense_intermediate;
    int ne = c.num_routed_experts;
    int Kexp = c.experts_per_tok;

    cudaMalloc(&K->d_hidden,         (size_t)H * 4);
    cudaMalloc(&K->d_hidden_norm,    (size_t)H * 4);
    cudaMalloc(&K->d_residual,       (size_t)H * 4);
    cudaMalloc(&K->d_attn_out,       (size_t)H * 4);
    cudaMalloc(&K->d_mlp_out,        (size_t)H * 4);

    cudaMalloc(&K->d_router_logits,  (size_t)ne * 4);
    cudaMalloc(&K->d_topk_idx,       (size_t)Kexp * 4);
    cudaMalloc(&K->d_topk_w,         (size_t)Kexp * 4);
    cudaMalloc(&K->d_moe_accum,      (size_t)H * 4);

    cudaMalloc(&K->d_expert_block,   (size_t)KIMI_EXPERT_BLOCK_BYTES);
    cudaMalloc(&K->d_gate_tmp,       (size_t)moe_int * 4);
    cudaMalloc(&K->d_up_tmp,         (size_t)moe_int * 4);
    cudaMalloc(&K->d_glu_tmp,        (size_t)moe_int * 4);
    cudaMalloc(&K->d_expert_out,     (size_t)H * 4);

    cudaMalloc(&K->d_sgate_tmp,      (size_t)moe_int * 4);
    cudaMalloc(&K->d_sup_tmp,        (size_t)moe_int * 4);
    cudaMalloc(&K->d_sglu_tmp,       (size_t)moe_int * 4);
    cudaMalloc(&K->d_shared_out,     (size_t)H * 4);

    cudaMalloc(&K->d_dgate_tmp,      (size_t)dense_int * 4);
    cudaMalloc(&K->d_dup_tmp,        (size_t)dense_int * 4);
    cudaMalloc(&K->d_dglu_tmp,       (size_t)dense_int * 4);

    cudaMalloc(&K->d_logits,         (size_t)c.vocab_size * 4);
}

// ---------------------------------------------------------------------------
// Open packed_experts/layer_N.bin for each MoE layer and cache FDs.
// ---------------------------------------------------------------------------
static inline bool kimi_open_expert_files(KimiModel* K, const std::string& packed_dir) {
    const KimiConfig& c = K->cfg;
    K->expert_fds.assign(c.num_layers, -1);
    for (int li = c.first_k_dense; li < c.num_layers; li++) {
        char name[64];
        std::snprintf(name, sizeof name, "layer_%d.bin", li);
        std::string path = packed_dir + "/" + name;
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) { fprintf(stderr, "kimi: cannot open %s\n", path.c_str()); return false; }
        K->expert_fds[li] = fd;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Build the full KimiModel: config, all layers' non-expert weights, globals,
// yarn tables, scratch, expert file descriptors.
//
// max_seq_len: pre-allocates KV cache for all layers (bf16 [max_seq, 576] per layer).
// ---------------------------------------------------------------------------
static inline bool kimi_build_model(
    const std::string& model_dir, const std::string& packed_dir,
    int max_seq_len, KimiModel* K)
{
    if (!kimi_load_full_config(model_dir, &K->cfg)) return false;
    K->max_seq_len = max_seq_len;
    printf("Kimi config: L=%d first_dense=%d experts=%d K=%d moe_int=%d dense_int=%d vocab=%d scale=%.3f\n",
        K->cfg.num_layers, K->cfg.first_k_dense, K->cfg.num_routed_experts,
        K->cfg.experts_per_tok, K->cfg.moe_intermediate, K->cfg.dense_intermediate,
        K->cfg.vocab_size, K->cfg.routed_scaling_factor);

    KimiModelDir M{};
    if (!kimi_open(&M, model_dir)) return false;

    if (!kimi_load_globals(&M, K->cfg, K)) return false;
    printf("  loaded globals (embed, lm_head, final_norm)\n");

    K->layers.resize(K->cfg.num_layers);
    for (int li = 0; li < K->cfg.num_layers; li++) {
        if (!kimi_load_mla_layer(&M, K->cfg.mla, li, max_seq_len, &K->layers[li].mla)) return false;
        if (!kimi_load_layer_non_expert(&M, K->cfg, li, &K->layers[li])) return false;
        if (li == 0 || li == K->cfg.num_layers - 1 || li % 10 == 0)
            printf("  layer %d/%d loaded\n", li, K->cfg.num_layers);
    }
    kimi_close(&M);

    // Yarn tables
    std::vector<float> h_cos, h_sin;
    mla_yarn_tables_precompute(K->cfg.mla, max_seq_len, h_cos, h_sin);
    cudaMalloc(&K->d_cos_table, h_cos.size() * 4);
    cudaMalloc(&K->d_sin_table, h_sin.size() * 4);
    cudaMemcpy(K->d_cos_table, h_cos.data(), h_cos.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(K->d_sin_table, h_sin.data(), h_sin.size() * 4, cudaMemcpyHostToDevice);

    kimi_alloc_scratch(K);

    if (!kimi_open_expert_files(K, packed_dir)) return false;
    printf("  opened %d expert files in %s\n",
        K->cfg.num_layers - K->cfg.first_k_dense, packed_dir.c_str());
    return true;
}
