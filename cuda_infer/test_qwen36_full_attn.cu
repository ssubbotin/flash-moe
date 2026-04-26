/*
 * test_qwen36_full_attn.cu — substep validation of Qwen3.6 full-attention
 * (layer 3 of layer_types ['linear','linear','linear','full',...]).
 *
 * Per token t in the prompt:
 *    1. h_norm = rms_norm_plus_one(L03_input[t], input_layernorm.w)
 *    2. q_full = fp8_matvec(q_proj.w, h_norm)            [16*256*2 = 8192]
 *    3. k_full = fp8_matvec(k_proj.w, h_norm)            [2*256 = 512]
 *    4. v_full = fp8_matvec(v_proj.w, h_norm)            [2*256 = 512]
 *    5. split q_full into Q [16,256] and gate [16,256]   (interleaved per head)
 *    6. q_norm: per-head rms_norm_plus_one over head_dim=256, weight q_norm.w
 *    7. k_norm: same on K, weight k_norm.w
 *    8. RoPE: rotate first 64 dims of each Q-head and K-head at position t
 *    9. write into K_cache[t,:], V_cache[t,:]
 *
 * Then for each query position t:
 *   10. attn_scores: Q_t @ K_cache[0..t]^T * 1/sqrt(256)
 *   11. attn_softmax over [0..t]
 *   12. attn_values: scores @ V_cache[0..t] → out_t [16*256=4096]
 *   13. out_t *= sigmoid(gate_t)
 *   14. attn_out_t = fp8_matvec(o_proj.w, out_t)         [2048]
 *
 * Compare attn_out_t against L03_attn_out[t] for each t.
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

// ---------- oracle / compare helpers (same as MoE test) ----------
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
    printf("  [%s] %-22s n=%zu  max_abs=%.4e L2rel=%.4e  L2(act)=%.3f L2(ref)=%.3f  worst[%d] act=%.4f ref=%.4f\n",
           mark, tag, n, max_abs, l2rel,
           std::sqrt(l2_a), std::sqrt(l2_r),
           worst, h_actual[worst], h_ref[worst]);
}

static uint8_t* upload_bytes(st::ModelDir* M, const std::string& name) {
    std::vector<uint8_t> buf;
    if (!st::read_bytes(M, name, buf)) std::exit(1);
    uint8_t* d; CUDA_OK(cudaMalloc(&d, buf.size()));
    CUDA_OK(cudaMemcpy(d, buf.data(), buf.size(), cudaMemcpyHostToDevice));
    return d;
}

int main(int argc, char** argv) {
    std::string model_dir, oracle_dir;
    int layer_idx = 3;  // first full-attention layer
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--model-dir" && i + 1 < argc) model_dir = argv[++i];
        else if (a == "--oracle"    && i + 1 < argc) oracle_dir = argv[++i];
        else if (a == "--layer"     && i + 1 < argc) layer_idx  = std::atoi(argv[++i]);
        else { fprintf(stderr, "usage: %s --model-dir DIR --oracle DIR [--layer N]\n", argv[0]); return 1; }
    }
    if (model_dir.empty() || oracle_dir.empty()) return 1;

    cudaDeviceProp p{}; cudaGetDeviceProperties(&p, 0);
    printf("device: %s sm_%d%d  layer_idx=%d\n", p.name, p.major, p.minor, layer_idx);

    st::ModelDir M{}; if (!st::open(&M, model_dir)) return 1;
    Oracle O{}; if (!oracle_open(&O, oracle_dir)) return 1;

    // ---- shapes ----
    const uint32_t H        = 2048;
    const uint32_t NQ       = 16;             // num_attention_heads
    const uint32_t NKV      = 2;              // num_key_value_heads
    const uint32_t HEAD_D   = 256;
    const uint32_t Q_DIM    = NQ * HEAD_D;        // 4096
    const uint32_t Q_RAW_DIM= NQ * HEAD_D * 2;    // 8192 (Q + gate per head)
    const uint32_t KV_DIM   = NKV * HEAD_D;       // 512
    const uint32_t HEADS_PER_KV = NQ / NKV;       // 8
    const uint32_t ROPE_DIM = HEAD_D / 4;         // partial_rotary_factor=0.25
    const uint32_t ROPE_HALF= ROPE_DIM / 2;       // 32
    const float    ROPE_THETA = 10000000.0f;
    const float    RMS_EPS  = 1e-6f;
    const float    SCALE    = 1.0f / std::sqrt((float)HEAD_D);

    // ---- weights ----
    char prefix[128];
    snprintf(prefix, sizeof(prefix), "model.language_model.layers.%d", layer_idx);
    auto T = [&](const std::string& nm){ return std::string(prefix) + "." + nm; };

    uint16_t* d_in_ln_w   = (uint16_t*)upload_bytes(&M, T("input_layernorm.weight"));
    uint8_t*  d_q_w       = upload_bytes(&M, T("self_attn.q_proj.weight"));
    __nv_bfloat16* d_q_s  = (__nv_bfloat16*)upload_bytes(&M, T("self_attn.q_proj.weight_scale_inv"));
    uint8_t*  d_k_w       = upload_bytes(&M, T("self_attn.k_proj.weight"));
    __nv_bfloat16* d_k_s  = (__nv_bfloat16*)upload_bytes(&M, T("self_attn.k_proj.weight_scale_inv"));
    uint8_t*  d_v_w       = upload_bytes(&M, T("self_attn.v_proj.weight"));
    __nv_bfloat16* d_v_s  = (__nv_bfloat16*)upload_bytes(&M, T("self_attn.v_proj.weight_scale_inv"));
    uint8_t*  d_o_w       = upload_bytes(&M, T("self_attn.o_proj.weight"));
    __nv_bfloat16* d_o_s  = (__nv_bfloat16*)upload_bytes(&M, T("self_attn.o_proj.weight_scale_inv"));
    uint16_t* d_q_norm_w  = (uint16_t*)upload_bytes(&M, T("self_attn.q_norm.weight"));
    uint16_t* d_k_norm_w  = (uint16_t*)upload_bytes(&M, T("self_attn.k_norm.weight"));

    // ---- oracle ----
    char tag[64];
    auto K = [&](const char* sub){ snprintf(tag, sizeof(tag), "L%02d_%s", layer_idx, sub); return std::string(tag); };

    std::vector<float> ora_input, ora_h_norm, ora_q_proj, ora_k_proj, ora_v_proj, ora_attn_out;
    if (!oracle_read_f32(&O, K("input"),       ora_input))    return 1;
    if (!oracle_read_f32(&O, K("h_norm"),      ora_h_norm))   return 1;
    if (!oracle_read_f32(&O, K("q_proj_out"),  ora_q_proj))   return 1;
    if (!oracle_read_f32(&O, K("k_proj_out"),  ora_k_proj))   return 1;
    if (!oracle_read_f32(&O, K("v_proj_out"),  ora_v_proj))   return 1;
    if (!oracle_read_f32(&O, K("attn_out"),    ora_attn_out)) return 1;
    const auto& sh = O.manifest[K("input")].shape;
    const uint32_t SEQ = (uint32_t)sh[1];
    printf("seq_len = %u\n", SEQ);

    // ---- precompute cos/sin table on host then upload ----
    std::vector<float> h_cos(SEQ * ROPE_HALF), h_sin(SEQ * ROPE_HALF);
    qwen36::rope_precompute_table(h_cos.data(), h_sin.data(), SEQ, ROPE_DIM, ROPE_THETA);
    float *d_cos, *d_sin;
    CUDA_OK(cudaMalloc(&d_cos, h_cos.size() * 4));
    CUDA_OK(cudaMalloc(&d_sin, h_sin.size() * 4));
    CUDA_OK(cudaMemcpy(d_cos, h_cos.data(), h_cos.size() * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_sin, h_sin.data(), h_sin.size() * 4, cudaMemcpyHostToDevice));

    // ---- scratch ----
    float *d_x, *d_h, *d_q_full, *d_k_full, *d_v_full;
    float *d_Q, *d_GATE, *d_K_norm, *d_V_packed;     // post-norm Q, gate, post-norm K, v
    float *d_K_cache, *d_V_cache;                     // [SEQ, KV_DIM]
    float *d_GATE_cache;                              // [SEQ, Q_DIM]
    float *d_scores;                                  // [NQ, SEQ]  (stride = SEQ)
    float *d_attn_per_head;                           // [NQ, HEAD_D] = [Q_DIM]
    float *d_attn_out;                                // [H]
    CUDA_OK(cudaMalloc(&d_x,            H * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_h,            H * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_q_full,       Q_RAW_DIM * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_k_full,       KV_DIM    * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_v_full,       KV_DIM    * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_Q,            Q_DIM     * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_GATE,         Q_DIM     * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_K_norm,       KV_DIM    * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_V_packed,     KV_DIM    * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_K_cache,      SEQ * KV_DIM * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_V_cache,      SEQ * KV_DIM * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_GATE_cache,   SEQ * Q_DIM  * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_scores,       NQ  * SEQ    * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_attn_per_head,Q_DIM * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_attn_out,     H * sizeof(float)));

    // We will store all SEQ tokens of Q (post norm + RoPE) too, to attend per-token.
    float* d_Q_cache; CUDA_OK(cudaMalloc(&d_Q_cache, SEQ * Q_DIM * sizeof(float)));

    // ===== PASS 1: per-token projections + norm + RoPE, fill K/V/Q caches =====
    for (uint32_t t = 0; t < SEQ; t++) {
        printf("\n=== prefill prep t=%u ===\n", t);

        CUDA_OK(cudaMemcpy(d_x, ora_input.data() + t * H, H * 4, cudaMemcpyHostToDevice));
        qwen36::launch_rms_norm_bf16_plus_one(d_x, d_in_ln_w, d_h, H, RMS_EPS);
        if (t == 0) compare("rms_norm", d_h, ora_h_norm.data() + t * H, H);

        // Q raw  [16, 512] = [Q[256] | gate[256]] per head, packed in head-major order.
        qwen36::launch_dequant_matvec_fp8_block128(d_q_w, d_q_s, d_h, d_q_full, Q_RAW_DIM, H);
        if (t == 0) compare("q_proj", d_q_full, ora_q_proj.data() + t * Q_RAW_DIM, Q_RAW_DIM);

        // K raw  [2, 256]
        qwen36::launch_dequant_matvec_fp8_block128(d_k_w, d_k_s, d_h, d_k_full, KV_DIM, H);
        if (t == 0) compare("k_proj", d_k_full, ora_k_proj.data() + t * KV_DIM, KV_DIM);

        // V raw  [2, 256]
        qwen36::launch_dequant_matvec_fp8_block128(d_v_w, d_v_s, d_h, d_v_full, KV_DIM, H);
        if (t == 0) compare("v_proj", d_v_full, ora_v_proj.data() + t * KV_DIM, KV_DIM);

        // Split q_full into d_Q [16,256] and d_GATE [16,256] using cudaMemcpy2D.
        // For each of 16 heads h: head occupies 512 floats in d_q_full at offset h*512;
        // first 256 → d_Q[h*256..(h+1)*256], next 256 → d_GATE[h*256..].
        CUDA_OK(cudaMemcpy2D(d_Q,                  HEAD_D * 4,
                             d_q_full,             HEAD_D * 2 * 4,
                             HEAD_D * 4, NQ, cudaMemcpyDeviceToDevice));
        CUDA_OK(cudaMemcpy2D(d_GATE,               HEAD_D * 4,
                             d_q_full + HEAD_D,    HEAD_D * 2 * 4,
                             HEAD_D * 4, NQ, cudaMemcpyDeviceToDevice));

        // q_norm per head (16 rows of head_dim=256) — (1+w) form, weight is shared
        qwen36::launch_rms_norm_per_row_plus_one(d_Q, d_q_norm_w, d_Q, NQ, HEAD_D, RMS_EPS);
        // k_norm per kv-head (2 rows)
        qwen36::launch_rms_norm_per_row_plus_one(d_k_full, d_k_norm_w, d_K_norm, NKV, HEAD_D, RMS_EPS);

        // RoPE (partial: rotate first 64 dims of each head) at position t
        const float* d_cos_t = d_cos + t * ROPE_HALF;
        const float* d_sin_t = d_sin + t * ROPE_HALF;
        qwen36::launch_rope_partial_inplace(d_Q,      d_cos_t, d_sin_t, NQ,  HEAD_D, ROPE_DIM);
        qwen36::launch_rope_partial_inplace(d_K_norm, d_cos_t, d_sin_t, NKV, HEAD_D, ROPE_DIM);

        // Cache: K [t, 0..KV_DIM-1], V [t, 0..KV_DIM-1], Q [t, 0..Q_DIM-1], gate [t, 0..Q_DIM-1]
        CUDA_OK(cudaMemcpyAsync(d_K_cache    + t * KV_DIM, d_K_norm,  KV_DIM * 4, cudaMemcpyDeviceToDevice));
        CUDA_OK(cudaMemcpyAsync(d_V_cache    + t * KV_DIM, d_v_full,  KV_DIM * 4, cudaMemcpyDeviceToDevice));
        CUDA_OK(cudaMemcpyAsync(d_Q_cache    + t * Q_DIM,  d_Q,       Q_DIM  * 4, cudaMemcpyDeviceToDevice));
        CUDA_OK(cudaMemcpyAsync(d_GATE_cache + t * Q_DIM,  d_GATE,    Q_DIM  * 4, cudaMemcpyDeviceToDevice));
    }

    // ===== PASS 2: per-query-token attention + gate + o_proj =====
    for (uint32_t t = 0; t < SEQ; t++) {
        printf("\n=== query t=%u ===\n", t);
        const uint32_t L = t + 1;            // causal: attend to positions 0..t

        const float* d_Q_t    = d_Q_cache    + t * Q_DIM;
        const float* d_GATE_t = d_GATE_cache + t * Q_DIM;

        // Scores: Q_t @ K_cache[0..t]^T * scale.
        // attn_scores grid: blockIdx.x = h * num_seq_tgs + pos, num_seq_tgs = L.
        {
            dim3 grid(NQ * L), block(256);
            attn_scores<<<grid, block>>>(
                d_Q_t, d_K_cache, d_scores,
                HEAD_D, KV_DIM, L,            // seq_len
                /*seq_stride=*/SEQ,
                SCALE, HEADS_PER_KV, /*num_seq_tgs=*/L);
        }

        // Softmax
        {
            dim3 grid(NQ), block(256);
            attn_softmax<<<grid, block>>>(d_scores, L, /*seq_stride=*/SEQ);
        }

        // Values
        {
            dim3 block(256), grid((Q_DIM + 255) / 256);
            attn_values<<<grid, block>>>(
                d_scores, d_V_cache, d_attn_per_head,
                HEAD_D, KV_DIM, L, /*seq_stride=*/SEQ, HEADS_PER_KV);
        }

        // Apply sigmoid gate (gate[h, d]) elementwise.  d_attn_per_head is laid out
        // [NQ heads × HEAD_D] head-major, same as d_GATE_t — sigmoid_gate matches.
        {
            dim3 block(256), grid((Q_DIM + 255) / 256);
            sigmoid_gate<<<grid, block>>>(d_attn_per_head, d_GATE_t, Q_DIM);
        }

        // o_proj: FP8 [H, Q_DIM] @ d_attn_per_head → d_attn_out
        qwen36::launch_dequant_matvec_fp8_block128(d_o_w, d_o_s, d_attn_per_head, d_attn_out, H, Q_DIM);

        compare("attn_out", d_attn_out, ora_attn_out.data() + t * H, H);
    }

    st::close(&M);
    printf("\ndone.\n");
    return 0;
}
