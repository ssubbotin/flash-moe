/*
 * test_qwen36_linear_attn.cu — substep-by-substep validation of Qwen3.6 layer 0
 * GatedDeltaNet against the python oracle.
 *
 * For each token t in the prompt, runs and checks:
 *   1.  rms_norm(input_layernorm)            vs L00_h_norm
 *   2.  in_proj_qkv (FP8 matvec)             vs L00_qkv_pre_conv
 *   3.  conv1d(kernel=4, depthwise) + SiLU   vs L00_qkv_post_conv[:, t]
 *   4.  in_proj_z   (FP8 matvec)             vs L00_z
 *   5.  in_proj_a   (BF16 matvec)            vs L00_a_raw
 *   6.  in_proj_b   (BF16 matvec)            vs L00_b_raw
 *   7.  delta-net step (with L2 norm + Q scale)  vs L00_core_attn_out
 *   8.  RMSNormGated with z                  vs L00_gated_norm_out
 *   9.  out_proj (FP8 matvec)                vs L00_attn_out
 *  10.  attn_out + input                     vs L00_post_attn_resid
 *
 * Build:  make test_qwen36_linear_attn
 * Run:    ./test_qwen36_linear_attn --model-dir /home/user1/qwen36-fp8 \
 *                                   --oracle    /home/user1/qwen36-oracle/france_bin
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <unordered_map>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "safetensors_io.cuh"
#include "qwen36_kernels.cuh"
#include "kernels.cuh"

#define CUDA_OK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA: %s @ %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    std::exit(1); } } while (0)

static const char* LAYER0 = "model.language_model.layers.0";

// ---------------------------------------------------------------------------
// Oracle bin loader
// ---------------------------------------------------------------------------
struct OracleEntry {
    std::vector<uint64_t> shape;
    std::string dtype;
    uint64_t nbytes;
};

struct Oracle {
    std::string dir;
    std::unordered_map<std::string, OracleEntry> manifest;
};

static bool oracle_open(Oracle* O, const std::string& dir) {
    O->dir = dir;
    std::string mpath = dir + "/manifest.json";
    FILE* f = std::fopen(mpath.c_str(), "rb");
    if (!f) { fprintf(stderr, "oracle: cannot open %s\n", mpath.c_str()); return false; }
    std::fseek(f, 0, SEEK_END); long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    std::string body(n, 0); std::fread(&body[0], 1, n, f); std::fclose(f);

    // very tiny json walker — looks for "key": { ... "shape": [...], "dtype": "..." }
    size_t i = 0;
    while (i < body.size()) {
        size_t kq = body.find('"', i);
        if (kq == std::string::npos) break;
        size_t kqe = body.find('"', kq + 1);
        std::string key = body.substr(kq + 1, kqe - kq - 1);
        size_t obj_open = body.find('{', kqe);
        if (obj_open == std::string::npos) break;
        // find matching close brace
        int depth = 1; size_t j = obj_open + 1;
        for (; j < body.size() && depth > 0; j++) {
            if (body[j] == '{') depth++;
            else if (body[j] == '}') depth--;
        }
        std::string obj = body.substr(obj_open, j - obj_open);
        i = j;

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
        if (nb != std::string::npos) {
            e.nbytes = std::strtoull(obj.c_str() + nb + 9, nullptr, 10);
        }
        if (!e.dtype.empty()) O->manifest.emplace(key, std::move(e));
    }
    fprintf(stderr, "oracle: %zu tensors loaded from %s\n", O->manifest.size(), dir.c_str());
    return true;
}

static bool oracle_read_f32(Oracle* O, const std::string& key, std::vector<float>& out) {
    auto it = O->manifest.find(key);
    if (it == O->manifest.end()) { fprintf(stderr, "oracle: missing key %s\n", key.c_str()); return false; }
    if (it->second.dtype != "f32") { fprintf(stderr, "oracle: %s is %s, not f32\n", key.c_str(), it->second.dtype.c_str()); return false; }
    std::string p = O->dir + "/" + key + ".bin";
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "oracle: open %s\n", p.c_str()); return false; }
    out.resize(it->second.nbytes / 4);
    if (std::fread(out.data(), 1, it->second.nbytes, f) != it->second.nbytes) {
        std::fclose(f); fprintf(stderr, "oracle: short read %s\n", p.c_str()); return false;
    }
    std::fclose(f);
    return true;
}

// ---------------------------------------------------------------------------
// Compare device tensor against host (oracle slice)
// ---------------------------------------------------------------------------
static void compare(const char* tag, const float* d_actual, const float* h_ref,
                    size_t n, double l2rel_thresh = 0.05)
{
    std::vector<float> h_actual(n);
    CUDA_OK(cudaMemcpy(h_actual.data(), d_actual, n * sizeof(float), cudaMemcpyDeviceToHost));
    double max_abs = 0;
    double l2_a = 0, l2_r = 0, l2_d = 0;
    int worst = 0;
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

// ---------------------------------------------------------------------------
// Helpers: load tensor from safetensors as raw bytes onto device
// ---------------------------------------------------------------------------
static uint8_t* upload_bytes(st::ModelDir* M, const std::string& name, size_t* out_bytes = nullptr) {
    std::vector<uint8_t> buf;
    if (!st::read_bytes(M, name, buf)) std::exit(1);
    uint8_t* d; CUDA_OK(cudaMalloc(&d, buf.size()));
    CUDA_OK(cudaMemcpy(d, buf.data(), buf.size(), cudaMemcpyHostToDevice));
    if (out_bytes) *out_bytes = buf.size();
    return d;
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    std::string model_dir, oracle_dir;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--model-dir" && i + 1 < argc) model_dir  = argv[++i];
        else if (a == "--oracle"    && i + 1 < argc) oracle_dir = argv[++i];
        else { fprintf(stderr, "usage: %s --model-dir DIR --oracle DIR\n", argv[0]); return 1; }
    }
    if (model_dir.empty() || oracle_dir.empty()) {
        fprintf(stderr, "need --model-dir and --oracle\n"); return 1;
    }

    cudaDeviceProp p{}; cudaGetDeviceProperties(&p, 0);
    printf("device: %s sm_%d%d\n", p.name, p.major, p.minor);

    // ---- open model + oracle ----
    st::ModelDir M{}; if (!st::open(&M, model_dir)) return 1;
    Oracle O{}; if (!oracle_open(&O, oracle_dir)) return 1;

    // shapes
    const uint32_t H = 2048;
    const uint32_t QKV_DIM = 8192;          // 2048 (Q) + 2048 (K) + 4096 (V)
    const uint32_t Z_DIM   = 4096;
    const uint32_t NV = 32, NK = 16, HEAD_K = 128, HEAD_V = 128;
    const uint32_t KEY_DIM_TOT = NK * HEAD_K;     // 2048
    const uint32_t VAL_DIM_TOT = NV * HEAD_V;     // 4096
    const uint32_t K_HEADS_PER_V = NV / NK;       // 2
    const uint32_t CONV_K = 4;
    const float    RMS_EPS = 1e-6f;

    // ---- weights ----
    auto T = [&](const std::string& nm){ return std::string(LAYER0) + "." + nm; };
    uint16_t* d_in_ln_w   = (uint16_t*)upload_bytes(&M, T("input_layernorm.weight"));
    uint8_t*  d_qkv_w     = upload_bytes(&M, T("linear_attn.in_proj_qkv.weight"));
    __nv_bfloat16* d_qkv_s = (__nv_bfloat16*)upload_bytes(&M, T("linear_attn.in_proj_qkv.weight_scale_inv"));
    uint8_t*  d_z_w       = upload_bytes(&M, T("linear_attn.in_proj_z.weight"));
    __nv_bfloat16* d_z_s  = (__nv_bfloat16*)upload_bytes(&M, T("linear_attn.in_proj_z.weight_scale_inv"));
    uint16_t* d_a_w       = (uint16_t*)upload_bytes(&M, T("linear_attn.in_proj_a.weight"));
    uint16_t* d_b_w       = (uint16_t*)upload_bytes(&M, T("linear_attn.in_proj_b.weight"));
    uint16_t* d_conv_w    = (uint16_t*)upload_bytes(&M, T("linear_attn.conv1d.weight"));
    uint16_t* d_A_log_bf  = (uint16_t*)upload_bytes(&M, T("linear_attn.A_log"));
    uint16_t* d_dt_bias_bf= (uint16_t*)upload_bytes(&M, T("linear_attn.dt_bias"));
    uint16_t* d_norm_w    = (uint16_t*)upload_bytes(&M, T("linear_attn.norm.weight"));
    uint8_t*  d_out_w     = upload_bytes(&M, T("linear_attn.out_proj.weight"));
    __nv_bfloat16* d_out_s= (__nv_bfloat16*)upload_bytes(&M, T("linear_attn.out_proj.weight_scale_inv"));

    // A_log is bf16 in the checkpoint — convert to f32 once for compute_decay_beta
    float* d_A_log; CUDA_OK(cudaMalloc(&d_A_log, NV * sizeof(float)));
    {
        std::vector<uint16_t> h_bf(NV);
        CUDA_OK(cudaMemcpy(h_bf.data(), d_A_log_bf, NV * 2, cudaMemcpyDeviceToHost));
        std::vector<float> h_f(NV);
        for (uint32_t i = 0; i < NV; i++) h_f[i] = st::bf16_to_f32(h_bf[i]);
        CUDA_OK(cudaMemcpy(d_A_log, h_f.data(), NV * 4, cudaMemcpyHostToDevice));
    }

    // ---- oracle slices ----
    std::vector<float> ora_input, ora_h_norm, ora_qkv_pre, ora_qkv_post, ora_z,
                       ora_a_raw, ora_b_raw, ora_core, ora_gated, ora_attn_out,
                       ora_post_resid;
    auto load = [&](const char* k, std::vector<float>& v){
        if (!oracle_read_f32(&O, k, v)) std::exit(1);
    };
    load("L00_input",            ora_input);
    load("L00_h_norm",           ora_h_norm);
    load("L00_qkv_pre_conv",     ora_qkv_pre);
    load("L00_qkv_post_conv",    ora_qkv_post);
    load("L00_z",                ora_z);
    load("L00_a_raw",            ora_a_raw);
    load("L00_b_raw",            ora_b_raw);
    load("L00_core_attn_out",    ora_core);
    load("L00_gated_norm_out",   ora_gated);
    load("L00_attn_out",         ora_attn_out);
    load("L00_post_attn_resid",  ora_post_resid);

    const auto& sh_input = O.manifest["L00_input"].shape;
    const uint32_t SEQ = (uint32_t)sh_input[1];
    printf("seq_len = %u\n", SEQ);

    // ---- device scratch ----
    float *d_x, *d_h_norm, *d_qkv_pre, *d_qkv_post_full, *d_q_rep, *d_k_rep, *d_v_full;
    float *d_z_out, *d_a, *d_b, *d_g, *d_beta, *d_core_out, *d_norm_out, *d_attn_out, *d_resid;
    CUDA_OK(cudaMalloc(&d_x,            H * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_h_norm,       H * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_qkv_pre,      QKV_DIM * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_qkv_post_full,QKV_DIM * sizeof(float)));   // post-conv, post-SiLU
    CUDA_OK(cudaMalloc(&d_q_rep,        VAL_DIM_TOT * sizeof(float)));   // 32 * 128
    CUDA_OK(cudaMalloc(&d_k_rep,        VAL_DIM_TOT * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_v_full,       VAL_DIM_TOT * sizeof(float)));   // 4096
    CUDA_OK(cudaMalloc(&d_z_out,        Z_DIM * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_a,            NV * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_b,            NV * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_g,            NV * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_beta,         NV * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_core_out,     VAL_DIM_TOT * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_norm_out,     VAL_DIM_TOT * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_attn_out,     H * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_resid,        H * sizeof(float)));

    // conv state: [3, QKV_DIM] (last 3 inputs), zeroed for token 0
    float* d_conv_state; CUDA_OK(cudaMalloc(&d_conv_state, 3 * QKV_DIM * sizeof(float)));
    CUDA_OK(cudaMemset(d_conv_state, 0, 3 * QKV_DIM * sizeof(float)));

    // delta state: [NV, HEAD_K, HEAD_V] = 32*128*128 floats
    float* d_delta_state; CUDA_OK(cudaMalloc(&d_delta_state, NV * HEAD_K * HEAD_V * sizeof(float)));
    CUDA_OK(cudaMemset(d_delta_state, 0, NV * HEAD_K * HEAD_V * sizeof(float)));

    const float Q_SCALE = 1.0f / std::sqrt((float)HEAD_K);

    // ---- per-token loop ----
    for (uint32_t t = 0; t < SEQ; t++) {
        printf("\n=== token t=%u ===\n", t);

        // upload input for this token
        CUDA_OK(cudaMemcpy(d_x, ora_input.data() + t * H, H * 4, cudaMemcpyHostToDevice));

        // 1. rms_norm
        if (t == 0) {
            std::vector<float> tmp_x(8); std::vector<uint16_t> tmp_w(8);
            CUDA_OK(cudaMemcpy(tmp_x.data(), d_x, 8*4, cudaMemcpyDeviceToHost));
            CUDA_OK(cudaMemcpy(tmp_w.data(), d_in_ln_w, 8*2, cudaMemcpyDeviceToHost));
            printf("  d_x[0..7] = "); for (int i=0;i<8;i++) printf("%.4f ", tmp_x[i]); printf("\n");
            printf("  d_w[0..7] = "); for (int i=0;i<8;i++) printf("%.4f ", st::bf16_to_f32(tmp_w[i])); printf("\n");
            printf("  ora_x[0..7] = "); for (int i=0;i<8;i++) printf("%.4f ", ora_input[i]); printf("\n");
            printf("  ora_h_norm[0..7] = "); for (int i=0;i<8;i++) printf("%.4f ", ora_h_norm[i]); printf("\n");
        }
        qwen36::launch_rms_norm_bf16_plus_one(d_x, d_in_ln_w, d_h_norm, H, RMS_EPS);
        if (t == 0) {
            std::vector<float> tmp_h(8);
            CUDA_OK(cudaMemcpy(tmp_h.data(), d_h_norm, 8*4, cudaMemcpyDeviceToHost));
            printf("  d_h_norm[0..7] = "); for (int i=0;i<8;i++) printf("%.4f ", tmp_h[i]); printf("\n");
        }
        compare("rms_norm", d_h_norm, ora_h_norm.data() + t * H, H);

        // 2. in_proj_qkv (FP8)
        qwen36::launch_dequant_matvec_fp8_block128(d_qkv_w, d_qkv_s, d_h_norm, d_qkv_pre, QKV_DIM, H);
        compare("in_proj_qkv", d_qkv_pre, ora_qkv_pre.data() + t * QKV_DIM, QKV_DIM);

        // 3. conv1d step + SiLU
        // conv1d_step takes input[QKV_DIM], rolls history, outputs SiLU(conv).
        dim3 cb(256), cg((QKV_DIM + 255) / 256);
        conv1d_step<<<cg, cb>>>(d_conv_state, d_qkv_pre, d_conv_w, d_qkv_post_full, QKV_DIM);
        // Oracle qkv_post_conv: shape [1, 8192, seq+padding=8] captured BEFORE the
        // [..., :seq_len] slice and BEFORE SiLU. The model takes positions 0..4 of the
        // 8 outputs (rest is the right-padded tail it discards). Position t of output
        // corresponds to inputs [t-3..t] with leading zeros for negative indices —
        // exactly what our rolling-state conv1d_step produces. So compare against
        // SiLU(oracle[c, t]).
        {
            std::vector<float> ref(QKV_DIM);
            const float* base = ora_qkv_post.data();
            const uint32_t SEQ_PAD = (uint32_t)O.manifest["L00_qkv_post_conv"].shape[2];  // 8
            for (uint32_t c = 0; c < QKV_DIM; c++) {
                float v = base[c * SEQ_PAD + t];
                ref[c] = v / (1.0f + std::exp(-v));   // SiLU
            }
            compare("conv1d+silu", d_qkv_post_full, ref.data(), QKV_DIM);
        }

        // 4. in_proj_z (FP8)
        qwen36::launch_dequant_matvec_fp8_block128(d_z_w, d_z_s, d_h_norm, d_z_out, Z_DIM, H);
        compare("in_proj_z", d_z_out, ora_z.data() + t * Z_DIM, Z_DIM);

        // 5. in_proj_a (BF16)
        launch_matvec_bf16(d_a_w, d_h_norm, d_a, NV, H);
        compare("in_proj_a", d_a, ora_a_raw.data() + t * NV, NV);

        // 6. in_proj_b (BF16)
        launch_matvec_bf16(d_b_w, d_h_norm, d_b, NV, H);
        compare("in_proj_b", d_b, ora_b_raw.data() + t * NV, NV);

        // 7. compute decay + beta gate
        compute_decay_beta<<<1, NV>>>(d_a, d_b, d_A_log, d_dt_bias_bf, d_g, d_beta);

        // 7b. q/k/v split + repeat-interleave Q,K from 16 to 32 heads (chunked layout).
        // After conv: layout is q[2048] | k[2048] | v[4096] in d_qkv_post_full.
        // Reshape q -> [16 heads, 128 dim], repeat to [32, 128] = head_id/2 chunked.
        {
            // For chunked repeat-interleave, copy q row k_h to v_h = 2*k_h and 2*k_h+1.
            // Simpler: write a small kernel inline. Use cudaMemcpy 2D.
            // src: [16 rows, 128 cols] stride 128*4 = 512
            // dst: [32 rows, 128 cols] stride 128*4 = 512
            // Pattern: dst[2k]=src[k], dst[2k+1]=src[k]. Use cudaMemcpy2D twice with offset.
            CUDA_OK(cudaMemcpy2D(d_q_rep,             2 * HEAD_K * 4,
                                 d_qkv_post_full,         HEAD_K * 4,
                                 HEAD_K * 4, NK, cudaMemcpyDeviceToDevice));
            CUDA_OK(cudaMemcpy2D(d_q_rep + HEAD_K,    2 * HEAD_K * 4,
                                 d_qkv_post_full,         HEAD_K * 4,
                                 HEAD_K * 4, NK, cudaMemcpyDeviceToDevice));
            CUDA_OK(cudaMemcpy2D(d_k_rep,             2 * HEAD_K * 4,
                                 d_qkv_post_full + KEY_DIM_TOT, HEAD_K * 4,
                                 HEAD_K * 4, NK, cudaMemcpyDeviceToDevice));
            CUDA_OK(cudaMemcpy2D(d_k_rep + HEAD_K,    2 * HEAD_K * 4,
                                 d_qkv_post_full + KEY_DIM_TOT, HEAD_K * 4,
                                 HEAD_K * 4, NK, cudaMemcpyDeviceToDevice));
            CUDA_OK(cudaMemcpyAsync(d_v_full, d_qkv_post_full + KEY_DIM_TOT * 2,
                                    VAL_DIM_TOT * 4, cudaMemcpyDeviceToDevice));
        }

        // 7c. L2 norm Q and K per head (eps=1e-6)
        l2_norm_qk<<<NV, 128>>>(d_q_rep, d_k_rep, HEAD_K);

        // 7d. Scale Q by 1/sqrt(HEAD_K)
        vec_scale<<<(VAL_DIM_TOT + 255) / 256, 256>>>(d_q_rep, Q_SCALE, VAL_DIM_TOT);

        // 7e. delta_net step.
        //
        // gated_delta_net_step internally maps each value-head h to a K-head via
        //   kh = h / k_heads_per_v
        // and reads q/k at offset kh*128. Since we already did the K-head
        // replication on the host (Q,K reshaped from [16,128] → [32,128] with
        // each k-head duplicated to two consecutive v-head slots), the kernel
        // must address d_q_rep / d_k_rep with one slot per v-head — i.e.
        // k_heads_per_v=1.
        gated_delta_net_step<<<NV, HEAD_V>>>(
            d_delta_state, d_q_rep, d_k_rep, d_v_full, d_g, d_beta, d_core_out, /*k_heads_per_v=*/1);
        if (t == 0) {
            std::vector<float> tmp_q(8), tmp_k(8), tmp_v(8), tmp_g(4), tmp_b(4), tmp_o(8);
            CUDA_OK(cudaMemcpy(tmp_q.data(), d_q_rep, 8*4, cudaMemcpyDeviceToHost));
            CUDA_OK(cudaMemcpy(tmp_k.data(), d_k_rep, 8*4, cudaMemcpyDeviceToHost));
            CUDA_OK(cudaMemcpy(tmp_v.data(), d_v_full, 8*4, cudaMemcpyDeviceToHost));
            CUDA_OK(cudaMemcpy(tmp_g.data(), d_g, 4*4, cudaMemcpyDeviceToHost));
            CUDA_OK(cudaMemcpy(tmp_b.data(), d_beta, 4*4, cudaMemcpyDeviceToHost));
            CUDA_OK(cudaMemcpy(tmp_o.data(), d_core_out, 8*4, cudaMemcpyDeviceToHost));
            printf("  d_q_rep[0..7]   = "); for (auto x:tmp_q) printf("%.5f ", x); printf("\n");
            printf("  d_k_rep[0..7]   = "); for (auto x:tmp_k) printf("%.5f ", x); printf("\n");
            printf("  d_v_full[0..7]  = "); for (auto x:tmp_v) printf("%.5f ", x); printf("\n");
            printf("  d_g[0..3]       = "); for (auto x:tmp_g) printf("%.5f ", x); printf("\n");
            printf("  d_beta[0..3]    = "); for (auto x:tmp_b) printf("%.5f ", x); printf("\n");
            printf("  d_core_out[0..7]= "); for (auto x:tmp_o) printf("%.5e ", x); printf("\n");
            printf("  ora_core[0..7]  = "); for (int i=0;i<8;i++) printf("%.5e ", ora_core[i]); printf("\n");
            // per-head L2 of our output vs oracle
            std::vector<float> all(VAL_DIM_TOT);
            CUDA_OK(cudaMemcpy(all.data(), d_core_out, VAL_DIM_TOT * 4, cudaMemcpyDeviceToHost));
            printf("  per-head L2 (act vs ref):\n");
            for (uint32_t h = 0; h < NV; h++) {
                double la = 0, lr = 0;
                for (uint32_t v = 0; v < HEAD_V; v++) {
                    float a = all[h * HEAD_V + v], r = ora_core[h * HEAD_V + v];
                    la += a*a; lr += r*r;
                }
                printf("    h=%2u  act=%.5f  ref=%.5f  ratio=%.3f%s\n",
                       h, std::sqrt(la), std::sqrt(lr),
                       std::sqrt(la) / (std::sqrt(lr) + 1e-9),
                       std::abs(std::sqrt(la) - std::sqrt(lr)) > 0.01 ? "  *" : "");
            }
        }

        compare("delta_step", d_core_out, ora_core.data() + (size_t)t * VAL_DIM_TOT, VAL_DIM_TOT);

        // 8. RMSNormGated with z
        gated_rms_norm<<<NV, HEAD_V>>>(d_core_out, d_z_out, d_norm_w, d_norm_out, HEAD_V, RMS_EPS);
        compare("gated_norm", d_norm_out, ora_gated.data() + (size_t)t * VAL_DIM_TOT, VAL_DIM_TOT);

        // 9. out_proj (FP8)
        qwen36::launch_dequant_matvec_fp8_block128(d_out_w, d_out_s, d_norm_out, d_attn_out, H, VAL_DIM_TOT);
        compare("out_proj", d_attn_out, ora_attn_out.data() + t * H, H);

        // 10. residual
        launch_residual_add(d_x, d_attn_out, d_resid, H);
        compare("residual", d_resid, ora_post_resid.data() + t * H, H);
    }

    st::close(&M);
    printf("\ndone.\n");
    return 0;
}
