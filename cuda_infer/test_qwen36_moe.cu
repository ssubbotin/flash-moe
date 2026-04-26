/*
 * test_qwen36_moe.cu — substep validation of Qwen3.6 layer 0 MoE block against
 * the oracle.
 *
 * Per token t in the prompt:
 *   1. h_norm2 = rms_norm_plus_one(post_attn_resid[t], post_attention_layernorm.weight)
 *   2. router_logits = bf16_matvec(gate.weight, h_norm2)                    [256]
 *   3. softmax + top-K=8  → indices, weights ; weights /= weights.sum()
 *   4. for each of K selected experts e:
 *        gate_pre = fp8_matvec(experts[e].gate_proj.weight, h_norm2)        [512]
 *        up_pre   = fp8_matvec(experts[e].up_proj.weight,   h_norm2)        [512]
 *        hidden_e = silu(gate_pre) * up_pre                                 [512]
 *        down_e   = fp8_matvec(experts[e].down_proj.weight, hidden_e)       [2048]
 *        expert_sum += routing_weight[e] * down_e
 *   5. shared expert: same MLP form, full strength (no routing weight)
 *   6. gated_shared = sigmoid(bf16_matvec(shared_expert_gate.weight, h_norm2)) * shared_out
 *   7. mlp_out = expert_sum + gated_shared
 *   8. compare against L00_mlp_out[t]
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

static const char* LAYER0 = "model.language_model.layers.0";

// ---------------------------------------------------------------------------
// Oracle bin loader (same as test_qwen36_linear_attn)
// ---------------------------------------------------------------------------
struct OracleEntry { std::vector<uint64_t> shape; std::string dtype; uint64_t nbytes; };
struct Oracle { std::string dir; std::unordered_map<std::string, OracleEntry> manifest; };

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

// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
struct ExpertW {
    uint8_t* gate_w;  __nv_bfloat16* gate_s;
    uint8_t* up_w;    __nv_bfloat16* up_s;
    uint8_t* down_w;  __nv_bfloat16* down_s;
};

int main(int argc, char** argv) {
    std::string model_dir, oracle_dir;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--model-dir" && i + 1 < argc) model_dir = argv[++i];
        else if (a == "--oracle"    && i + 1 < argc) oracle_dir = argv[++i];
        else { fprintf(stderr, "usage: %s --model-dir DIR --oracle DIR\n", argv[0]); return 1; }
    }
    if (model_dir.empty() || oracle_dir.empty()) { fprintf(stderr, "need --model-dir and --oracle\n"); return 1; }

    cudaDeviceProp p{}; cudaGetDeviceProperties(&p, 0);
    printf("device: %s sm_%d%d\n", p.name, p.major, p.minor);

    st::ModelDir M{}; if (!st::open(&M, model_dir)) return 1;
    Oracle O{}; if (!oracle_open(&O, oracle_dir)) return 1;

    const uint32_t H        = 2048;
    const uint32_t INTER    = 512;       // moe_intermediate_size
    const uint32_t N_EXPERTS= 256;
    const uint32_t TOP_K    = 8;
    const float    RMS_EPS  = 1e-6f;

    // ---- weights ----
    auto T = [&](const std::string& nm){ return std::string(LAYER0) + "." + nm; };
    uint16_t* d_post_ln_w   = (uint16_t*)upload_bytes(&M, T("post_attention_layernorm.weight"));
    uint16_t* d_router_w    = (uint16_t*)upload_bytes(&M, T("mlp.gate.weight"));               // [256, 2048] bf16
    uint16_t* d_shared_gate = (uint16_t*)upload_bytes(&M, T("mlp.shared_expert_gate.weight")); // [1, 2048] bf16

    fprintf(stderr, "loading 256 routed experts...\n");
    std::vector<ExpertW> EX(N_EXPERTS);
    for (uint32_t e = 0; e < N_EXPERTS; e++) {
        char buf[256];
        auto exT = [&](const char* nm){
            snprintf(buf, sizeof(buf), "mlp.experts.%u.%s", e, nm);
            return T(buf);
        };
        EX[e].gate_w = upload_bytes(&M, exT("gate_proj.weight"));
        EX[e].gate_s = (__nv_bfloat16*)upload_bytes(&M, exT("gate_proj.weight_scale_inv"));
        EX[e].up_w   = upload_bytes(&M, exT("up_proj.weight"));
        EX[e].up_s   = (__nv_bfloat16*)upload_bytes(&M, exT("up_proj.weight_scale_inv"));
        EX[e].down_w = upload_bytes(&M, exT("down_proj.weight"));
        EX[e].down_s = (__nv_bfloat16*)upload_bytes(&M, exT("down_proj.weight_scale_inv"));
    }
    ExpertW SH;
    SH.gate_w = upload_bytes(&M, T("mlp.shared_expert.gate_proj.weight"));
    SH.gate_s = (__nv_bfloat16*)upload_bytes(&M, T("mlp.shared_expert.gate_proj.weight_scale_inv"));
    SH.up_w   = upload_bytes(&M, T("mlp.shared_expert.up_proj.weight"));
    SH.up_s   = (__nv_bfloat16*)upload_bytes(&M, T("mlp.shared_expert.up_proj.weight_scale_inv"));
    SH.down_w = upload_bytes(&M, T("mlp.shared_expert.down_proj.weight"));
    SH.down_s = (__nv_bfloat16*)upload_bytes(&M, T("mlp.shared_expert.down_proj.weight_scale_inv"));

    fprintf(stderr, "expert load done.\n");

    // ---- oracle ----
    std::vector<float> ora_post_resid, ora_mlp_out;
    if (!oracle_read_f32(&O, "L00_post_attn_resid", ora_post_resid)) return 1;
    if (!oracle_read_f32(&O, "L00_mlp_out",         ora_mlp_out))    return 1;
    const auto& sh = O.manifest["L00_post_attn_resid"].shape;
    const uint32_t SEQ = (uint32_t)sh[1];
    printf("seq_len = %u\n", SEQ);

    // ---- scratch ----
    float *d_resid_in, *d_h, *d_router_logits, *d_gate_pre, *d_up_pre, *d_hidden_e, *d_down_e;
    float *d_expert_sum, *d_shared_out, *d_shared_gate_logit, *d_mlp_out;
    CUDA_OK(cudaMalloc(&d_resid_in,         H * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_h,                H * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_router_logits,    N_EXPERTS * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_gate_pre,         INTER * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_up_pre,           INTER * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_hidden_e,         INTER * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_down_e,           H * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_expert_sum,       H * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_shared_out,       H * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_shared_gate_logit,sizeof(float)));
    CUDA_OK(cudaMalloc(&d_mlp_out,          H * sizeof(float)));

    for (uint32_t t = 0; t < SEQ; t++) {
        printf("\n=== token t=%u ===\n", t);

        // 1. post-attn norm
        CUDA_OK(cudaMemcpy(d_resid_in, ora_post_resid.data() + t * H, H * 4, cudaMemcpyHostToDevice));
        qwen36::launch_rms_norm_bf16_plus_one(d_resid_in, d_post_ln_w, d_h, H, RMS_EPS);

        // 2. router logits
        launch_matvec_bf16(d_router_w, d_h, d_router_logits, N_EXPERTS, H);

        // 3. softmax + top-K (host side)
        std::vector<float> logits(N_EXPERTS);
        CUDA_OK(cudaMemcpy(logits.data(), d_router_logits, N_EXPERTS * 4, cudaMemcpyDeviceToHost));
        // softmax
        float lmax = *std::max_element(logits.begin(), logits.end());
        std::vector<double> probs(N_EXPERTS); double psum = 0;
        for (uint32_t i = 0; i < N_EXPERTS; i++) { probs[i] = std::exp((double)logits[i] - lmax); psum += probs[i]; }
        for (uint32_t i = 0; i < N_EXPERTS; i++) probs[i] /= psum;
        // top-K
        std::vector<int> idx(N_EXPERTS); for (uint32_t i = 0; i < N_EXPERTS; i++) idx[i] = i;
        std::partial_sort(idx.begin(), idx.begin() + TOP_K, idx.end(),
                          [&](int a, int b){ return probs[a] > probs[b]; });
        std::vector<int>   topk_idx(TOP_K);
        std::vector<float> topk_w(TOP_K);
        double wsum = 0;
        for (uint32_t k = 0; k < TOP_K; k++) { topk_idx[k] = idx[k]; topk_w[k] = (float)probs[idx[k]]; wsum += topk_w[k]; }
        for (uint32_t k = 0; k < TOP_K; k++) topk_w[k] = (float)((double)topk_w[k] / wsum);

        printf("  top-K experts:");
        for (uint32_t k = 0; k < TOP_K; k++) printf(" %d(%.3f)", topk_idx[k], topk_w[k]);
        printf("\n");

        // 4. accumulate routed expert outputs
        CUDA_OK(cudaMemset(d_expert_sum, 0, H * sizeof(float)));
        for (uint32_t k = 0; k < TOP_K; k++) {
            int e = topk_idx[k];
            qwen36::launch_dequant_matvec_fp8_block128(EX[e].gate_w, EX[e].gate_s, d_h, d_gate_pre, INTER, H);
            qwen36::launch_dequant_matvec_fp8_block128(EX[e].up_w,   EX[e].up_s,   d_h, d_up_pre,   INTER, H);
            // SwiGLU: hidden = silu(gate) * up
            launch_swiglu(d_gate_pre, d_up_pre, d_hidden_e, INTER);
            qwen36::launch_dequant_matvec_fp8_block128(EX[e].down_w, EX[e].down_s, d_hidden_e, d_down_e, H, INTER);
            // expert_sum += weight * down_e
            // No fused kernel handy; reuse vec_scale + residual_add. Cheaper to just write a tiny lambda.
            // Use a kernel via residual_add+ scale via a temp.
            // Simpler: do (d_expert_sum, d_down_e * w) via a one-off kernel below.
            // We can use existing "moe_combine_residual"-style logic; here just do a manual axpy.
            // Inline kernel:
            {
                float w = topk_w[k];
                int B = 256, G = (H + B - 1) / B;
                vec_scale<<<G, B>>>(d_down_e, w, H);
                launch_residual_add(d_expert_sum, d_down_e, d_expert_sum, H);
            }
        }

        // 5. shared expert (same MLP form, no routing weight)
        qwen36::launch_dequant_matvec_fp8_block128(SH.gate_w, SH.gate_s, d_h, d_gate_pre, INTER, H);
        qwen36::launch_dequant_matvec_fp8_block128(SH.up_w,   SH.up_s,   d_h, d_up_pre,   INTER, H);
        launch_swiglu(d_gate_pre, d_up_pre, d_hidden_e, INTER);
        qwen36::launch_dequant_matvec_fp8_block128(SH.down_w, SH.down_s, d_hidden_e, d_shared_out, H, INTER);

        // 6. shared_expert_gate: scalar = sigmoid(W_gate · h), W_gate is [1, 2048]
        launch_matvec_bf16(d_shared_gate, d_h, d_shared_gate_logit, 1, H);
        float h_logit = 0;
        CUDA_OK(cudaMemcpy(&h_logit, d_shared_gate_logit, 4, cudaMemcpyDeviceToHost));
        float h_sig = 1.0f / (1.0f + std::exp(-h_logit));
        // shared_out *= h_sig
        {
            int B = 256, G = (H + B - 1) / B;
            vec_scale<<<G, B>>>(d_shared_out, h_sig, H);
        }

        // 7. final mlp_out = expert_sum + gated_shared
        launch_residual_add(d_expert_sum, d_shared_out, d_mlp_out, H);

        // 8. compare against L00_mlp_out
        compare("mlp_out", d_mlp_out, ora_mlp_out.data() + t * H, H);
    }

    st::close(&M);
    printf("\ndone.\n");
    return 0;
}
