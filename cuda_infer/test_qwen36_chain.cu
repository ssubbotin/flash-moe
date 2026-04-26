/*
 * test_qwen36_chain.cu — full 40-layer end-to-end inference of Qwen3.6-35B-A3B-FP8.
 *
 * Starts from L00_input (= embed for the prompt), runs every decoder layer,
 * applies the final RMSNorm and lm_head, and compares against the bf16 oracle:
 *   - per-layer: post_layer  vs  L{N+1}_input  (same tensor in HF)
 *                                                 — measures error growth
 *   - end:       final_norm  vs  oracle.final_norm
 *                logits[t,top1]  vs  oracle.logits[t,top1]   (top-1 token id)
 *
 * VRAM strategy: load each layer's weights, run forward over SEQ tokens, free
 * expert weights before loading the next layer. Non-expert weights are tiny
 * (~50MB/layer) and freed too. Final norm + lm_head loaded once after all
 * layers complete. Two ping-pong buffers swap input/output between layers.
 *
 * Build: make test_qwen36_chain
 * Run:   ./test_qwen36_chain --model-dir /path/to/qwen36-fp8 \
 *                            --oracle    /path/to/qwen36-oracle/france_bin
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

#include "safetensors_io.cuh"
#include "qwen36_layer_runner.cuh"

#define CUDA_OK_LOCAL(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA: %s @ %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    std::exit(1); } } while (0)

// ---- oracle loader (same as before) ----
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

static bool oracle_read_i64(Oracle* O, const std::string& key, std::vector<int64_t>& out) {
    auto it = O->manifest.find(key);
    if (it == O->manifest.end()) return false;
    std::string p = O->dir + "/" + key + ".bin";
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) return false;
    out.resize(it->second.nbytes / 8);
    if (std::fread(out.data(), 1, it->second.nbytes, f) != it->second.nbytes) { std::fclose(f); return false; }
    std::fclose(f);
    return true;
}

static double l2rel_per_token(const float* d_actual, const float* h_ref, uint32_t SEQ, uint32_t H) {
    std::vector<float> h_act(SEQ * H);
    CUDA_OK_LOCAL(cudaMemcpy(h_act.data(), d_actual, SEQ * H * 4, cudaMemcpyDeviceToHost));
    double l2_d = 0, l2_r = 0;
    for (uint32_t i = 0; i < SEQ * H; i++) {
        double d = (double)h_act[i] - (double)h_ref[i];
        l2_d += d*d; l2_r += (double)h_ref[i] * (double)h_ref[i];
    }
    return std::sqrt(l2_d) / (std::sqrt(l2_r) + 1e-9);
}

int main(int argc, char** argv) {
    std::string model_dir, oracle_dir;
    int stop_at = -1;  // optional early stop for debugging (-1 = all 40)
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--model-dir" && i + 1 < argc) model_dir  = argv[++i];
        else if (a == "--oracle"    && i + 1 < argc) oracle_dir = argv[++i];
        else if (a == "--stop-at"   && i + 1 < argc) stop_at    = std::atoi(argv[++i]);
        else { fprintf(stderr, "usage: %s --model-dir DIR --oracle DIR [--stop-at N]\n", argv[0]); return 1; }
    }
    if (model_dir.empty() || oracle_dir.empty()) return 1;

    cudaDeviceProp p{}; cudaGetDeviceProperties(&p, 0);
    printf("device: %s sm_%d%d  total mem=%.1f GB\n", p.name, p.major, p.minor,
           (double)p.totalGlobalMem / 1e9);

    st::ModelDir M{}; if (!st::open(&M, model_dir)) return 1;
    Oracle O{}; if (!oracle_open(&O, oracle_dir)) return 1;

    // ---- load oracle slices ----
    std::vector<float> ora_input, ora_final_norm, ora_logits;
    std::vector<int64_t> ora_input_ids;
    if (!oracle_read_f32(&O, "L00_input",   ora_input))      return 1;
    if (!oracle_read_f32(&O, "final_norm",  ora_final_norm)) return 1;
    if (!oracle_read_f32(&O, "logits",      ora_logits))     return 1;
    if (!oracle_read_i64(&O, "input_ids",   ora_input_ids))  return 1;

    const auto& sh = O.manifest["L00_input"].shape;
    const uint32_t SEQ = (uint32_t)sh[1];
    printf("seq_len = %u  prompt token ids:", SEQ);
    for (uint32_t i = 0; i < SEQ; i++) printf(" %lld", (long long)ora_input_ids[i]);
    printf("\n");

    // ---- preload all per-layer post_layer oracle for chain validation ----
    std::vector<std::vector<float>> ora_post_layer(q36::NUM_LAYERS);
    for (uint32_t li = 0; li < q36::NUM_LAYERS; li++) {
        char k[64]; snprintf(k, sizeof(k), "L%02u_post_layer", li);
        if (!oracle_read_f32(&O, k, ora_post_layer[li])) return 1;
    }

    // ---- ping-pong buffers ----
    float *d_buf_a, *d_buf_b;
    CUDA_OK_LOCAL(cudaMalloc(&d_buf_a, SEQ * q36::H * 4));
    CUDA_OK_LOCAL(cudaMalloc(&d_buf_b, SEQ * q36::H * 4));
    float *d_resid1; CUDA_OK_LOCAL(cudaMalloc(&d_resid1, SEQ * q36::H * 4));
    CUDA_OK_LOCAL(cudaMemcpy(d_buf_a, ora_input.data(), SEQ * q36::H * 4, cudaMemcpyHostToDevice));

    // ---- shared scratch (small) ----
    q36::LinearAttnScratch  Lscratch{}; q36::alloc_linear_attn_scratch(&Lscratch);
    q36::FullAttnScratch    Fscratch{}; q36::alloc_full_attn_scratch  (&Fscratch, SEQ);
    q36::MoEScratch         Mscratch{}; q36::alloc_moe_scratch        (&Mscratch);

    const uint32_t LAST = (stop_at < 0) ? q36::NUM_LAYERS : (uint32_t)stop_at;
    printf("running %u layers...\n", LAST);

    float* d_in  = d_buf_a;
    float* d_out = d_buf_b;
    for (uint32_t li = 0; li < LAST; li++) {
        std::string prefix = q36::layer_prefix(li);
        bool full = q36::is_full_attn(li);
        printf("\n[layer %2u %s]\n", li, full ? "full" : "linear");

        // attention path
        if (full) {
            q36::FullAttnW F{}; q36::load_full_attn(&M, prefix, &F);
            q36::run_full_attn(F, SEQ, d_in, d_resid1, &Fscratch);
            q36::free_full_attn(&F);
        } else {
            q36::LinearAttnW Lw{}; q36::load_linear_attn(&M, prefix, &Lw);
            q36::run_linear_attn(Lw, SEQ, d_in, d_resid1, &Lscratch);
            q36::free_linear_attn(&Lw);
        }

        // MoE path
        q36::MoEW X{}; q36::load_moe(&M, prefix, &X);
        q36::run_moe(X, SEQ, d_resid1, d_out, &Mscratch);
        q36::free_moe(&X);

        double rel = l2rel_per_token(d_out, ora_post_layer[li].data(), SEQ, q36::H);
        printf("  post_layer L%02u  L2rel=%.4e\n", li, rel);

        std::swap(d_in, d_out);
    }

    // d_in now holds the final residual stream after layer LAST-1.
    if (LAST < q36::NUM_LAYERS) {
        printf("\nstopped early at layer %u; skipping final norm + lm_head.\n", LAST);
        cudaFree(d_buf_a); cudaFree(d_buf_b); cudaFree(d_resid1);
        st::close(&M); return 0;
    }

    // ---- final norm ----
    printf("\nloading final_norm + lm_head...\n");
    uint16_t* d_final_norm_w = (uint16_t*)q36::upload_bytes(&M, "model.language_model.norm.weight");
    uint16_t* d_lm_head      = (uint16_t*)q36::upload_bytes(&M, "lm_head.weight");

    float* d_normed; CUDA_OK_LOCAL(cudaMalloc(&d_normed, SEQ * q36::H * 4));
    for (uint32_t t = 0; t < SEQ; t++) {
        qwen36::launch_rms_norm_bf16_plus_one(
            d_in     + (size_t)t * q36::H,
            d_final_norm_w,
            d_normed + (size_t)t * q36::H,
            q36::H, q36::RMS_EPS);
    }

    double final_rel = l2rel_per_token(d_normed, ora_final_norm.data(), SEQ, q36::H);
    printf("\nfinal_norm        L2rel=%.4e\n", final_rel);

    // ---- lm_head: per-token logits (vocab = 248320 ≈ 1MB output per token) ----
    float* d_logits; CUDA_OK_LOCAL(cudaMalloc(&d_logits, q36::VOCAB * 4));
    std::vector<float> h_logits(q36::VOCAB);
    int oracle_top1[16] = {0};
    int our_top1[16]    = {0};
    for (uint32_t t = 0; t < SEQ && t < 16; t++) {
        launch_matvec_bf16(d_lm_head, d_normed + (size_t)t * q36::H, d_logits, q36::VOCAB, q36::H);
        CUDA_OK_LOCAL(cudaMemcpy(h_logits.data(), d_logits, q36::VOCAB * 4, cudaMemcpyDeviceToHost));

        int our_best = 0; float our_max = h_logits[0];
        for (uint32_t v = 1; v < q36::VOCAB; v++) if (h_logits[v] > our_max) { our_max = h_logits[v]; our_best = (int)v; }

        const float* ora_t = ora_logits.data() + (size_t)t * q36::VOCAB;
        int ora_best = 0; float ora_max = ora_t[0];
        for (uint32_t v = 1; v < q36::VOCAB; v++) if (ora_t[v] > ora_max) { ora_max = ora_t[v]; ora_best = (int)v; }

        // L2-rel of full logits
        double l2_d = 0, l2_r = 0;
        for (uint32_t v = 0; v < q36::VOCAB; v++) {
            double d = (double)h_logits[v] - (double)ora_t[v];
            l2_d += d*d; l2_r += (double)ora_t[v] * (double)ora_t[v];
        }
        double rel = std::sqrt(l2_d) / (std::sqrt(l2_r) + 1e-9);

        printf("  t=%u  ours top1=%d (%.3f)   oracle top1=%d (%.3f)   logits L2rel=%.3e   %s\n",
               t, our_best, our_max, ora_best, ora_max, rel,
               our_best == ora_best ? "MATCH" : "DIFF");
        oracle_top1[t] = ora_best;
        our_top1[t]    = our_best;
    }

    // last-position next token (most relevant for sampling)
    printf("\n=== last position next-token ===\n");
    printf("oracle next id: %d\n", oracle_top1[SEQ - 1]);
    printf("ours next id:   %d\n", our_top1[SEQ - 1]);

    cudaFree(d_buf_a); cudaFree(d_buf_b); cudaFree(d_resid1); cudaFree(d_normed); cudaFree(d_logits);
    cudaFree(d_final_norm_w); cudaFree(d_lm_head);
    st::close(&M);
    printf("\ndone.\n");
    return 0;
}
