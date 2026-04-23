/*
 * test_kimi_mla.cu — loads one layer's MLA weights straight from a Kimi K2.6
 * model directory (no .bin refs), runs mla_attention_step on a seeded hidden-state,
 * and compares against the python reference dump.
 *
 * Usage:  ./test_kimi_mla <kimi_model_dir> <layer_idx> <ref_dir>
 *
 * The <ref_dir> must have been produced by ref_mla_layer.py with the SAME
 * --layer and a prior run with the same seed/seq-len encoded in meta.json.
 * The test bypasses the loader's .bin inputs and uses the shards directly.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include "kimi_loader.cuh"

#define CHECK(call) do {                                                 \
    cudaError_t e = (call);                                              \
    if (e != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                      \
                cudaGetErrorString(e), __FILE__, __LINE__);              \
        std::exit(1);                                                    \
    }                                                                    \
} while(0)

static std::vector<uint8_t> read_all(const std::string& p) {
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "open %s\n", p.c_str()); std::exit(1); }
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> b(n);
    if ((long)std::fread(b.data(), 1, n, f) != n) std::exit(1);
    std::fclose(f);
    return b;
}

static std::vector<float> read_f32(const std::string& p) {
    auto b = read_all(p); std::vector<float> v(b.size()/4);
    std::memcpy(v.data(), b.data(), b.size()); return v;
}

static std::vector<uint16_t> read_bf16(const std::string& p) {
    auto b = read_all(p); std::vector<uint16_t> v(b.size()/2);
    std::memcpy(v.data(), b.data(), b.size()); return v;
}

static double parse_num(const std::string& src, const std::string& key) {
    size_t p = src.find("\"" + key + "\"");
    if (p == std::string::npos) { fprintf(stderr, "meta key %s\n", key.c_str()); std::exit(1); }
    p = src.find(':', p);
    while (p < src.size() && !std::isdigit((unsigned char)src[p]) && src[p]!='-') p++;
    return std::strtod(src.c_str() + p, nullptr);
}

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <kimi_model_dir> <layer_idx> <ref_dir>\n", argv[0]);
        return 1;
    }
    std::string model_dir = argv[1];
    int layer_idx = std::atoi(argv[2]);
    std::string ref_dir = argv[3];
    auto P = [&](const char* n) { return ref_dir + "/" + n; };

    // 1) Load config.json via the loader
    MLAConfig cfg{};
    if (!kimi_load_mla_config(model_dir, &cfg)) {
        fprintf(stderr, "failed to read config.json from %s\n", model_dir.c_str()); return 1;
    }
    printf("cfg: H=%d nh=%d qlora=%d kvlora=%d d_nope=%d d_rope=%d d_v=%d rope_theta=%.0f factor=%.0f sscale=%.6f\n",
        cfg.H, cfg.num_heads, cfg.q_lora_rank, cfg.kv_lora_rank,
        cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim,
        cfg.rope_theta, cfg.yarn_factor, cfg.softmax_scale);

    // 2) Open model dir, build tensor index
    KimiModelDir M{};
    if (!kimi_open(&M, model_dir)) return 1;
    printf("indexed %zu tensors\n", M.tensors.size());

    // 3) Parse reference meta.json for pos/T, and load hidden_in/layer_out_ref
    auto meta_b = read_all(P("meta.json"));
    std::string meta((const char*)meta_b.data(), meta_b.size());
    int pos = (int)parse_num(meta, "pos");
    int T   = (int)parse_num(meta, "T");

    auto hidden_in     = read_f32(P("hidden_in.bin"));
    auto layer_out_ref = read_f32(P("layer_out_ref.bin"));
    auto kv_hist_bf16  = read_bf16(P("kv_history.bin"));   // [T-1, kvlora+d_rope] bf16

    // 4) Load the specified layer's MLA weights via the new loader
    MLALayer L{};
    if (!kimi_load_mla_layer(&M, cfg, layer_idx, T, &L)) {
        fprintf(stderr, "kimi_load_mla_layer failed\n"); return 1;
    }
    printf("loaded MLA layer %d + absorbed W_Q/W_O\n", layer_idx);

    // 5) Preload kv/kr caches with T-1 history tokens (split the packed [kvlora+d_rope] rows)
    int kvlora = cfg.kv_lora_rank, d_rope = cfg.qk_rope_head_dim;
    std::vector<uint16_t> hist_kv((T-1) * (size_t)kvlora);
    std::vector<uint16_t> hist_kr((T-1) * (size_t)d_rope);
    for (int t = 0; t < T - 1; t++) {
        const uint16_t* src = kv_hist_bf16.data() + (size_t)t * (kvlora + d_rope);
        std::memcpy(hist_kv.data() + (size_t)t * kvlora, src,            kvlora * 2);
        std::memcpy(hist_kr.data() + (size_t)t * d_rope, src + kvlora,   d_rope * 2);
    }
    CHECK(cudaMemcpy(L.d_kv_cache, hist_kv.data(), hist_kv.size()*2, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(L.d_kr_cache, hist_kr.data(), hist_kr.size()*2, cudaMemcpyHostToDevice));
    L.cache_len = T - 1;

    // 6) Yarn cos/sin table on device (for position `pos`)
    std::vector<float> h_cos, h_sin;
    mla_yarn_tables_precompute(cfg, T, h_cos, h_sin);
    int half = d_rope / 2;
    float *d_cos_table, *d_sin_table;
    CHECK(cudaMalloc(&d_cos_table, h_cos.size()*4));
    CHECK(cudaMalloc(&d_sin_table, h_sin.size()*4));
    CHECK(cudaMemcpy(d_cos_table, h_cos.data(), h_cos.size()*4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_sin_table, h_sin.data(), h_sin.size()*4, cudaMemcpyHostToDevice));
    const float* d_cos_pos = d_cos_table + (size_t)pos * half;
    const float* d_sin_pos = d_sin_table + (size_t)pos * half;

    // 7) hidden_in → device
    float* d_hidden;
    CHECK(cudaMalloc(&d_hidden, cfg.H * 4));
    CHECK(cudaMemcpy(d_hidden, hidden_in.data(), cfg.H * 4, cudaMemcpyHostToDevice));

    float* d_layer_out;
    CHECK(cudaMalloc(&d_layer_out, cfg.H * 4));

    // 8) Run the step
    mla_attention_step(cfg, &L, d_cos_pos, d_sin_pos, d_hidden, pos, d_layer_out);
    CHECK(cudaDeviceSynchronize());

    std::vector<float> got(cfg.H);
    CHECK(cudaMemcpy(got.data(), d_layer_out, cfg.H*4, cudaMemcpyDeviceToHost));

    // 9) Compare
    float max_abs = 0; size_t worst = 0;
    double mean = 0; float ref_max = 0;
    for (int i = 0; i < cfg.H; i++) {
        float d = std::fabs(got[i] - layer_out_ref[i]);
        if (d > max_abs) { max_abs = d; worst = i; }
        mean += d;
        if (std::fabs(layer_out_ref[i]) > ref_max) ref_max = std::fabs(layer_out_ref[i]);
    }
    printf("layer_out via loader: max_abs=%.4g  mean_abs=%.4g  worst@%zu got=%.6g ref=%.6g  ref_max=%.4g  rel=%.3f%%\n",
        max_abs, mean / cfg.H, worst, got[worst], layer_out_ref[worst], ref_max,
        ref_max > 0 ? 100.f * max_abs / ref_max : 0.f);

    kimi_free_mla_layer(&L);
    kimi_close(&M);
    cudaFree(d_cos_table); cudaFree(d_sin_table);
    cudaFree(d_hidden); cudaFree(d_layer_out);
    return (ref_max > 0 && (max_abs / ref_max) < 0.02f) ? 0 : 2;
}
