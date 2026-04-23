/*
 * test_mla.cu — end-to-end correctness for the MLA forward path via
 * mla_forward.cuh against reference dumps from ref_mla_layer.py.
 *
 * Usage:  ./test_mla <ref_dir>
 */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include "mla_forward.cuh"

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
    auto b = read_all(p);
    std::vector<float> v(b.size()/4);
    std::memcpy(v.data(), b.data(), b.size());
    return v;
}

static std::vector<uint16_t> read_bf16_raw(const std::string& p) {
    auto b = read_all(p);
    std::vector<uint16_t> v(b.size()/2);
    std::memcpy(v.data(), b.data(), b.size());
    return v;
}

static double parse_num(const std::string& src, const std::string& key) {
    size_t p = src.find("\"" + key + "\"");
    if (p == std::string::npos) { fprintf(stderr, "meta key %s\n", key.c_str()); std::exit(1); }
    p = src.find(':', p);
    while (p < src.size() && !std::isdigit((unsigned char)src[p]) && src[p]!='-') p++;
    return std::strtod(src.c_str() + p, nullptr);
}

static std::vector<uint16_t> f32_to_bf16(const std::vector<float>& v) {
    std::vector<uint16_t> o(v.size());
    for (size_t i = 0; i < v.size(); i++) {
        uint32_t u; std::memcpy(&u, &v[i], 4);
        uint32_t rb = 0x00007FFFu + ((u >> 16) & 1u);
        o[i] = (uint16_t)((u + rb) >> 16);
    }
    return o;
}

static float max_abs(const std::vector<float>& a, const std::vector<float>& b, size_t* worst=nullptr) {
    float m = 0.0f; size_t w = 0;
    size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; i++) {
        float d = std::fabs(a[i] - b[i]);
        if (d > m) { m = d; w = i; }
    }
    if (worst) *worst = w;
    return m;
}
static float mean_abs(const std::vector<float>& a, const std::vector<float>& b) {
    double s = 0.0;
    size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; i++) s += std::fabs(a[i] - b[i]);
    return (float)(s / n);
}

int main(int argc, char** argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s <ref_dir>\n", argv[0]); return 1; }
    std::string D = argv[1];
    auto P = [&](const char* n) { return D + "/" + n; };

    // --- Parse meta.json
    auto meta_b = read_all(P("meta.json"));
    std::string meta((const char*)meta_b.data(), meta_b.size());
    int pos    = (int)parse_num(meta, "pos");
    int T      = (int)parse_num(meta, "T");

    MLAConfig cfg{};
    cfg.H                = (int)parse_num(meta, "H");
    cfg.num_heads        = (int)parse_num(meta, "num_heads");
    cfg.q_lora_rank      = (int)parse_num(meta, "q_lora_rank");
    cfg.kv_lora_rank     = (int)parse_num(meta, "kv_lora_rank");
    cfg.qk_nope_head_dim = (int)parse_num(meta, "qk_nope_head_dim");
    cfg.qk_rope_head_dim = (int)parse_num(meta, "qk_rope_head_dim");
    cfg.v_head_dim       = (int)parse_num(meta, "v_head_dim");
    cfg.rope_theta       = 50000.0f;
    cfg.yarn_factor      = 64.0f;
    cfg.yarn_beta_fast   = 32.0f;
    cfg.yarn_beta_slow   = 1.0f;
    cfg.yarn_orig_max    = 4096;
    cfg.yarn_mscale      = 1.0f;
    cfg.yarn_mscale_all  = 1.0f;
    cfg.softmax_scale    = mla_effective_softmax_scale(cfg);

    printf("pos=%d T=%d  softmax_scale=%.6f  effective=%.6f (delta=%.2g)\n",
        pos, T, (float)parse_num(meta, "softmax_scale"),
        cfg.softmax_scale,
        (float)parse_num(meta, "softmax_scale") - cfg.softmax_scale);

    int H      = cfg.H, nh = cfg.num_heads;
    int qlora  = cfg.q_lora_rank;
    int kvlora = cfg.kv_lora_rank;
    int d_rope = cfg.qk_rope_head_dim;

    // --- Load inputs + weights + references
    auto hidden_in   = read_f32(P("hidden_in.bin"));
    auto Wqa_f       = read_f32(P("Wqa.bin"));
    auto qa_norm_f   = read_f32(P("qa_norm_w.bin"));
    auto Wqb_rope_f  = read_f32(P("Wqb_rope.bin"));
    auto W_Q_abs_f   = read_f32(P("W_Q_abs.bin"));    // [nh, qlora, kvlora]
    auto Wkva_f      = read_f32(P("Wkva.bin"));
    auto kva_norm_f  = read_f32(P("kva_norm_w.bin"));
    auto W_O_abs_f   = read_f32(P("W_O_abs.bin"));    // [nh*kvlora, H]

    auto layer_out_ref = read_f32(P("layer_out_ref.bin"));
    auto kv_hist_bf16  = read_bf16_raw(P("kv_history.bin"));

    // --- Permute/transpose absorbed weights to match MLALayer storage
    // W_Q_abs (ref) [nh, qlora, kvlora]  →  kernel [nh*kvlora, qlora]
    std::vector<float> W_Q_abs_kernel(nh * (size_t)kvlora * qlora);
    for (int h = 0; h < nh; h++)
        for (int k = 0; k < kvlora; k++)
            for (int q = 0; q < qlora; q++)
                W_Q_abs_kernel[(size_t)(h*kvlora + k) * qlora + q] =
                    W_Q_abs_f[((size_t)h * qlora + q) * kvlora + k];

    // W_O_abs (ref) [nh*kvlora, H]  →  kernel [H, nh*kvlora]   (transpose)
    int Din = nh * kvlora;
    std::vector<float> W_O_abs_kernel((size_t)H * Din);
    for (int i = 0; i < Din; i++)
        for (int o = 0; o < H; o++)
            W_O_abs_kernel[(size_t)o * Din + i] = W_O_abs_f[(size_t)i * H + o];

    // --- Allocate MLALayer + upload weights
    MLALayer L{};
    auto up_bf = [&](const std::vector<float>& v, uint16_t** d) {
        auto bf = f32_to_bf16(v);
        CHECK(cudaMalloc(d, bf.size()*2));
        CHECK(cudaMemcpy(*d, bf.data(), bf.size()*2, cudaMemcpyHostToDevice));
    };
    up_bf(Wqa_f,         &L.d_Wqa);
    up_bf(qa_norm_f,     &L.d_qa_norm);
    up_bf(Wqb_rope_f,    &L.d_Wqb_rope);
    up_bf(Wkva_f,        &L.d_Wkva);
    up_bf(kva_norm_f,    &L.d_kva_norm);
    up_bf(W_Q_abs_kernel,&L.d_W_Q_abs);
    up_bf(W_O_abs_kernel,&L.d_W_O_abs);

    mla_layer_alloc_scratch(&L, cfg);
    CHECK(cudaMalloc(&L.d_kv_cache, (size_t)T * kvlora * 2));
    CHECK(cudaMalloc(&L.d_kr_cache, (size_t)T * d_rope  * 2));

    // Preload T-1 history tokens into the cache (split the [512+64] packed into separate caches)
    std::vector<uint16_t> hist_kv((T-1) * (size_t)kvlora);
    std::vector<uint16_t> hist_kr((T-1) * (size_t)d_rope);
    for (int t = 0; t < T - 1; t++) {
        const uint16_t* src = kv_hist_bf16.data() + (size_t)t * (kvlora + d_rope);
        std::memcpy(hist_kv.data() + (size_t)t * kvlora, src,                 kvlora * 2);
        std::memcpy(hist_kr.data() + (size_t)t * d_rope, src + kvlora,        d_rope  * 2);
    }
    CHECK(cudaMemcpy(L.d_kv_cache, hist_kv.data(), hist_kv.size()*2, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(L.d_kr_cache, hist_kr.data(), hist_kr.size()*2, cudaMemcpyHostToDevice));
    L.cache_len = T - 1;

    // --- yarn cos/sin table, upload entire table then select pos
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

    // --- hidden_in → device
    float* d_hidden;
    CHECK(cudaMalloc(&d_hidden, H * 4));
    CHECK(cudaMemcpy(d_hidden, hidden_in.data(), H * 4, cudaMemcpyHostToDevice));

    // --- layer_out allocation
    float* d_layer_out;
    CHECK(cudaMalloc(&d_layer_out, H * 4));

    // === DRIVE ===
    mla_attention_step(cfg, &L, d_cos_pos, d_sin_pos, d_hidden, pos, d_layer_out);
    CHECK(cudaDeviceSynchronize());

    std::vector<float> got_layer_out(H);
    CHECK(cudaMemcpy(got_layer_out.data(), d_layer_out, H*4, cudaMemcpyDeviceToHost));

    // === Compare ===
    size_t w;
    float m  = max_abs(got_layer_out, layer_out_ref, &w);
    float mn = mean_abs(got_layer_out, layer_out_ref);
    float ref_max = 0.0f;
    for (float v : layer_out_ref) if (std::fabs(v) > ref_max) ref_max = std::fabs(v);
    printf("layer_out: max_abs=%.4g  mean_abs=%.4g  worst@%zu got=%.6g ref=%.6g  ref_max=%.4g  rel=%.3g%%\n",
        m, mn, w, got_layer_out[w], layer_out_ref[w], ref_max,
        (ref_max > 0 ? 100.0f * m / ref_max : 0.0f));

    return (ref_max > 0 && (m / ref_max) < 0.01f) ? 0 : 2;  // pass if < 1% rel
}
