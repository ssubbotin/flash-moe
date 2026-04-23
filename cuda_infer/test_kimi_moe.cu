/*
 * test_kimi_moe.cu — drive the Kimi MoE path using a packed layer file
 * produced by repack_experts_kimi.py, and compare against ref_moe_layer.py.
 *
 * Usage:
 *     ./test_kimi_moe <kimi_model_dir> <layer_idx> <packed_experts_dir> <ref_dir>
 *
 * Reads routing inputs from the python reference (hidden_in, gate_logits/bias),
 * performs routing on GPU, pread's the K=8 selected experts from
 *   <packed_experts_dir>/layer_<layer_idx>.bin
 * runs expert forward for each, weighted-accumulates, also reads the layer's
 * shared-expert weights from the shards, runs shared forward, and combines.
 */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <fcntl.h>
#include <unistd.h>
#include <vector>
#include <string>
#include "kimi_loader.cuh"
#include "kimi_moe.cuh"

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
static std::vector<int32_t> read_i32(const std::string& p) {
    auto b = read_all(p); std::vector<int32_t> v(b.size()/4);
    std::memcpy(v.data(), b.data(), b.size()); return v;
}
static float max_abs(const std::vector<float>& a, const std::vector<float>& b, size_t* w=nullptr) {
    float m = 0; size_t ww = 0;
    size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; i++) {
        float d = std::fabs(a[i] - b[i]);
        if (d > m) { m = d; ww = i; }
    }
    if (w) *w = ww; return m;
}

int main(int argc, char** argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s <model_dir> <layer_idx> <packed_dir> <ref_dir>\n", argv[0]);
        return 1;
    }
    std::string model_dir   = argv[1];
    int         layer_idx   = std::atoi(argv[2]);
    std::string packed_dir  = argv[3];
    std::string ref_dir     = argv[4];
    auto Pref = [&](const char* n) { return ref_dir + "/" + n; };

    // Load reference inputs + expected outputs
    auto hidden_in       = read_f32(Pref("hidden_in.bin"));
    auto gate_logits_ref = read_f32(Pref("gate_logits.bin"));
    auto gate_bias_ref   = read_f32(Pref("gate_bias.bin"));
    auto topk_idx_ref    = read_i32(Pref("topk_indices.bin"));
    auto topk_w_ref      = read_f32(Pref("topk_weights.bin"));
    auto shared_out_ref  = read_f32(Pref("shared_out.bin"));
    auto moe_accum_ref   = read_f32(Pref("moe_accum_ref.bin"));
    auto layer_out_ref   = read_f32(Pref("layer_out_ref.bin"));

    int H        = (int)hidden_in.size();
    int num_exp  = (int)gate_logits_ref.size();
    int K        = (int)topk_idx_ref.size();
    int moe_int  = KIMI_MOE_INT;
    float scaling_factor = 2.827f;
    printf("H=%d num_experts=%d K=%d\n", H, num_exp, K);

    // Upload routing inputs
    float *d_logits, *d_bias;
    int   *d_topk_idx;
    float *d_topk_w;
    CHECK(cudaMalloc(&d_logits,   num_exp * 4));
    CHECK(cudaMalloc(&d_bias,     num_exp * 4));
    CHECK(cudaMalloc(&d_topk_idx, K * 4));
    CHECK(cudaMalloc(&d_topk_w,   K * 4));
    CHECK(cudaMemcpy(d_logits, gate_logits_ref.data(), num_exp*4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_bias,   gate_bias_ref.data(),   num_exp*4, cudaMemcpyHostToDevice));

    launch_kimi_moe_routing_noaux_tc(
        d_logits, d_bias, d_topk_idx, d_topk_w,
        (uint32_t)num_exp, (uint32_t)K,
        scaling_factor, /*renormalize=*/1);
    CHECK(cudaDeviceSynchronize());

    // Pull back the routing for verification
    std::vector<int32_t> topk_idx(K);
    std::vector<float>   topk_w(K);
    CHECK(cudaMemcpy(topk_idx.data(), d_topk_idx, K*4, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(topk_w.data(),   d_topk_w,   K*4, cudaMemcpyDeviceToHost));

    int mismatch = 0;
    for (int k = 0; k < K; k++) {
        if (topk_idx[k] != topk_idx_ref[k]) {
            printf("routing idx mismatch k=%d: got=%d ref=%d\n", k, topk_idx[k], topk_idx_ref[k]);
            mismatch++;
        }
    }
    float w_err = max_abs(topk_w, topk_w_ref);
    printf("routing: %s (idx match=%d/%d, weight max_abs=%.4g)\n",
           mismatch == 0 && w_err < 1e-4f ? "OK" : "MISMATCH",
           K - mismatch, K, w_err);

    // -----------------------------------------------------------------------
    // Run K expert forwards, streaming each from the packed layer file.
    // -----------------------------------------------------------------------
    std::string layer_bin = packed_dir + "/layer_" + std::to_string(layer_idx) + ".bin";
    int fd = ::open(layer_bin.c_str(), O_RDONLY);
    if (fd < 0) { fprintf(stderr, "open %s\n", layer_bin.c_str()); return 1; }

    // Device expert block scratch (reuse across all K experts)
    uint8_t* d_block;
    CHECK(cudaMalloc(&d_block, KIMI_EXPERT_BLOCK_BYTES));

    float *d_x, *d_gate_tmp, *d_up_tmp, *d_glu_tmp, *d_expert_out, *d_moe_accum;
    CHECK(cudaMalloc(&d_x,           H * 4));
    CHECK(cudaMalloc(&d_gate_tmp,    moe_int * 4));
    CHECK(cudaMalloc(&d_up_tmp,      moe_int * 4));
    CHECK(cudaMalloc(&d_glu_tmp,     moe_int * 4));
    CHECK(cudaMalloc(&d_expert_out,  H * 4));
    CHECK(cudaMalloc(&d_moe_accum,   H * 4));
    CHECK(cudaMemcpy(d_x, hidden_in.data(), H*4, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_moe_accum, 0, H * 4));

    std::vector<uint8_t> host_block(KIMI_EXPERT_BLOCK_BYTES);
    for (int k = 0; k < K; k++) {
        int e = topk_idx[k];
        off_t off = (off_t)e * KIMI_EXPERT_BLOCK_BYTES;
        ssize_t got = ::pread(fd, host_block.data(), KIMI_EXPERT_BLOCK_BYTES, off);
        if (got != (ssize_t)KIMI_EXPERT_BLOCK_BYTES) {
            fprintf(stderr, "pread short: %zd\n", got); return 1;
        }
        CHECK(cudaMemcpy(d_block, host_block.data(), KIMI_EXPERT_BLOCK_BYTES, cudaMemcpyHostToDevice));

        kimi_expert_forward_from_block(
            d_block, d_x,
            d_gate_tmp, d_up_tmp, d_glu_tmp, d_expert_out);

        launch_kimi_weighted_accum_dw(
            d_moe_accum, d_expert_out, d_topk_w + k, (uint32_t)H);
    }
    CHECK(cudaDeviceSynchronize());
    ::close(fd);

    // Compare moe_accum vs reference
    std::vector<float> got_moe(H);
    CHECK(cudaMemcpy(got_moe.data(), d_moe_accum, H*4, cudaMemcpyDeviceToHost));
    size_t w;
    float mm = max_abs(got_moe, moe_accum_ref, &w);
    float ref_max = 0; for (float v : moe_accum_ref) if (std::fabs(v) > ref_max) ref_max = std::fabs(v);
    printf("moe_accum: max_abs=%.4g  worst@%zu got=%.6g ref=%.6g  ref_max=%.4g  rel=%.3f%%\n",
        mm, w, got_moe[w], moe_accum_ref[w], ref_max,
        ref_max > 0 ? 100.f * mm / ref_max : 0.f);

    // -----------------------------------------------------------------------
    // Shared expert forward (bf16) + combine
    // -----------------------------------------------------------------------
    KimiModelDir M{};
    if (!kimi_open(&M, model_dir)) return 1;

    char prefix[128];
    std::snprintf(prefix, sizeof prefix, "language_model.model.layers.%d.mlp.shared_experts.", layer_idx);
    std::vector<uint8_t> sg_raw, su_raw, sd_raw;
    if (!kimi_read_tensor_bytes(&M, std::string(prefix) + "gate_proj.weight", sg_raw)) return 1;
    if (!kimi_read_tensor_bytes(&M, std::string(prefix) + "up_proj.weight",   su_raw)) return 1;
    if (!kimi_read_tensor_bytes(&M, std::string(prefix) + "down_proj.weight", sd_raw)) return 1;

    uint16_t* d_Wsg = kimi_upload_bf16((const uint16_t*)sg_raw.data(), (size_t)moe_int * H);
    uint16_t* d_Wsu = kimi_upload_bf16((const uint16_t*)su_raw.data(), (size_t)moe_int * H);
    uint16_t* d_Wsd = kimi_upload_bf16((const uint16_t*)sd_raw.data(), (size_t)H * moe_int);

    float *d_sgate, *d_sup, *d_sglu, *d_shared_out;
    CHECK(cudaMalloc(&d_sgate,      moe_int * 4));
    CHECK(cudaMalloc(&d_sup,        moe_int * 4));
    CHECK(cudaMalloc(&d_sglu,       moe_int * 4));
    CHECK(cudaMalloc(&d_shared_out, H * 4));

    launch_matvec_bf16(d_Wsg, d_x, d_sgate, (uint32_t)moe_int, (uint32_t)H);
    launch_matvec_bf16(d_Wsu, d_x, d_sup,   (uint32_t)moe_int, (uint32_t)H);
    launch_swiglu(d_sgate, d_sup, d_sglu, (uint32_t)moe_int);
    launch_matvec_bf16(d_Wsd, d_sglu, d_shared_out, (uint32_t)H, (uint32_t)moe_int);
    CHECK(cudaDeviceSynchronize());

    std::vector<float> got_shared(H);
    CHECK(cudaMemcpy(got_shared.data(), d_shared_out, H*4, cudaMemcpyDeviceToHost));
    float sh_max = 0; for (float v : shared_out_ref) if (std::fabs(v) > sh_max) sh_max = std::fabs(v);
    float sh_err = max_abs(got_shared, shared_out_ref, &w);
    printf("shared_out: max_abs=%.4g  ref_max=%.4g  rel=%.3f%%\n",
        sh_err, sh_max, sh_max > 0 ? 100.f * sh_err / sh_max : 0.f);

    // Combine: h_out = h_in + shared + scaling * moe_accum
    // NOTE: ref_moe_layer.py multiplied topk_w by scaling_factor already, so its
    // moe_accum already includes the scaling. Our GPU moe_accum also incorporates
    // scaling via topk_w (the routing kernel applies it). Therefore combine here
    // uses scaling_factor=1.0.
    float* d_layer_out;
    CHECK(cudaMalloc(&d_layer_out, H * 4));
    launch_kimi_moe_combine(d_x, d_shared_out, d_moe_accum, d_layer_out, 1.0f, (uint32_t)H);
    CHECK(cudaDeviceSynchronize());

    std::vector<float> got_layer_out(H);
    CHECK(cudaMemcpy(got_layer_out.data(), d_layer_out, H*4, cudaMemcpyDeviceToHost));
    float lo_max = 0; for (float v : layer_out_ref) if (std::fabs(v) > lo_max) lo_max = std::fabs(v);
    float lo_err = max_abs(got_layer_out, layer_out_ref, &w);
    printf("layer_out: max_abs=%.4g  worst@%zu got=%.6g ref=%.6g  ref_max=%.4g  rel=%.3f%%\n",
        lo_err, w, got_layer_out[w], layer_out_ref[w], lo_max,
        lo_max > 0 ? 100.f * lo_err / lo_max : 0.f);

    kimi_close(&M);
    return (lo_max > 0 && (lo_err / lo_max) < 0.02f) ? 0 : 2;
}
