/*
 * test_fp8_block.cu — verify FP8 e4m3 + 128x128 block-scale matvec kernel
 * against a CPU reference using a real Qwen3.6-35B-A3B-FP8 expert tensor.
 *
 * Loads from a Qwen3.6 FP8 model directory:
 *   - model.language_model.layers.0.mlp.experts.0.gate_proj.weight    (F8_E4M3, [512, 2048])
 *   - model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv  (BF16, [4, 16])
 *
 * Generates a random f32 input vector x[2048], runs both the GPU kernel and a
 * CPU reference, prints max_abs / max_rel error.  Also runs a synthetic test
 * with known weights to catch arithmetic bugs independently of the loader.
 *
 * Build:
 *   make test_fp8_block
 *
 * Run:
 *   ./test_fp8_block --model-dir /home/user1/qwen36-fp8
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "safetensors_io.cuh"
#include "qwen36_kernels.cuh"

#define CUDA_OK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA: %s @ %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    std::exit(1); } } while (0)

// ---------------------------------------------------------------------------
// fp8/bf16 host conversions
// ---------------------------------------------------------------------------
static inline float host_fp8_e4m3_to_f32(uint8_t raw) {
    // E4M3FN: sign(1) | exp(4) | man(3), bias=7, no inf, NaN = 0x7f / 0xff
    uint32_t s = (raw >> 7) & 0x1;
    uint32_t e = (raw >> 3) & 0xF;
    uint32_t m = raw & 0x7;
    float f;
    if (e == 0) {
        // subnormal: value = (-1)^s * 2^-6 * (m/8)
        f = std::ldexp((float)m / 8.0f, -6);
    } else if (e == 0xF && m == 0x7) {
        // NaN
        f = std::nanf("");
    } else {
        // normal: value = (-1)^s * 2^(e-7) * (1 + m/8)
        f = std::ldexp(1.0f + (float)m / 8.0f, (int)e - 7);
    }
    return s ? -f : f;
}

static inline float host_bf16_to_f32(uint16_t u) {
    uint32_t w = (uint32_t)u << 16;
    float v; std::memcpy(&v, &w, 4); return v;
}

// ---------------------------------------------------------------------------
// CPU reference matvec: y = W . x  with W stored in fp8 + per-128x128 bf16 scale.
// ---------------------------------------------------------------------------
static void cpu_dequant_matvec_fp8_block128(
    const uint8_t* W_fp8, const uint16_t* Sinv_bf16,
    const float* x, float* y, uint32_t N, uint32_t K)
{
    uint32_t Kb = K / 128;
    for (uint32_t row = 0; row < N; row++) {
        uint32_t row_blk = row / 128;
        double acc = 0.0;
        for (uint32_t cb = 0; cb < Kb; cb++) {
            float scale = host_bf16_to_f32(Sinv_bf16[row_blk * Kb + cb]);
            for (uint32_t k = 0; k < 128; k++) {
                uint32_t col = cb * 128 + k;
                float w = host_fp8_e4m3_to_f32(W_fp8[(size_t)row * K + col]);
                acc += (double)(w * scale) * (double)x[col];
            }
        }
        y[row] = (float)acc;
    }
}

// ---------------------------------------------------------------------------
// One test case: given W (host), scales (host), x (host), shapes, run GPU
// kernel + CPU ref and print errors.
// ---------------------------------------------------------------------------
static void run_one(const char* name,
                    const std::vector<uint8_t>& W,
                    const std::vector<uint16_t>& Sinv,
                    const std::vector<float>& x,
                    uint32_t N, uint32_t K)
{
    if (W.size() != (size_t)N * K) {
        fprintf(stderr, "%s: W size mismatch (%zu vs %u*%u)\n", name, W.size(), N, K); std::exit(1);
    }
    if (Sinv.size() != (size_t)(N / 128) * (K / 128)) {
        fprintf(stderr, "%s: scale size mismatch (%zu vs %u*%u)\n", name, Sinv.size(), N/128, K/128); std::exit(1);
    }
    if (x.size() != K) {
        fprintf(stderr, "%s: x size mismatch (%zu vs %u)\n", name, x.size(), K); std::exit(1);
    }

    std::vector<float> y_cpu(N);
    cpu_dequant_matvec_fp8_block128(W.data(), Sinv.data(), x.data(), y_cpu.data(), N, K);

    uint8_t* d_W; __nv_bfloat16* d_S; float* d_x; float* d_y;
    CUDA_OK(cudaMalloc(&d_W, W.size()));
    CUDA_OK(cudaMalloc(&d_S, Sinv.size() * sizeof(__nv_bfloat16)));
    CUDA_OK(cudaMalloc(&d_x, K * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_y, N * sizeof(float)));
    CUDA_OK(cudaMemcpy(d_W, W.data(),       W.size(),                  cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_S, Sinv.data(),    Sinv.size() * 2,           cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_x, x.data(),       K * sizeof(float),         cudaMemcpyHostToDevice));

    qwen36::launch_dequant_matvec_fp8_block128(d_W, d_S, d_x, d_y, N, K);
    CUDA_OK(cudaDeviceSynchronize());

    std::vector<float> y_gpu(N);
    CUDA_OK(cudaMemcpy(y_gpu.data(), d_y, N * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_W); cudaFree(d_S); cudaFree(d_x); cudaFree(d_y);

    double max_abs = 0, max_rel = 0;
    int worst_row = 0;
    for (uint32_t i = 0; i < N; i++) {
        double a = std::fabs(y_cpu[i]);
        double d = std::fabs(y_cpu[i] - y_gpu[i]);
        if (d > max_abs) { max_abs = d; worst_row = (int)i; }
        double rel = d / (a + 1e-6);
        if (rel > max_rel) max_rel = rel;
    }
    printf("[%s] N=%u K=%u  max_abs=%.4e  max_rel=%.4e  worst row %d (cpu=%.6f gpu=%.6f)\n",
           name, N, K, max_abs, max_rel, worst_row, y_cpu[worst_row], y_gpu[worst_row]);
    printf("       sample y[0..4]  cpu={%.6f %.6f %.6f %.6f}  gpu={%.6f %.6f %.6f %.6f}\n",
           y_cpu[0], y_cpu[1], y_cpu[2], y_cpu[3],
           y_gpu[0], y_gpu[1], y_gpu[2], y_gpu[3]);
}

// ---------------------------------------------------------------------------
// Synthetic test: known W = +1 in slot 0 of every row (after dequant), random x.
// Expected y[row] = scale[row_blk, 0] * x[0] for that pattern.
// ---------------------------------------------------------------------------
static void test_synthetic() {
    const uint32_t N = 512, K = 2048;
    std::mt19937 rng(0xc0ffee);

    std::vector<uint8_t> W(N * K);
    std::vector<uint16_t> Sinv((N/128) * (K/128));
    std::vector<float> x(K);

    // Random fp8 e4m3 bytes, but skip the NaN encodings (0x7f, 0xff)
    std::uniform_int_distribution<int> ub(0, 255);
    for (auto& b : W) {
        int v;
        do { v = ub(rng); } while ((v & 0x7f) == 0x7f);
        b = (uint8_t)v;
    }

    // Random small bf16 scales near 1.0 (typical FP8 dequant scales)
    std::uniform_real_distribution<float> us(0.5f, 2.0f);
    for (auto& s : Sinv) {
        float f = us(rng);
        uint32_t u; std::memcpy(&u, &f, 4);
        s = (uint16_t)(u >> 16);
    }

    std::uniform_real_distribution<float> ux(-1.0f, 1.0f);
    for (auto& v : x) v = ux(rng);

    run_one("synthetic-512x2048", W, Sinv, x, N, K);
}

// ---------------------------------------------------------------------------
// Real Qwen3.6 FP8 expert: load gate_proj for layer 0 expert 0.
// ---------------------------------------------------------------------------
static void test_real(const std::string& model_dir) {
    st::ModelDir M{};
    if (!st::open(&M, model_dir)) {
        fprintf(stderr, "st::open(%s) failed — skipping real-weight test\n", model_dir.c_str());
        return;
    }
    auto try_one = [&](const std::string& tname){
        // weight_scale_inv is named with .weight_scale_inv replacing .weight, not as a suffix.
        std::string sname = tname;
        size_t dot = sname.rfind(".weight");
        if (dot != std::string::npos) sname = sname.substr(0, dot) + ".weight_scale_inv";
        else sname = tname + "_scale_inv";
        const st::Tensor* tw = st::info(&M, tname);
        const st::Tensor* ts = st::info(&M, sname);
        if (!tw || !ts) {
            fprintf(stderr, "missing %s or %s — skipping\n", tname.c_str(), sname.c_str());
            return;
        }
        printf("\nreal: %s\n  weight dtype=%s shape=", tname.c_str(), tw->dtype.c_str());
        for (auto d : tw->shape) printf("%lu ", (unsigned long)d);
        printf("\n  scale  dtype=%s shape=", ts->dtype.c_str());
        for (auto d : ts->shape) printf("%lu ", (unsigned long)d);
        printf("\n");

        if (tw->dtype != "F8_E4M3" || ts->dtype != "BF16") {
            fprintf(stderr, "  unexpected dtypes; skipping\n"); return;
        }
        if (tw->shape.size() != 2 || ts->shape.size() != 2) {
            fprintf(stderr, "  expected rank-2 tensors; skipping\n"); return;
        }
        uint32_t N = (uint32_t)tw->shape[0], K = (uint32_t)tw->shape[1];

        std::vector<uint8_t> wbuf, sbuf;
        if (!st::read_bytes(&M, tname, wbuf)) { fprintf(stderr, "read weight failed\n"); return; }
        if (!st::read_bytes(&M, sname, sbuf)) { fprintf(stderr, "read scale failed\n"); return; }
        if (wbuf.size() != (size_t)N * K) {
            fprintf(stderr, "weight bytes mismatch %zu vs %u*%u\n", wbuf.size(), N, K); return;
        }
        std::vector<uint16_t> sinv(sbuf.size() / 2);
        std::memcpy(sinv.data(), sbuf.data(), sbuf.size());

        // Random input
        std::vector<float> x(K);
        std::mt19937 rng(0x12345);
        std::uniform_real_distribution<float> ux(-1.0f, 1.0f);
        for (auto& v : x) v = ux(rng);

        run_one(tname.c_str(), wbuf, sinv, x, N, K);
    };

    try_one("model.language_model.layers.0.mlp.experts.0.gate_proj.weight");
    try_one("model.language_model.layers.0.mlp.experts.0.up_proj.weight");
    try_one("model.language_model.layers.0.mlp.experts.0.down_proj.weight");
    try_one("model.language_model.layers.0.linear_attn.in_proj_qkv.weight");
    try_one("model.language_model.layers.0.linear_attn.in_proj_z.weight");
    try_one("model.language_model.layers.0.linear_attn.out_proj.weight");
    try_one("model.language_model.layers.0.mlp.shared_expert.gate_proj.weight");

    st::close(&M);
}

int main(int argc, char** argv) {
    std::string model_dir;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--model-dir" && i + 1 < argc) model_dir = argv[++i];
        else if (a == "--help") { fprintf(stderr, "usage: %s [--model-dir DIR]\n", argv[0]); return 0; }
    }

    int dev = 0;
    cudaDeviceProp p{};
    CUDA_OK(cudaGetDeviceProperties(&p, dev));
    printf("device: %s  sm_%d%d  cap %d.%d  global mem %.1f GB\n",
           p.name, p.major, p.minor, p.major, p.minor,
           (double)p.totalGlobalMem / 1e9);

    test_synthetic();
    if (!model_dir.empty()) test_real(model_dir);
    else printf("\n(no --model-dir given; skipping real-weight test)\n");
    return 0;
}
