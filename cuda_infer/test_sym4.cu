/*
 * test_sym4.cu — correctness test for dequant_matvec_sym4_g32 against real Kimi K2.6
 * compressed-tensors symmetric int4 weights.
 *
 * Usage:  ./test_sym4 <shard.safetensors> <tensor_prefix> [seed]
 *   prefix example: language_model.model.layers.1.mlp.experts.0.gate_proj
 *
 * Loads weight_packed (I32) and weight_scale (BF16) from the shard, generates a
 * deterministic input x, computes a CPU reference, then runs the GPU kernel and
 * reports the worst-case error.
 */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <cstdio>
#include "kernels.cuh"

#define CHECK(call) do {                                                   \
    cudaError_t e = (call);                                                \
    if (e != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                        \
                cudaGetErrorString(e), __FILE__, __LINE__);                \
        std::exit(1);                                                      \
    }                                                                      \
} while(0)

// --- Minimal safetensors reader ------------------------------------------------

struct TensorInfo {
    std::string dtype;
    std::vector<uint64_t> shape;
    uint64_t data_offset;      // absolute file offset to tensor bytes
    uint64_t nbytes;
};

static bool st_extract_field(const std::string& obj, const char* key,
                             std::string& val, bool is_string) {
    std::string pat = std::string("\"") + key + "\":";
    size_t p = obj.find(pat);
    if (p == std::string::npos) return false;
    p += pat.size();
    while (p < obj.size() && (obj[p] == ' ' || obj[p] == '\t')) p++;
    if (is_string) {
        if (obj[p] != '"') return false;
        size_t q = obj.find('"', p + 1);
        if (q == std::string::npos) return false;
        val = obj.substr(p + 1, q - p - 1);
    } else if (obj[p] == '[') {
        // Array — consume up to matching ']'
        int depth = 0; size_t q = p;
        while (q < obj.size()) {
            if (obj[q] == '[') depth++;
            else if (obj[q] == ']') { depth--; if (depth == 0) { q++; break; } }
            q++;
        }
        val = obj.substr(p, q - p);
    } else {
        size_t q = p;
        while (q < obj.size() && obj[q] != ',' && obj[q] != '}') q++;
        val = obj.substr(p, q - p);
    }
    return true;
}

static std::vector<uint64_t> parse_shape(const std::string& s) {
    std::vector<uint64_t> out;
    size_t p = s.find('['); if (p == std::string::npos) return out;
    size_t q = s.find(']', p); if (q == std::string::npos) return out;
    std::string body = s.substr(p + 1, q - p - 1);
    size_t i = 0;
    while (i < body.size()) {
        while (i < body.size() && (body[i] == ' ' || body[i] == ',')) i++;
        if (i >= body.size()) break;
        out.push_back(std::strtoull(body.c_str() + i, nullptr, 10));
        while (i < body.size() && body[i] != ',') i++;
    }
    return out;
}

static TensorInfo st_find(FILE* f, const std::string& name) {
    uint64_t hdr_len = 0;
    std::fseek(f, 0, SEEK_SET);
    if (std::fread(&hdr_len, 1, 8, f) != 8) { fprintf(stderr, "short hdr len\n"); std::exit(1); }
    std::vector<char> hdr(hdr_len);
    if (std::fread(hdr.data(), 1, hdr_len, f) != hdr_len) { fprintf(stderr, "short hdr\n"); std::exit(1); }
    std::string H(hdr.begin(), hdr.end());
    uint64_t data_base = 8 + hdr_len;

    std::string key = "\"" + name + "\":";
    size_t p = H.find(key);
    if (p == std::string::npos) {
        fprintf(stderr, "tensor not found: %s\n", name.c_str()); std::exit(1);
    }
    p += key.size();
    size_t brace = H.find('{', p);
    size_t end = H.find('}', brace);
    std::string obj = H.substr(brace, end - brace + 1);

    TensorInfo t;
    std::string s;
    st_extract_field(obj, "dtype", t.dtype, true);
    st_extract_field(obj, "shape", s, false); t.shape = parse_shape(s);
    std::string off;
    st_extract_field(obj, "data_offsets", off, false);
    size_t lb = off.find('['), rb = off.find(']');
    std::string body = off.substr(lb + 1, rb - lb - 1);
    uint64_t beg = std::strtoull(body.c_str(), nullptr, 10);
    size_t comma = body.find(',');
    uint64_t fin = std::strtoull(body.c_str() + comma + 1, nullptr, 10);
    t.data_offset = data_base + beg;
    t.nbytes = fin - beg;
    return t;
}

static std::vector<uint8_t> st_read(FILE* f, const TensorInfo& t) {
    std::vector<uint8_t> buf(t.nbytes);
    std::fseek(f, (long)t.data_offset, SEEK_SET);
    if (std::fread(buf.data(), 1, t.nbytes, f) != t.nbytes) {
        fprintf(stderr, "short tensor read\n"); std::exit(1);
    }
    return buf;
}

// --- Helpers ------------------------------------------------------------------

static inline float bf16_host(uint16_t u) {
    uint32_t w = (uint32_t)u << 16;
    float v; std::memcpy(&v, &w, 4); return v;
}

// Xorshift32 — deterministic and stdlib-free
struct XS32 { uint32_t s; float next() {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    return ((s >> 8) / (float)(1u << 24)) * 2.0f - 1.0f;  // (-1, 1)
} };

// --- CPU reference ------------------------------------------------------------

static void cpu_sym4_matvec(
    const int32_t* packed, const uint16_t* scale_bf16,
    const float* x, float* out,
    uint32_t out_dim, uint32_t in_dim)
{
    uint32_t packed_cols = in_dim / 8;
    uint32_t groups      = in_dim / 32;
    for (uint32_t r = 0; r < out_dim; r++) {
        double acc = 0.0;
        const int32_t* wr = packed + (size_t)r * packed_cols;
        const uint16_t* sr = scale_bf16 + (size_t)r * groups;
        for (uint32_t c = 0; c < packed_cols; c++) {
            uint32_t g = c / 4;           // 4 packed_cols per group of 32
            float scale = bf16_host(sr[g]);
            uint32_t packed_v = (uint32_t)wr[c];
            for (uint32_t k = 0; k < 8; k++) {
                // compressed-tensors biased rep: signed = unsigned - 8
                int32_t nib = (int32_t)((packed_v >> (4*k)) & 0xFu) - 8;
                acc += (double)nib * (double)scale * (double)x[c * 8 + k];
            }
        }
        out[r] = (float)acc;
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
          "usage: %s <shard.safetensors> <tensor_prefix> [seed]\n"
          "  prefix e.g. language_model.model.layers.1.mlp.experts.0.gate_proj\n",
          argv[0]);
        return 1;
    }
    const char* shard = argv[1];
    std::string prefix = argv[2];
    uint32_t seed = argc > 3 ? (uint32_t)std::strtoul(argv[3], nullptr, 10) : 42u;

    FILE* f = std::fopen(shard, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", shard); return 1; }

    TensorInfo tp = st_find(f, prefix + ".weight_packed");
    TensorInfo ts = st_find(f, prefix + ".weight_scale");
    if (tp.dtype != "I32" || ts.dtype != "BF16" || tp.shape.size() != 2 || ts.shape.size() != 2) {
        fprintf(stderr, "unexpected dtypes/shapes: packed=%s%zu scale=%s%zu\n",
            tp.dtype.c_str(), tp.shape.size(), ts.dtype.c_str(), ts.shape.size());
        std::fclose(f); return 1;
    }
    uint32_t out_dim = (uint32_t)tp.shape[0];
    uint32_t packed_cols = (uint32_t)tp.shape[1];
    uint32_t in_dim = packed_cols * 8;
    if (ts.shape[0] != out_dim || ts.shape[1] != in_dim / 32) {
        fprintf(stderr, "scale shape mismatch: expected [%u,%u], got [%llu,%llu]\n",
            out_dim, in_dim/32,
            (unsigned long long)ts.shape[0], (unsigned long long)ts.shape[1]);
        std::fclose(f); return 1;
    }
    printf("tensor: %s  out_dim=%u  in_dim=%u\n", prefix.c_str(), out_dim, in_dim);

    auto packed_b = st_read(f, tp);
    auto scale_b  = st_read(f, ts);
    std::fclose(f);

    std::vector<float> x(in_dim);
    XS32 rng{ seed ? seed : 0x9E3779B9u };
    for (uint32_t i = 0; i < in_dim; i++) x[i] = rng.next() * 0.1f;

    std::vector<float> ref(out_dim);
    cpu_sym4_matvec((const int32_t*)packed_b.data(),
                    (const uint16_t*)scale_b.data(),
                    x.data(), ref.data(), out_dim, in_dim);

    uint32_t *d_packed; uint16_t *d_scales; float *d_x, *d_out;
    CHECK(cudaMalloc(&d_packed, packed_b.size()));
    CHECK(cudaMalloc(&d_scales, scale_b.size()));
    CHECK(cudaMalloc(&d_x,      x.size() * sizeof(float)));
    CHECK(cudaMalloc(&d_out,    out_dim * sizeof(float)));
    CHECK(cudaMemcpy(d_packed, packed_b.data(), packed_b.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_scales, scale_b.data(),  scale_b.size(),  cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_x,      x.data(), x.size()*sizeof(float), cudaMemcpyHostToDevice));

    launch_dequant_matvec_sym4(d_packed, d_scales, d_x, d_out, out_dim, in_dim);
    CHECK(cudaDeviceSynchronize());

    std::vector<float> got(out_dim);
    CHECK(cudaMemcpy(got.data(), d_out, out_dim*sizeof(float), cudaMemcpyDeviceToHost));

    double max_abs = 0.0, sum_abs = 0.0;
    uint32_t worst = 0;
    for (uint32_t i = 0; i < out_dim; i++) {
        double d = std::fabs((double)got[i] - (double)ref[i]);
        sum_abs += d;
        if (d > max_abs) { max_abs = d; worst = i; }
    }
    printf("max_abs=%.6g  mean_abs=%.6g  worst@%u  got=%.6g  ref=%.6g\n",
        max_abs, sum_abs/out_dim, worst, got[worst], ref[worst]);

    cudaFree(d_packed); cudaFree(d_scales); cudaFree(d_x); cudaFree(d_out);
    return (max_abs < 5e-4 ? 0 : 2);
}
