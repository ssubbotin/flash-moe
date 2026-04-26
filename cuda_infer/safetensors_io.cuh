/*
 * safetensors_io.cuh — minimal sharded-safetensors index + tensor reader.
 *
 * Carved out of kimi_loader.cuh so that small tests (e.g. test_fp8_block.cu)
 * can read tensors from a HF safetensors model directory without pulling in
 * the rest of the Kimi/MLA forward stack and its WMMA-heavy kernels.cuh.
 *
 * Public API (in namespace `st`):
 *   st::Tensor                    — descriptor for one tensor
 *   st::ModelDir                  — open model dir (index + per-shard FILE*)
 *   st::open(M, model_dir)        — read model.safetensors.index.json + headers
 *   st::close(M)
 *   st::info(M, name) -> Tensor*
 *   st::read_bytes(M, name, out)  — copy raw bytes of a tensor into `out`
 *   st::bf16_to_f32 / f32_to_bf16
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

namespace st {

struct Tensor {
    std::string shard;
    uint64_t    offset;
    uint64_t    nbytes;
    std::string dtype;
    std::vector<uint64_t> shape;
};

struct ModelDir {
    std::string model_dir;
    std::unordered_map<std::string, Tensor> tensors;
    std::unordered_map<std::string, FILE*> shard_fp;
};

// --- tiny JSON helpers (just enough for the safetensors header format) -----

static inline std::string json_read_string(const char*& p, const char* end) {
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

static inline std::string json_read_object(const char*& p, const char* end) {
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

// --- open / close ---------------------------------------------------------

static inline bool open(ModelDir* M, const std::string& model_dir) {
    M->model_dir = model_dir;
    std::string idx_path = model_dir + "/model.safetensors.index.json";
    FILE* f = std::fopen(idx_path.c_str(), "rb");
    if (!f) { fprintf(stderr, "st::open: cannot open %s\n", idx_path.c_str()); return false; }
    std::fseek(f, 0, SEEK_END); long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    std::vector<char> body(n);
    if ((long)std::fread(body.data(), 1, n, f) != n) { std::fclose(f); return false; }
    std::fclose(f);

    std::string src(body.data(), body.size());
    size_t wm = src.find("\"weight_map\"");
    if (wm == std::string::npos) { fprintf(stderr, "st::open: no weight_map\n"); return false; }
    size_t open_brace = src.find('{', wm);
    size_t close_brace = open_brace;
    int depth = 0;
    for (size_t i = open_brace; i < src.size(); i++) {
        if (src[i] == '{') depth++;
        else if (src[i] == '}') { depth--; if (depth == 0) { close_brace = i; break; } }
    }

    std::unordered_map<std::string, std::string> name_to_shard;
    const char* p = src.data() + open_brace + 1;
    const char* e = src.data() + close_brace;
    while (p < e) {
        while (p < e && (*p == ' ' || *p == '\n' || *p == ',' || *p == '\t' || *p == '\r')) p++;
        if (p >= e || *p != '"') break;
        std::string name = json_read_string(p, e);
        while (p < e && (*p == ' ' || *p == ':' || *p == '\t')) p++;
        std::string shard = json_read_string(p, e);
        if (!name.empty() && !shard.empty()) name_to_shard.emplace(std::move(name), std::move(shard));
    }

    std::unordered_map<std::string, std::vector<std::string>> shard_to_names;
    for (auto& kv : name_to_shard) shard_to_names[kv.second].push_back(kv.first);

    for (auto& kv : shard_to_names) {
        const std::string& shard = kv.first;
        std::string path = model_dir + "/" + shard;
        FILE* sf = std::fopen(path.c_str(), "rb");
        if (!sf) { fprintf(stderr, "st::open: cannot open shard %s\n", path.c_str()); return false; }

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
                fprintf(stderr, "st::open: missing %s in %s\n", tn.c_str(), shard.c_str());
                return false;
            }
            const char* q = H.data() + px + pat.size();
            const char* qe = H.data() + H.size();
            std::string obj = json_read_object(q, qe);
            Tensor t; t.shard = shard;

            size_t dt = obj.find("\"dtype\":");
            if (dt != std::string::npos) {
                const char* qp = obj.data() + dt + 8;
                const char* qe2 = obj.data() + obj.size();
                t.dtype = json_read_string(qp, qe2);
            }
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

static inline void close(ModelDir* M) {
    for (auto& kv : M->shard_fp) if (kv.second) std::fclose(kv.second);
    M->shard_fp.clear();
    M->tensors.clear();
}

static inline const Tensor* info(const ModelDir* M, const std::string& name) {
    auto it = M->tensors.find(name);
    return (it == M->tensors.end()) ? nullptr : &it->second;
}

static inline bool read_bytes(ModelDir* M, const std::string& name,
                              std::vector<uint8_t>& out) {
    auto it = M->tensors.find(name);
    if (it == M->tensors.end()) {
        fprintf(stderr, "st::read_bytes: missing %s\n", name.c_str());
        return false;
    }
    const Tensor& t = it->second;
    FILE*& fp = M->shard_fp[t.shard];
    if (!fp) {
        std::string path = M->model_dir + "/" + t.shard;
        fp = std::fopen(path.c_str(), "rb");
        if (!fp) { fprintf(stderr, "st::read_bytes: open %s\n", path.c_str()); return false; }
    }
    out.resize(t.nbytes);
    int fd = fileno(fp);
    size_t total = 0;
    while (total < t.nbytes) {
        size_t want = t.nbytes - total;
        if (want > (size_t)0x40000000) want = (size_t)0x40000000;
        ssize_t got = ::pread(fd, out.data() + total, want, (off_t)(t.offset + total));
        if (got <= 0) {
            fprintf(stderr, "st::read_bytes: pread %s got=%zd after %zu/%" PRIu64 "\n",
                    name.c_str(), got, total, (uint64_t)t.nbytes);
            return false;
        }
        total += (size_t)got;
    }
    return true;
}

// --- bf16 helpers ---------------------------------------------------------

static inline float bf16_to_f32(uint16_t u) {
    uint32_t w = (uint32_t)u << 16;
    float v; std::memcpy(&v, &w, 4); return v;
}

static inline uint16_t f32_to_bf16(float x) {
    uint32_t u; std::memcpy(&u, &x, 4);
    uint32_t rb = 0x00007FFFu + ((u >> 16) & 1u);
    return (uint16_t)((u + rb) >> 16);
}

} // namespace st
