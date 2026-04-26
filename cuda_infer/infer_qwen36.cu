/*
 * infer_qwen36.cu — full inference driver for Qwen3.6-35B-A3B-FP8.
 *
 * Loads everything except the routed experts up front to VRAM, then for each
 * token streams the K=8 active experts per layer from disk on demand via the
 * safetensors index. Per-layer recurrence state (conv_state + delta_state for
 * linear-attn layers, K/V cache for full-attn layers) lives across calls.
 *
 * Tokenizer: spawns a long-lived `python3 kimi_tokenize.py serve` child
 * pointed at the Qwen3.6 model directory (the script is generic — it just
 * loads whatever HF tokenizer is there). Reused verbatim.
 *
 * CLI mirrors infer_kimi.cu:
 *   ./infer_qwen36 --model-dir DIR (--prompt TEXT | --chat ROLE TEXT ... | --tokens "id,id,...")
 *                  [--max-tokens N] [--max-seq L]
 *                  [--greedy | --temp T --top-p P --top-k K] [--rep-pen R] [--seed N]
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <ctime>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <unordered_map>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <fcntl.h>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "safetensors_io.cuh"
#include "qwen36_layer_runner.cuh"

#ifndef CUDA_OK
#define CUDA_OK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA: %s @ %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    std::exit(1); } } while (0)
#endif

// ---------------------------------------------------------------------------
// Tokenizer subprocess client (verbatim from infer_kimi.cu)
// ---------------------------------------------------------------------------
struct TokClient {
    pid_t pid = -1;
    FILE* to_child   = nullptr;
    FILE* from_child = nullptr;
    int   vocab_size = 0;
    int   bos_id     = -1;
    std::vector<int> eos_ids;
};

static const char* b64_chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static std::string b64encode(const std::string& in) {
    std::string out; out.reserve(((in.size() + 2) / 3) * 4);
    int val = 0, valb = -6;
    for (unsigned char c : in) {
        val = (val << 8) | c; valb += 8;
        while (valb >= 0) { out.push_back(b64_chars[(val >> valb) & 0x3F]); valb -= 6; }
    }
    if (valb > -6) out.push_back(b64_chars[((val << 8) >> (valb + 8)) & 0x3F]);
    while (out.size() % 4) out.push_back('=');
    return out;
}
static std::string b64decode(const std::string& in) {
    std::vector<int> T(256, -1);
    for (int i = 0; i < 64; i++) T[(unsigned char)b64_chars[i]] = i;
    std::string out; int val = 0, valb = -8;
    for (unsigned char c : in) {
        if (c == '=' || T[c] == -1) continue;
        val = (val << 6) | T[c]; valb += 6;
        if (valb >= 0) { out.push_back(char((val >> valb) & 0xFF)); valb -= 8; }
    }
    return out;
}

static std::string tok_call(TokClient* tc, const std::string& cmd, bool* ok) {
    fputs(cmd.c_str(), tc->to_child); fputc('\n', tc->to_child); fflush(tc->to_child);
    char* line = nullptr; size_t cap = 0;
    ssize_t n = getline(&line, &cap, tc->from_child);
    if (n <= 0) { if (line) free(line); *ok = false; return ""; }
    while (n > 0 && (line[n-1] == '\n' || line[n-1] == '\r')) line[--n] = 0;
    std::string s(line, (size_t)n); free(line);
    if (s.size() >= 3 && s.compare(0, 3, "OK\t") == 0) { *ok = true;  return s.substr(3); }
    if (s == "OK")                                      { *ok = true;  return ""; }
    if (s.size() >= 4 && s.compare(0, 4, "ERR\t") == 0) { *ok = false; fprintf(stderr, "tok ERR: %s\n", s.c_str()+4); return ""; }
    fprintf(stderr, "tok unexpected: %s\n", s.c_str()); *ok = false; return "";
}

static bool tok_open(TokClient* tc, const std::string& model_dir) {
    int in_pipe[2], out_pipe[2];
    if (pipe(in_pipe) != 0 || pipe(out_pipe) != 0) { perror("pipe"); return false; }
    pid_t pid = fork();
    if (pid < 0) return false;
    if (pid == 0) {
        dup2(in_pipe[0], 0); close(in_pipe[1]);
        dup2(out_pipe[1], 1); close(out_pipe[0]);
        close(in_pipe[0]); close(out_pipe[1]);
        const char* py = std::getenv("KIMI_PYTHON");
        if (!py || !*py) py = "python3";
        char self[4096]; ssize_t sn = readlink("/proc/self/exe", self, sizeof(self) - 1);
        std::string script;
        if (sn > 0) {
            self[sn] = 0; std::string s = self; auto sl = s.find_last_of('/');
            script = (sl == std::string::npos ? "" : s.substr(0, sl + 1)) + "kimi_tokenize.py";
        } else script = "kimi_tokenize.py";
        execlp(py, py, script.c_str(), "--model-dir", model_dir.c_str(), "serve", (char*)nullptr);
        _exit(127);
    }
    close(in_pipe[0]); close(out_pipe[1]);
    tc->pid = pid;
    tc->to_child   = fdopen(in_pipe[1],  "w");
    tc->from_child = fdopen(out_pipe[0], "r");
    char* line = nullptr; size_t cap = 0;
    if (getline(&line, &cap, tc->from_child) <= 0) { if (line) free(line); return false; }
    free(line);
    bool ok; std::string s = tok_call(tc, "info", &ok);
    if (!ok) return false;
    int p1 = s.find('\t'), p2 = s.find('\t', p1 + 1);
    tc->vocab_size = std::atoi(s.substr(0, p1).c_str());
    tc->bos_id     = std::atoi(s.substr(p1 + 1, p2 - p1 - 1).c_str());
    std::string eos = s.substr(p2 + 1);
    size_t i = 0;
    while (i < eos.size()) {
        size_t j = eos.find(',', i); if (j == std::string::npos) j = eos.size();
        int id = std::atoi(eos.substr(i, j - i).c_str());
        if (id >= 0) tc->eos_ids.push_back(id);
        i = j + 1;
    }
    fprintf(stderr, "tokenizer ready: vocab=%d bos=%d eos=", tc->vocab_size, tc->bos_id);
    for (size_t k = 0; k < tc->eos_ids.size(); k++) fprintf(stderr, "%s%d", k ? "," : "", tc->eos_ids[k]);
    fprintf(stderr, "\n");
    return true;
}

static void tok_close(TokClient* tc) {
    if (tc->to_child) { bool ok; tok_call(tc, "quit", &ok); fclose(tc->to_child); tc->to_child = nullptr; }
    if (tc->from_child) { fclose(tc->from_child); tc->from_child = nullptr; }
    if (tc->pid > 0) { int st; waitpid(tc->pid, &st, 0); tc->pid = -1; }
}

static std::vector<int> parse_csv_ids(const std::string& s) {
    std::vector<int> out; size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && !(std::isdigit((unsigned char)s[i]) || s[i] == '-')) i++;
        if (i >= s.size()) break;
        char* end; long v = std::strtol(s.c_str()+i, &end, 10);
        out.push_back((int)v); i = (size_t)(end - s.c_str());
    }
    return out;
}
static std::vector<int> tok_encode(TokClient* tc, const std::string& text) {
    bool ok; std::string s = tok_call(tc, "encode\t" + b64encode(text), &ok);
    if (!ok) return {}; return parse_csv_ids(s);
}
static std::vector<int> tok_chat(TokClient* tc, const std::vector<std::pair<std::string,std::string>>& msgs) {
    std::string j = "[";
    for (size_t i = 0; i < msgs.size(); i++) {
        if (i) j += ",";
        j += "{\"role\":\""; for (char c : msgs[i].first)  { if (c == '"' || c == '\\') j += '\\'; j += c; }
        j += "\",\"content\":\"";
        for (char c : msgs[i].second) {
            if (c == '"' || c == '\\') { j += '\\'; j += c; }
            else if (c == '\n') j += "\\n"; else if (c == '\r') j += "\\r"; else if (c == '\t') j += "\\t";
            else if ((unsigned char)c < 0x20) { char b[8]; snprintf(b, sizeof(b), "\\u%04x", c); j += b; }
            else j += c;
        }
        j += "\"}";
    }
    j += "]";
    bool ok; std::string s = tok_call(tc, "chat\t" + b64encode(j), &ok);
    if (!ok) return {}; return parse_csv_ids(s);
}
static void tok_decode_reset(TokClient* tc) { bool ok; tok_call(tc, "decode_stream_reset", &ok); }
static std::string tok_decode_push(TokClient* tc, int id) {
    char buf[64]; snprintf(buf, sizeof(buf), "decode_stream_push\t%d", id);
    bool ok; std::string s = tok_call(tc, buf, &ok);
    if (!ok) return ""; return b64decode(s);
}

// ---------------------------------------------------------------------------
// Sampling (verbatim from infer_kimi.cu)
// ---------------------------------------------------------------------------
struct SampleCfg { bool greedy=false; float temp=0.7f, top_p=0.9f; int top_k=0; float rep_pen=1.0f; };

static int sample_next(const std::vector<float>& logits, const std::vector<int>& history,
                       const SampleCfg& cfg, std::mt19937& rng) {
    int V = (int)logits.size();
    if (cfg.greedy || cfg.temp <= 0.0f) {
        int best = 0; float bv = logits[0];
        for (int i = 1; i < V; i++) if (logits[i] > bv) { bv = logits[i]; best = i; }
        return best;
    }
    std::vector<float> z(logits);
    if (cfg.rep_pen != 1.0f) for (int id : history) {
        if (id < 0 || id >= V) continue;
        z[id] = z[id] > 0 ? z[id] / cfg.rep_pen : z[id] * cfg.rep_pen;
    }
    float inv_t = 1.0f / cfg.temp;
    for (int i = 0; i < V; i++) z[i] *= inv_t;
    std::vector<int> idx(V); for (int i = 0; i < V; i++) idx[i] = i;
    int keep = V;
    if (cfg.top_k > 0 && cfg.top_k < V) {
        std::partial_sort(idx.begin(), idx.begin() + cfg.top_k, idx.end(),
                          [&](int a, int b){ return z[a] > z[b]; });
        keep = cfg.top_k;
    } else {
        std::sort(idx.begin(), idx.end(), [&](int a, int b){ return z[a] > z[b]; });
    }
    float zmax = z[idx[0]];
    std::vector<float> p(keep); double sum = 0;
    for (int i = 0; i < keep; i++) { p[i] = std::exp(z[idx[i]] - zmax); sum += p[i]; }
    for (int i = 0; i < keep; i++) p[i] = (float)(p[i] / sum);
    if (cfg.top_p > 0.0f && cfg.top_p < 1.0f) {
        double cum = 0; int cut = keep;
        for (int i = 0; i < keep; i++) { cum += p[i]; if (cum >= cfg.top_p) { cut = i + 1; break; } }
        keep = cut; sum = 0; for (int i = 0; i < keep; i++) sum += p[i];
        for (int i = 0; i < keep; i++) p[i] = (float)(p[i] / sum);
    }
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    float r = U(rng), c = 0;
    for (int i = 0; i < keep; i++) { c += p[i]; if (r <= c) return idx[i]; }
    return idx[keep - 1];
}

// ---------------------------------------------------------------------------
// Embed kernel: copy bf16 row into f32 buffer
// ---------------------------------------------------------------------------
__global__ void embed_bf16_to_f32(const uint16_t* embed, int token, float* out, uint32_t H) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H) return;
    out[i] = __uint_as_float((uint32_t)embed[(size_t)token * H + i] << 16);
}

// ---------------------------------------------------------------------------
// Per-layer expert streamer (just-in-time loader from FP8 safetensors)
// ---------------------------------------------------------------------------
struct ExpertStreamer {
    st::ModelDir* M;
    int layer_idx;
    // device buffers reused per call
    uint8_t* d_gw; __nv_bfloat16* d_gs;
    uint8_t* d_uw; __nv_bfloat16* d_us;
    uint8_t* d_dw; __nv_bfloat16* d_ds;
    // host scratch
    std::vector<uint8_t> h_buf;

    void alloc() {
        // gate/up: [INTER=512, H=2048] FP8 = 1 MiB; scale [4, 16] bf16 = 128 B
        // down:    [H=2048, INTER=512] FP8 = 1 MiB; scale [16, 4] bf16 = 128 B
        CUDA_OK(cudaMalloc(&d_gw, q36::INTER * q36::H));
        CUDA_OK(cudaMalloc(&d_uw, q36::INTER * q36::H));
        CUDA_OK(cudaMalloc(&d_dw, q36::H     * q36::INTER));
        CUDA_OK(cudaMalloc(&d_gs, 4 * 16 * 2));
        CUDA_OK(cudaMalloc(&d_us, 4 * 16 * 2));
        CUDA_OK(cudaMalloc(&d_ds, 16 * 4 * 2));
    }
    void free_() {
        cudaFree(d_gw); cudaFree(d_uw); cudaFree(d_dw);
        cudaFree(d_gs); cudaFree(d_us); cudaFree(d_ds);
    }

    void operator()(int e,
                    const uint8_t** out_gw, const __nv_bfloat16** out_gs,
                    const uint8_t** out_uw, const __nv_bfloat16** out_us,
                    const uint8_t** out_dw, const __nv_bfloat16** out_ds)
    {
        char buf[256];
        std::string p = q36::layer_prefix(layer_idx);
        auto load = [&](const char* sub, void* dst, size_t n_max) {
            snprintf(buf, sizeof(buf), "%s.mlp.experts.%d.%s", p.c_str(), e, sub);
            std::vector<uint8_t> v;
            if (!st::read_bytes(M, buf, v)) std::exit(1);
            if (v.size() > n_max) { fprintf(stderr, "expert tensor %s too big\n", buf); std::exit(1); }
            CUDA_OK(cudaMemcpy(dst, v.data(), v.size(), cudaMemcpyHostToDevice));
        };
        load("gate_proj.weight",           d_gw, q36::INTER * q36::H);
        load("gate_proj.weight_scale_inv", d_gs, 4 * 16 * 2);
        load("up_proj.weight",             d_uw, q36::INTER * q36::H);
        load("up_proj.weight_scale_inv",   d_us, 4 * 16 * 2);
        load("down_proj.weight",           d_dw, q36::H * q36::INTER);
        load("down_proj.weight_scale_inv", d_ds, 16 * 4 * 2);
        *out_gw = d_gw; *out_gs = d_gs;
        *out_uw = d_uw; *out_us = d_us;
        *out_dw = d_dw; *out_ds = d_ds;
    }
};

// ---------------------------------------------------------------------------
// Loaded model: persistent everything except routed experts
// ---------------------------------------------------------------------------
struct Model {
    st::ModelDir M;
    uint16_t* d_embed     = nullptr;   // [VOCAB, H] bf16
    uint16_t* d_lm_head   = nullptr;
    uint16_t* d_final_norm= nullptr;

    q36::LinearAttnW lin_attn[q36::NUM_LAYERS];
    q36::FullAttnW   full_attn[q36::NUM_LAYERS];
    q36::MoEW        moe[q36::NUM_LAYERS];      // post_ln + router + shared_gate + shared_expert
    bool             is_full[q36::NUM_LAYERS];

    // per-layer scratch (one set, reused since layers run in series)
    q36::LinearAttnScratch Lscratch{};
    q36::FullAttnScratch   Fscratch{};
    q36::MoEScratch        Mscratch{};

    // per-layer recurrence state (persistent across tokens)
    float* d_conv_state[q36::NUM_LAYERS];     // [3, QKV_DIM]; only for linear layers
    float* d_delta_state[q36::NUM_LAYERS];    // [LIN_NV, LIN_HEAD, LIN_HEAD]; linear only
    float* d_K_cache[q36::NUM_LAYERS];        // [max_seq, FA_KV_DIM]; full only
    float* d_V_cache[q36::NUM_LAYERS];        // [max_seq, FA_KV_DIM]; full only

    // ping-pong hidden state buffers
    float* d_h_in;
    float* d_h_out;
    float* d_resid1;
    float* d_h_norm;     // post final-norm
    float* d_logits;

    std::vector<ExpertStreamer> streamers;    // one per layer (different layer_idx each)
};

static void load_model(Model* G, const std::string& model_dir, uint32_t max_seq) {
    if (!st::open(&G->M, model_dir)) std::exit(1);

    fprintf(stderr, "loading globals: embed, lm_head, final_norm...\n");
    G->d_embed      = (uint16_t*)q36::upload_bytes(&G->M, "model.language_model.embed_tokens.weight");
    G->d_lm_head    = (uint16_t*)q36::upload_bytes(&G->M, "lm_head.weight");
    G->d_final_norm = (uint16_t*)q36::upload_bytes(&G->M, "model.language_model.norm.weight");

    fprintf(stderr, "loading per-layer non-expert weights...\n");
    for (uint32_t li = 0; li < q36::NUM_LAYERS; li++) {
        std::string p = q36::layer_prefix(li);
        G->is_full[li] = q36::is_full_attn(li);
        if (G->is_full[li]) q36::load_full_attn  (&G->M, p, &G->full_attn[li]);
        else                q36::load_linear_attn(&G->M, p, &G->lin_attn[li]);
        // MoE non-expert: post_ln + router + shared_gate + shared_expert (3 FP8 tensors).
        // Reuse load_moe but skip the 256-expert loop by loading just the small parts manually.
        auto T = [&](const std::string& nm){ return p + "." + nm; };
        G->moe[li].post_ln_w     = (uint16_t*)q36::upload_bytes(&G->M, T("post_attention_layernorm.weight"));
        G->moe[li].router_w      = (uint16_t*)q36::upload_bytes(&G->M, T("mlp.gate.weight"));
        G->moe[li].shared_gate_w = (uint16_t*)q36::upload_bytes(&G->M, T("mlp.shared_expert_gate.weight"));
        G->moe[li].shared.gate_w = q36::upload_bytes(&G->M, T("mlp.shared_expert.gate_proj.weight"));
        G->moe[li].shared.gate_s = (__nv_bfloat16*)q36::upload_bytes(&G->M, T("mlp.shared_expert.gate_proj.weight_scale_inv"));
        G->moe[li].shared.up_w   = q36::upload_bytes(&G->M, T("mlp.shared_expert.up_proj.weight"));
        G->moe[li].shared.up_s   = (__nv_bfloat16*)q36::upload_bytes(&G->M, T("mlp.shared_expert.up_proj.weight_scale_inv"));
        G->moe[li].shared.down_w = q36::upload_bytes(&G->M, T("mlp.shared_expert.down_proj.weight"));
        G->moe[li].shared.down_s = (__nv_bfloat16*)q36::upload_bytes(&G->M, T("mlp.shared_expert.down_proj.weight_scale_inv"));
        if (li % 5 == 0) fprintf(stderr, "  layer %u/%u\n", li, q36::NUM_LAYERS);
    }

    fprintf(stderr, "allocating scratch + recurrence state (max_seq=%u)...\n", max_seq);
    q36::alloc_linear_attn_scratch(&G->Lscratch);
    q36::alloc_full_attn_scratch  (&G->Fscratch, max_seq);
    q36::alloc_moe_scratch        (&G->Mscratch);

    // Per-layer recurrence state
    for (uint32_t li = 0; li < q36::NUM_LAYERS; li++) {
        if (G->is_full[li]) {
            CUDA_OK(cudaMalloc(&G->d_K_cache[li], (size_t)max_seq * q36::FA_KV_DIM * 4));
            CUDA_OK(cudaMalloc(&G->d_V_cache[li], (size_t)max_seq * q36::FA_KV_DIM * 4));
            G->d_conv_state[li]  = nullptr; G->d_delta_state[li] = nullptr;
        } else {
            CUDA_OK(cudaMalloc(&G->d_conv_state[li],  3 * q36::QKV_DIM * 4));
            CUDA_OK(cudaMalloc(&G->d_delta_state[li], q36::LIN_NV * q36::LIN_HEAD * q36::LIN_HEAD * 4));
            CUDA_OK(cudaMemset(G->d_conv_state[li],  0, 3 * q36::QKV_DIM * 4));
            CUDA_OK(cudaMemset(G->d_delta_state[li], 0, q36::LIN_NV * q36::LIN_HEAD * q36::LIN_HEAD * 4));
            G->d_K_cache[li] = nullptr; G->d_V_cache[li] = nullptr;
        }
    }

    CUDA_OK(cudaMalloc(&G->d_h_in,   q36::H * 4));
    CUDA_OK(cudaMalloc(&G->d_h_out,  q36::H * 4));
    CUDA_OK(cudaMalloc(&G->d_resid1, q36::H * 4));
    CUDA_OK(cudaMalloc(&G->d_h_norm, q36::H * 4));
    CUDA_OK(cudaMalloc(&G->d_logits, q36::VOCAB * 4));

    // Expert streamers (one per layer; share `M` index)
    G->streamers.resize(q36::NUM_LAYERS);
    for (uint32_t li = 0; li < q36::NUM_LAYERS; li++) {
        G->streamers[li].M = &G->M;
        G->streamers[li].layer_idx = (int)li;
        G->streamers[li].alloc();
    }
    fprintf(stderr, "model loaded.\n");
}

// ---------------------------------------------------------------------------
// One-token forward: embed + 40 layers + final norm + lm_head → logits.
// `pos` is the token's absolute position (used by full-attn KV cache).
// On entry the per-layer recurrence state must reflect tokens [0..pos-1].
// On exit `out_logits` (host vector) holds the [VOCAB] logits for this token.
// ---------------------------------------------------------------------------
static void forward_one(Model* G, int token, uint32_t pos, uint32_t max_seq,
                        std::vector<float>& out_logits)
{
    // embed
    {
        dim3 b(256), g((q36::H + 255) / 256);
        embed_bf16_to_f32<<<g, b>>>(G->d_embed, token, G->d_h_in, q36::H);
    }

    for (uint32_t li = 0; li < q36::NUM_LAYERS; li++) {
        // attention substep, using the layer's persistent state
        if (G->is_full[li]) {
            // splice the layer-specific KV cache into Fscratch
            G->Fscratch.d_K_cache = G->d_K_cache[li];
            G->Fscratch.d_V_cache = G->d_V_cache[li];
            q36::run_full_attn_step(G->full_attn[li], G->d_h_in, G->d_resid1, &G->Fscratch, pos, max_seq);
        } else {
            G->Lscratch.d_conv_state  = G->d_conv_state[li];
            G->Lscratch.d_delta_state = G->d_delta_state[li];
            q36::run_linear_attn_step(G->lin_attn[li], G->d_h_in, G->d_resid1, &G->Lscratch);
        }

        // MoE substep, with experts streamed from disk
        q36::run_moe_step(G->moe[li], G->d_resid1, G->d_h_out, &G->Mscratch, G->streamers[li]);

        // swap hidden buffers for next layer
        std::swap(G->d_h_in, G->d_h_out);
    }

    // final norm + lm_head
    qwen36::launch_rms_norm_bf16_plus_one(G->d_h_in, G->d_final_norm, G->d_h_norm, q36::H, q36::RMS_EPS);
    launch_matvec_bf16(G->d_lm_head, G->d_h_norm, G->d_logits, q36::VOCAB, q36::H);
    out_logits.resize(q36::VOCAB);
    CUDA_OK(cudaMemcpy(out_logits.data(), G->d_logits, q36::VOCAB * 4, cudaMemcpyDeviceToHost));
}

// ---------------------------------------------------------------------------
static void usage(const char* prog) {
    fprintf(stderr,
        "usage: %s --model-dir DIR \\\n"
        "          (--prompt TEXT | --chat ROLE TEXT [ROLE TEXT ...] | --tokens \"id,id,...\") \\\n"
        "          [--max-tokens N] [--max-seq L]\\\n"
        "          [--greedy | --temp T --top-p P --top-k K] [--rep-pen R] [--seed N]\n",
        prog);
}

int main(int argc, char** argv) {
    std::string model_dir;
    std::string tokens_csv, prompt_text;
    std::vector<std::pair<std::string,std::string>> chat_msgs;
    int max_tokens = 64, max_seq = 2048;
    SampleCfg samp{}; uint64_t seed = (uint64_t)time(nullptr);

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--model-dir"  && i + 1 < argc) model_dir = argv[++i];
        else if (a == "--tokens"     && i + 1 < argc) tokens_csv = argv[++i];
        else if (a == "--prompt"     && i + 1 < argc) prompt_text = argv[++i];
        else if (a == "--chat") {
            while (i + 2 < argc && argv[i+1][0] != '-') {
                chat_msgs.push_back({argv[i+1], argv[i+2]}); i += 2;
            }
        }
        else if (a == "--max-tokens" && i + 1 < argc) max_tokens = std::atoi(argv[++i]);
        else if (a == "--max-seq"    && i + 1 < argc) max_seq    = std::atoi(argv[++i]);
        else if (a == "--greedy")                     samp.greedy = true;
        else if (a == "--temp"       && i + 1 < argc) samp.temp   = (float)std::atof(argv[++i]);
        else if (a == "--top-p"      && i + 1 < argc) samp.top_p  = (float)std::atof(argv[++i]);
        else if (a == "--top-k"      && i + 1 < argc) samp.top_k  = std::atoi(argv[++i]);
        else if (a == "--rep-pen"    && i + 1 < argc) samp.rep_pen= (float)std::atof(argv[++i]);
        else if (a == "--seed"       && i + 1 < argc) seed        = (uint64_t)std::atoll(argv[++i]);
        else { usage(argv[0]); return 1; }
    }
    int input_modes = (int)!tokens_csv.empty() + (int)!prompt_text.empty() + (int)!chat_msgs.empty();
    if (model_dir.empty() || input_modes != 1) { usage(argv[0]); return 1; }

    cudaDeviceProp p{}; cudaGetDeviceProperties(&p, 0);
    fprintf(stderr, "device: %s sm_%d%d  total mem=%.1f GB\n", p.name, p.major, p.minor,
            (double)p.totalGlobalMem / 1e9);

    TokClient tok{};
    bool need_tok = tokens_csv.empty();
    if (need_tok && !tok_open(&tok, model_dir)) {
        fprintf(stderr, "failed to spawn tokenizer\n"); return 1;
    }

    std::vector<int> prompt_ids;
    if (!tokens_csv.empty())       prompt_ids = parse_csv_ids(tokens_csv);
    else if (!prompt_text.empty()) prompt_ids = tok_encode(&tok, prompt_text);
    else                           prompt_ids = tok_chat(&tok, chat_msgs);
    if (prompt_ids.empty()) { fprintf(stderr, "no prompt tokens\n"); return 1; }

    fprintf(stderr, "prompt: %zu tokens\n", prompt_ids.size());

    Model G{};
    load_model(&G, model_dir, (uint32_t)max_seq);

    std::mt19937 rng((uint32_t)(seed ^ (seed >> 32)));

    // Prefill — feed each prompt token through, advancing per-layer state.
    fprintf(stderr, "prefill...\n");
    std::vector<float> logits;
    for (size_t i = 0; i < prompt_ids.size(); i++) {
        forward_one(&G, prompt_ids[i], (uint32_t)i, (uint32_t)max_seq, logits);
        fprintf(stderr, "\r  prefill %zu/%zu", i + 1, prompt_ids.size());
    }
    fprintf(stderr, "\n");

    // Sample first generated token from prefill logits.
    std::vector<int> history(prompt_ids);
    int next = sample_next(logits, history, samp, rng);
    history.push_back(next);

    if (need_tok) {
        tok_decode_reset(&tok);
        for (int id : prompt_ids) tok_decode_push(&tok, id);
    }

    auto is_eos = [&](int id){ for (int e : tok.eos_ids) if (id == e) return true; return false; };

    int pos = (int)prompt_ids.size();
    int generated = 0;
    while (generated < max_tokens && pos < max_seq) {
        if (is_eos(next)) { fprintf(stderr, "\n[eos %d]\n", next); break; }
        if (need_tok) {
            std::string frag = tok_decode_push(&tok, next);
            fwrite(frag.data(), 1, frag.size(), stdout); fflush(stdout);
        } else {
            printf(" %d", next); fflush(stdout);
        }
        forward_one(&G, next, (uint32_t)pos++, (uint32_t)max_seq, logits);
        history.push_back(next);
        next = sample_next(logits, history, samp, rng);
        generated++;
    }
    printf("\n");

    if (need_tok) tok_close(&tok);
    return 0;
}
