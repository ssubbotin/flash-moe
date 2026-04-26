/*
 * infer_kimi.cu — single-file inference driver for Kimi K2.6.
 *
 * Uses kimi_loader.cuh + kimi_forward.cuh + kimi_moe.cuh + mla_forward.cuh +
 * kernels.cuh. All heavy lifting lives in headers.
 *
 * Tokenization: a long-lived python child process running
 *   `python3 cuda_infer/kimi_tokenize.py serve --model-dir <model-dir>`
 * speaks a line-based protocol over a pair of pipes. See kimi_tokenize.py for
 * the wire format.
 *
 * Usage:
 *   ./infer_kimi --model-dir kimi-k2.6 --packed-dir kimi-k2.6/packed_experts \
 *                --prompt "Hello, how are you?" --max-tokens 64
 *
 *   ./infer_kimi --model-dir kimi-k2.6 --packed-dir kimi-k2.6/packed_experts \
 *                --chat user "What is 2+2?" --max-tokens 64
 *
 *   # Skip tokenizer entirely (legacy / debug):
 *   ./infer_kimi --tokens "1,2,3" --max-tokens 8
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <ctime>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include "kimi_loader.cuh"
#include "kimi_forward.cuh"

// ---------------------------------------------------------------------------
// Tokenizer subprocess client
// ---------------------------------------------------------------------------
struct TokClient {
    pid_t pid = -1;
    FILE* to_child   = nullptr;   // parent writes -> child stdin
    FILE* from_child = nullptr;   // parent reads  <- child stdout
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
    std::string out;
    int val = 0, valb = -8;
    for (unsigned char c : in) {
        if (c == '=' || T[c] == -1) continue;
        val = (val << 6) | T[c]; valb += 6;
        if (valb >= 0) { out.push_back(char((val >> valb) & 0xFF)); valb -= 8; }
    }
    return out;
}

// Send "verb\t...\n" and read back one "OK\t...\n" or "ERR\t...\n" line.
// Returns the part after "OK\t" on success, or "" with *ok=false on error.
static std::string tok_call(TokClient* tc, const std::string& cmd, bool* ok) {
    fputs(cmd.c_str(), tc->to_child);
    fputc('\n', tc->to_child);
    fflush(tc->to_child);

    char* line = nullptr; size_t cap = 0;
    ssize_t n = getline(&line, &cap, tc->from_child);
    if (n <= 0) {
        fprintf(stderr, "tok_call: child closed pipe\n");
        if (line) free(line);
        *ok = false; return "";
    }
    while (n > 0 && (line[n-1] == '\n' || line[n-1] == '\r')) line[--n] = 0;
    std::string s(line, (size_t)n);
    free(line);

    if (s.size() >= 3 && s.compare(0, 3, "OK\t") == 0) { *ok = true;  return s.substr(3); }
    if (s == "OK")                                      { *ok = true;  return ""; }
    if (s.size() >= 4 && s.compare(0, 4, "ERR\t") == 0) { *ok = false; fprintf(stderr, "tok ERR: %s\n", s.c_str() + 4); return ""; }
    fprintf(stderr, "tok unexpected reply: %s\n", s.c_str());
    *ok = false; return "";
}

static bool tok_open(TokClient* tc, const std::string& model_dir) {
    int in_pipe[2], out_pipe[2];
    if (pipe(in_pipe) != 0 || pipe(out_pipe) != 0) { perror("pipe"); return false; }

    pid_t pid = fork();
    if (pid < 0) { perror("fork"); return false; }
    if (pid == 0) {
        // child
        dup2(in_pipe[0],  0); close(in_pipe[1]);
        dup2(out_pipe[1], 1); close(out_pipe[0]);
        close(in_pipe[0]); close(out_pipe[1]);
        const char* py = std::getenv("KIMI_PYTHON");
        if (!py || !*py) py = "python3";
        // The script lives next to this binary (cuda_infer/kimi_tokenize.py).
        // Resolve via /proc/self/exe so we work regardless of cwd.
        char self[4096]; ssize_t sn = readlink("/proc/self/exe", self, sizeof(self) - 1);
        std::string script;
        if (sn > 0) {
            self[sn] = 0;
            std::string s = self;
            auto slash = s.find_last_of('/');
            script = (slash == std::string::npos ? "" : s.substr(0, slash + 1)) + "kimi_tokenize.py";
        } else {
            script = "kimi_tokenize.py";
        }
        execlp(py, py, script.c_str(), "--model-dir", model_dir.c_str(), "serve", (char*)nullptr);
        perror("execlp python3");
        _exit(127);
    }
    // parent
    close(in_pipe[0]); close(out_pipe[1]);
    tc->pid = pid;
    tc->to_child   = fdopen(in_pipe[1],  "w");
    tc->from_child = fdopen(out_pipe[0], "r");
    if (!tc->to_child || !tc->from_child) { perror("fdopen"); return false; }

    // Wait for "READY\n" handshake.
    char* line = nullptr; size_t cap = 0;
    if (getline(&line, &cap, tc->from_child) <= 0) {
        fprintf(stderr, "tokenizer subprocess died before READY\n");
        if (line) free(line);
        return false;
    }
    free(line);

    bool ok;
    std::string s = tok_call(tc, "info", &ok);
    if (!ok) return false;
    // payload: "<vocab>\t<bos>\t<eos1,eos2,...>"
    int p1 = s.find('\t'), p2 = s.find('\t', p1 + 1);
    tc->vocab_size = std::atoi(s.substr(0, p1).c_str());
    tc->bos_id     = std::atoi(s.substr(p1 + 1, p2 - p1 - 1).c_str());
    std::string eos = s.substr(p2 + 1);
    size_t i = 0;
    while (i < eos.size()) {
        size_t j = eos.find(',', i);
        if (j == std::string::npos) j = eos.size();
        int id = std::atoi(eos.substr(i, j - i).c_str());
        if (id >= 0) tc->eos_ids.push_back(id);
        i = j + 1;
    }
    fprintf(stderr, "tokenizer ready: vocab=%d bos=%d eos=", tc->vocab_size, tc->bos_id);
    for (size_t k = 0; k < tc->eos_ids.size(); k++)
        fprintf(stderr, "%s%d", k ? "," : "", tc->eos_ids[k]);
    fprintf(stderr, "\n");
    return true;
}

static void tok_close(TokClient* tc) {
    if (tc->to_child) {
        bool ok; tok_call(tc, "quit", &ok);
        fclose(tc->to_child);   tc->to_child = nullptr;
    }
    if (tc->from_child) { fclose(tc->from_child); tc->from_child = nullptr; }
    if (tc->pid > 0) { int st = 0; waitpid(tc->pid, &st, 0); tc->pid = -1; }
}

static std::vector<int> parse_csv_ids(const std::string& s) {
    std::vector<int> out;
    size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && !(std::isdigit((unsigned char)s[i]) || s[i] == '-')) i++;
        if (i >= s.size()) break;
        char* end; long v = std::strtol(s.c_str() + i, &end, 10);
        out.push_back((int)v);
        i = (size_t)(end - s.c_str());
    }
    return out;
}

static std::vector<int> tok_encode(TokClient* tc, const std::string& text) {
    bool ok; std::string s = tok_call(tc, "encode\t" + b64encode(text), &ok);
    if (!ok) return {};
    return parse_csv_ids(s);
}

// chat_msgs is alternating role/content pairs.
static std::vector<int> tok_chat(TokClient* tc, const std::vector<std::pair<std::string,std::string>>& msgs) {
    // build a tiny json by hand to avoid pulling in a json lib
    std::string j = "[";
    for (size_t i = 0; i < msgs.size(); i++) {
        if (i) j += ",";
        j += "{\"role\":\"";
        for (char c : msgs[i].first)  { if (c == '"' || c == '\\') j += '\\'; j += c; }
        j += "\",\"content\":\"";
        for (char c : msgs[i].second) {
            if (c == '"' || c == '\\') { j += '\\'; j += c; }
            else if (c == '\n') j += "\\n";
            else if (c == '\r') j += "\\r";
            else if (c == '\t') j += "\\t";
            else if ((unsigned char)c < 0x20) { char buf[8]; snprintf(buf, sizeof(buf), "\\u%04x", c); j += buf; }
            else j += c;
        }
        j += "\"}";
    }
    j += "]";
    bool ok; std::string s = tok_call(tc, "chat\t" + b64encode(j), &ok);
    if (!ok) return {};
    return parse_csv_ids(s);
}

static void tok_decode_reset(TokClient* tc) {
    bool ok; tok_call(tc, "decode_stream_reset", &ok);
}

static std::string tok_decode_push(TokClient* tc, int id) {
    char buf[64]; snprintf(buf, sizeof(buf), "decode_stream_push\t%d", id);
    bool ok; std::string s = tok_call(tc, buf, &ok);
    if (!ok) return "";
    return b64decode(s);
}

// ---------------------------------------------------------------------------
// Sampling (replaces the GPU argmax; runs on host over copied logits).
// ---------------------------------------------------------------------------
struct SampleCfg {
    bool   greedy   = false;
    float  temp     = 0.7f;
    float  top_p    = 0.9f;
    int    top_k    = 0;     // 0 = disabled
    float  rep_pen  = 1.0f;  // 1 = no penalty
};

static int sample_next(const std::vector<float>& logits,
                       const std::vector<int>& history,
                       const SampleCfg& cfg,
                       std::mt19937& rng) {
    int V = (int)logits.size();
    if (cfg.greedy || cfg.temp <= 0.0f) {
        int best = 0; float bv = logits[0];
        for (int i = 1; i < V; i++) if (logits[i] > bv) { bv = logits[i]; best = i; }
        return best;
    }

    std::vector<float> z(logits);
    if (cfg.rep_pen != 1.0f) {
        for (int id : history) {
            if (id < 0 || id >= V) continue;
            z[id] = z[id] > 0 ? z[id] / cfg.rep_pen : z[id] * cfg.rep_pen;
        }
    }
    float inv_t = 1.0f / cfg.temp;
    for (int i = 0; i < V; i++) z[i] *= inv_t;

    std::vector<int> idx(V);
    for (int i = 0; i < V; i++) idx[i] = i;
    int keep = V;
    if (cfg.top_k > 0 && cfg.top_k < V) {
        std::partial_sort(idx.begin(), idx.begin() + cfg.top_k, idx.end(),
                          [&](int a, int b){ return z[a] > z[b]; });
        keep = cfg.top_k;
    } else {
        std::sort(idx.begin(), idx.end(),
                  [&](int a, int b){ return z[a] > z[b]; });
    }

    float zmax = z[idx[0]];
    std::vector<float> p(keep);
    double sum = 0;
    for (int i = 0; i < keep; i++) { p[i] = std::exp(z[idx[i]] - zmax); sum += p[i]; }
    for (int i = 0; i < keep; i++) p[i] = (float)(p[i] / sum);

    if (cfg.top_p > 0.0f && cfg.top_p < 1.0f) {
        double cum = 0; int cut = keep;
        for (int i = 0; i < keep; i++) { cum += p[i]; if (cum >= cfg.top_p) { cut = i + 1; break; } }
        keep = cut;
        sum = 0; for (int i = 0; i < keep; i++) sum += p[i];
        for (int i = 0; i < keep; i++) p[i] = (float)(p[i] / sum);
    }

    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    float r = U(rng), c = 0;
    for (int i = 0; i < keep; i++) { c += p[i]; if (r <= c) return idx[i]; }
    return idx[keep - 1];
}

// ---------------------------------------------------------------------------
// Forward variants: with-argmax (legacy) + logits-only (used by sampler).
// ---------------------------------------------------------------------------
static int kimi_forward_with_logits(KimiModel* K, int token, int pos,
                                    std::vector<float>* out_logits) {
    int next = kimi_forward(K, token, pos);  // runs the GPU path + argmax inside
    if (out_logits) {
        out_logits->resize(K->cfg.vocab_size);
        cudaMemcpy(out_logits->data(), K->d_logits,
                   (size_t)K->cfg.vocab_size * sizeof(float), cudaMemcpyDeviceToHost);
    }
    return next;
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------
static void usage(const char* prog) {
    fprintf(stderr,
        "usage: %s --model-dir DIR --packed-dir DIR \\\n"
        "          (--prompt TEXT | --chat ROLE TEXT [ROLE TEXT ...] | --tokens \"id,id,...\") \\\n"
        "          [--max-tokens N] [--max-seq L]\\\n"
        "          [--greedy | --temp T --top-p P --top-k K] [--rep-pen R] [--seed N]\n",
        prog);
}

int main(int argc, char** argv) {
    std::string model_dir  = "kimi-k2.6";
    std::string packed_dir = "kimi-k2.6/packed_experts";
    std::string tokens_csv;
    std::string prompt_text;
    std::vector<std::pair<std::string,std::string>> chat_msgs;
    int max_tokens = 64;
    int max_seq    = 2048;
    SampleCfg samp{};
    samp.greedy = false;
    uint64_t seed = (uint64_t)time(nullptr);

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--model-dir"  && i + 1 < argc) model_dir  = argv[++i];
        else if (a == "--packed-dir" && i + 1 < argc) packed_dir = argv[++i];
        else if (a == "--tokens"     && i + 1 < argc) tokens_csv = argv[++i];
        else if (a == "--prompt"     && i + 1 < argc) prompt_text = argv[++i];
        else if (a == "--chat") {
            // pairs of role text until next --flag or end of argv
            while (i + 2 < argc && argv[i+1][0] != '-') {
                chat_msgs.push_back({argv[i+1], argv[i+2]});
                i += 2;
            }
        }
        else if (a == "--max-tokens" && i + 1 < argc) max_tokens = std::atoi(argv[++i]);
        else if (a == "--max-seq"    && i + 1 < argc) max_seq    = std::atoi(argv[++i]);
        else if (a == "--greedy")                     samp.greedy = true;
        else if (a == "--temp"       && i + 1 < argc) samp.temp   = (float)std::atof(argv[++i]);
        else if (a == "--top-p"      && i + 1 < argc) samp.top_p  = (float)std::atof(argv[++i]);
        else if (a == "--top-k"      && i + 1 < argc) samp.top_k  = std::atoi(argv[++i]);
        else if (a == "--rep-pen"    && i + 1 < argc) samp.rep_pen = (float)std::atof(argv[++i]);
        else if (a == "--seed"       && i + 1 < argc) seed         = (uint64_t)std::atoll(argv[++i]);
        else { usage(argv[0]); return 1; }
    }

    int input_modes = (int)!tokens_csv.empty() + (int)!prompt_text.empty() + (int)!chat_msgs.empty();
    if (input_modes != 1) {
        fprintf(stderr, "error: pass exactly one of --tokens / --prompt / --chat\n");
        usage(argv[0]); return 1;
    }

    if (const char* d = std::getenv("KIMI_DEBUG")) g_kimi_debug = std::atoi(d);

    // Spawn tokenizer subprocess if we need it.
    TokClient tok{};
    bool need_tok = tokens_csv.empty();
    if (need_tok) {
        if (!tok_open(&tok, model_dir)) {
            fprintf(stderr, "failed to start tokenizer subprocess\n"); return 1;
        }
    }

    std::vector<int> prompt_ids;
    if (!tokens_csv.empty()) {
        prompt_ids = parse_csv_ids(tokens_csv);
    } else if (!prompt_text.empty()) {
        prompt_ids = tok_encode(&tok, prompt_text);
    } else {
        prompt_ids = tok_chat(&tok, chat_msgs);
    }
    if (prompt_ids.empty()) { fprintf(stderr, "no prompt tokens\n"); return 1; }

    fprintf(stderr, "prompt: %zu tokens", prompt_ids.size());
    if (!prompt_ids.empty()) fprintf(stderr, " (first=%d last=%d)", prompt_ids.front(), prompt_ids.back());
    fprintf(stderr, "\n");

    fprintf(stderr, "loading model from %s (packed %s) max_seq=%d\n",
            model_dir.c_str(), packed_dir.c_str(), max_seq);
    KimiModel K{};
    if (!kimi_build_model(model_dir, packed_dir, max_seq, &K)) {
        fprintf(stderr, "kimi_build_model failed\n");
        if (need_tok) tok_close(&tok);
        return 1;
    }

    std::mt19937 rng((uint32_t)(seed ^ (seed >> 32)));

    // ---- Prefill ----
    fprintf(stderr, "prefill...\n"); fflush(stderr);
    std::vector<float> logits;
    int next = -1;
    for (size_t i = 0; i < prompt_ids.size(); i++) {
        bool last = (i + 1 == prompt_ids.size());
        if (last) next = kimi_forward_with_logits(&K, prompt_ids[i], (int)i, &logits);
        else      kimi_forward(&K, prompt_ids[i], (int)i);
    }

    // Sample first generated token from prefill logits.
    std::vector<int> history(prompt_ids);
    next = sample_next(logits, history, samp, rng);
    history.push_back(next);

    // ---- Decode ----
    if (need_tok) {
        tok_decode_reset(&tok);
        // seed the streaming decoder with the prompt so byte boundaries align
        for (int id : prompt_ids) tok_decode_push(&tok, id);
    }

    auto is_eos = [&](int id){
        for (int e : tok.eos_ids) if (id == e) return true;
        return false;
    };

    int pos = (int)prompt_ids.size();
    int generated = 0;
    while (generated < max_tokens && pos < max_seq) {
        if (is_eos(next)) { fprintf(stderr, "\n[eos %d]\n", next); break; }

        if (need_tok) {
            std::string frag = tok_decode_push(&tok, next);
            fwrite(frag.data(), 1, frag.size(), stdout);
            fflush(stdout);
        } else {
            printf(" %d", next); fflush(stdout);
        }

        // forward this token, sample the next
        kimi_forward_with_logits(&K, next, pos++, &logits);
        history.push_back(next);
        next = sample_next(logits, history, samp, rng);
        generated++;
    }
    printf("\n");

    if (need_tok) tok_close(&tok);
    return 0;
}
