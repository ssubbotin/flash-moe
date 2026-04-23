/*
 * infer_kimi.cu — single-file inference driver for Kimi K2.6.
 *
 * Uses kimi_loader.cuh + kimi_forward.cuh + kimi_moe.cuh + mla_forward.cuh +
 * kernels.cuh. All heavy lifting lives in headers.
 *
 * Usage:
 *   ./infer_kimi --model-dir kimi-k2.6 --packed-dir kimi-k2.6/packed_experts \
 *                --tokens "1,2,3" --max-tokens 10 --max-seq 2048
 *
 * Tokenizer is not yet ported (see task 9). CLI accepts explicit token IDs;
 * outputs decoded token IDs.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <string>
#include <vector>
#include "kimi_loader.cuh"
#include "kimi_forward.cuh"

static std::vector<int> parse_tokens_csv(const std::string& s) {
    std::vector<int> out; size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && !(std::isdigit((unsigned char)s[i]) || s[i] == '-')) i++;
        if (i >= s.size()) break;
        char* end; long v = std::strtol(s.c_str() + i, &end, 10);
        out.push_back((int)v);
        i = (size_t)(end - s.c_str());
    }
    return out;
}

int main(int argc, char** argv) {
    std::string model_dir  = "kimi-k2.6";
    std::string packed_dir = "kimi-k2.6/packed_experts";
    std::string tokens_csv = "1";
    int max_tokens = 8;
    int max_seq    = 2048;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--model-dir"  && i + 1 < argc) model_dir  = argv[++i];
        else if (a == "--packed-dir" && i + 1 < argc) packed_dir = argv[++i];
        else if (a == "--tokens"     && i + 1 < argc) tokens_csv = argv[++i];
        else if (a == "--max-tokens" && i + 1 < argc) max_tokens = std::atoi(argv[++i]);
        else if (a == "--max-seq"    && i + 1 < argc) max_seq    = std::atoi(argv[++i]);
        else {
            fprintf(stderr,
                "usage: %s --model-dir D --packed-dir P --tokens ids [--max-tokens N] [--max-seq L]\n",
                argv[0]);
            return 1;
        }
    }

    auto prompt_ids = parse_tokens_csv(tokens_csv);
    if (prompt_ids.empty()) { fprintf(stderr, "no prompt tokens\n"); return 1; }

    if (const char* d = std::getenv("KIMI_DEBUG")) g_kimi_debug = std::atoi(d);

    printf("Loading Kimi model from %s (packed %s) max_seq=%d\n",
           model_dir.c_str(), packed_dir.c_str(), max_seq);
    KimiModel K{};
    if (!kimi_build_model(model_dir, packed_dir, max_seq, &K)) {
        fprintf(stderr, "kimi_build_model failed\n"); return 1;
    }

    // Prefill
    printf("prefill:");
    int next = -1;
    for (size_t i = 0; i < prompt_ids.size(); i++) {
        printf(" %d", prompt_ids[i]); fflush(stdout);
        next = kimi_forward(&K, prompt_ids[i], (int)i);
    }
    printf(" -> %d\n", next);

    // Decode
    printf("decode:");
    int pos = (int)prompt_ids.size();
    for (int t = 0; t < max_tokens && pos < max_seq; t++) {
        printf(" %d", next); fflush(stdout);
        next = kimi_forward(&K, next, pos++);
    }
    printf("\n");
    return 0;
}
