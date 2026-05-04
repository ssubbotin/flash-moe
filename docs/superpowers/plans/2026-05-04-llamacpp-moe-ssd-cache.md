# llama.cpp MoE SSD Streaming + GPU LRU Cache — Hackathon Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land a focused PR (or PR series) on `ggml-org/llama.cpp` that adds GPU-resident LRU caching for MoE expert weights with optional SSD streaming, plus an MI300X-tuned fused MoE kernel — and ship a small MIT-licensed companion repo with an agent demo. Targets the AMD Developer Hackathon (deadline 2026-05-10).

**Architecture (Option D after 2026-05-04 recon — supersedes the original Phase 2/3/4 plan; see "Plan revision note" below):**

- New CUDA/HIP-backend buffer type **`ggml_backend_buffer_type_moe_stream`**. Tensors allocated through it are MoE expert weights; their `data` is a stable pseudo-pointer (the buffer-type handles all reads), and the bytes live in a 3-tier cache: VRAM LRU pool (the Phase 1 cache), OS page cache (kernel-managed), SSD pack file (per layer).
- The CUDA backend's `supports_op(MUL_MAT_ID)` returns true for ops whose src0 sits in `moe_stream` buffers and dispatches a **custom MoE forward path** instead of the existing MMVQ/MMQ/MMVF/MMF kernels: routing → 3-tier expert lookup for K active experts → fused dequant+SwiGLU+down kernel (the Sergey-authored kernel ported from `mi300-opt/rocm_infer/kernels_fused_moe.hip.h`) → write outputs.
- This mirrors flash-moe's `infer.hip` `layer_forward` pattern 1-to-1: VRAM cache → page cache fallback → SSD pread → custom kernel on K device pointers. It also mirrors llama.cpp's existing `--n-cpu-moe` shape (load-time tensor-buffer-type override), so upstream reviewers have a precedent.
- Backend kernel work lands in `ggml/src/ggml-cuda/` with HIP guards (which is how `ggml-hip` already shares CUDA sources). gfx942-specific tunings gated by `defined(__HIP_PLATFORM_AMD__) && defined(__gfx942__)`.
- All code written from public references (vLLM, AITER MIT, llama.cpp itself, e1n00r/tinyserve, the Qwen3.5 paper) — and from the three Sergey-authored files on `mi300-opt` whose copyright header explicitly carries `Sergey Subbotin`. **Zero lines copied from danveloper/flash-moe** to keep our PR (and our companion repo) MIT-licensable.

**Plan revision note:** The original Phases 2/3/4 assumed we could intercept per-expert weight pointers inside `mmid.cu`. Recon (2026-05-04) showed the real decode path uses kernels (`mmvq`/`mmq`/`mmvf`/`mmf`) that take a single contiguous expert-slab pointer + ids tensor; the per-expert loop in `ggml-cuda.cu:2585` is reached only on prefill / large batch, not on decode. The buffer-type + custom-dispatch architecture (Option D) sidesteps the kernel-level intercept entirely, runs on the decode path, and matches both flash-moe and `--n-cpu-moe`. Phase 1 stands unchanged — the Phase 1 LRU is now the VRAM tier of the new buffer type.

**Tech Stack:** llama.cpp upstream (MIT), ggml (MIT), HIP/ROCm 7.2 for MI300X path, CMake, ctest. Companion demo repo: Python 3.11+ with `openai` SDK (MIT) and `httpx`.

---

## License & provenance — read before writing any code

This is the constraint that makes the rest of the plan possible. Violating it sinks the submission.

1. **Source materials we may consult and cite**:
   - llama.cpp itself (MIT) — read freely, edit freely.
   - vLLM (Apache 2.0) — read for architecture reference, cite in commit message + code comments where ideas come from. RFC #38256 and PR #37190 are the canonical references.
   - AITER (MIT) — read for HIP MFMA / fused MoE kernel patterns.
   - llama.cpp Adreno MoE PR #22301 (MIT, just merged) — read for the MUL_MAT_ID dispatch hooks.
   - tinyserve (MIT, by e1n00r) — read for the `ExpertWeightProvider` ABC pattern.
   - Qwen3.5 / Qwen3-Next paper — read for model architecture (already implemented in llama.cpp).

2. **Source materials we may NOT consult**:
   - `danveloper/flash-moe` and any of its forks (no license = all rights reserved).
   - **Including our own `mi300-opt` branch** — that branch carries Dan's `infer.hip` / `kernels.hip.h` lineage. We can re-derive the same techniques from the public sources above, but we cannot copy or paraphrase from `mi300-opt`'s tree.
   - **Exception**: the three files we wrote with our own copyright header on `mi300-opt` — `kernels_aotriton_attn.hip.h`, `kernels_fla_gdn.hip.h`, `kernels_fused_moe.hip.h`. These are wholly Sergey's (`Copyright (c) 2026 Sergey Subbotin`) and may be reused, refactored, and relicensed MIT in the new PR.

3. **Provenance audit policy** for each PR commit:
   - Commit body must list every external reference consulted.
   - Inline code comments must cite specific upstream files (vLLM `model_executor/layers/fused_moe/fused_moe.py`, etc.) where a non-obvious algorithm was learned from.
   - No mention of "flash-moe" or "Dan Woods" anywhere in the PR — the PR exists on its own merits.
   - Companion repo's README cites prior art (ktransformers, tinyserve, vLLM RFC, llama.cpp's existing CPU offload `--n-cpu-moe`) but does not reference flash-moe.

4. **Self-check before each commit**: open `mi300-opt/rocm_infer/{infer.hip,kernels.hip.h}` is **not allowed** while writing PR code. If a piece of math needs to be re-derived, derive it from the Qwen paper or the HF transformers reference (`transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py` — Apache 2.0).

---

## File Structure

### A. llama.cpp PR branch — `feature/moe-expert-gpu-cache`

```
ggml/include/
  ggml-backend.h                      # Modify: add ggml_moe_cache_* API surface
ggml/src/
  ggml-moe-cache.cpp                  # Create: portable LRU + index + miss source dispatch
  ggml-moe-cache.h                    # Create: opaque struct + C API
  ggml-cuda/
    ggml-cuda.cu                      # Modify: register cache hook in CUDA backend init
    moe-cache.cu                      # Create: GPU pool alloc, async H2D copy, slot pointer
    moe-cache.cuh                     # Create: header for above
    mmid.cu                           # Modify: route expert weight ptrs through cache
    fused-moe-amd.cu                  # Create: gfx942-specific fused gate+up+SwiGLU+down
    fused-moe-amd.cuh                 # Create: header for above
  ggml-hip/
    CMakeLists.txt                    # Modify: include the new fused-moe-amd.cu when AMD
src/
  llama-model-loader.cpp              # Modify: optional `--moe-ssd-cache` parsing
  llama-cparams.cpp                   # Modify: store cache config on context params
  llama-context.cpp                   # Modify: instantiate cache after backend ready
common/
  arg.cpp                             # Modify: add CLI flags for cache size + SSD path
  common.h                            # Modify: struct fields for cache config
tests/
  test-moe-cache.cpp                  # Create: unit tests for LRU + miss source
  test-moe-cache.cmake                # Create: ctest registration
docs/
  ops.md                              # Modify: add MOE_CACHE op note (if needed)
  multimodal/                         # untouched
  ...
README.md                             # Modify: short paragraph in "Features"
```

### B. Companion repo — `flash-moe-rocm-demo` (MIT, new repo on github.com/ssubbotin)

```
LICENSE                               # MIT, dated 2026, copyright Sergey Subbotin
README.md                             # What/why/how, perf numbers, video link
.gitignore                            # standard Python
pyproject.toml                        # uv / pip metadata
src/agent_demo/
  __init__.py
  __main__.py                         # CLI entry: `python -m agent_demo --task TASK`
  client.py                           # OpenAI SDK wrapper around local llama-server
  tools.py                            # Tool implementations: read_file, run_shell, http_get
  agent.py                            # 1-step + multi-step loop with tool dispatch
  prompts.py                          # System prompts for each agent profile
scripts/
  build_llamacpp.sh                   # Clones llama.cpp + checks out our PR branch + builds for ROCm
  run_server.sh                       # Starts llama-server with --moe-ssd-cache
  bench.sh                            # tok/s + cache-hit-rate measurements
  record_demo.sh                      # asciinema/ffmpeg recording helper
docs/
  architecture.md                     # Cache architecture, miss path, gfx942 kernel
  benchmarks.md                       # Numbers: baseline llama.cpp vs llama.cpp+our-PR
  prior_art.md                        # Citations: vLLM RFC, ktransformers, tinyserve
tests/
  test_client.py                      # Smoke test against a running server
```

---

## Phase 0 — environment + baseline

### Task 0.1: Provision a fresh worktree for llama.cpp work

**Files:**
- Create: `~/llamacpp-moe-cache/` (worktree directory outside flash-moe repo)

- [ ] **Step 1: Clone llama.cpp upstream into a sibling directory**

```bash
cd ~ && git clone --depth=200 https://github.com/ggml-org/llama.cpp.git llamacpp-moe-cache
cd ~/llamacpp-moe-cache && git log --oneline -1
```

Expected: a recent commit hash from `master`.

- [ ] **Step 2: Create the working branch**

```bash
git -C ~/llamacpp-moe-cache checkout -b feature/moe-expert-gpu-cache
git -C ~/llamacpp-moe-cache branch --show-current
```

Expected: `feature/moe-expert-gpu-cache`.

- [ ] **Step 3: Stop reading flash-moe code**

```bash
echo "Closing all editor tabs that show /home/sergey/flash-moe/* paths."
# Manual checkpoint, no command.
```

This is the license firewall. From this step on, the only flash-moe files we may open are the three named in section "License & provenance — read before writing any code", point 2.

### Task 0.2: Build llama.cpp ROCm baseline on mi300

**Files:**
- Modify: `~/llamacpp-moe-cache/CMakePresets.json` (if needed for gfx942)

- [ ] **Step 1: Sync llama.cpp source to mi300**

```bash
rsync -az --delete --exclude=.git --exclude=build ~/llamacpp-moe-cache/ mi300:~/llamacpp-moe-cache/
ssh mi300 "ls ~/llamacpp-moe-cache | head"
```

Expected: standard llama.cpp tree.

- [ ] **Step 2: Build with ROCm (gfx942) on mi300**

```bash
ssh mi300 "cd ~/llamacpp-moe-cache && cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx942 -DCMAKE_BUILD_TYPE=Release && cmake --build build -j --target llama-server llama-bench 2>&1 | tail -10"
```

Expected: `llama-server` and `llama-bench` binaries under `build/bin/`.

- [ ] **Step 3: Quick smoke test with a tiny GGUF** (download `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` if not present)

```bash
ssh mi300 "cd ~/llamacpp-moe-cache && [ -f models/qwen2.5-0.5b-instruct-q4_k_m.gguf ] || (mkdir -p models && wget -qO models/qwen2.5-0.5b-instruct-q4_k_m.gguf 'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf')"
ssh mi300 "cd ~/llamacpp-moe-cache && build/bin/llama-bench -m models/qwen2.5-0.5b-instruct-q4_k_m.gguf -ngl 99 -t 4 -p 128 -n 64"
```

Expected: a tok/s row from `llama-bench`. This proves ROCm path works.

### Task 0.3: Pick the MoE model for benchmarking + download GGUF

**Files:**
- Create: `scripts/get_model.sh` (in companion repo, deferred to Phase 5)

- [ ] **Step 1: Decide model**

Use **Qwen3-30B-A3B** (Q4_K_M GGUF, ~17 GB) as the primary benchmark — small enough to fit MI300X VRAM entirely (so cache effect is real) without demanding SSD streaming yet, and llama.cpp upstream supports it natively.

For the SSD streaming demo we'll target **Qwen3-235B-A22B** (Q4_K_M, ~140 GB) — fits in MI300X VRAM but wouldn't fit on consumer hardware where SSD streaming matters.

- [ ] **Step 2: Download Qwen3-30B-A3B Q4_K_M**

```bash
ssh mi300 "cd ~/llamacpp-moe-cache/models && [ -f qwen3-30b-a3b-q4_k_m.gguf ] || huggingface-cli download Qwen/Qwen3-30B-A3B-GGUF qwen3-30b-a3b-q4_k_m.gguf --local-dir . --local-dir-use-symlinks False"
```

- [ ] **Step 3: Baseline llama-bench on Qwen3-30B-A3B**

```bash
ssh mi300 "cd ~/llamacpp-moe-cache && build/bin/llama-bench -m models/qwen3-30b-a3b-q4_k_m.gguf -ngl 99 -t 8 -p 512 -n 128 2>&1 | tee baseline-30b.log"
```

Expected: a tok/s row. Save it — this is the "stock llama.cpp on MI300X" number we'll compare against.

- [ ] **Step 4: Commit**

```bash
git -C ~/llamacpp-moe-cache add -A
git -C ~/llamacpp-moe-cache commit -m "wip: baseline numbers on MI300X (gfx942), Qwen3-30B-A3B Q4_K_M"
# This commit lives only on our local branch; we'll never push the baseline commit.
```

---

## Phase 1 — portable MoE expert cache infrastructure

### Task 1.1: Define the C API in `ggml-moe-cache.h`

**Files:**
- Create: `ggml/src/ggml-moe-cache.h`

- [ ] **Step 1: Write the header**

```c
// ggml/src/ggml-moe-cache.h — portable LRU cache for MoE expert weights.
// References (consulted, MIT/Apache):
//   - vLLM RFC #38256 ExpertWeightProvider abstraction (Apache 2.0)
//   - tinyserve providers/ (MIT, e1n00r)
//   - llama.cpp PR #11397 tensor buffer override (MIT)
#pragma once
#include "ggml.h"
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

struct ggml_moe_cache;

// Source of expert weights on a cache miss. The cache copies `bytes`
// bytes for expert (layer_id, expert_id) into `dst` (already on the
// device). Returns 0 on success, non-zero on error.
typedef int (*ggml_moe_miss_fn)(
    void *user_data,
    int   layer_id,
    int   expert_id,
    void *dst,
    size_t bytes);

struct ggml_moe_cache_params {
    int       num_layers;
    int       num_experts;
    size_t    expert_bytes;          // size of one expert's weight blob in bytes
    size_t    cache_bytes;           // total VRAM budget for the cache pool
    ggml_moe_miss_fn miss_fn;
    void *    miss_user_data;
    int       prefetch_lookahead;    // 0 = disabled; otherwise number of layers
                                     // to prefetch ahead in async stream
};

struct ggml_moe_cache *
ggml_moe_cache_init(struct ggml_backend *backend,
                    const struct ggml_moe_cache_params *params);
void  ggml_moe_cache_free(struct ggml_moe_cache *c);

// Returns a device pointer to the expert's weights, populating the cache
// from the miss source if needed. The pointer remains valid until the
// next call that may evict it.
void *ggml_moe_cache_get(struct ggml_moe_cache *c, int layer_id, int expert_id);

// Cheap stats for telemetry.
struct ggml_moe_cache_stats {
    uint64_t hits;
    uint64_t misses;
    uint64_t evictions;
    uint64_t bytes_in_use;
};
struct ggml_moe_cache_stats ggml_moe_cache_stats(const struct ggml_moe_cache *c);

#ifdef __cplusplus
}
#endif
```

- [ ] **Step 2: Add include into `ggml/CMakeLists.txt`** so the header is exported

Find the `set(GGML_PUBLIC_HEADERS …)` block and append `ggml-moe-cache.h`. Verify with:

```bash
grep -n GGML_PUBLIC_HEADERS ~/llamacpp-moe-cache/ggml/CMakeLists.txt | head -3
```

- [ ] **Step 3: Compile-only check**

```bash
ssh mi300 "cd ~/llamacpp-moe-cache && cmake --build build -j --target ggml 2>&1 | tail -5"
```

Expected: `ggml` target builds clean (header alone shouldn't cause errors).

- [ ] **Step 4: Commit**

```bash
git add ggml/src/ggml-moe-cache.h ggml/CMakeLists.txt
git commit -m "ggml: add public header for portable MoE expert cache (no impl yet)"
```

### Task 1.2: Implement the LRU + miss-dispatch core in `ggml-moe-cache.cpp`

**Files:**
- Create: `ggml/src/ggml-moe-cache.cpp`

- [ ] **Step 1: Write the implementation**

```cpp
// ggml/src/ggml-moe-cache.cpp — LRU MoE expert cache (backend-agnostic core).
// The actual device pool lives in the backend's memory; this file owns only
// the bookkeeping + miss dispatch. Thread-safety: caller (graph compute
// thread) is the only writer; not designed for concurrent ggml_moe_cache_get.
#include "ggml-moe-cache.h"
#include "ggml-backend.h"
#include <unordered_map>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cstdint>

namespace {
struct Slot {
    int      layer_id   = -1;
    int      expert_id  = -1;
    uint64_t last_used  = 0;
    uint32_t access_cnt = 0;
};

inline uint64_t key(int layer, int expert) {
    return ((uint64_t)layer << 32) | (uint32_t)expert;
}
} // namespace

struct ggml_moe_cache {
    ggml_backend_t        backend;
    ggml_moe_cache_params params;
    void *                pool = nullptr;
    size_t                pool_bytes = 0;
    int                   capacity = 0;
    std::vector<Slot>     slots;
    std::unordered_map<uint64_t, int> index; // (layer,expert) -> slot
    uint64_t              clock = 0;
    ggml_moe_cache_stats  stats {};
};

extern "C" {

struct ggml_moe_cache *
ggml_moe_cache_init(struct ggml_backend *backend,
                    const struct ggml_moe_cache_params *p) {
    if (!backend || !p || !p->miss_fn || p->expert_bytes == 0) return nullptr;
    auto *c = new ggml_moe_cache;
    c->backend = backend;
    c->params  = *p;
    c->capacity = (int)(p->cache_bytes / p->expert_bytes);
    if (c->capacity > p->num_layers * p->num_experts) {
        c->capacity = p->num_layers * p->num_experts;
    }
    c->pool_bytes = (size_t)c->capacity * p->expert_bytes;
    c->pool = ggml_backend_buft_alloc(ggml_backend_get_default_buffer_type(backend),
                                      c->pool_bytes, "moe_cache_pool");
    if (!c->pool) { delete c; return nullptr; }
    c->slots.assign(c->capacity, Slot{});
    c->index.reserve(c->capacity * 2);
    return c;
}

void ggml_moe_cache_free(struct ggml_moe_cache *c) {
    if (!c) return;
    if (c->pool) ggml_backend_buft_free(ggml_backend_get_default_buffer_type(c->backend),
                                        c->pool, c->pool_bytes);
    delete c;
}

void *ggml_moe_cache_get(struct ggml_moe_cache *c, int layer_id, int expert_id) {
    c->clock++;
    uint64_t k = key(layer_id, expert_id);
    auto it = c->index.find(k);
    if (it != c->index.end()) {
        int s = it->second;
        c->slots[s].last_used  = c->clock;
        c->slots[s].access_cnt++;
        c->stats.hits++;
        return (char *)c->pool + (size_t)s * c->params.expert_bytes;
    }
    // Miss — pick a slot. Frequency-weighted LRU.
    int free_slot = -1;
    if ((int)c->index.size() < c->capacity) {
        free_slot = (int)c->index.size();
    } else {
        uint64_t worst_score = UINT64_MAX;
        for (int s = 0; s < c->capacity; ++s) {
            uint64_t score = (uint64_t)c->slots[s].access_cnt * 10ULL +
                             c->slots[s].last_used;
            if (score < worst_score) { worst_score = score; free_slot = s; }
        }
        // Drop old entry from index.
        c->index.erase(key(c->slots[free_slot].layer_id,
                           c->slots[free_slot].expert_id));
        c->stats.evictions++;
    }
    void *dst = (char *)c->pool + (size_t)free_slot * c->params.expert_bytes;
    int err = c->params.miss_fn(c->params.miss_user_data, layer_id, expert_id,
                                 dst, c->params.expert_bytes);
    if (err) {
        fprintf(stderr, "[moe-cache] miss_fn failed for L%d E%d (err=%d)\n",
                layer_id, expert_id, err);
        return nullptr;
    }
    c->slots[free_slot] = {layer_id, expert_id, c->clock, 1};
    c->index[k] = free_slot;
    c->stats.misses++;
    c->stats.bytes_in_use = (size_t)c->index.size() * c->params.expert_bytes;
    return dst;
}

struct ggml_moe_cache_stats
ggml_moe_cache_stats(const struct ggml_moe_cache *c) {
    return c ? c->stats : ggml_moe_cache_stats{};
}

} // extern "C"
```

- [ ] **Step 2: Wire the source file into `ggml/CMakeLists.txt`**

Find the `add_library(ggml …)` source list and append `src/ggml-moe-cache.cpp`. Compile:

```bash
ssh mi300 "cd ~/llamacpp-moe-cache && cmake --build build -j --target ggml 2>&1 | tail -10"
```

Expected: builds clean. (Note `ggml_backend_buft_alloc` is the existing portable alloc helper in `ggml-backend.h`; verify its signature with `grep -n 'ggml_backend_buft_alloc' ggml/include/ggml-backend.h` before committing in case API name changed.)

- [ ] **Step 3: Commit**

```bash
git add ggml/src/ggml-moe-cache.cpp ggml/CMakeLists.txt
git commit -m "ggml: portable LRU MoE expert cache core (bookkeeping only, no kernel hook)"
```

### Task 1.3: Unit-test the LRU policy with a fake backend + fake miss source

**Files:**
- Create: `tests/test-moe-cache.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Write the failing test**

```cpp
// tests/test-moe-cache.cpp
// Tests the cache bookkeeping without any GPU. Uses ggml's CPU backend.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-moe-cache.h"
#include <cassert>
#include <cstring>
#include <vector>
#include <cstdio>

static int g_miss_calls = 0;
static int counting_miss(void *ud, int l, int e, void *dst, size_t n) {
    (void)ud; (void)l; (void)e;
    g_miss_calls++;
    memset(dst, l * 7 + e, n);
    return 0;
}

int main() {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    assert(cpu);
    ggml_moe_cache_params p {
        /*num_layers*/ 4, /*num_experts*/ 8, /*expert_bytes*/ 64,
        /*cache_bytes*/ 64 * 16,  // capacity = 16 of 32 experts
        counting_miss, nullptr, 0
    };
    auto *c = ggml_moe_cache_init(cpu, &p);
    assert(c);

    // First touch: 4 layers × 4 experts each = 16 distinct → all misses.
    for (int l = 0; l < 4; ++l)
        for (int e = 0; e < 4; ++e)
            assert(ggml_moe_cache_get(c, l, e));
    auto s = ggml_moe_cache_stats(c);
    assert(s.misses == 16); assert(s.hits == 0); assert(s.evictions == 0);
    assert(g_miss_calls == 16);

    // Re-touch the first 4 → all hits.
    for (int e = 0; e < 4; ++e) ggml_moe_cache_get(c, 0, e);
    s = ggml_moe_cache_stats(c);
    assert(s.hits == 4); assert(s.evictions == 0);

    // Force evictions: insert 16 more unique entries → 16 evictions.
    for (int l = 0; l < 4; ++l)
        for (int e = 4; e < 8; ++e) ggml_moe_cache_get(c, l, e);
    s = ggml_moe_cache_stats(c);
    assert(s.evictions == 16);

    ggml_moe_cache_free(c);
    ggml_backend_free(cpu);
    printf("test-moe-cache OK (hits=%llu misses=%llu evictions=%llu)\n",
           (unsigned long long)s.hits, (unsigned long long)s.misses,
           (unsigned long long)s.evictions);
    return 0;
}
```

- [ ] **Step 2: Register the test in `tests/CMakeLists.txt`**

```cmake
llama_target_and_test(test-moe-cache.cpp)
```

- [ ] **Step 3: Build and run the test**

```bash
ssh mi300 "cd ~/llamacpp-moe-cache && cmake --build build -j --target test-moe-cache && ctest --test-dir build -R test-moe-cache --output-on-failure"
```

Expected: `test-moe-cache OK …` and ctest PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test-moe-cache.cpp tests/CMakeLists.txt
git commit -m "tests: LRU policy + miss/hit/eviction accounting for ggml-moe-cache"
```

---

## Phase 2 — MoE streaming buffer type (skeleton)

The buffer type is the integration surface. Tensors allocated through it carry MoE expert metadata and a per-layer file descriptor; they don't have device-resident data of their own. Reads from these tensors, when they happen via the kernel dispatch in Phase 4, route through the cache+stream path. Phase 2 lands the buffer-type machinery with a no-op miss handler so the build is clean and the model loader can be exercised; Phase 3 wires the real 3-tier source.

### Task 2.1: Reread the recon (no code, build shared understanding)

The 2026-05-04 recon found:
- `ggml/src/ggml-cuda/common.cuh:1365` — `struct ggml_backend_cuda_context` (room for new fields, but no public reach-in from `llama-context.cpp`).
- `ggml/src/ggml-cuda/ggml-cuda.cu:2470` — `ggml_cuda_mul_mat_id`. Two paths: fast (decode batch=1, takes one slab + ids, lines 2484-2509) and slow (per-expert loop, line 2585+). The fast path is what we need to claim.
- `ggml/src/ggml-cuda/{mmvq,mmq,mmvf,mmf}.{cu,cuh}` — kernel-level launches that consume `src0->data` as a single base. Untouched in Option D.
- `common/arg.cpp:2308-2321` — `--n-cpu-moe` is a `tensor_buft_overrides` push (regex match on `blk.<i>.ffn_*_exps`, route to `ggml_backend_cpu_buffer_type()`). Our `--moe-stream-dir` mirrors this exactly: same regex, route to `ggml_backend_cuda_buffer_type_moe_stream(dev, dir)`.

No commit for this task — it's a controller checkpoint that the implementer has the recon report in hand before writing buffer-type code.

### Task 2.2: Define the buffer-type C API

**Files:**
- Create: `ggml/src/ggml-cuda/buft-moe-stream.cuh`
- Create: `ggml/src/ggml-cuda/buft-moe-stream.cu`

- [ ] **Step 1: Header**

```cpp
// ggml/src/ggml-cuda/buft-moe-stream.cuh
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Sergey Subbotin
#pragma once
#include "common.cuh"
#include "ggml-moe-cache.h"

// Per-tensor metadata held by buffers of this buffer type.
struct ggml_cuda_moe_stream_meta {
    int          layer_id;       // parsed from tensor name `blk.<L>.ffn_*_exps.weight`
    int          fd;             // open() of <ssd_dir>/layer_<NN>.bin (one fd per layer, refcounted by buffer)
    size_t       expert_bytes;   // bytes per expert blob in the file
    int          num_experts;
};

// Public API: returns a singleton-per-(device, ssd_dir) buffer type.
// `ssd_dir` may be empty — in that case Phase 3's miss handler will fall
// back to the GGUF mmap source.
ggml_backend_buffer_type_t
ggml_backend_cuda_buffer_type_moe_stream(int device, const char *ssd_dir);

// Returns the per-tensor metadata if `tensor` lives in a moe_stream buffer,
// otherwise nullptr. Used by the dispatch hook in Phase 4 to recognize
// "this MUL_MAT_ID is mine".
const ggml_cuda_moe_stream_meta *
ggml_cuda_moe_stream_get_meta(const ggml_tensor *tensor);

// Get the cache instance attached to a moe_stream buffer (Phase 1 LRU).
// Phase 4 dispatch calls _get on it for each active expert.
struct ggml_moe_cache *
ggml_cuda_moe_stream_get_cache(ggml_backend_buffer_t buf);
```

- [ ] **Step 2: Implementation skeleton (no-op miss handler)**

```cpp
// ggml/src/ggml-cuda/buft-moe-stream.cu
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Sergey Subbotin
//
// Custom CUDA/HIP buffer type for streamed MoE expert weights.
// References consulted: vLLM RFC #38256 (Apache 2.0), llama.cpp PR #11397
// tensor_buft_overrides (MIT), AITER (MIT). Phase 1 LRU is reused as the
// VRAM tier of the 3-tier cache here.
#include "buft-moe-stream.cuh"
#include "ggml-impl.h"
#include <cstring>
#include <cstdio>
#include <fcntl.h>
#include <unistd.h>
#include <map>
#include <mutex>
#include <string>

namespace {

struct buft_context {
    int         device;
    std::string ssd_dir;
    // Per-(device, ssd_dir) singleton — the cache itself attaches to the
    // first buffer allocated; subsequent buffers share it via the singleton.
    ggml_moe_cache *cache = nullptr;
    size_t          cache_pool_bytes = 0;
};

struct buf_context {
    buft_context *bt;
    // For each tensor allocated in this buffer, we hold its metadata so
    // ggml_cuda_moe_stream_get_meta can recover it from a tensor pointer.
    std::map<const ggml_tensor *, ggml_cuda_moe_stream_meta> tensors;
    void *base = nullptr;     // pseudo-base (we hand out unique offsets here)
    size_t size = 0;
    size_t cursor = 0;        // bump allocator within `base`
};

// Phase 2 miss handler: stub. Phase 3 replaces this with the 3-tier source.
int stub_miss(void *user_data, int layer_id, int expert_id, void *dst, size_t bytes) {
    (void)user_data; (void)layer_id; (void)expert_id; (void)dst; (void)bytes;
    GGML_LOG_ERROR("ggml-cuda moe-stream: Phase 2 stub miss handler hit (L=%d E=%d) — Phase 3 not landed yet\n",
                   layer_id, expert_id);
    return 1;
}

} // namespace

// ---------------- buffer iface ----------------

static const char * gfx_buf_name(ggml_backend_buffer_t buf) {
    return "CUDA_MOE_STREAM";
}

static void gfx_buf_free(ggml_backend_buffer_t buf) {
    auto *ctx = (buf_context *)buf->context;
    delete ctx;
}

static void * gfx_buf_get_base(ggml_backend_buffer_t buf) {
    auto *ctx = (buf_context *)buf->context;
    return ctx->base;
}

static void gfx_buf_init_tensor(ggml_backend_buffer_t buf, ggml_tensor *tensor) {
    auto *ctx = (buf_context *)buf->context;
    // Parse layer index from tensor name `blk.<L>.ffn_*_exps.weight`.
    int layer = -1;
    int n = sscanf(tensor->name, "blk.%d.", &layer);
    if (n != 1 || layer < 0) {
        // Not a MoE expert tensor — shouldn't be allocated here, but be defensive.
        return;
    }
    ggml_cuda_moe_stream_meta meta {};
    meta.layer_id    = layer;
    meta.num_experts = (int)tensor->ne[2];
    meta.expert_bytes = ggml_nbytes(tensor) / meta.num_experts;
    meta.fd = -1;  // Phase 3 will open <ssd_dir>/layer_<NN>.bin and dup its fd here
    ctx->tensors[tensor] = meta;
    // Pseudo-pointer — the dispatch in Phase 4 only ever uses tensor->data
    // as a key to recover meta, never dereferences it on host.
    tensor->data = (char *)ctx->base + ctx->cursor;
    ctx->cursor += ggml_nbytes(tensor);
}

static void gfx_buf_set_tensor(ggml_backend_buffer_t buf, ggml_tensor *tensor,
                               const void *data, size_t offset, size_t size) {
    // Phase 2 stub — Phase 3 writes the SSD file here on first set.
    auto *ctx = (buf_context *)buf->context;
    auto it = ctx->tensors.find(tensor);
    if (it == ctx->tensors.end()) return;
    GGML_LOG_DEBUG("moe-stream set_tensor: tensor=%s offset=%zu size=%zu (Phase 2 stub, ignored)\n",
                   tensor->name, offset, size);
    // We intentionally do not copy to VRAM here — the data lives on disk.
    // Phase 3 streams it back in on cache miss.
}

static void gfx_buf_get_tensor(ggml_backend_buffer_t buf, const ggml_tensor *tensor,
                               void *data, size_t offset, size_t size) {
    GGML_LOG_ERROR("moe-stream get_tensor not supported (Phase 2)\n");
}

static bool gfx_buf_cpy_tensor(ggml_backend_buffer_t buf, const ggml_tensor *src,
                               ggml_tensor *dst) {
    return false;  // never participate in tensor copies
}

static void gfx_buf_clear(ggml_backend_buffer_t buf, uint8_t value) {
    // No-op — we don't own real device memory directly (the LRU pool does).
}

static const ggml_backend_buffer_i moe_stream_buf_iface = {
    /* .free_buffer  */ gfx_buf_free,
    /* .get_base     */ gfx_buf_get_base,
    /* .init_tensor  */ gfx_buf_init_tensor,
    /* .memset_tensor*/ nullptr,
    /* .set_tensor   */ gfx_buf_set_tensor,
    /* .get_tensor   */ gfx_buf_get_tensor,
    /* .cpy_tensor   */ gfx_buf_cpy_tensor,
    /* .clear        */ gfx_buf_clear,
    /* .reset        */ nullptr,
};

// ---------------- buffer-type iface ----------------

static const char * gfx_buft_name(ggml_backend_buffer_type_t buft) {
    return "CUDA_MOE_STREAM";
}

static ggml_backend_buffer_t
gfx_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    auto *bt = (buft_context *)buft->context;
    auto *bc = new buf_context;
    bc->bt   = bt;
    bc->size = size;
    // Reserve a pseudo-address space — we hand out distinct offsets for
    // each init_tensor call. Real device memory is held by the LRU pool
    // (bt->cache), allocated lazily on first cache_init.
    static uintptr_t next_pseudo = 0x1000;
    bc->base = (void *)(next_pseudo);
    next_pseudo += size + 0x1000;
    bc->cursor = 0;
    return ggml_backend_buffer_init(buft, moe_stream_buf_iface, bc, size);
}

static size_t gfx_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    return 256;
}

static size_t gfx_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    return SIZE_MAX;
}

static bool gfx_buft_is_host(ggml_backend_buffer_type_t buft) {
    return false;
}

static const ggml_backend_buffer_type_i moe_stream_buft_iface = {
    /* .get_name        */ gfx_buft_name,
    /* .alloc_buffer    */ gfx_buft_alloc_buffer,
    /* .get_alignment   */ gfx_buft_get_alignment,
    /* .get_max_size    */ gfx_buft_get_max_size,
    /* .get_alloc_size  */ nullptr,
    /* .is_host         */ gfx_buft_is_host,
};

ggml_backend_buffer_type_t
ggml_backend_cuda_buffer_type_moe_stream(int device, const char *ssd_dir) {
    static std::mutex                                                         mu;
    static std::map<std::pair<int, std::string>, ggml_backend_buffer_type *>  cache;
    std::lock_guard<std::mutex> lk(mu);
    auto key = std::make_pair(device, std::string(ssd_dir ? ssd_dir : ""));
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    auto *bt = new buft_context;
    bt->device  = device;
    bt->ssd_dir = key.second;
    auto *buft = new ggml_backend_buffer_type{
        /* .iface   */ moe_stream_buft_iface,
        /* .device  */ ggml_backend_cuda_reg_get_device(device),
        /* .context */ bt,
    };
    cache[key] = buft;
    return buft;
}

const ggml_cuda_moe_stream_meta *
ggml_cuda_moe_stream_get_meta(const ggml_tensor *tensor) {
    if (!tensor || !tensor->buffer) return nullptr;
    if (tensor->buffer->iface.get_name != gfx_buf_name) return nullptr;
    auto *ctx = (buf_context *)tensor->buffer->context;
    auto it = ctx->tensors.find(tensor);
    return (it == ctx->tensors.end()) ? nullptr : &it->second;
}

struct ggml_moe_cache *
ggml_cuda_moe_stream_get_cache(ggml_backend_buffer_t buf) {
    auto *ctx = (buf_context *)buf->context;
    return ctx->bt->cache;  // may be nullptr until Phase 3 attaches it
}
```

- [ ] **Step 3: Wire into the CUDA build (CMake or source list)**

If `ggml/src/ggml-cuda/CMakeLists.txt` globs `*.cu`, the new files are picked up automatically. If it has an explicit list, append both. Verify with:

```bash
grep -n 'buft-moe-stream\|*.cu' ~/llamacpp-moe-cache/ggml/src/ggml-cuda/CMakeLists.txt
```

- [ ] **Step 4: Build on mi300**

```bash
rsync -az --exclude=.git --exclude=build --exclude=build-local --exclude=models ~/llamacpp-moe-cache/ mi300:~/llamacpp-moe-cache/
ssh mi300 "cd ~/llamacpp-moe-cache && cmake --build build -j --target ggml-hip 2>&1 | tail -10"
```

Expected: clean build. (Target is `ggml-hip` when `GGML_HIP=ON`; if cmake renamed it, the `tail -10` will show `make: *** No rule to make target` — try `ggml` or `ggml-cuda`.)

- [ ] **Step 5: Commit**

```bash
git add ggml/src/ggml-cuda/buft-moe-stream.{cu,cuh} ggml/src/ggml-cuda/CMakeLists.txt
git commit -m "ggml-cuda: skeleton MoE streaming buffer type (no-op miss, parses layer-id from tensor name)"
```

### Task 2.3: `--moe-stream-dir` and `--moe-cache-mb` CLI flags

**Files:**
- Modify: `common/arg.cpp`
- Modify: `common/common.h`
- Modify: `common/common.cpp` (the regex push that mirrors `--n-cpu-moe`)

- [ ] **Step 1: Add `moe_stream_dir` (string) and `moe_cache_mb` (int) to `common_params`**

```cpp
// common/common.h, inside struct common_params (place near n_cpu_moe)
std::string moe_stream_dir;   // empty = disabled
int         moe_cache_mb = 0; // 0 = disabled
```

- [ ] **Step 2: Add the args**

```cpp
// common/arg.cpp — add after the n_cpu_moe block
add_opt(common_arg(
    {"--moe-stream-dir"}, "DIR",
    "stream MoE expert weights from per-layer files in DIR via the CUDA moe-stream buffer type",
    [](common_params & params, const std::string & v) { params.moe_stream_dir = v; }
).set_env("LLAMA_ARG_MOE_STREAM_DIR"));

add_opt(common_arg(
    {"--moe-cache-mb"}, "MB",
    "MoE expert VRAM cache size in MiB (0 = disabled). Requires --moe-stream-dir.",
    [](common_params & params, int v) { params.moe_cache_mb = v; }
).set_env("LLAMA_ARG_MOE_CACHE_MB"));
```

- [ ] **Step 3: When `moe_stream_dir` is non-empty, push a `tensor_buft_overrides` entry**

In `common/common.cpp`, find where `n_cpu_moe` pushes its overrides. Add a parallel block:

```cpp
// common/common.cpp — near the n_cpu_moe override push
if (!params.moe_stream_dir.empty()) {
    int dev = 0;  // Phase 2 wires device 0; multi-GPU lands later
    auto buft = ggml_backend_cuda_buffer_type_moe_stream(dev, params.moe_stream_dir.c_str());
    if (!buft) {
        throw std::runtime_error("--moe-stream-dir requires CUDA/HIP backend");
    }
    common_params_handle_buft_override::add_layer_regex_override(
        params.tensor_buft_overrides, "blk\\.\\d+\\.ffn_(gate|up|down)_exps\\.weight", buft);
}
```

(The exact helper name varies by upstream commit — `grep -n 'tensor_buft_overrides' common/common.cpp` will find the existing pattern from `--n-cpu-moe` to mirror.)

- [ ] **Step 4: Smoke build + run with the flags (no model touch yet — flag parsing only)**

```bash
ssh mi300 "cd ~/llamacpp-moe-cache && cmake --build build -j --target llama-server llama-bench 2>&1 | tail -5"
ssh mi300 "cd ~/llamacpp-moe-cache && build/bin/llama-server --help 2>&1 | grep -E 'moe-(stream-dir|cache-mb)'"
```

Expected: both flags appear in `--help`. Smoke run with the flags but no model file yet would crash; that's fine — the flag is parsed.

- [ ] **Step 5: Commit**

```bash
git add common/arg.cpp common/common.h common/common.cpp
git commit -m "common: --moe-stream-dir and --moe-cache-mb flags, route MoE expert tensors through the streaming buffer type"
```

### Task 2.4: Test that model load with `--moe-stream-dir` puts experts in our buffer type

**Files:**
- Modify: `tests/test-moe-cache.cpp` to also exercise the buffer type's tensor-init path

Phase 2 has no real cache lookup (stub miss), so we can't run inference. We can still verify the loader pushes experts into our buffer type by checking `ggml_cuda_moe_stream_get_meta(tensor) != nullptr` for an `ffn_gate_exps` tensor after model load.

- [ ] **Step 1: Add a unit test that loads a tiny MoE GGUF with the override**

```cpp
// Append to tests/test-moe-cache.cpp under a `#ifdef GGML_USE_CUDA` (or HIP) guard.
// Uses Qwen2-0.5B-MoE or similar tiny MoE; if no tiny MoE GGUF is present,
// skip the test. The test asserts that for an "ffn_gate_exps" tensor in the
// loaded model, ggml_cuda_moe_stream_get_meta returns non-null with the
// right num_experts and expert_bytes parsed from the tensor.
//
// (Implementer: pick the smallest MoE GGUF available locally; if none,
// skip the test with a `printf("[skip] no MoE GGUF available\n"); return 0;`.)
```

If tracing this through the model loader is tangled, ship without this unit test and rely on the Phase 4 end-to-end test instead.

- [ ] **Step 2: Commit (only if test added)**

```bash
git add tests/test-moe-cache.cpp
git commit -m "tests: assert MoE expert tensors land in moe_stream buffer type when --moe-stream-dir is set"
```

---

## Phase 3 — 3-tier expert source (LRU + page cache + SSD pread)

The buffer type is in place; this phase fills in the actual miss handler so the LRU has somewhere to fetch from. The 3 tiers, in order:

1. **VRAM LRU** (Phase 1 cache) — `_get(key)` returns a slot pointer or nullptr.
2. **OS page cache** — first miss into VRAM falls back to a CPU mmap of the SSD pack file. Subsequent reads go through the kernel page cache. flash-moe's "trust the OS" principle.
3. **SSD pread** — only on cold-cold miss when the page hasn't been pulled before, the kernel issues a real disk read.

In our impl all three tiers collapse into a single miss path: `mmap` the per-layer SSD file at first access, then any `pread` (or pointer dereference) hits the page cache after warmup.

### Task 3.1: Document the per-layer SSD layout

**Files:**
- Create: `docs/moe-stream-layout.md`

- [ ] **Step 1: Write the layout doc**

```markdown
# moe-stream per-layer file layout

Each MoE layer's expert weights live in `<ssd_dir>/layer_<NN>.bin`. Within
the file, expert `e` starts at byte offset `e * expert_bytes` and is
exactly `expert_bytes` long. No header, no metadata, no padding.

`expert_bytes` is parsed from the GGUF tensor's `nbytes / ne[2]` at model
load time and stored in `ggml_cuda_moe_stream_meta`. The file's expert
ordering matches the GGUF tensor's expert axis (axis 2).

For Q4_K_M MoE models: `expert_bytes = ne[0] * ne[1] / QK_K * sizeof(block_q4_K)`
≈ 2.4 MB per expert per gate/up/down for typical 4096×1024 shapes.

For Qwen3-235B-A22B Q4_K_M: 94 layers × 128 experts × ~7 MB per expert per
proj × 3 projs ≈ 250 GB on disk. Fits comfortably on a 1 TB SSD.

Files are written by `tools/moe-pack/moe-pack` (Task 3.3). It reads any
GGUF MoE checkpoint and writes one `layer_<NN>.bin` per MoE layer with
the gate/up/down expert blobs concatenated per expert in the order
[gate_w, up_w, down_w]. The order is fixed; if a model uses a different
GGUF tensor ordering, the converter normalizes.
```

- [ ] **Step 2: Commit**

```bash
git add docs/moe-stream-layout.md
git commit -m "docs: per-layer SSD layout for moe-stream buffer type"
```

### Task 3.2: Implement the 3-tier miss handler

**Files:**
- Create: `ggml/src/ggml-cuda/moe-stream-source.cu`
- Create: `ggml/src/ggml-cuda/moe-stream-source.cuh`
- Modify: `ggml/src/ggml-cuda/buft-moe-stream.cu` — replace `stub_miss` with the real handler, allocate the LRU cache lazily, open per-layer fds in `init_tensor`.

- [ ] **Step 1: Header**

```cpp
// ggml/src/ggml-cuda/moe-stream-source.cuh
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Sergey Subbotin
#pragma once
#include "common.cuh"
#include <stddef.h>

struct moe_stream_source {
    int          fd;             // per-layer
    size_t       expert_bytes;
    void *       mmap_base;      // host mmap of the layer file
    size_t       mmap_bytes;
    void *       pinned_staging; // pinned host buffer, expert_bytes
    cudaStream_t stream;
};

// Open <ssd_dir>/layer_<NN>.bin, mmap it read-only, allocate pinned staging.
// Returns 0 on success, fills `*out`. Releases everything via destructor.
int moe_stream_source_open(const char *ssd_dir, int layer_id, size_t expert_bytes,
                           cudaStream_t stream, moe_stream_source *out);
void moe_stream_source_close(moe_stream_source *s);

// Miss handler signature compatible with ggml_moe_miss_fn. user_data is a
// `moe_stream_source *`. Copies expert_id's bytes from page cache /
// pread to `dst` (which is in the LRU pool, device memory).
int moe_stream_miss(void *user_data, int layer_id, int expert_id,
                    void *dst, size_t bytes);
```

- [ ] **Step 2: Implementation**

```cpp
// ggml/src/ggml-cuda/moe-stream-source.cu
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Sergey Subbotin
#include "moe-stream-source.cuh"
#include "ggml-impl.h"
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <cstring>

int moe_stream_source_open(const char *ssd_dir, int layer_id, size_t expert_bytes,
                           cudaStream_t stream, moe_stream_source *out) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/layer_%02d.bin", ssd_dir, layer_id);
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        GGML_LOG_ERROR("moe-stream: cannot open %s: %s\n", path, strerror(errno));
        return 1;
    }
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return 1; }
    void *m = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (m == MAP_FAILED) { close(fd); return 1; }
    out->fd            = fd;
    out->expert_bytes  = expert_bytes;
    out->mmap_base     = m;
    out->mmap_bytes    = st.st_size;
    out->stream        = stream;
    if (cudaMallocHost(&out->pinned_staging, expert_bytes) != cudaSuccess) {
        munmap(m, st.st_size); close(fd);
        return 1;
    }
    return 0;
}

void moe_stream_source_close(moe_stream_source *s) {
    if (!s) return;
    if (s->pinned_staging) cudaFreeHost(s->pinned_staging);
    if (s->mmap_base)      munmap(s->mmap_base, s->mmap_bytes);
    if (s->fd >= 0)        close(s->fd);
    *s = {};
    s->fd = -1;
}

int moe_stream_miss(void *user_data, int layer_id, int expert_id,
                    void *dst, size_t bytes) {
    auto *s = (moe_stream_source *)user_data;
    if (bytes != s->expert_bytes) {
        GGML_LOG_ERROR("moe-stream: miss size mismatch (req=%zu have=%zu)\n",
                       bytes, s->expert_bytes);
        return 1;
    }
    // Tier 2/3 collapsed: read from mmap (page-cache hit if warm; pread on cold).
    const char *src = (const char *)s->mmap_base + (size_t)expert_id * s->expert_bytes;
    memcpy(s->pinned_staging, src, bytes);
    if (cudaMemcpyAsync(dst, s->pinned_staging, bytes,
                        cudaMemcpyHostToDevice, s->stream) != cudaSuccess) {
        return 1;
    }
    return cudaStreamSynchronize(s->stream) == cudaSuccess ? 0 : 1;
}
```

- [ ] **Step 3: Wire into `buft-moe-stream.cu`**

In `init_tensor`, after parsing `layer_id`, open the per-layer source. Stash the `moe_stream_source` in the buffer-type context's `std::map<int, moe_stream_source>` keyed by `layer_id`. On first allocation in a buffer type, also call `ggml_moe_cache_init` with `cache_bytes = bt->cache_pool_bytes` (from `--moe-cache-mb`), `expert_bytes = meta.expert_bytes`, and `miss_fn = moe_stream_miss` with `miss_user_data = &source` — but note: the cache holds ONE miss_fn per cache, so the user_data can't change per-layer. Resolution: lift the `moe_stream_source` lookup into a wrapper miss function that takes `layer_id` and looks up the right source from a global-per-cache map.

```cpp
// In buft-moe-stream.cu near the cache-init site
struct cache_user_data {
    std::map<int, moe_stream_source> *layer_sources;
};
static int dispatching_miss(void *ud, int layer_id, int expert_id,
                            void *dst, size_t bytes) {
    auto *u = (cache_user_data *)ud;
    auto it = u->layer_sources->find(layer_id);
    if (it == u->layer_sources->end()) return 1;
    return moe_stream_miss(&it->second, layer_id, expert_id, dst, bytes);
}
```

- [ ] **Step 4: Build + smoke test (just init, no kernel dispatch yet)**

```bash
rsync -az --exclude=.git --exclude=build --exclude=build-local --exclude=models ~/llamacpp-moe-cache/ mi300:~/llamacpp-moe-cache/
ssh mi300 "cd ~/llamacpp-moe-cache && cmake --build build -j --target llama-server 2>&1 | tail -5"
```

End-to-end model load with the SSD dir set requires Phase 3.3's `moe-pack` to have produced files. Defer the run-test to after Task 3.3.

- [ ] **Step 5: Commit**

```bash
git add ggml/src/ggml-cuda/moe-stream-source.{cu,cuh} ggml/src/ggml-cuda/buft-moe-stream.cu
git commit -m "ggml-cuda: 3-tier MoE expert source (LRU + page cache + SSD pread)"
```

### Task 3.3: `tools/moe-pack` — GGUF → per-layer files

**Files:**
- Create: `tools/moe-pack/moe-pack.cpp`
- Create: `tools/moe-pack/CMakeLists.txt`
- Modify: `tools/CMakeLists.txt`

- [ ] **Step 1: Implement the converter**

```cpp
// tools/moe-pack/moe-pack.cpp
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Sergey Subbotin
//
// Converts a GGUF MoE checkpoint into per-layer files for use with
// llama.cpp --moe-stream-dir. One file per MoE layer, name layer_NN.bin,
// experts laid out contiguously at offset e * expert_bytes.
#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <regex>
#include <string>
#include <vector>
#include <map>

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: moe-pack input.gguf out_dir\n");
        return 2;
    }
    const char *gguf_path = argv[1];
    const char *out_dir   = argv[2];
    mkdir(out_dir, 0755);

    gguf_init_params p { /*no_alloc*/ false, /*ctx*/ nullptr };
    gguf_context *gguf = gguf_init_from_file(gguf_path, p);
    if (!gguf) { fprintf(stderr, "gguf_init failed\n"); return 1; }

    // Group MoE expert tensors by (layer, projection-name).
    std::regex re("blk\\.(\\d+)\\.ffn_(gate|up|down)_exps\\.weight");
    std::map<int, std::map<std::string, int>> by_layer;  // layer -> {proj -> tensor_id}
    int n = gguf_get_n_tensors(gguf);
    for (int i = 0; i < n; ++i) {
        const char *name = gguf_get_tensor_name(gguf, i);
        std::cmatch m;
        if (!std::regex_match(name, m, re)) continue;
        int layer = std::stoi(m[1]);
        by_layer[layer][m[2]] = i;
    }

    // For each layer, write a file containing per-expert blobs in
    // [gate, up, down] order.
    for (auto &kv : by_layer) {
        int layer = kv.first;
        char path[1024];
        snprintf(path, sizeof(path), "%s/layer_%02d.bin", out_dir, layer);
        FILE *f = fopen(path, "wb");
        if (!f) { fprintf(stderr, "fopen %s failed\n", path); return 1; }
        for (const char *proj : {"gate", "up", "down"}) {
            auto it = kv.second.find(proj);
            if (it == kv.second.end()) continue;
            int t = it->second;
            // Use gguf API to read raw bytes for tensor t and write them.
            // (Exact API: gguf_get_tensor_offset + read from file at that
            // offset for size = gguf_get_tensor_size — adapt to the
            // version of gguf the build uses.)
            // For now: implementer fills in based on the actual gguf.h
            // signature in this checkout.
        }
        fclose(f);
        printf("[moe-pack] L%02d: wrote %s\n", layer, path);
    }
    gguf_free(gguf);
    return 0;
}
```

- [ ] **Step 2: CMakeLists**

```cmake
# tools/moe-pack/CMakeLists.txt
add_executable(moe-pack moe-pack.cpp)
target_link_libraries(moe-pack PRIVATE ggml)
```

```cmake
# tools/CMakeLists.txt — append
add_subdirectory(moe-pack)
```

- [ ] **Step 3: Build + run on Qwen3-30B-A3B**

```bash
ssh mi300 "cd ~/llamacpp-moe-cache && cmake --build build -j --target moe-pack && build/bin/moe-pack models/qwen3-30b-a3b-q4_k_m.gguf /mnt/scratch/moe-stream-30b/"
ssh mi300 "ls /mnt/scratch/moe-stream-30b/ | head -5 && du -sh /mnt/scratch/moe-stream-30b/"
```

Expected: 48 layer files, total size matches the expert subset of the GGUF (the non-expert tensors stay in the GGUF, served by the normal mmap path).

- [ ] **Step 4: Commit**

```bash
git add tools/moe-pack/ tools/CMakeLists.txt
git commit -m "tools: moe-pack — convert GGUF MoE expert tensors to per-layer files for --moe-stream-dir"
```

### Task 3.4: End-to-end load test (model loads, no inference yet)

- [ ] **Step 1: Run llama-server with the streaming flags**

```bash
ssh mi300 "cd ~/llamacpp-moe-cache && build/bin/llama-server -m models/qwen3-30b-a3b-q4_k_m.gguf -ngl 99 --moe-stream-dir /mnt/scratch/moe-stream-30b/ --moe-cache-mb 8192 --port 8080 2>&1 | head -40"
```

Expected: model loads, init prints `[moe-stream] tier-1 LRU cache initialized: capacity=NNN experts (8192 MB pool)`. No inference request yet — Phase 4 wires the kernel.

If the server crashes at load (most likely cause: tensor data is needed during graph build but our buffer type's `set_tensor` is a no-op), the right fix is to land the simplest version of Phase 4's dispatch as a placeholder so the graph compiles, then iterate. Report BLOCKED if so.

- [ ] **Step 2: No commit unless something needed adjusting in code.**

---

## Phase 4 — custom MoE forward dispatch (gfx942 fused kernel)

The buffer type and source are in place; this phase makes the CUDA backend recognize MUL_MAT_ID where src0 lives in the moe_stream buffer type and dispatches our fused kernel against the K cached expert pointers.

### Task 4.1: Port the Sergey-authored fused MoE kernel

**Files:**
- Create: `ggml/src/ggml-cuda/fused-moe-amd.cu`
- Create: `ggml/src/ggml-cuda/fused-moe-amd.cuh`

- [ ] **Step 1: Read the source kernel** (license-allowed)

The fused kernel is in `/home/sergey/flash-moe/.worktrees/mi300-opt/rocm_infer/kernels_fused_moe.hip.h`. **You may read this file** — it has a `Sergey Subbotin` copyright header and is one of the three files explicitly approved for reuse. Do NOT open `infer.hip`, `kernels.hip.h`, or any other file in `/home/sergey/flash-moe/`.

Port the `fused_moe_gate_up_swiglu_mlx` and `fused_moe_down_mlx` kernels into the new files, with these adaptations:
- Replace MLX-specific weight layout (4-bit packed nibbles + bf16 scale + bf16 bias per group of 64) with **Q4_K_M block layout** (`block_q4_K`, 144 bytes per 256 elements). Use llama.cpp's existing `dequantize_q4_K` helpers from `ggml/src/ggml-cuda/dequantize.cuh` to drive the inner loop — saves us re-deriving Q4_K dequant.
- Take **K device pointers** (one per active expert) as `const void * const *` argument, not a packed weight blob.
- Take the `ids` tensor pointer + the routing weights as additional inputs.
- Output shape matches the existing MUL_MAT_ID output.
- HIP guards: gate the gfx942-specific tunings behind `defined(__HIP_PLATFORM_AMD__) && defined(__gfx942__)`. Provide a generic CUDA fallback path (just calls llama.cpp's existing per-expert MMVQ in a loop) so `--moe-stream-dir` works on NVIDIA too, just slower.

- [ ] **Step 2: Header**

```cpp
// ggml/src/ggml-cuda/fused-moe-amd.cuh
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Sergey Subbotin
#pragma once
#include "common.cuh"

// Dispatches MUL_MAT_ID for tensors whose src0 is in the moe_stream buffer
// type. K active experts per token; their device pointers come from the
// LRU cache. Output is `dst` (one row per token, ne01 == hidden_dim).
//
// Returns false if the inputs aren't supported on this device (caller
// falls back to a generic loop).
bool ggml_cuda_moe_stream_mul_mat_id(
    ggml_backend_cuda_context &ctx,
    const ggml_tensor *src0,    // expert tensor (in moe_stream buffer)
    const ggml_tensor *src1,    // input activations
    const ggml_tensor *ids,     // routing decisions (top-K)
    ggml_tensor *dst);
```

- [ ] **Step 3: Implementation skeleton + Q4_K_M kernel**

This is the largest single task in the plan (~400 LOC). The implementer should land it in 3 sub-commits:
- a) skeleton + dispatch wiring (calls into the existing per-expert loop as a placeholder)
- b) replace placeholder with the fused kernel for the gate/up/SwiGLU path
- c) replace with the fused kernel for the down path

After each, smoke-test with `llama-server` + a one-token completion.

- [ ] **Step 4: Wire into `ggml_cuda_mul_mat_id`**

In `ggml/src/ggml-cuda/ggml-cuda.cu` (around line 2470, where the dispatch decisions are made), insert at the top:

```cpp
if (ggml_cuda_moe_stream_get_meta(src0)) {
    if (ggml_cuda_moe_stream_mul_mat_id(ctx, src0, src1, ids, dst)) {
        return;
    }
    // Fall through if our path declined.
}
```

- [ ] **Step 5: End-to-end smoke test**

```bash
ssh mi300 "cd ~/llamacpp-moe-cache && build/bin/llama-server -m models/qwen3-30b-a3b-q4_k_m.gguf -ngl 99 --moe-stream-dir /mnt/scratch/moe-stream-30b/ --moe-cache-mb 8192 --port 8080 &"
sleep 60
curl -s -m 60 -X POST http://mi300:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"local","messages":[{"role":"user","content":"Reply with OK only."}],"max_tokens":4}'
ssh mi300 "pkill -f llama-server"
```

Expected: a sane response containing "OK". If garbled, the dequant or routing logic is wrong — diff against the no-cache baseline.

- [ ] **Step 6: Bench**

```bash
for cfg in 'baseline' 'stream-8g' 'stream-32g'; do
  case $cfg in
    baseline)   args="";;
    stream-8g)  args="--moe-stream-dir /mnt/scratch/moe-stream-30b/ --moe-cache-mb 8192";;
    stream-32g) args="--moe-stream-dir /mnt/scratch/moe-stream-30b/ --moe-cache-mb 32768";;
  esac
  ssh mi300 "cd ~/llamacpp-moe-cache && build/bin/llama-bench -m models/qwen3-30b-a3b-q4_k_m.gguf -ngl 99 -t 8 -p 256 -n 64 $args 2>&1 | tail -3"
done | tee /tmp/bench-30b.log
scp mi300:/tmp/bench-30b.log docs/bench-30b.log || cp /tmp/bench-30b.log docs/bench-30b.log
```

Decision rule: keep if `stream-32g` is within 5% of baseline tg/s on Qwen3-30B (proves no regression when the model fits VRAM), AND the agent demo on Qwen3-235B (Phase 5) actually runs. The 30B model fitting in VRAM means cache adds no win there; the win is enabling >VRAM models to run at all.

- [ ] **Step 7: Commit (3 sub-commits per Step 3 + bench commit)**

```bash
git add docs/bench-30b.log
git commit -m "bench: Qwen3-30B-A3B baseline vs --moe-stream-dir cache 8 GB / 32 GB on MI300X"
```

### Task 4.2: Demo target — DeepSeek-V3 671B Q4_K_M (the headline)

This is the model where streaming actually matters. ~340 GB Q4_K_M, exceeds MI300X's 192 GB VRAM. The whole pipeline (buffer type + source + fused kernel + cache) is exercised because experts must be evicted and refetched.

- [ ] **Step 1: Download + pack** (do this in `screen` — multi-hour download)

```bash
ssh mi300 "cd /mnt/scratch/llamacpp-moe-cache/models/ && screen -dmS dl bash -c 'HF_HUB_ENABLE_HF_TRANSFER=1 hf download unsloth/DeepSeek-V3-GGUF DeepSeek-V3-Q4_K_M --local-dir . 2>&1 | tee /root/dl.log'"
# (Actual repo / filename varies; pick the smallest available Q4_K_M variant from a verified uploader.)
ssh mi300 "ls -la /mnt/scratch/llamacpp-moe-cache/models/DeepSeek-V3-Q4_K_M*.gguf"
ssh mi300 "cd ~/llamacpp-moe-cache && build/bin/moe-pack models/DeepSeek-V3-Q4_K_M.gguf /mnt/scratch/moe-stream-dsv3/"
```

- [ ] **Step 2: Run + bench**

```bash
ssh mi300 "cd ~/llamacpp-moe-cache && build/bin/llama-bench -m models/DeepSeek-V3-Q4_K_M.gguf -ngl 99 --moe-stream-dir /mnt/scratch/moe-stream-dsv3/ --moe-cache-mb 131072 -p 128 -n 64 2>&1 | tail -3"
```

Expected: model that wouldn't otherwise fit in 192 GB VRAM produces tokens. Headline number for the hackathon submission.

- [ ] **Step 3: Commit benchmark result**

```bash
git add docs/bench-dsv3.log
git commit -m "bench: DeepSeek-V3 671B Q4_K_M on a single MI300X via --moe-stream-dir"
```


## Phase 5 — companion repo + agent demo + submission

### Task 5.1: Create the public companion repo `flash-moe-rocm-demo`

**Files:**
- Create new public repo on github.com/ssubbotin (NOT a fork of flash-moe)

- [ ] **Step 1: Create the repo**

```bash
gh repo create ssubbotin/flash-moe-rocm-demo --public --description "MI300X-optimized MoE inference: agent demo + benchmarks for our llama.cpp PR" --license MIT
git -C ~/llamacpp-moe-cache log --oneline | head -10
mkdir -p ~/flash-moe-rocm-demo && cd ~/flash-moe-rocm-demo && git init -b main
gh repo set-default ssubbotin/flash-moe-rocm-demo
```

- [ ] **Step 2: Add MIT LICENSE (the `gh repo create --license MIT` should have created it; verify)**

```bash
ls ~/flash-moe-rocm-demo/LICENSE && head -3 ~/flash-moe-rocm-demo/LICENSE
```

Expected: MIT License text with `Copyright (c) 2026 Sergey Subbotin`.

- [ ] **Step 3: Skeleton README + dirs**

```bash
cd ~/flash-moe-rocm-demo
mkdir -p src/agent_demo scripts docs tests
cat > README.md <<'EOF'
# flash-moe-rocm-demo

MI300X-optimized MoE inference using a [llama.cpp PR branch](https://github.com/ggml-org/llama.cpp/pull/PRNUM) that adds a GPU-resident expert cache and optional SSD streaming.

This repo contains the agent demo, benchmark scripts, and writeup for the AMD Developer Hackathon 2026 submission.

The actual inference engine changes are upstream in llama.cpp. This repo wraps `llama-server` with an OpenAI-compatible Python agent that exercises tool calling on a real MI300X.

## Prior art

- vLLM RFC #38256 (Apache 2.0) — `ExpertWeightProvider` abstraction
- tinyserve by e1n00r (MIT) — independent CPU-offload reference impl
- ktransformers (Apache 2.0) — earlier ROCm/CPU MoE offload work
- llama.cpp itself (MIT) — host engine; we extend its MoE dispatch
EOF
git add README.md && git commit -m "init: repo skeleton + prior art credits"
```

### Task 5.2: Implement the Python agent

**Files:**
- Create: `src/agent_demo/{__init__.py,__main__.py,client.py,tools.py,agent.py,prompts.py}`
- Create: `pyproject.toml`

- [ ] **Step 1: pyproject.toml**

```toml
[project]
name = "flash-moe-agent-demo"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "openai>=1.30",
    "httpx>=0.27",
]

[project.scripts]
agent-demo = "agent_demo.__main__:main"
```

- [ ] **Step 2: tools.py — three concrete tools**

```python
# src/agent_demo/tools.py
import subprocess, pathlib, json, httpx
TOOLS_SPEC = [
    {"type":"function","function":{"name":"read_file","description":"Read a UTF-8 text file.","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},
    {"type":"function","function":{"name":"run_shell","description":"Run a shell command and return stdout (truncated 4 KB).","parameters":{"type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"]}}},
    {"type":"function","function":{"name":"http_get","description":"GET a URL and return the body (truncated 4 KB).","parameters":{"type":"object","properties":{"url":{"type":"string"}},"required":["url"]}}},
]
def read_file(path: str) -> str:
    return pathlib.Path(path).read_text(errors="replace")[:8192]
def run_shell(cmd: str) -> str:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
    return (r.stdout + r.stderr)[:4096]
def http_get(url: str) -> str:
    return httpx.get(url, timeout=20).text[:4096]
DISPATCH = {"read_file": read_file, "run_shell": run_shell, "http_get": http_get}
```

- [ ] **Step 3: agent.py — minimal multi-step loop**

```python
# src/agent_demo/agent.py
import json
from openai import OpenAI
from .tools import TOOLS_SPEC, DISPATCH
from .prompts import SYSTEM
def run_agent(task: str, base_url: str, model: str, max_steps: int = 8) -> str:
    client = OpenAI(base_url=base_url, api_key="local")
    msgs = [{"role":"system","content":SYSTEM},{"role":"user","content":task}]
    for step in range(max_steps):
        r = client.chat.completions.create(model=model, messages=msgs, tools=TOOLS_SPEC, tool_choice="auto")
        m = r.choices[0].message
        msgs.append({"role":"assistant","content":m.content,"tool_calls":m.tool_calls})
        if not m.tool_calls:
            return m.content or ""
        for tc in m.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            try:    out = DISPATCH[name](**args)
            except Exception as e: out = f"ERROR: {e}"
            msgs.append({"role":"tool","tool_call_id":tc.id,"content":out})
    return "[max_steps reached]"
```

- [ ] **Step 4: __main__.py CLI**

```python
import argparse
from .agent import run_agent
def main():
    p = argparse.ArgumentParser(prog="agent-demo")
    p.add_argument("--task", required=True)
    p.add_argument("--base-url", default="http://localhost:8080/v1")
    p.add_argument("--model", default="local")
    print(run_agent(p.parse_args().task, p.parse_args().base_url, p.parse_args().model))
if __name__ == "__main__": main()
```

- [ ] **Step 5: prompts.py**

```python
SYSTEM = """You are a careful engineering assistant running locally on a MacBook-class
machine via llama.cpp + MI300X. You may call tools to read files, run shell commands,
or fetch URLs. Use tools sparingly: prefer one tool call, then a final answer."""
```

- [ ] **Step 6: Smoke test against running llama-server**

```bash
# (server should still be up from Task 3.4; if not, restart with --moe-cache-mb 8192)
cd ~/flash-moe-rocm-demo && pip install -e .
ssh -fN -L 18080:localhost:8080 mi300
agent-demo --task "What is the size in bytes of /etc/hostname on this machine?" --base-url http://localhost:18080/v1
```

Expected: a final answer containing the byte count, after the agent calls `run_shell` or `read_file`.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/
git commit -m "agent: 3-tool agent loop (read_file, run_shell, http_get) over OpenAI API"
```

### Task 5.3: Benchmarks doc

**Files:**
- Create: `docs/benchmarks.md`

- [ ] **Step 1: Run a fixed bench matrix**

```bash
ssh mi300 "cd ~/llamacpp-moe-cache && for cfg in 'baseline' 'cache-8g' 'cache-8g-ssd' 'cache-8g-ssd-fused'; do echo === $cfg ===; ...; done | tee /tmp/bench-matrix.log"
scp mi300:/tmp/bench-matrix.log ~/flash-moe-rocm-demo/docs/raw-bench.log
```

- [ ] **Step 2: Write `docs/benchmarks.md` with a table**

```markdown
# Benchmarks — MI300X (gfx942), Qwen3-30B-A3B Q4_K_M

| Configuration | tok/s (decode 128) | cache hit rate | Notes |
|---|---:|---:|---|
| llama.cpp upstream | X.XX | n/a | baseline |
| + `--moe-cache-mb 8192` | X.XX | XX % | LRU only |
| + SSD streaming | X.XX | XX % | --moe-ssd-dir |
| + gfx942 fused MoE | X.XX | XX % | all flags |

(Numbers filled in by Task 5.3 step 1.)
```

- [ ] **Step 3: Commit**

```bash
git add docs/benchmarks.md docs/raw-bench.log
git commit -m "docs: benchmarks vs upstream llama.cpp on MI300X"
```

### Task 5.4: Record the demo video

**Files:**
- Create: `docs/demo.mp4` (or link to YouTube/Loom)

- [ ] **Step 1: Storyboard (under 5 min)**
  1. 0:00–0:30 Title card + claim ("Qwen3-30B running on a single MI300X with 8 GB cache + SSD streaming, agent doing real work")
  2. 0:30–1:30 Show `llama-server` start with `--moe-cache-mb 8192 --moe-ssd-dir …`, hit-rate live log
  3. 1:30–3:00 `agent-demo --task "…"` — agent reads a file, runs a shell command, returns answer
  4. 3:00–4:00 Bench table + before/after tok/s
  5. 4:00–4:45 Quick architecture sketch (cache → miss → SSD pread → kernel)
  6. 4:45–5:00 Submission card (links to repo + PR)

- [ ] **Step 2: Record + upload**

Use OBS or asciinema → ffmpeg. Upload to YouTube unlisted or Loom; put the link in `README.md`.

- [ ] **Step 3: Commit link**

```bash
sed -i 's|RECORDING_LINK|https://youtu.be/XXXX|' README.md
git add README.md && git commit -m "docs: demo video link"
```

### Task 5.5: Open the llama.cpp PR

- [ ] **Step 1: Push the branch**

```bash
ssh -F /dev/null github.com 2>/dev/null  # noop, just to make sure ssh agent is alive
gh repo fork ggml-org/llama.cpp --remote=true --remote-name=fork
git push -u fork feature/moe-expert-gpu-cache
```

- [ ] **Step 2: Open PR with body referencing the demo repo + benchmarks**

```bash
gh pr create --repo ggml-org/llama.cpp --base master --head ssubbotin:feature/moe-expert-gpu-cache \
  --title "ggml: GPU-resident MoE expert cache + optional SSD streaming + gfx942 fused kernel" \
  --body "$(cat <<'EOF'
Adds a portable GPU LRU cache for MoE expert weights. Cache is opaque to
the kernel — it just hands a device pointer that points at either the
existing `src0->data` slice or a cache slot populated from a configurable
miss source. Two miss sources land in this PR: the default CPU copy and
an opt-in SSD pread per layer.

Companion benchmarks + agent demo: https://github.com/ssubbotin/flash-moe-rocm-demo

References:
- vLLM RFC #38256 (Apache 2.0) — ExpertWeightProvider abstraction
- e1n00r/tinyserve (MIT) — independent reference implementation
- llama.cpp #11532 (closed) — original feature ask
- AITER (MIT) — inspiration for the gfx942 fused MoE kernel
EOF
)"
```

- [ ] **Step 3: Verify CI is running**

```bash
gh pr view --repo ggml-org/llama.cpp ssubbotin:feature/moe-expert-gpu-cache --json statusCheckRollup
```

Watch for any compile failures across the matrix; iterate fixes on the branch.

### Task 5.6: Submit on lablab.ai

- [ ] **Step 1: Fill the submission form** at lablab.ai with:
  - Project name: "flash-moe-rocm-demo"
  - GitHub: `https://github.com/ssubbotin/flash-moe-rocm-demo`
  - Upstream PR: link to the llama.cpp PR
  - Demo video: link from Task 5.4
  - License: MIT (verifiable in repo)
  - Description: 1 paragraph from `README.md`

- [ ] **Step 2: Sanity-check the submission renders correctly** (preview before final submit).

---

## Self-review checklist

Run before declaring the plan done:

1. **Spec coverage**: Does each item in section "What's missing for a clean submission" from the conversation map to a task?
   - MIT LICENSE → Task 5.1 step 2 ✓
   - Agent/tool-calling demo → Tasks 5.2 ✓
   - Hackathon-facing README → Task 5.1 step 3 + Task 5.4 step 3 ✓
   - Demo video < 5 min → Task 5.4 ✓
   - lablab.ai submission → Task 5.6 ✓

2. **Placeholder scan**: search the plan for `TBD`, `TODO`, `…`, `(omitted)`, `XXX`. The only remaining ellipsis is a deliberate "fill numbers from bench" in Task 5.3 step 2, marked. Any others to fix?

3. **Type consistency**:
   - `ggml_moe_cache_params.miss_fn` (Task 1.1) matches `ggml_moe_miss_fn` typedef ✓
   - `ggml_cuda_moe_cpu_src` (Task 2.2) matches its use in `moe-cache-ssd.cu` (Task 3.2)? **No** — Task 2.2's `cpu_src` has a `base + expert_stride` pair, but `ggml_cuda_moe_miss_from_cpu` ignores the layer offset (comment says "layer offset is the caller's responsibility — per-layer cpu_src in context"). When Task 2.4 instantiates the cache, ensure the wiring stores one `cpu_src` per layer or extends the struct with `layer_strides[]`. **Action: in Task 2.4 step 2 implementation, allocate `std::vector<ggml_cuda_moe_cpu_src> per_layer_src(num_moe_layers)`.**
   - `--moe-cache-mb` (Task 2.4) and `--moe-ssd-dir` (Task 3.2 step 3) both flow into `common_params` then into `cparams` via the existing llama_context build path. Verify by `grep` after Task 2.4 step 2.

4. **License firewall**: every Phase has at least one explicit reminder ("read from public references only") in its tasks where flash-moe code might be tempting? Phase 4 task 4.1 is the highest-risk one — it explicitly approves reading the three Sergey-authored files only.

If self-review surfaces a real issue past the type-consistency note above, fix it inline.

---

## Out of scope for this plan

- MTP speculative decoding (Task 5.1 of the original mi300-opt plan). Blocked anyway by missing weights in the MLX repo, would require a separate base model.
- aotriton flash attention path. Marginal at decode shape (Sq=1); not worth the dependency churn for a hackathon PR.
- MFMA path. Empirically -7% on N=1 matvec; defer to a future "prefill batching" PR.
- Q5_K, Q6_K, Q8_0 dequant paths in the fused MoE kernel. Phase 4 lands Q4_K_M only (the GGUF format Qwen3 ships in). Other dtypes can be future PRs.
- `--n-cpu-moe` interaction. The new cache is orthogonal: cache lives on GPU, `--n-cpu-moe` keeps CPU-resident layers. We won't touch its code path.
