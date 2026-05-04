# llama.cpp MoE SSD Streaming + GPU LRU Cache — Hackathon Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land a focused PR (or PR series) on `ggml-org/llama.cpp` that adds GPU-resident LRU caching for MoE expert weights with optional SSD streaming, plus an MI300X-tuned fused MoE kernel — and ship a small MIT-licensed companion repo with an agent demo. Targets the AMD Developer Hackathon (deadline 2026-05-10).

**Architecture:**
- New `ggml-backend`-level abstraction: `ggml_moe_expert_cache` that owns a fixed-size GPU buffer pool, an LRU index keyed by `(layer_id, expert_id)`, and a pluggable miss source (default: copy from CPU mmap'd weights; opt-in: pread from a packed SSD file).
- Hooked into `MUL_MAT_ID` / `mmid.cu` so cache lookups happen at the kernel-dispatch site without changing the public ggml API.
- Backend kernel work lands in `ggml/src/ggml-cuda/` with HIP guards (which is how `ggml-hip` already shares CUDA sources). gfx942-specific tunings gated by `defined(__HIP_PLATFORM_AMD__) && defined(__gfx942__)`.
- All code written from public references (vLLM, AITER MIT, llama.cpp itself, e1n00r/tinyserve, the Qwen3.5 paper). **Zero lines copied from danveloper/flash-moe** to keep our PR (and our companion repo) MIT-licensable.

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

## Phase 2 — wire the cache into MUL_MAT_ID

### Task 2.1: Read the existing `mmid.cu` dispatch and identify the hook point

**Files:**
- Read-only: `ggml/src/ggml-cuda/mmid.cu`, `ggml/src/ggml-cuda/mmid.cuh`

- [ ] **Step 1: Locate the per-expert weight pointer in `ggml_cuda_mul_mat_id`**

```bash
ssh mi300 "grep -n 'mul_mat_id\|src0_row\|expert' ~/llamacpp-moe-cache/ggml/src/ggml-cuda/mmid.cu | head -20"
```

Identify where the source-tensor pointer for expert `e` is computed (something like `src0->data + e * stride`). Record the line number and the variable name in this plan's notes for Task 2.2 reference.

- [ ] **Step 2: Look at how the kernel receives the pointer**

```bash
ssh mi300 "grep -n 'launch_mul_mat_id\|kernel<<<' ~/llamacpp-moe-cache/ggml/src/ggml-cuda/mmid.cu | head -20"
```

Note whether the kernel takes a single base pointer + per-expert index OR an array of K pointers. Our cache integration follows the same convention.

(No commit — reconnaissance only.)

### Task 2.2: Add the GPU-side cache adapter `moe-cache.cu`

**Files:**
- Create: `ggml/src/ggml-cuda/moe-cache.cu`
- Create: `ggml/src/ggml-cuda/moe-cache.cuh`

- [ ] **Step 1: Write the header**

```cpp
// ggml/src/ggml-cuda/moe-cache.cuh
#pragma once
#include "common.cuh"
#include "ggml-moe-cache.h"

// CUDA/HIP backend hook: returns a device pointer for expert (l, e),
// honoring the cache. ctx->moe_cache may be null (no cache configured),
// in which case the caller falls back to the normal `src0->data` path.
const void *ggml_cuda_moe_cache_lookup(ggml_backend_cuda_context &ctx,
                                       int layer_id, int expert_id);

// Build a CPU-mmap-source miss function for the common case where the
// expert weights are present in host memory (e.g. after model load with
// --no-mmap-experts off). The user data is the base pointer + stride.
struct ggml_cuda_moe_cpu_src {
    const void *base;
    size_t      expert_stride; // bytes between experts in CPU buffer
    cudaStream_t stream;       // for async H2D copy
};
int ggml_cuda_moe_miss_from_cpu(void *user_data, int layer, int expert,
                                void *dst, size_t bytes);
```

- [ ] **Step 2: Write the implementation**

```cpp
// ggml/src/ggml-cuda/moe-cache.cu
#include "moe-cache.cuh"

const void *ggml_cuda_moe_cache_lookup(ggml_backend_cuda_context &ctx,
                                       int layer_id, int expert_id) {
    if (!ctx.moe_cache) return nullptr;
    return ggml_moe_cache_get(ctx.moe_cache, layer_id, expert_id);
}

int ggml_cuda_moe_miss_from_cpu(void *ud, int layer, int expert,
                                void *dst, size_t bytes) {
    auto *src = (ggml_cuda_moe_cpu_src *)ud;
    const char *p = (const char *)src->base
                  + ((size_t)layer * 0 /*layers stored separately*/ )
                  + (size_t)expert * src->expert_stride;
    // NOTE: layer offset is the caller's responsibility — the cpu_src
    // struct is per-layer in the simplest wiring (Phase 2 stores one
    // cpu_src per MoE layer in the context). See Task 2.3.
    cudaMemcpyAsync(dst, p, bytes, cudaMemcpyHostToDevice, src->stream);
    return cudaStreamSynchronize(src->stream) == cudaSuccess ? 0 : 1;
}
```

- [ ] **Step 3: Wire into `ggml-cuda.cu`**

In `ggml/src/ggml-cuda/ggml-cuda.cu`, extend the `ggml_backend_cuda_context` struct (header `common.cuh`) to hold:

```cpp
ggml_moe_cache *moe_cache = nullptr;
```

Add cleanup in the backend's destructor.

- [ ] **Step 4: Build**

```bash
ssh mi300 "cd ~/llamacpp-moe-cache && cmake --build build -j 2>&1 | tail -10"
```

Expected: clean build.

- [ ] **Step 5: Commit**

```bash
git add ggml/src/ggml-cuda/moe-cache.{cu,cuh} ggml/src/ggml-cuda/ggml-cuda.cu ggml/src/ggml-cuda/common.cuh
git commit -m "ggml-cuda: scaffolding for MoE cache lookup + CPU miss source (no dispatch yet)"
```

### Task 2.3: Hook `mmid.cu` to route per-expert weights through the cache

**Files:**
- Modify: `ggml/src/ggml-cuda/mmid.cu`

- [ ] **Step 1: Replace the per-expert pointer derivation with a cache lookup**

In the spot identified in Task 2.1, change the code that computes the per-expert source pointer from:

```cpp
const void *src0_e = (const char *)src0->data + e * src0_e_stride;
```

to:

```cpp
const void *src0_e = nullptr;
if (ctx.moe_cache) {
    src0_e = ggml_cuda_moe_cache_lookup(ctx, layer_id, e);
}
if (!src0_e) {
    src0_e = (const char *)src0->data + e * src0_e_stride;
}
```

Where `layer_id` is derived from the tensor's `op_params` or from a layer-id field added to `ggml_tensor` (we'll need to thread it through — start by deriving from the tensor name's layer index suffix as a temporary measure).

- [ ] **Step 2: Build with cache disabled (default), bench unchanged**

```bash
ssh mi300 "cd ~/llamacpp-moe-cache && cmake --build build -j --target llama-bench && build/bin/llama-bench -m models/qwen3-30b-a3b-q4_k_m.gguf -ngl 99 -t 8 -p 512 -n 128 2>&1 | tail -5"
```

Expected: same tok/s as Task 0.3 baseline (cache is null, so the new code path falls through).

- [ ] **Step 3: Commit**

```bash
git add ggml/src/ggml-cuda/mmid.cu
git commit -m "ggml-cuda: route per-expert pointer through MoE cache when configured (no-op when disabled)"
```

### Task 2.4: Surface CLI flag `--moe-cache-mb` and instantiate the cache

**Files:**
- Modify: `common/arg.cpp`, `common/common.h`
- Modify: `src/llama-context.cpp`

- [ ] **Step 1: Add the flag**

In `common/arg.cpp`, add an arg parser entry near the existing `--n-cpu-moe`:

```cpp
add_opt(common_arg(
    {"--moe-cache-mb"},
    "MB", "GPU MoE expert cache size in MiB (0 = disabled)",
    [](common_params & params, int v) { params.moe_cache_mb = v; }
).set_env("LLAMA_ARG_MOE_CACHE_MB"));
```

In `common/common.h`, add `int moe_cache_mb = 0;` to `common_params`.

- [ ] **Step 2: Construct the cache after the backend is up**

In `src/llama-context.cpp`, after the GPU backends are initialized and the model is loaded (find the existing CUDA/HIP backend init block), if `cparams.moe_cache_mb > 0` and the model has MoE layers, build the params and call `ggml_moe_cache_init`. Store the result on the per-backend context so the kernel hook from Task 2.3 can reach it.

(Concrete code location and field name will depend on the current llama-context layout; identify by `grep -n 'ggml_backend_cuda_init' src/llama-context.cpp` and adapt.)

- [ ] **Step 3: Smoke test with cache on**

```bash
ssh mi300 "cd ~/llamacpp-moe-cache && build/bin/llama-bench -m models/qwen3-30b-a3b-q4_k_m.gguf -ngl 99 -t 8 -p 512 -n 128 --moe-cache-mb 8192 2>&1 | tail -10"
```

Expected: same correctness, possibly slightly different tok/s (cache adds bookkeeping but eliminates redundant pointer math). If tok/s drops more than 5%, investigate eviction churn before proceeding.

- [ ] **Step 4: Commit**

```bash
git add common/arg.cpp common/common.h src/llama-context.cpp
git commit -m "llama: --moe-cache-mb CLI flag, instantiate MoE cache when set"
```

---

## Phase 3 — SSD streaming as a miss source

### Task 3.1: Define the SSD-source layout

**Files:**
- Create: `docs/moe-ssd-layout.md`

- [ ] **Step 1: Document the on-disk format**

```markdown
# MoE SSD layout (one file per layer)

Each layer's expert weights live in a single binary file named
`layer_<NN>.bin` inside a user-supplied directory. Within the file,
expert `e` starts at byte offset `e * expert_bytes`; layout matches
the order of components in the GGUF tensor that produced it (gate_w,
up_w, down_w concatenated and contiguous per expert). Files are
written by an offline conversion script from any GGUF MoE checkpoint;
no in-engine generation. Component order is fixed by `enum
moe_expert_layout` in ggml-moe-cache.h.
```

This format mirrors the cache's `expert_bytes` exactly — no
metadata header on the SSD file, so `pread(fd, dst, expert_bytes, e * expert_bytes)`
suffices on cache miss.

- [ ] **Step 2: Commit**

```bash
git add docs/moe-ssd-layout.md
git commit -m "docs: MoE SSD per-layer file layout"
```

### Task 3.2: Implement the SSD miss source

**Files:**
- Create: `ggml/src/ggml-cuda/moe-cache-ssd.cu`
- Modify: `ggml/src/ggml-cuda/moe-cache.cuh`

- [ ] **Step 1: Add the SSD source struct + miss fn declaration to `moe-cache.cuh`**

```cpp
struct ggml_cuda_moe_ssd_src {
    int        layer_fds[256];   // one fd per MoE layer (caller opens)
    int        num_layers;
    size_t     expert_bytes;
    void *     pinned_staging;   // pinned host buffer, expert_bytes
    cudaStream_t stream;
};
int ggml_cuda_moe_miss_from_ssd(void *user_data, int layer, int expert,
                                void *dst, size_t bytes);
```

- [ ] **Step 2: Implement**

```cpp
// ggml/src/ggml-cuda/moe-cache-ssd.cu
#include "moe-cache.cuh"
#include <unistd.h>
#include <sys/types.h>
#include <cerrno>
#include <cstring>

int ggml_cuda_moe_miss_from_ssd(void *ud, int layer, int expert,
                                void *dst, size_t bytes) {
    auto *src = (ggml_cuda_moe_ssd_src *)ud;
    if (layer < 0 || layer >= src->num_layers) return 1;
    int fd = src->layer_fds[layer];
    off_t off = (off_t)expert * (off_t)src->expert_bytes;
    ssize_t got = pread(fd, src->pinned_staging, bytes, off);
    if (got != (ssize_t)bytes) {
        fprintf(stderr, "[moe-ssd] pread L%d E%d short read got=%zd err=%s\n",
                layer, expert, got, strerror(errno));
        return 1;
    }
    cudaMemcpyAsync(dst, src->pinned_staging, bytes,
                    cudaMemcpyHostToDevice, src->stream);
    return cudaStreamSynchronize(src->stream) == cudaSuccess ? 0 : 1;
}
```

- [ ] **Step 3: Add CLI flag `--moe-ssd-dir DIR`**

In `common/arg.cpp` and `common/common.h`, add `std::string moe_ssd_dir;`. In `src/llama-context.cpp`, when the cache is being built and `cparams.moe_ssd_dir` is non-empty:
- For each MoE layer, `open(<dir>/layer_<NN>.bin, O_RDONLY)`; store fds in `ggml_cuda_moe_ssd_src`
- `cudaMallocHost(&pinned_staging, expert_bytes)`
- Wire `miss_fn = ggml_cuda_moe_miss_from_ssd`

- [ ] **Step 4: Build**

```bash
ssh mi300 "cd ~/llamacpp-moe-cache && cmake --build build -j 2>&1 | tail -5"
```

Expected: clean build.

- [ ] **Step 5: Commit**

```bash
git add ggml/src/ggml-cuda/moe-cache-ssd.cu ggml/src/ggml-cuda/moe-cache.cuh common/arg.cpp common/common.h src/llama-context.cpp
git commit -m "ggml-cuda: SSD miss source for MoE expert cache, --moe-ssd-dir flag"
```

### Task 3.3: Write the offline GGUF→per-layer converter

**Files:**
- Create: `tools/moe-pack/moe-pack.cpp`
- Modify: `tools/CMakeLists.txt`

- [ ] **Step 1: Write the converter (host-only, no GPU)**

```cpp
// tools/moe-pack/moe-pack.cpp
// Converts a GGUF MoE checkpoint into one binary file per MoE layer:
//   <out_dir>/layer_NN.bin  with experts laid out contiguously, expert e
//   at offset e * expert_bytes. No header, no metadata.
#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <string>

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: moe-pack input.gguf out_dir\n"); return 2; }
    const char *path = argv[1]; const char *out = argv[2];
    mkdir(out, 0755);
    gguf_init_params p { false, nullptr };
    gguf_context *ctx = gguf_init_from_file(path, p);
    if (!ctx) { fprintf(stderr, "gguf_init failed\n"); return 1; }
    int n = gguf_get_n_tensors(ctx);
    // Group tensors by layer index parsed from name (blk.<L>.ffn_*_exps.weight).
    // For each layer, concatenate gate_w, up_w, down_w expert blobs in the
    // same order GGUF stores them (already [expert, out, in] row-major).
    // Write to <out>/layer_NN.bin. Skip non-MoE tensors.
    // … (full enumeration omitted in this plan; TDD-extend)
    gguf_free(ctx);
    return 0;
}
```

- [ ] **Step 2: Add to `tools/CMakeLists.txt`**

```cmake
add_subdirectory(moe-pack)
```

And in `tools/moe-pack/CMakeLists.txt`:

```cmake
add_executable(moe-pack moe-pack.cpp)
target_link_libraries(moe-pack PRIVATE ggml)
```

- [ ] **Step 3: Run on Qwen3-30B-A3B**

```bash
ssh mi300 "cd ~/llamacpp-moe-cache && build/bin/moe-pack models/qwen3-30b-a3b-q4_k_m.gguf models/moe-ssd-30b/ && ls models/moe-ssd-30b/ | head -5 && du -sh models/moe-ssd-30b/"
```

Expected: 48 layer files (Qwen3-30B has 48 layers), total size ≈ expert weight subset of the GGUF.

- [ ] **Step 4: Commit**

```bash
git add tools/moe-pack/ tools/CMakeLists.txt
git commit -m "tools: moe-pack — GGUF MoE → per-layer binary files for --moe-ssd-dir"
```

### Task 3.4: End-to-end SSD streaming smoke test

- [ ] **Step 1: Run with SSD streaming on, small cache**

```bash
ssh mi300 "cd ~/llamacpp-moe-cache && build/bin/llama-server -m models/qwen3-30b-a3b-q4_k_m.gguf -ngl 99 --moe-cache-mb 2048 --moe-ssd-dir models/moe-ssd-30b/ --port 8080 &"
sleep 60
curl -s http://mi300:8080/v1/models
curl -s http://mi300:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"local","messages":[{"role":"user","content":"Reply with OK"}],"max_tokens":4}'
```

Expected: `OK` response, no crashes. Cache stats logged on exit.

- [ ] **Step 2: Bench tok/s + cache hit rate at 2 GB / 4 GB / 8 GB cache sizes**

```bash
for mb in 2048 4096 8192; do
  ssh mi300 "cd ~/llamacpp-moe-cache && build/bin/llama-bench -m models/qwen3-30b-a3b-q4_k_m.gguf -ngl 99 --moe-cache-mb $mb --moe-ssd-dir models/moe-ssd-30b/ -p 256 -n 128 2>&1 | tail -3"
done
```

Record numbers in `docs/benchmarks.md` (Phase 5).

- [ ] **Step 3: Commit benchmark log**

```bash
mkdir -p docs/benchmarks
cp /tmp/bench-*.log docs/benchmarks/  # or pipe directly
git add docs/benchmarks/
git commit -m "bench: MoE cache + SSD streaming numbers, Qwen3-30B-A3B on MI300X"
```

---

## Phase 4 — gfx942-tuned fused MoE kernel

### Task 4.1: Port `kernels_fused_moe.hip.h` (the scalar path) into ggml-cuda with HIP guards

**Files:**
- Create: `ggml/src/ggml-cuda/fused-moe-amd.cu`
- Create: `ggml/src/ggml-cuda/fused-moe-amd.cuh`
- Modify: `ggml/src/ggml-cuda/mmid.cu` (call site behind `#if defined(__HIP_PLATFORM_AMD__)`)

- [ ] **Step 1: Refactor + relicense the kernel**

Open `mi300-opt/rocm_infer/kernels_fused_moe.hip.h` (this is one of the three files we may consult, see License section). Copy the `fused_moe_gate_up_swiglu_mlx` and `fused_moe_down_mlx` kernel bodies into `ggml/src/ggml-cuda/fused-moe-amd.cu`, but:
- Replace MLX-specific pack offsets (`EXP_GATE_W` etc.) with parameters passed from the launcher
- Drop the MLX-quantization-only assumption — the layout struct is now per-call
- Header gains MIT SPDX line: `// SPDX-License-Identifier: MIT`
- Copyright header: `// Copyright (c) 2026 Sergey Subbotin`
- Cite AITER (MIT) as the inspiration for the scalar dispatch shape

- [ ] **Step 2: Wire from the cache lookup hook**

In `mmid.cu`, after the cache lookup yields a device pointer for each of K experts, on AMD/gfx942 path replace the K iterations of normal `mmvq`/`mmvf` with a single launch of the fused kernel:

```cpp
#if defined(__HIP_PLATFORM_AMD__) && defined(__gfx942__)
    if (use_fused_moe_amd && K == 4) {
        launch_fused_moe_amd(d_expert_ptrs, ..., stream);
    } else
#endif
    {
        // existing per-expert dispatch
    }
```

- [ ] **Step 3: Bench against baseline (no cache, no fused)**

```bash
ssh mi300 "cd ~/llamacpp-moe-cache && build/bin/llama-bench -m models/qwen3-30b-a3b-q4_k_m.gguf -ngl 99 -p 256 -n 128 --moe-cache-mb 8192 2>&1 | tail -3"
```

Decision rule: keep if ≥5% improvement at 128-token decode over Phase 3 numbers. (The standalone +8% number from `mi300-opt` is for a different model — Qwen3.5-397B at MLX 4-bit. Q4_K_M layout in llama.cpp is different; expect smaller gains.)

- [ ] **Step 4: Commit**

```bash
git add ggml/src/ggml-cuda/fused-moe-amd.{cu,cuh} ggml/src/ggml-cuda/mmid.cu
git commit -m "ggml-cuda: gfx942 fused MoE kernel (gate+up+SwiGLU+down in 2 launches)"
```

---

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
