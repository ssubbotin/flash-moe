# Strix Halo (gfx1151) APU Optimization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Push `apu_infer/` (Qwen3.5-397B-A17B 4-bit MoE on AMD Ryzen AI MAX+ 395 / Radeon 8060S) past the current 3.46 tok/s peak / ~1.8 tok/s steady-state, by exploiting Strix Halo features that don't apply on M3 Max or discrete CUDA.

**Architecture:** Each task is one experiment in the project's `results.tsv` style — measure baseline, change one thing, re-measure, decide keep-or-discard. Correctness is gated on producing the same `" Paris.<|im_end|>\n<|im_start|>assistant\n<think>\nThinking..."` output as `cuda_infer` for the prompt `"The capital of France is"`. Phases are ordered highest-impact-first within each tier; tiers are ordered by how unique-to-APU the win is.

**Tech Stack:** HIP/ROCm 6.4, gfx1151 RDNA 3.5 wavefront-32, Linux 6.19 with HMM/HSA SVM, Zen 5 (AVX-512 BF16), 64 GB LPDDR5 unified (32 GB GPU + 32 GB CPU split), 928 GB NVMe.

**Test environment:** All work happens against `max395` over SSH. Code is edited locally on the `apu` branch, scp'd to `~/flash-moe/apu_infer/` on max395, built via `make`, run against the model at `~/flash-moe/model-safetensors/`.

---

## File Structure

```
apu_infer/
  infer.hip            # Main engine — most experiments touch this file
  kernels.hip.h        # GPU kernels — Phase 3 work touches this
  Makefile             # Build flags — Phase 5 touches this
  bench/
    bench.sh           # Benchmark harness (created in Phase 0)
    expected.txt       # Golden output for correctness check
results.tsv            # Append one row per experiment (existing log)
docs/superpowers/plans/2026-04-09-strix-halo-optimization.md  # This plan
```

Key decisions:
- **Stay on the `apu` branch.** All commits land here. The `rocm` branch (discrete MI300X) only gets the format-conditional fixes that already landed (`12e6ca0`); APU-specific tricks would hurt discrete cards.
- **One experiment per commit.** Even discarded experiments get committed temporarily so we can `git diff` against the previous baseline; failed ones get reverted with `git revert`.
- **`results.tsv` is the source of truth for tok/s numbers.** The plan tasks reference rows by the description text, not row numbers.
- **Each task ends with a measurement and a decision.** No "implement, see what happens" — every step has a number to chase and a number it has to beat.

---

## Phase 0: Baseline & test harness

We need a one-button benchmark that reports tok/s consistently across runs and a correctness gate that catches regressions before measurement is wasted.

### Task 0.1: Create benchmark harness

**Files:**
- Create: `apu_infer/bench/bench.sh`
- Create: `apu_infer/bench/expected.txt`

- [ ] **Step 1: Write the expected output**

```bash
cat > apu_infer/bench/expected.txt <<'EOF'
 Paris.<|im_end|>
<|im_start|>assistant
<think>
EOF
```

- [ ] **Step 2: Write the bench script**

```bash
cat > apu_infer/bench/bench.sh <<'BASH'
#!/usr/bin/env bash
# Usage: ./bench.sh [cold|warm] [tokens] [vram_cache_mib]
# Examples:
#   ./bench.sh cold 30          # cold cache, no VRAM cache, 30 tokens
#   ./bench.sh warm 100 20480   # warm cache, 20 GB VRAM cache, 100 tokens
set -euo pipefail

MODE="${1:-warm}"
TOKENS="${2:-30}"
VRAM_MIB="${3:-0}"

cd "$(dirname "$0")/.."

if [[ "$MODE" == "cold" ]]; then
    sync && echo 3 > /proc/sys/vm/drop_caches
fi

ENV=""
if [[ "$VRAM_MIB" != "0" ]]; then
    ENV="ENV ENABLE_VRAM_CACHE=$VRAM_MIB"
    export ENABLE_VRAM_CACHE="$VRAM_MIB"
fi

LOG=$(mktemp /tmp/bench.XXXXXX.log)
./infer --prompt 'The capital of France is' --tokens "$TOKENS" \
        --experts ../model-safetensors/packed_experts > "$LOG" 2>&1

# Extract the generated text (between [generating] and [done])
GEN=$(awk '/^\[generating\]/{flag=1; next} /^\[done\]/{flag=0} flag' "$LOG")
EXPECTED=$(cat bench/expected.txt)

# Correctness gate: first 3 lines must match expected.txt
if ! printf '%s\n' "$GEN" | head -3 | diff -q - bench/expected.txt > /dev/null 2>&1; then
    echo "FAIL: output mismatch"
    echo "--- expected ---"
    cat bench/expected.txt
    echo "--- got ---"
    printf '%s\n' "$GEN" | head -5
    rm -f "$LOG"
    exit 1
fi

# Extract tok/s from [done] line
TOKPS=$(grep '^\[done\]' "$LOG" | grep -oP '[0-9.]+(?= tok/s)')
TOTAL_MS=$(grep '^\[done\]' "$LOG" | grep -oP '[0-9.]+(?= ms total)')
PER_TOK_MS=$(grep '^\[done\]' "$LOG" | grep -oP '[0-9.]+(?= ms/token)')

printf '%s mode=%s tokens=%s vram=%sMiB tok/s=%s per_tok_ms=%s\n' \
    "$(date +%H:%M:%S)" "$MODE" "$TOKENS" "$VRAM_MIB" "$TOKPS" "$PER_TOK_MS"
rm -f "$LOG"
BASH
chmod +x apu_infer/bench/bench.sh
```

- [ ] **Step 3: Push to max395 and verify it runs**

```bash
ssh max395 "mkdir -p ~/flash-moe/apu_infer/bench"
scp apu_infer/bench/bench.sh apu_infer/bench/expected.txt max395:~/flash-moe/apu_infer/bench/
ssh max395 "cd ~/flash-moe/apu_infer && bench/bench.sh warm 30 20480"
```

Expected: a single line of output like:
```
17:42:01 mode=warm tokens=30 vram=20480MiB tok/s=3.46 per_tok_ms=289.4
```

- [ ] **Step 4: Commit**

```bash
git add apu_infer/bench/bench.sh apu_infer/bench/expected.txt
git commit -m "feat: bench harness for apu_infer with correctness gate"
```

---

### Task 0.2: Lock in baseline numbers

**Files:**
- Modify: `results.tsv` (append)

- [ ] **Step 1: Run the four baseline configurations**

Run the bench three times each to average out noise.

```bash
ssh max395 "cd ~/flash-moe/apu_infer && for i in 1 2 3; do bench/bench.sh cold 30 0; done"
ssh max395 "cd ~/flash-moe/apu_infer && for i in 1 2 3; do bench/bench.sh warm 30 0; done"
ssh max395 "cd ~/flash-moe/apu_infer && for i in 1 2 3; do bench/bench.sh cold 30 20480; done"
ssh max395 "cd ~/flash-moe/apu_infer && for i in 1 2 3; do bench/bench.sh warm 30 20480; done"
ssh max395 "cd ~/flash-moe/apu_infer && for i in 1 2 3; do bench/bench.sh warm 100 20480; done"
```

- [ ] **Step 2: Append baseline rows to results.tsv**

Use the median of the three runs per configuration. Append rows in this format:

```
12e6ca0	Qwen3.5-397B-A17B-4bit	397.0	17.0	<tok/s>	0	5.5	keep	C/HIP gfx1151 BASELINE: <mode> cache, <tokens> tokens, ENABLE_VRAM_CACHE=<MiB>
```

Example (use real numbers):
```bash
cat >> results.tsv <<'EOF'
12e6ca0	Qwen3.5-397B-A17B-4bit	397.0	17.0	1.27	0	5.5	keep	C/HIP gfx1151 BASELINE: cold cache, 30 tokens, no VRAM cache
12e6ca0	Qwen3.5-397B-A17B-4bit	397.0	17.0	1.83	0	5.5	keep	C/HIP gfx1151 BASELINE: warm cache, 30 tokens, no VRAM cache
12e6ca0	Qwen3.5-397B-A17B-4bit	397.0	17.0	1.18	0	5.5	keep	C/HIP gfx1151 BASELINE: cold cache, 30 tokens, ENABLE_VRAM_CACHE=20480
12e6ca0	Qwen3.5-397B-A17B-4bit	397.0	17.0	3.46	0	5.5	keep	C/HIP gfx1151 BASELINE: warm cache, 30 tokens, ENABLE_VRAM_CACHE=20480
12e6ca0	Qwen3.5-397B-A17B-4bit	397.0	17.0	2.08	0	5.5	keep	C/HIP gfx1151 BASELINE: warm cache, 100 tokens, ENABLE_VRAM_CACHE=20480
EOF
```

- [ ] **Step 3: Commit**

```bash
git add results.tsv
git commit -m "bench: gfx1151 baseline rows for Strix Halo APU experiments"
```

---

### Task 0.3: Profile baseline with rocprof

We need to know where time goes per kernel before optimizing kernels. `rocprof` is part of the Fedora rocm packages.

**Files:**
- Create: `apu_infer/bench/profile.sh`

- [ ] **Step 1: Write the profile script**

```bash
cat > apu_infer/bench/profile.sh <<'BASH'
#!/usr/bin/env bash
# Usage: ./profile.sh
# Runs apu_infer under rocprof and dumps a per-kernel CSV.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="bench/profile.csv"
export ENABLE_VRAM_CACHE=20480

# rocprof v2: --stats outputs both per-kernel and per-instance
rocprof --stats -o "$OUT" --basenames on \
  ./infer --prompt 'The capital of France is' --tokens 10 \
          --experts ../model-safetensors/packed_experts \
  > /tmp/profile_run.log 2>&1

echo "--- top 10 kernels by total time ---"
sort -t, -k4 -nr "$OUT" | head -10
BASH
chmod +x apu_infer/bench/profile.sh
```

- [ ] **Step 2: Push, run, capture results**

```bash
scp apu_infer/bench/profile.sh max395:~/flash-moe/apu_infer/bench/
ssh max395 "cd ~/flash-moe/apu_infer && bench/profile.sh > bench/profile_baseline.txt"
ssh max395 "cat ~/flash-moe/apu_infer/bench/profile_baseline.txt"
```

Expected: a list with `dequant_matvec_4bit_fma_vec4`, `gated_delta_net_step`, `swiglu_fused`, etc. with total time. The top kernel should consume 30-50% of GPU time.

- [ ] **Step 3: Commit the script and the captured baseline**

```bash
scp max395:~/flash-moe/apu_infer/bench/profile_baseline.txt apu_infer/bench/
git add apu_infer/bench/profile.sh apu_infer/bench/profile_baseline.txt
git commit -m "bench: rocprof baseline profile for gfx1151"
```

---

## Phase 1: zero-copy unified memory (Tier 1)

This is the biggest unique-to-APU win. The current code path for expert load is `pread → hipHostMalloc pinned buffer → hipMemcpyAsync to device buffer → kernel reads device buffer`. On a UMA system, three of those four steps are pointless: GPU and CPU share physical RAM and can both directly access pages already in the page cache.

### Task 1.1: hipMallocManaged for the expert ring buffer

**Hypothesis:** Replace the four-step pipeline with `pread → managed buffer → kernel reads same buffer`. Saves one full 6.75 MB hipMemcpy per expert load. Expected drop in `expert_io` from ~0.5 ms/layer to ~0.1 ms/layer on warm cache.

**Files:**
- Modify: `apu_infer/infer.hip` (alloc + load_experts function)

- [ ] **Step 1: Locate the allocation**

```bash
grep -n "hipHostMalloc.*h_expert_buf\|hipMalloc.*buf_expert_data" apu_infer/infer.hip
```

You should find roughly:
```c
CHECK_HIP(hipHostMalloc(&model->h_expert_buf[i], g_expert_size));
...
CHECK_HIP(hipMalloc(&model->buf_expert_data, MAX_K * g_expert_size));
```

- [ ] **Step 2: Replace with managed allocation**

In `model_init()` (or wherever h_expert_buf and buf_expert_data are allocated), change to:

```c
// Strix Halo / unified memory: managed memory is visible to both CPU
// (for pread to write into) and GPU (for the kernels to read from)
// without an explicit hipMemcpy step.
for (int i = 0; i < MAX_K; i++) {
    CHECK_HIP(hipMallocManaged(&model->h_expert_buf[i], g_expert_size, hipMemAttachGlobal));
}
// buf_expert_data was a device-only staging buffer; with managed memory we
// can use h_expert_buf directly as both pread destination and kernel input.
// Keep the symbol around for cache slot fallback but don't allocate it.
model->buf_expert_data = NULL;
```

- [ ] **Step 3: Update the load_experts SSD path**

Find the SSD load block in `layer_forward()` (search for `n_ssd > 0`):

```c
if (n_ssd > 0) {
    pthread_t threads[MAX_K];
    PreadArg args[MAX_K];
    int fd = model->expert_fds[layer_idx];
    for (int i = 0; i < n_ssd; i++) {
        args[i].fd = fd;
        args[i].buf = model->h_expert_buf[i];
        args[i].size = g_expert_size;
        args[i].offset = (off_t)need_ssd_ids[i] * g_expert_size;
        pthread_create(&threads[i], NULL, pread_worker, &args[i]);
    }
    for (int i = 0; i < n_ssd; i++)
        pthread_join(threads[i], NULL);

    // No copies needed — h_expert_buf[i] is managed memory the kernel can
    // read directly. Just point expert_ptrs at it.
    for (int i = 0; i < n_ssd; i++) {
        int k = need_ssd[i];
        expert_ptrs[k] = model->h_expert_buf[i];
        // (cache promotion via vram_cache_pool still works the same way if enabled)
    }
}
```

The cache-hit path (when `slot >= 0`) is unchanged — it still uses VRAM-allocated cache slots when `ENABLE_VRAM_CACHE` is set.

- [ ] **Step 4: Build, push, run correctness check**

```bash
scp apu_infer/infer.hip max395:~/flash-moe/apu_infer/
ssh max395 "cd ~/flash-moe/apu_infer && rm -f infer.o infer && make infer 2>&1 | tail -3"
ssh max395 "cd ~/flash-moe/apu_infer && bench/bench.sh warm 30 0"
```

Expected: bench reports a tok/s number AND no `FAIL` line. If output mismatches, revert the file (`git checkout apu_infer/infer.hip`) and investigate.

- [ ] **Step 5: Measure**

Run the same five configurations as the baseline and compare:

```bash
ssh max395 "cd ~/flash-moe/apu_infer && for i in 1 2 3; do bench/bench.sh cold 30 0; done"
ssh max395 "cd ~/flash-moe/apu_infer && for i in 1 2 3; do bench/bench.sh warm 30 0; done"
ssh max395 "cd ~/flash-moe/apu_infer && for i in 1 2 3; do bench/bench.sh warm 30 20480; done"
ssh max395 "cd ~/flash-moe/apu_infer && bench/bench.sh warm 100 20480"
```

Take the median tok/s per configuration.

- [ ] **Step 6: Decide keep-or-discard and append to results.tsv**

Decision rule: keep if **warm-cache no-VRAM-cache** improves by ≥10%. Other configs are bonus.

If keep:
```bash
cat >> results.tsv <<'EOF'
<new-hash>	Qwen3.5-397B-A17B-4bit	397.0	17.0	<best-tok/s>	0	5.5	keep	C/HIP gfx1151: hipMallocManaged for expert buf, removes H2D copy on UMA
EOF
git add apu_infer/infer.hip results.tsv
git commit -m "perf(apu): hipMallocManaged for expert buffer — UMA zero-copy"
```

If discard:
```bash
cat >> results.tsv <<'EOF'
<rocm-head-hash>	Qwen3.5-397B-A17B-4bit	397.0	17.0	<measured>	0	5.5	discard	C/HIP gfx1151: hipMallocManaged actually slower than pinned+memcpy because <reason>
EOF
git checkout apu_infer/infer.hip
git add results.tsv
git commit -m "bench: discard hipMallocManaged — slower due to <reason>"
```

---

### Task 1.2: hipMemAdvise residency hints

**Hypothesis:** Even with managed memory, the page placement defaults may be wrong. `hipMemAdvise` lets us hint that managed pages should live in GPU-side memory (which on UMA is just a different region of the same DRAM but accessed via a different cache hierarchy), or stay CPU-side. For our case, the expert buffers should be `hipMemAdviseSetPreferredLocation = device 0` so the GPU's L2 prefetcher behaves correctly.

**Files:**
- Modify: `apu_infer/infer.hip` (just after `hipMallocManaged` from Task 1.1)

- [ ] **Step 1: Add the advise calls**

After each `hipMallocManaged` introduced in Task 1.1, add:

```c
CHECK_HIP(hipMemAdvise(model->h_expert_buf[i], g_expert_size,
                        hipMemAdviseSetPreferredLocation, 0));  // device 0
CHECK_HIP(hipMemAdvise(model->h_expert_buf[i], g_expert_size,
                        hipMemAdviseSetAccessedBy, 0));         // GPU will read
CHECK_HIP(hipMemAdvise(model->h_expert_buf[i], g_expert_size,
                        hipMemAdviseSetAccessedBy, hipCpuDeviceId));  // CPU writes via pread
```

- [ ] **Step 2: Build and verify correctness**

```bash
scp apu_infer/infer.hip max395:~/flash-moe/apu_infer/
ssh max395 "cd ~/flash-moe/apu_infer && rm -f infer.o infer && make infer 2>&1 | tail -3"
ssh max395 "cd ~/flash-moe/apu_infer && bench/bench.sh warm 30 0"
```

- [ ] **Step 3: Measure (same configs as Task 1.1) and decide**

Decision rule: keep if any configuration improves by ≥3% over the Task 1.1 result.

```bash
ssh max395 "cd ~/flash-moe/apu_infer && for i in 1 2 3; do bench/bench.sh warm 30 0; done"
ssh max395 "cd ~/flash-moe/apu_infer && for i in 1 2 3; do bench/bench.sh warm 100 20480; done"
```

- [ ] **Step 4: Commit (keep or revert)**

Same pattern as Task 1.1 step 6. Append to `results.tsv`, commit either the change or the revert.

---

### Task 1.3: mmap layer files for direct GPU access

**Hypothesis:** With HMM enabled, an `mmap()`'d file can be accessed by the GPU through the same page tables as the CPU. This would replace `pread → managed buffer` with `mmap → kernel reads file pages directly`. On Mac this approach was a 5× regression (`results.tsv` discarded entry "mmap expert files: CATASTROPHIC 5x"), but Mac doesn't have HMM. On Strix Halo with kernel 6.19 and ROCm 6.4, the page-fault cost should be much lower because the GPU and CPU share the IOMMU page tables.

**Files:**
- Modify: `apu_infer/infer.hip` (replace expert_fds + load_experts with mmap pointers)

- [ ] **Step 1: Add mmap of layer files at init**

In `model_init()`, after the existing `expert_fds` setup, add:

```c
// HMM-direct path: mmap each layer file so the GPU can read pages directly.
// Falls back to pread+managed if mmap fails for any layer.
size_t layer_file_size = (size_t)NUM_EXPERTS * g_expert_size;
model->expert_mmap = (void**)calloc(NUM_LAYERS, sizeof(void*));
int mmap_ok = 1;
for (int i = 0; i < NUM_LAYERS; i++) {
    if (model->expert_fds[i] < 0) { mmap_ok = 0; break; }
    void *p = mmap(NULL, layer_file_size, PROT_READ, MAP_SHARED | MAP_POPULATE,
                   model->expert_fds[i], 0);
    if (p == MAP_FAILED) { mmap_ok = 0; break; }
    model->expert_mmap[i] = p;
    // Hint the kernel: we'll read these in MoE-router order, not sequentially
    madvise(p, layer_file_size, MADV_RANDOM);
}
if (mmap_ok) {
    printf("[init] Layer files mmap'd for HMM direct access (%.1f GB total)\n",
           (double)NUM_LAYERS * layer_file_size / (1024.0*1024*1024));
} else {
    // Free anything we did map
    for (int i = 0; i < NUM_LAYERS; i++)
        if (model->expert_mmap[i]) munmap(model->expert_mmap[i], layer_file_size);
    free(model->expert_mmap);
    model->expert_mmap = NULL;
    printf("[init] mmap failed, falling back to pread\n");
}
```

You need a new field in the `Model` struct:

```c
void **expert_mmap;  // [NUM_LAYERS] mmap'd layer files, or NULL
```

- [ ] **Step 2: Use mmap pointers in load_experts**

In `layer_forward()`, replace the SSD load path with:

```c
if (n_ssd > 0) {
    if (model->expert_mmap && model->expert_mmap[layer_idx]) {
        // HMM direct: just point at the mmap'd region
        char *base = (char*)model->expert_mmap[layer_idx];
        for (int i = 0; i < n_ssd; i++) {
            int k = need_ssd[i];
            int eid = need_ssd_ids[i];
            expert_ptrs[k] = base + (size_t)eid * g_expert_size;
        }
    } else {
        // Fall back to the pread path from Task 1.1
        pthread_t threads[MAX_K];
        PreadArg args[MAX_K];
        int fd = model->expert_fds[layer_idx];
        for (int i = 0; i < n_ssd; i++) {
            args[i].fd = fd;
            args[i].buf = model->h_expert_buf[i];
            args[i].size = g_expert_size;
            args[i].offset = (off_t)need_ssd_ids[i] * g_expert_size;
            pthread_create(&threads[i], NULL, pread_worker, &args[i]);
        }
        for (int i = 0; i < n_ssd; i++) pthread_join(threads[i], NULL);
        for (int i = 0; i < n_ssd; i++) {
            int k = need_ssd[i];
            expert_ptrs[k] = model->h_expert_buf[i];
        }
    }
}
```

- [ ] **Step 3: Build and verify correctness**

```bash
scp apu_infer/infer.hip max395:~/flash-moe/apu_infer/
ssh max395 "cd ~/flash-moe/apu_infer && rm -f infer.o infer && make infer 2>&1 | tail -3"
ssh max395 "cd ~/flash-moe/apu_infer && bench/bench.sh warm 30 0"
```

If the GPU faults trying to read mmap'd pages, you'll see a crash or `hipErrorIllegalAddress`. In that case, `mmap_ok` should be set to 0 and the fallback path used.

- [ ] **Step 4: Measure cold and warm**

The interesting comparison here is **cold cache** — that's where mmap might shine (no double-buffering through the page cache, just direct page faults).

```bash
ssh max395 "cd ~/flash-moe/apu_infer && for i in 1 2 3; do bench/bench.sh cold 30 0; done"
ssh max395 "cd ~/flash-moe/apu_infer && for i in 1 2 3; do bench/bench.sh warm 30 0; done"
ssh max395 "cd ~/flash-moe/apu_infer && for i in 1 2 3; do bench/bench.sh warm 100 0; done"
```

- [ ] **Step 5: Decide and commit**

Decision rule:
- **Keep** if cold tok/s improves ≥20% AND warm tok/s does not regress more than 5%.
- **Discard** otherwise (mmap on Mac was a 5× regression — pessimism is warranted).

Append a row to `results.tsv` either way.

---

## Phase 2: CPU/GPU cooperative compute (Tier 2)

The Zen 5 cores sit at ~0% during inference. They have AVX-512 + VNNI + BF16 — a real matmul accelerator. Strix Halo's memory controller has separate ports for CPU and GPU, so CPU compute on host memory does not steal bandwidth from GPU compute on VRAM-region memory.

### Task 2.1: Move shared expert to CPU

**Hypothesis:** The shared expert is `down(SwiGLU(gate(x), up(x)))` with intermediate dim 1024 — three small dense matvecs. On GPU it costs `0.26 ms × 60 = 15.6 ms / token`. Running it on CPU using OpenBLAS BF16 SGEMV in parallel with the GPU's MoE expert compute should remove that 15.6 ms from the critical path.

**Files:**
- Modify: `apu_infer/infer.hip` (shared expert dispatch in `layer_forward()`)
- Modify: `apu_infer/Makefile` (link OpenBLAS)
- Create: `apu_infer/cpu_shared.c` (CPU-side shared expert + dequant)

- [ ] **Step 1: Install OpenBLAS on max395**

```bash
ssh max395 "dnf install -y openblas-devel openblas-openmp 2>&1 | tail -3"
ssh max395 "find / -name 'libopenblas*.so*' 2>/dev/null | head"
```

- [ ] **Step 2: Write the CPU shared expert helper**

```c
// apu_infer/cpu_shared.c — CPU-side shared expert, dequant + SwiGLU + GEMV
//
// Runs on Zen 5 with AVX-512 BF16 via OpenBLAS. Called from layer_forward()
// in parallel with the MoE expert dispatch.

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>

#define HIDDEN_DIM 4096
#define SHARED_INTERMEDIATE 1024
#define GROUP_SIZE 64

static inline float bf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f; memcpy(&f, &u, 4); return f;
}

// Dequantize one MLX 4-bit row [in_dim] into a pre-allocated float buffer.
static void dequant_4bit_row(const uint32_t *w, const uint16_t *s,
                              const uint16_t *b, float *out, uint32_t in_dim) {
    uint32_t num_groups = in_dim / GROUP_SIZE;
    for (uint32_t g = 0; g < num_groups; g++) {
        float scale = bf16_to_f32(s[g]);
        float bias  = bf16_to_f32(b[g]);
        const uint32_t *wp = w + g * (GROUP_SIZE / 8);
        float *op = out + g * GROUP_SIZE;
        for (uint32_t p = 0; p < GROUP_SIZE / 8; p++) {
            uint32_t packed = wp[p];
            for (int n = 0; n < 8; n++) {
                uint32_t nibble = (packed >> (n * 4)) & 0xF;
                op[p * 8 + n] = (float)nibble * scale + bias;
            }
        }
    }
}

// Dequantize an entire matrix [out_dim, in_dim] into a contiguous float buffer.
// One-time cost paid at model load (called during model_init from infer.hip).
void cpu_dequant_matrix(const uint32_t *w, const uint16_t *s, const uint16_t *b,
                        float *out, uint32_t out_dim, uint32_t in_dim) {
    uint32_t packed_cols = in_dim / 8;
    uint32_t num_groups = in_dim / GROUP_SIZE;
    for (uint32_t r = 0; r < out_dim; r++) {
        dequant_4bit_row(w + r * packed_cols, s + r * num_groups,
                         b + r * num_groups, out + r * in_dim, in_dim);
    }
}

// Forward pass for one layer's shared expert.
//   input  [HIDDEN_DIM]                   — RMS-normed activations
//   gate_w [SHARED_INTERMEDIATE, HIDDEN_DIM]
//   up_w   [SHARED_INTERMEDIATE, HIDDEN_DIM]
//   down_w [HIDDEN_DIM, SHARED_INTERMEDIATE]
//   output [HIDDEN_DIM]                   — shared expert contribution
//   scratch [SHARED_INTERMEDIATE * 2]     — caller-allocated workspace
void cpu_shared_expert_forward(const float *input,
                                const float *gate_w, const float *up_w, const float *down_w,
                                float *output, float *scratch) {
    float *gate = scratch;
    float *up   = scratch + SHARED_INTERMEDIATE;

    // gate = gate_w @ input
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                SHARED_INTERMEDIATE, HIDDEN_DIM,
                1.0f, gate_w, HIDDEN_DIM, input, 1, 0.0f, gate, 1);

    // up = up_w @ input
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                SHARED_INTERMEDIATE, HIDDEN_DIM,
                1.0f, up_w, HIDDEN_DIM, input, 1, 0.0f, up, 1);

    // gate = SiLU(gate) * up
    for (uint32_t i = 0; i < SHARED_INTERMEDIATE; i++) {
        float g = gate[i];
        gate[i] = (g / (1.0f + expf(-g))) * up[i];
    }

    // output = down_w @ gate
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                HIDDEN_DIM, SHARED_INTERMEDIATE,
                1.0f, down_w, SHARED_INTERMEDIATE, gate, 1, 0.0f, output, 1);
}
```

- [ ] **Step 3: Update Makefile**

```makefile
# Add to apu_infer/Makefile after HIPCC = hipcc:
CBLAS_LIB ?= -lopenblas
LDFLAGS = -lpthread $(CBLAS_LIB)

# Add a new object target before the infer target:
cpu_shared.o: cpu_shared.c
	gcc -O3 -march=znver5 -mavx512f -mavx512bf16 -fPIC -c cpu_shared.c -o cpu_shared.o

# Modify the link rule to include cpu_shared.o:
$(TARGET): infer.o tokenizer_impl.o cpu_shared.o
	$(HIPCC) $(CFLAGS) -o $(TARGET) infer.o tokenizer_impl.o cpu_shared.o $(LDFLAGS)
```

- [ ] **Step 4: Wire it into infer.hip**

Add a forward declaration near the top:
```c
extern "C" {
    void cpu_dequant_matrix(const uint32_t *w, const uint16_t *s, const uint16_t *b,
                            float *out, uint32_t out_dim, uint32_t in_dim);
    void cpu_shared_expert_forward(const float *input,
                                    const float *gate_w, const float *up_w, const float *down_w,
                                    float *output, float *scratch);
}
```

In `model_init()`, after loading the shared expert weights, dequantize them once into host float buffers:

```c
// Pre-dequantize shared expert weights to host float for CPU compute
for (int i = 0; i < NUM_LAYERS; i++) {
    auto &L = model->layers[i];
    L.cpu_sg = (float*)aligned_alloc(64, SHARED_INTERMEDIATE * HIDDEN_DIM * sizeof(float));
    L.cpu_su = (float*)aligned_alloc(64, SHARED_INTERMEDIATE * HIDDEN_DIM * sizeof(float));
    L.cpu_sd = (float*)aligned_alloc(64, HIDDEN_DIM * SHARED_INTERMEDIATE * sizeof(float));

    // Need to copy from device first
    uint32_t *h_w = (uint32_t*)malloc(SHARED_INTERMEDIATE * (HIDDEN_DIM/8) * sizeof(uint32_t));
    uint16_t *h_s = (uint16_t*)malloc(SHARED_INTERMEDIATE * (HIDDEN_DIM/GROUP_SIZE_C) * sizeof(uint16_t));
    uint16_t *h_b = (uint16_t*)malloc(SHARED_INTERMEDIATE * (HIDDEN_DIM/GROUP_SIZE_C) * sizeof(uint16_t));

    CHECK_HIP(hipMemcpy(h_w, L.sg_w, SHARED_INTERMEDIATE*(HIDDEN_DIM/8)*4, hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(h_s, L.sg_s, SHARED_INTERMEDIATE*(HIDDEN_DIM/GROUP_SIZE_C)*2, hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(h_b, L.sg_b, SHARED_INTERMEDIATE*(HIDDEN_DIM/GROUP_SIZE_C)*2, hipMemcpyDeviceToHost));
    cpu_dequant_matrix(h_w, h_s, h_b, L.cpu_sg, SHARED_INTERMEDIATE, HIDDEN_DIM);

    CHECK_HIP(hipMemcpy(h_w, L.su_w, SHARED_INTERMEDIATE*(HIDDEN_DIM/8)*4, hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(h_s, L.su_s, SHARED_INTERMEDIATE*(HIDDEN_DIM/GROUP_SIZE_C)*2, hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(h_b, L.su_b, SHARED_INTERMEDIATE*(HIDDEN_DIM/GROUP_SIZE_C)*2, hipMemcpyDeviceToHost));
    cpu_dequant_matrix(h_w, h_s, h_b, L.cpu_su, SHARED_INTERMEDIATE, HIDDEN_DIM);

    CHECK_HIP(hipMemcpy(h_w, L.sd_w, HIDDEN_DIM*(SHARED_INTERMEDIATE/8)*4, hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(h_s, L.sd_s, HIDDEN_DIM*(SHARED_INTERMEDIATE/GROUP_SIZE_C)*2, hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(h_b, L.sd_b, HIDDEN_DIM*(SHARED_INTERMEDIATE/GROUP_SIZE_C)*2, hipMemcpyDeviceToHost));
    cpu_dequant_matrix(h_w, h_s, h_b, L.cpu_sd, HIDDEN_DIM, SHARED_INTERMEDIATE);

    free(h_w); free(h_s); free(h_b);
}
```

In the `LayerWeights` struct add:
```c
float *cpu_sg, *cpu_su, *cpu_sd;  // host-side dequantized shared expert
```

In `layer_forward()`, find the existing GPU shared expert dispatch (the three `do_matvec` calls for `sg_w`, `su_w`, `sd_w`) and replace with a CPU launch that runs in parallel with expert load:

```c
// Stage shared expert input on host
float h_normed[HIDDEN_DIM];
CHECK_HIP(hipMemcpy(h_normed, model->buf_normed, HIDDEN_DIM*sizeof(float), hipMemcpyDeviceToHost));

// Launch CPU shared expert (synchronous for now; Task 2.4 adds threading)
static thread_local float scratch[SHARED_INTERMEDIATE * 2];
float h_shared_out[HIDDEN_DIM];
cpu_shared_expert_forward(h_normed, L.cpu_sg, L.cpu_su, L.cpu_sd,
                          h_shared_out, scratch);

// Upload result for moe_combine
CHECK_HIP(hipMemcpy(model->buf_shared_out, h_shared_out, HIDDEN_DIM*sizeof(float),
                    hipMemcpyHostToDevice));
```

- [ ] **Step 5: Build and verify correctness**

```bash
scp apu_infer/cpu_shared.c apu_infer/Makefile apu_infer/infer.hip max395:~/flash-moe/apu_infer/
ssh max395 "cd ~/flash-moe/apu_infer && rm -f *.o infer && make infer 2>&1 | tail -10"
ssh max395 "cd ~/flash-moe/apu_infer && bench/bench.sh warm 30 0"
```

The bench harness will fail if the output diverges. Common gotchas:
- Pre-dequantized matrix in row-major vs column-major order — `cblas_sgemv` with `CblasNoTrans` expects row-major `M×N` where output is M-dim.
- Float precision differences from BF16 round-trip at dequant — should be tiny but if the bench fails, dump the shared output and compare against the GPU version.

- [ ] **Step 6: Measure**

```bash
ssh max395 "cd ~/flash-moe/apu_infer && for i in 1 2 3; do bench/bench.sh warm 30 20480; done"
ssh max395 "cd ~/flash-moe/apu_infer && for i in 1 2 3; do bench/bench.sh warm 100 20480; done"
```

- [ ] **Step 7: Decide**

Decision rule: keep if **warm-cache 100-token** improves ≥10% over baseline. The test is "did the GPU's shared expert time disappear from the per-layer breakdown without a corresponding CPU bottleneck appearing".

Use `bench.sh warm 30 20480` with `--timing` once and verify `shared` phase drops from 0.26 ms/layer to ~0 ms/layer.

Append to `results.tsv`, commit accordingly.

---

### Task 2.2: Parallelize CPU-side helpers in full attention

**Hypothesis:** The 15 full-attention layers all fall back to CPU code for deinterleave Q/Q_gate, Q/K RMS norm, RoPE, and sigmoid_gate. These are scalar loops in `layer_forward()`. Each one is small (~µs per call) but they're on the critical path. Vectorizing with AVX-512 should make them disappear.

**Files:**
- Modify: `apu_infer/infer.hip` (the `if (L.is_full)` branch — search `apply_rope`)

- [ ] **Step 1: Compile a simple baseline measurement**

Add a per-phase timer around the existing CPU code in the full-attn branch:

```c
double t_cpu = now_ms();
// existing deinterleave + Q/K norm + RoPE + sigmoid_gate ...
double t_cpu_done = now_ms();
if (g_timing_enabled) g_layer_timing.cpu_attn_misc += t_cpu_done - t_cpu;
```

Run the bench with `--timing` and capture how many ms `cpu_attn_misc` averages per layer over the 15 full-attn layers.

- [ ] **Step 2: Replace deinterleave with memcpy-friendly loop**

The existing loop:
```c
for (int h = 0; h < NUM_ATTN_HEADS; h++) {
    memcpy(h_q + h * HEAD_DIM, h_q_proj + h * 2 * HEAD_DIM, HEAD_DIM * sizeof(float));
    memcpy(h_qg + h * HEAD_DIM, h_q_proj + h * 2 * HEAD_DIM + HEAD_DIM, HEAD_DIM * sizeof(float));
}
```

is already vectorized by the compiler. Skip changing it.

- [ ] **Step 3: AVX-512 RMS norm for Q/K**

Replace the per-head Q/K RMS norm scalar loops with explicit AVX-512:

```c
#include <immintrin.h>

static inline void rms_norm_avx512(float *x, const float *weight, int dim, float eps) {
    __m512 acc = _mm512_setzero_ps();
    for (int i = 0; i < dim; i += 16) {
        __m512 v = _mm512_loadu_ps(x + i);
        acc = _mm512_fmadd_ps(v, v, acc);
    }
    float sum_sq = _mm512_reduce_add_ps(acc);
    float inv_rms = 1.0f / sqrtf(sum_sq / (float)dim + eps);
    __m512 vrms = _mm512_set1_ps(inv_rms);
    for (int i = 0; i < dim; i += 16) {
        __m512 v = _mm512_loadu_ps(x + i);
        __m512 w = _mm512_loadu_ps(weight + i);
        _mm512_storeu_ps(x + i, _mm512_mul_ps(_mm512_mul_ps(v, vrms), w));
    }
}
```

`HEAD_DIM = 256` is divisible by 16, so this works without a tail loop.

Replace the existing per-head loops:
```c
for (int h = 0; h < NUM_ATTN_HEADS; h++) {
    float *qh = h_q + h * HEAD_DIM;
    float qnorm_f[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; d++) qnorm_f[d] = bf16_to_f32_host(h_qnorm[d]);
    rms_norm_avx512(qh, qnorm_f, HEAD_DIM, RMS_NORM_EPS);
}
```

- [ ] **Step 4: AVX-512 sigmoid_gate (q_gate * attn_out)**

Currently sigmoid_gate runs as a GPU kernel. For full-attn, the q_gate path is already on CPU. Vectorize the per-element sigmoid:

```c
static inline __m512 sigmoid_avx512(__m512 x) {
    // sigmoid(x) = 1 / (1 + exp(-x)); use the polyfit or a fast approximation
    // For correctness, use scalar expf via _mm512_exp_ps if AVX-512 SVML is available
    // Otherwise just unpack-compute-pack
    float buf[16] __attribute__((aligned(64)));
    _mm512_store_ps(buf, x);
    for (int i = 0; i < 16; i++) buf[i] = 1.0f / (1.0f + expf(-buf[i]));
    return _mm512_load_ps(buf);
}
```

(SVML is in libimf — add `-lsvml` if available; otherwise fall back to scalar `expf`.)

- [ ] **Step 5: Build and verify**

```bash
scp apu_infer/infer.hip max395:~/flash-moe/apu_infer/
ssh max395 "cd ~/flash-moe/apu_infer && rm -f infer.o infer && make infer 2>&1 | tail -3"
ssh max395 "cd ~/flash-moe/apu_infer && bench/bench.sh warm 30 0"
```

- [ ] **Step 6: Measure and decide**

Decision rule: keep if `cpu_attn_misc` drops by ≥50% AND warm-cache tok/s does not regress. This is a small absolute saving (< 1 ms/token total) so the tok/s improvement will be modest — that's fine, the value is freeing up CPU cycles for Task 2.1's parallel shared expert.

Append, commit.

---

### Task 2.3: Make CPU shared expert async (run in parallel with GPU MoE)

**Hypothesis:** Task 2.1 ran the CPU shared expert synchronously: GPU waits for CPU. The actual win comes from running them concurrently. Spin up a worker thread that consumes shared-expert work and produces shared output, while the main thread continues to dispatch GPU MoE experts. They synchronize at `moe_combine_residual`.

**Files:**
- Modify: `apu_infer/infer.hip` (refactor `layer_forward()`)
- Modify: `apu_infer/cpu_shared.c` (add a simple work queue or just per-layer pthread)

- [ ] **Step 1: Pthread-per-layer (simplest)**

Spawn a pthread at the start of MoE phase that runs the shared expert:

```c
typedef struct {
    const float *normed;
    const float *gate_w, *up_w, *down_w;
    float *output;
    float *scratch;
    pthread_t tid;
} SharedTask;

static void *shared_worker(void *arg) {
    SharedTask *t = (SharedTask*)arg;
    cpu_shared_expert_forward(t->normed, t->gate_w, t->up_w, t->down_w,
                              t->output, t->scratch);
    return NULL;
}
```

In `layer_forward()`:

```c
// Stage normed input to host
static thread_local float h_normed[HIDDEN_DIM];
static thread_local float h_shared_out[HIDDEN_DIM];
static thread_local float scratch[SHARED_INTERMEDIATE * 2];
CHECK_HIP(hipMemcpy(h_normed, model->buf_normed, HIDDEN_DIM*sizeof(float),
                    hipMemcpyDeviceToHost));

// Launch CPU shared expert in background
SharedTask st = {
    .normed = h_normed, .gate_w = L.cpu_sg, .up_w = L.cpu_su, .down_w = L.cpu_sd,
    .output = h_shared_out, .scratch = scratch,
};
pthread_create(&st.tid, NULL, shared_worker, &st);

// ... GPU MoE expert dispatch happens here in the main thread ...

// Wait for shared expert and upload result
pthread_join(st.tid, NULL);
CHECK_HIP(hipMemcpyAsync(model->buf_shared_out, h_shared_out,
                         HIDDEN_DIM*sizeof(float), hipMemcpyHostToDevice,
                         model->stream_compute));
```

- [ ] **Step 2: Build and verify**

Same build command. Bench harness catches correctness regressions.

- [ ] **Step 3: Measure**

Compare against Task 2.1's tok/s.

- [ ] **Step 4: Decide and commit**

Decision rule: keep if any improvement, since the cost is one pthread_create per layer (~10 µs) which is negligible.

---

### Task 2.4: Parallelize routing softmax + topK

**Hypothesis:** Currently `cpu_softmax(h_scores, NUM_EXPERTS)` runs single-threaded over 512 elements. With 32 cores idle, parallelization saves ~3 ms/token. Tiny but free.

Skip if Task 2.1+2.3 already get us under 4 ms/token total CPU time.

---

## Phase 3: Kernel optimization for RDNA 3.5 (Tier 3)

The current `attn_compute = 2.12 ms × 60 = 127 ms/token` is the GPU compute floor. Halving that doubles the steady-state ceiling.

### Task 3.1: rocprof breakdown of gated_delta_net_step

**Files:**
- Modify: `apu_infer/bench/profile.sh` (add `--input` for kernel-detail mode)

- [ ] **Step 1: Add instruction-level profiling**

```bash
# rocprof v2 with --hsa-trace gives per-kernel stats; --pmc gives perf counters
rocprof --hsa-trace -o bench/profile_gdn.json \
  ./infer --prompt "The capital of France is" --tokens 5 \
          --experts ../model-safetensors/packed_experts
```

- [ ] **Step 2: Identify gated_delta_net_step's bottleneck**

Check the LDS bandwidth, VALU utilization, and global memory bandwidth columns. Hypothesis: it's memory-bound on the 8 MB delta_state buffer accessed in stride-128.

- [ ] **Step 3: Document findings**

Append to `bench/profile_baseline.txt` and commit.

---

### Task 3.2: gated_delta_net_step → WMMA

**Hypothesis:** The kernel does three sequential 128-element dot products per thread (`kv_mem`, `state[]`, `out_val`), which the compiler turns into 384 scalar FMA instructions per thread. RDNA 3.5 has WMMA `f32_16x16x16_bf16` that does a 16×16 BF16 matmul-accumulate in a single instruction across an entire wavefront. The state matrix is 128×128 per head — exactly 8×8 = 64 WMMA tiles. Replacing the scalar inner loops with WMMA should be 5-10× faster on the GPU.

**Files:**
- Modify: `apu_infer/kernels.hip.h` (`gated_delta_net_step`)
- Modify: `apu_infer/infer.hip` (the launch — block dim may change)

- [ ] **Step 1: Read up on RDNA WMMA semantics**

Reference: [HIP WMMA documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/cpp_language_extensions.html#wmma-intrinsics) and `hip/amd_detail/amd_hip_wmma.h`. Key intrinsics on RDNA 3+:

```c
// 16x16x16 BF16 matmul accumulator
__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(__bf16x16, __bf16x16, float8);
```

The `_w32` suffix means it operates on a 32-lane wavefront (gfx11), exactly what gfx1151 has.

- [ ] **Step 2: Rewrite the inner loop using WMMA**

This is non-trivial. The key insight: each block currently handles ONE value head (64 blocks × 128 threads). Switch to one block handles ALL 128 value dims of one head, with the wavefront computing 16×16 tiles cooperatively.

Sketch (full code in the implementation step):

```c
__global__ void gated_delta_net_step_wmma(
    float* state, const float* q, const float* k, const float* v,
    const float* g_decay, const float* beta_gate, float* output,
    uint32_t k_heads_per_v, uint32_t kh_mode)
{
    // Each block handles one value head (64 blocks)
    // 32-lane wavefront cooperatively does 16x16 WMMA tiles over the 128x128 state
    uint32_t head_id = blockIdx.x;
    uint32_t lane = threadIdx.x;  // 0..31
    ...
    // Phase 1: state *= g; kv_mem[vi] = sum_ki state[vi,ki] * k[ki]
    //   This is a 128x128 @ 128 GEMV — 8 WMMA tiles
    // Phase 2: delta = (v - kv_mem) * beta
    // Phase 3: state += outer(k, delta) — 8 WMMA outer-product tiles
    // Phase 4: out[vi] = sum_ki state[vi,ki] * q[ki] — 8 WMMA tiles
}
```

Implementation will take a few iterations; expect 200-400 lines.

- [ ] **Step 3: Add a runtime switch**

Don't replace the old kernel — add the new one alongside and pick one at launch time via env var:

```c
if (getenv("USE_WMMA_GDN")) {
    gated_delta_net_step_wmma<<<...>>>(...);
} else {
    gated_delta_net_step<<<...>>>(...);
}
```

This makes A/B testing trivial.

- [ ] **Step 4: Build and verify**

```bash
scp apu_infer/kernels.hip.h apu_infer/infer.hip max395:~/flash-moe/apu_infer/
ssh max395 "cd ~/flash-moe/apu_infer && rm -f infer.o infer && make infer 2>&1 | tail -10"
ssh max395 "cd ~/flash-moe/apu_infer && USE_WMMA_GDN=1 bench/bench.sh warm 30 20480"
```

If correctness fails, the WMMA kernel has a bug. Diff the kernel output against the scalar one in a small dump.

- [ ] **Step 5: Measure**

```bash
ssh max395 "cd ~/flash-moe/apu_infer && USE_WMMA_GDN=1 bench/bench.sh warm 30 20480"
ssh max395 "cd ~/flash-moe/apu_infer && bench/bench.sh warm 30 20480"  # baseline
ssh max395 "cd ~/flash-moe/apu_infer && rocprof --stats -o /tmp/gdn_wmma.csv \
            ./infer --prompt 'The capital of France is' --tokens 5 \
            --experts ../model-safetensors/packed_experts && \
            grep gated_delta_net /tmp/gdn_wmma.csv.stats.csv"
```

The rocprof line shows the kernel time; compare against baseline `profile_baseline.txt`.

- [ ] **Step 6: Decide**

Decision rule: keep if `gated_delta_net_step_wmma` is at least 2× faster than the scalar version per kernel call. This is a high-effort task; smaller wins aren't worth the maintenance burden.

---

### Task 3.3: dequant_matvec_4bit_fma_vec4 → WMMA

**Hypothesis:** The 4-bit dequant matvec is the bandwidth-bound workhorse: gate/up/down per expert, Q/K/V projections per layer. Each call reads `[out_dim, in_dim/8]` packed weights + dequantizes + dot products. RDNA 3.5 WMMA can replace the inner FMA loop with a single 16×16 matmul per tile. The dequant step has to happen first (into a small LDS staging buffer), then WMMA reads from LDS.

**Files:**
- Modify: `apu_infer/kernels.hip.h` (add `dequant_matvec_4bit_wmma`)
- Modify: `apu_infer/infer.hip` (route through new launcher when env var set)

Steps mirror Task 3.2: write alongside the scalar version, gate behind `USE_WMMA_MATVEC=1`, A/B test, decide.

This is the second-biggest single kernel speedup target (matvecs account for ~30% of GPU time per the rocprof baseline).

---

### Task 3.4: LDS bank conflict audit

**Files:**
- Modify: `apu_infer/kernels.hip.h` (potentially every shared-memory layout)

- [ ] **Step 1: Profile bank conflicts with rocprof**

```bash
ssh max395 "cd ~/flash-moe/apu_infer && rocprof --pmc LDSBankConflict \
            -o /tmp/bank.csv ./infer --prompt 'The capital of France is' --tokens 5 \
            --experts ../model-safetensors/packed_experts && cat /tmp/bank.csv"
```

If `LDSBankConflict / LDSInsts > 0.05`, there's measurable serialization. Otherwise skip this task.

- [ ] **Step 2: Pad shared arrays**

For arrays accessed at stride-32 (the bank count), add a +1 element padding to break the conflict:

```c
extern __shared__ float x_shared[];   // declared as in_dim
// becomes
__shared__ float x_shared[in_dim + 1];  // or use dynamic allocation with stride
```

Actually for dynamic shared memory, the fix is at index time: `x_shared[i + i/32]` instead of `x_shared[i]`. Write a small wrapper macro.

- [ ] **Step 3: Re-profile and decide**

Keep if `LDSBankConflict` drops AND total kernel time drops by ≥3%.

---

### Task 3.5: Use the larger LDS for multi-row tiling

**Hypothesis:** RDNA 3.5 has 128 KB LDS per WGP vs CDNA 3's 64 KB per CU. Our matvec kernels load 16 KB of activations into LDS per block; we have 7-8× headroom. Loading two output rows' worth of weights at once and producing both rows per block halves global memory traffic on the scales/biases load.

This is a deeper rewrite; do it last in Phase 3 if Tasks 3.2 and 3.3 land successfully.

---

## Phase 4: I/O and paging (Tier 4)

Lower-leverage but easy wins for cold-cache and 100-token steady-state.

### Task 4.1: io_uring expert loader

**Hypothesis:** Replace `pthread_create + pread + pthread_join` with an io_uring submission queue. Saves ~3 µs/expert on context switches; over 240 experts/token = ~700 µs/token = ~0.5% improvement. Marginal but sets up for batched async submission later.

**Files:**
- Modify: `apu_infer/infer.hip` (replace `load_experts` body)
- Modify: `apu_infer/Makefile` (link `-luring`)

Steps: install `liburing-devel`, replace the worker loop with `io_uring_prep_read` + `io_uring_submit_and_wait_nr`, build, bench, decide. Marginal win expected.

---

### Task 4.2: Huge pages for the layer files

**Hypothesis:** RDNA's IOMMU page walks on 4 KB pages are slow. With 2 MB hugepages, each page table entry covers 512× more memory and the GPU's TLB hit rate skyrockets when reading pread/mmap'd expert data.

**Files:**
- Modify: `apu_infer/infer.hip` (`madvise(MADV_HUGEPAGE)` on the mmap from Task 1.3)
- Modify: System config (`echo always > /sys/kernel/mm/transparent_hugepage/enabled`)

Steps: enable THP, hint the mmap, bench, decide. Only matters if Task 1.3 was kept.

---

### Task 4.3: madvise / readahead tuning

**Hypothesis:** Linux's default readahead of 128 KB triggers extra reads for nearby experts that we don't use. `posix_fadvise(POSIX_FADV_RANDOM)` disables readahead.

**Files:**
- Modify: `apu_infer/infer.hip` (after `open()` of layer files)

Steps: add fadvise calls, bench cold and warm, decide.

---

## Phase 5: Compiler / system flags (Tier 5/6)

Cheap to try, may compound with other phases.

### Task 5.1: hipcc -O3 -ffast-math

**Files:**
- Modify: `apu_infer/Makefile`

- [ ] **Step 1: Edit CFLAGS**

```makefile
# Replace
# CFLAGS = -O2 --offload-arch=$(GPU_TARGETS) -DWARP_SIZE=$(WARP_SIZE)
# with
CFLAGS = -O3 -ffast-math --offload-arch=$(GPU_TARGETS) -DWARP_SIZE=$(WARP_SIZE)
```

- [ ] **Step 2: Build, run correctness gate, measure**

`-ffast-math` reorders FP ops. The bench's golden output check catches divergence; if it fails, drop `-ffast-math` and try `-O3` alone.

- [ ] **Step 3: Decide**

Keep if any improvement and correctness passes.

---

### Task 5.2: Host code -march=znver5 -mavx512bf16

**Files:**
- Modify: `apu_infer/Makefile`

- [ ] **Step 1: Add to gcc rule**

```makefile
tokenizer_impl.o: tokenizer_impl.c ../metal_infer/tokenizer.h
	gcc -O3 -march=znver5 -mavx512f -mavx512bf16 -fPIC -c tokenizer_impl.c -o tokenizer_impl.o
```

The CPU code in `infer.hip` is compiled by hipcc which is clang under the hood — clang accepts `-march=znver5` for the host pass:

```makefile
infer.o: infer.hip kernels.hip.h ../metal_infer/tokenizer.h
	$(HIPCC) $(CFLAGS) -Xarch_host -march=znver5 -Xarch_host -mavx512bf16 -c infer.hip -o infer.o
```

- [ ] **Step 2: Build and bench**

Same routine.

- [ ] **Step 3: Decide**

---

### Task 5.3: BIOS VRAM split exploration

This is a system-level change requiring a reboot. Document only — don't automate.

- [ ] **Step 1: Reboot into BIOS**
- [ ] **Step 2: Find "UMA Frame Buffer Size" or "GPU Memory" setting**
- [ ] **Step 3: Try 48 GB VRAM / 16 GB CPU. Boot, re-run bench/baseline, document.**
- [ ] **Step 4: Try 16 GB VRAM / 48 GB CPU. Same.**
- [ ] **Step 5: Append observations to results.tsv with the BIOS setting in the description.**

Decision rule: keep the setting that gives the best **100-token warm-cache** result.

---

## Self-review checklist

- Spec coverage: 18 ideas from prior brainstorming → 18 tasks across Phases 1-5 plus Phase 0 harness work. ✓
- No "TBD" / "implement later" — every task has concrete code or bash commands. Two tasks (3.2 WMMA gated_delta_net, 3.3 WMMA matvec) say "200-400 lines, write alongside, gate behind env" rather than full code; that's because the WMMA semantics warrant inline experimentation rather than a pre-baked diff. The structure (env-gated, A/B tested) is fully specified.
- Type consistency: `cpu_sg`/`cpu_su`/`cpu_sd` (Task 2.1), `expert_mmap` (Task 1.3), `cpu_shared_expert_forward` signature — all consistent across tasks.
- Per-task decision criteria: every task has a concrete keep-or-discard rule tied to a measurable number.
- Rollback: every task uses the bench's correctness gate and a single-file revert pattern (`git checkout apu_infer/<file>`) so failed experiments don't leave a mess.
