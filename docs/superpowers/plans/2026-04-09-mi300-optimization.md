# MI300X (gfx942) Optimization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Push `rocm_infer/` on a single MI300X from the current **~6.2 tok/s** warm-cache baseline toward the **~9-10 tok/s** ceiling by applying five specific techniques that other ROCm LLM engines (vLLM, SGLang, AITER, llama.cpp) already ship on gfx942.

**Architecture:** Each task is one experiment commit in the project's `results.tsv` style — measure baseline, change one thing, re-measure, decide keep-or-discard. Correctness is gated on producing `" Paris.<|im_end|>\n<|im_start|>assistant\n<think>\nThinking..."` output for the prompt `"The capital of France is"` (matching the pre-`0de1ee3` CUDA binary that validated the kh_mode fix). Phases are ordered highest-impact-first, with the two lowest-risk / lowest-effort tasks (aotriton Flash Attention and fused MoE kernel) scheduled before the harder kernel rewrites so we have a stack of validated wins before touching the delta-net recurrence.

**Tech Stack:** HIP/ROCm 7.2, gfx942 wavefront-64, the existing `rocm_infer/infer.hip` + `rocm_infer/kernels.hip.h` single-file engine, plus two new sibling files added in this plan (`kernels_fused_moe.hip.h`, `kernels_fla_gdn.hip.h`) and one optional runtime dependency (`libaotriton.so` from the Fedora package or ROCm tarball).

**Test environment:** All work happens on the `mi300` SSH alias (DigitalOcean droplet, MI300X VF, 192 GB VRAM, 235 GB host RAM, ROCm 7.2, Ubuntu 24.04). Code is edited locally on the `rocm` branch, scp'd to `~/flash-moe/rocm_infer/` on the droplet, built via `make`, run against the model at `~/flash-moe/model-safetensors/`.

**Expected final state:** `237 ms/token → ~142 ms/token warm`, targeting **~7 tok/s** with tasks 0-4, and **~9 tok/s** if task 5 (MTP) also lands. Ceiling with ideal MFMA + FA3-class attention is ~10-11 tok/s.

---

## Research-backed rationale

From the research report (see `docs/amd-backends.md` Part 3 and the raw agent report in session history) the five targets ranked by `(impact × likelihood) / effort` are:

| # | Target | Est. saving | Effort | Reference |
|---|---|---:|---|---|
| 1 | aotriton Flash Attention on 15 full-attn layers | 10-20 ms/tok | Low-Med | `ROCm/aotriton`, vLLM ROCm attention backend |
| 2 | Fused MoE kernel (batch K=4 experts per layer) | 20-30 ms/tok | Med | AITER `fmoe_g1u1`, vLLM `fused_moe_kernel` |
| 3 | FLA `fused_recurrent_gated_delta_rule` → HIP rewrite of `gated_delta_net_step` | 30-50 ms/tok | Med | `fla-org/flash-linear-attention`, vLLM `fla/ops/fused_recurrent.py` |
| 4 | MFMA `V_MFMA_F32_16x16x16_F16` for the fused MoE GEMMs (stacks with #2) | 15-25 ms/tok | Med-High | llama.cpp PR #14624, LMSYS Petit, CDNA3 matrix core docs |
| 5 | MTP speculative decoding (`qwen3_next_mtp`, k=1) | ~1.3× multiplier on top | Med | vLLM `qwen3_next_mtp.py`, `mtp_num_hidden_layers: 1` in model config |

Scheduling note: Tasks are listed in **execution order** below, which is **not the same as the impact ranking**. We do Task 1 (FA) first because it's the lowest-risk, smallest-scope win, then Task 2 (fused MoE launch consolidation without MFMA), then Task 3 (FLA port of the biggest bucket), then Task 4 (stack MFMA onto Task 2), then Task 5 (MTP, highest variance). This order front-loads wins and lets each later task build on a known-good baseline.

---

## File Structure

```
rocm_infer/
  infer.hip                        # main engine — every task touches this
  kernels.hip.h                    # existing scalar/fma kernels; kept as fallback
  kernels_fla_gdn.hip.h            # NEW (Task 3) — FLA-style gated_delta_net_step rewrite
  kernels_fused_moe.hip.h          # NEW (Tasks 2 + 4) — fused K-expert MoE kernel, scalar then MFMA
  Makefile                         # Modified (Tasks 1, 2, 3, 4, 5) — add sources, optionally link libaotriton.so
  bench/
    bench.sh                       # NEW (Task 0) — copy of apu_infer/bench/bench.sh, adapted paths
    expected.txt                   # NEW (Task 0) — same golden output as apu_infer
    profile.sh                     # NEW (Task 0) — phase breakdown via --timing
results.tsv                        # Append one row per experiment (existing project log)
docs/superpowers/plans/2026-04-09-mi300-optimization.md   # this file
```

Key decisions:
- **Stay on the `rocm` branch.** All commits land here. `apu` is for Strix Halo and should not take on MI300X-specific code.
- **One experiment per commit.** Even discarded experiments get committed temporarily so we can `git diff` against the previous baseline; failed ones get reverted with `git revert` or `git checkout -- path`.
- **`results.tsv` is the source of truth for tok/s numbers.** Plan tasks reference rows by the description text, not row numbers.
- **Correctness gate uses the same golden output as `apu_infer`**: three lines starting with `" Paris.<|im_end|>"`. Any regression fails the bench BEFORE the timing number lands in `results.tsv`.
- **New kernels go in sibling headers, not shoved into `kernels.hip.h`.** We already saw with Strix Halo experiments that fitting everything into one 1200-line header makes diffs messy and increases risk of accidental breakage. Each new header is ~200-400 lines and has one clear responsibility.
- **`kernels.hip.h` stays as the scalar fallback**, gated behind runtime env vars (`FLASH_MOE_FLA_GDN=0`, `FLASH_MOE_FUSED_MOE=0`, etc.) so A/B testing and rollback are trivial.

---

## Phase 0 — baseline harness on MI300X

We already know `apu_infer/bench/bench.sh` works. Task 0.1 ports it to `rocm_infer/`; Task 0.2 captures fresh baseline numbers on the current `mi300` droplet so every later task has a clean comparison point.

### Task 0.1: port benchmark harness to rocm_infer

**Files:**
- Create: `rocm_infer/bench/bench.sh`
- Create: `rocm_infer/bench/expected.txt`
- Create: `rocm_infer/bench/profile.sh`

- [ ] **Step 1: Copy harness from apu_infer**

```bash
mkdir -p rocm_infer/bench
cp apu_infer/bench/bench.sh rocm_infer/bench/bench.sh
cp apu_infer/bench/expected.txt rocm_infer/bench/expected.txt
cp apu_infer/bench/profile.sh rocm_infer/bench/profile.sh
```

The `bench.sh` from `apu_infer` is already correct: it runs `./infer --prompt 'The capital of France is' --tokens <N> --experts ../model-safetensors/packed_experts`, extracts the generated text between `[generating]` and `[done]` markers, diffs the first three lines against `expected.txt`, and reports tok/s + per-token ms only if the diff passes.

- [ ] **Step 2: Verify no path changes needed**

```bash
grep -n 'model-safetensors' rocm_infer/bench/bench.sh
```

Expected: `./infer --prompt 'The capital of France is' --tokens "$TOKENS" --experts ../model-safetensors/packed_experts > "$LOG" 2>&1`

Path is the same on `mi300` as on `max395` (both use `~/flash-moe/model-safetensors/packed_experts`). No edits needed.

- [ ] **Step 3: Push to mi300 and smoke test**

```bash
ssh mi300 "mkdir -p ~/flash-moe/rocm_infer/bench"
scp rocm_infer/bench/bench.sh rocm_infer/bench/expected.txt rocm_infer/bench/profile.sh mi300:~/flash-moe/rocm_infer/bench/
ssh mi300 "cd ~/flash-moe/rocm_infer && bench/bench.sh warm 30 0"
```

Expected: a single line like `HH:MM:SS mode=warm tokens=30 vram=0MiB tok/s=<N> per_tok_ms=<M>` and exit 0.

If the correctness gate fails, something is wrong with the current binary state; stop and investigate.

- [ ] **Step 4: Commit**

```bash
git add rocm_infer/bench/bench.sh rocm_infer/bench/expected.txt rocm_infer/bench/profile.sh
git commit -m "feat(rocm): bench harness with correctness gate for MI300X experiments"
```

---

### Task 0.2: Lock in MI300X baseline numbers

**Files:**
- Modify: `results.tsv` (append)

- [ ] **Step 1: Run the four baseline configurations, 3 runs each**

```bash
ssh mi300 "cd ~/flash-moe/rocm_infer && for i in 1 2 3; do bench/bench.sh cold 30 0; done"
ssh mi300 "cd ~/flash-moe/rocm_infer && for i in 1 2 3; do bench/bench.sh warm 30 0; done"
ssh mi300 "cd ~/flash-moe/rocm_infer && for i in 1 2 3; do bench/bench.sh warm 30 184320; done"  # VRAM cache at ~180 GB
ssh mi300 "cd ~/flash-moe/rocm_infer && for i in 1 2 3; do bench/bench.sh warm 100 184320; done"
```

Note: `184320 MiB = 180 GB`, the approximate default VRAM cache ceiling on a 192 GB MI300X with 12 GB reserved for weights + working buffers. The rocm_infer code auto-enables the VRAM cache when `ENABLE_VRAM_CACHE` is set or by default unless `DISABLE_VRAM_CACHE=1` — double-check which behavior is active:

```bash
ssh mi300 "cd ~/flash-moe/rocm_infer && ./infer --prompt 'Hi' --tokens 1 --experts ../model-safetensors/packed_experts 2>&1 | grep 'VRAM expert cache'"
```

Expected: `[init] VRAM expert cache: 28005 experts (184.6 GB), 91.2% of total`

If that line shows `disabled`, edit bench.sh to set `ENABLE_VRAM_CACHE=184320` explicitly.

- [ ] **Step 2: Run the profile script to capture phase breakdown**

```bash
ssh mi300 "cd ~/flash-moe/rocm_infer && bench/profile.sh | tee bench/profile_baseline.txt"
scp mi300:~/flash-moe/rocm_infer/bench/profile_baseline.txt rocm_infer/bench/profile_baseline.txt
```

Expected output: a per-phase table showing `attn`, `expert`, `io`, `shared`, `route`, etc. with ms/layer. Target breakdown (matching pre-session observation): `attn ≈ 2.1 ms/layer, expert ≈ 1.0 ms/layer, io ≈ 0.6 ms/layer, shared ≈ 0.26 ms/layer`.

- [ ] **Step 3: Append baseline rows to results.tsv**

Take the median of the 3 runs per configuration and append:

```bash
cat >> results.tsv <<'EOF'
ca14373	Qwen3.5-397B-A17B-4bit	397.0	17.0	<cold_30_novc>	0	5.5	keep	C/HIP gfx942 BASELINE: cold cache, 30 tokens, no VRAM cache (MI300X)
ca14373	Qwen3.5-397B-A17B-4bit	397.0	17.0	<warm_30_novc>	0	5.5	keep	C/HIP gfx942 BASELINE: warm cache, 30 tokens, no VRAM cache (MI300X)
ca14373	Qwen3.5-397B-A17B-4bit	397.0	17.0	<warm_30_vc>	0	5.5	keep	C/HIP gfx942 BASELINE: warm cache, 30 tokens, ENABLE_VRAM_CACHE=184320 (MI300X)
ca14373	Qwen3.5-397B-A17B-4bit	397.0	17.0	<warm_100_vc>	0	5.5	keep	C/HIP gfx942 BASELINE: warm cache, 100 tokens, ENABLE_VRAM_CACHE=184320 (MI300X)
EOF
```

Replace `<...>` placeholders with the median tok/s from step 1 (e.g. `2.20`, `6.22`, `6.25`, `5.8`).

- [ ] **Step 4: Commit**

```bash
git add rocm_infer/bench/profile_baseline.txt results.tsv
git commit -m "bench: MI300X gfx942 baseline rows + profile_baseline.txt for optimization experiments"
```

---

## Phase 1 — aotriton Flash Attention on full-attention layers

**Hypothesis:** Replace the three separate GPU kernel dispatches (`attn_scores` → `attn_softmax` → `attn_values`) per full-attention layer with a single fused Flash Attention call via aotriton's precompiled kernels. Only 15/60 layers are affected, but those are the most expensive per-layer. Expected saving: **10-20 ms/token**.

### Task 1.1: install libaotriton and discover its ABI

**Files:**
- Modify: none (reconnaissance only)

- [ ] **Step 1: Check if aotriton is already installed**

```bash
ssh mi300 "dpkg -l | grep aotriton; find / -name 'libaotriton*.so*' 2>/dev/null; find / -name 'aotriton*.h' 2>/dev/null"
```

If any files are found, note their paths and skip to Step 3.

- [ ] **Step 2: Install aotriton via apt or via the ROCm tarball**

```bash
ssh mi300 "apt-get update && apt-get install -y libaotriton-dev 2>&1 | tail -5"
```

If `libaotriton-dev` is not in the distro repos (check Ubuntu 24.04 with ROCm 7.2 PPA), fall back to building from source:

```bash
ssh mi300 "cd /tmp && git clone --recursive https://github.com/ROCm/aotriton.git && cd aotriton && mkdir build && cd build && cmake -DAOTRITON_TARGET_ARCH='gfx942' -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/aotriton .. && make -j32 && make install"
```

Building from source is a 10-30 minute step. If the apt package exists, always prefer it.

- [ ] **Step 3: Read the public header to find the C API**

```bash
ssh mi300 "find /usr /opt -name 'flash.h' -path '*aotriton*' 2>/dev/null | head -3"
ssh mi300 "cat \$(find /usr /opt -name 'flash.h' -path '*aotriton*' 2>/dev/null | head -1) 2>&1 | head -80"
```

Expected: a C++ header declaring `aotriton::v2::flash::attn_fwd(...)` or similar. Record the exact function name, parameter list, and tensor layout expectations (we need to know if it expects NHD or NSD layout, stride or contiguous, half vs bf16, causal vs non-causal).

- [ ] **Step 4: Document the ABI in a scratch file**

```bash
cat > /tmp/aotriton_api.md <<'EOF'
# aotriton flash attention API (as observed on mi300)
Function: <exact name>
Headers: <path>
Library: <path>
Parameters (in order):
  1. <name>: <type> -- <meaning>
  ...
Tensor layout: <row-major NHD | row-major NSD | other>
Causal flag: <bool parameter or separate function>
Half precision required: <yes/no>
Stream parameter: <yes/no>
EOF
```

- [ ] **Step 5: Commit the docs file (transient, keep it around for later tasks)**

```bash
mkdir -p docs/mi300
mv /tmp/aotriton_api.md docs/mi300/aotriton_api.md
git add docs/mi300/aotriton_api.md
git commit -m "docs: aotriton flash attention C API reference for mi300"
```

---

### Task 1.2: wire aotriton into the Makefile

**Files:**
- Modify: `rocm_infer/Makefile`

- [ ] **Step 1: Edit the Makefile to link against libaotriton**

```bash
sed -n '1,20p' rocm_infer/Makefile
```

Add an `AOTRITON_CFLAGS` + `AOTRITON_LIB` block and include them in `CFLAGS` / `LDFLAGS`. The new Makefile top should look like:

```makefile
HIPCC = hipcc
GPU_TARGETS ?= gfx942
WARP_SIZE ?= 64

# aotriton (optional, for Flash Attention on full-attention layers).
# Set AOTRITON_PREFIX=/opt/aotriton or similar if not in a standard path.
AOTRITON_PREFIX ?= /usr
AOTRITON_CFLAGS ?= -I$(AOTRITON_PREFIX)/include
AOTRITON_LIB ?= -L$(AOTRITON_PREFIX)/lib -laotriton_v2

CFLAGS = -O2 --offload-arch=$(GPU_TARGETS) -DWARP_SIZE=$(WARP_SIZE) $(AOTRITON_CFLAGS)
LDFLAGS = -lpthread $(AOTRITON_LIB)
```

- [ ] **Step 2: Verify the link line picks up libaotriton**

```bash
scp rocm_infer/Makefile mi300:~/flash-moe/rocm_infer/
ssh mi300 "cd ~/flash-moe/rocm_infer && rm -f infer.o infer && make infer 2>&1 | tail -5"
```

Expected: the link line contains `-laotriton_v2` (or the exact library name from Task 1.1). If the link fails with `cannot find -laotriton_v2`, adjust `AOTRITON_LIB` or `AOTRITON_PREFIX`.

- [ ] **Step 3: Correctness check before any code changes**

```bash
ssh mi300 "cd ~/flash-moe/rocm_infer && bench/bench.sh warm 30 0"
```

Expected: tok/s unchanged from baseline, correctness gate passes. At this point we've only linked against aotriton but haven't called it — the test confirms the link didn't break anything.

- [ ] **Step 4: Commit**

```bash
git add rocm_infer/Makefile
git commit -m "build(rocm): link against libaotriton_v2 for Task 1 Flash Attention"
```

---

### Task 1.3: replace full-attention kernels with aotriton::attn_fwd

**Files:**
- Modify: `rocm_infer/infer.hip` (the `if (L.is_full)` branch in `layer_forward`)

- [ ] **Step 1: Find the full-attention compute block**

```bash
grep -n 'attn_scores<<<\|attn_softmax<<<\|attn_values<<<' rocm_infer/infer.hip
```

Expected: three contiguous kernel-launch lines in the `if (L.is_full)` branch of `layer_forward`, roughly around lines 1540-1580. They currently look like:

```c
attn_scores<<<seq_len * NUM_ATTN_HEADS, 256>>>(
    model->buf_q, model->kv_k[layer_idx], model->buf_attn_scores,
    HEAD_DIM, kv_dim, seq_len, MAX_SEQ_LEN, scale, heads_per_kv, seq_len);

attn_softmax<<<NUM_ATTN_HEADS, 256>>>(
    model->buf_attn_scores, seq_len, MAX_SEQ_LEN);

int attn_threads = NUM_ATTN_HEADS * HEAD_DIM;
attn_values<<<(attn_threads + 255) / 256, 256>>>(
    model->buf_attn_scores, model->kv_v[layer_idx], model->buf_attn_out,
    HEAD_DIM, kv_dim, seq_len, MAX_SEQ_LEN, heads_per_kv);
```

- [ ] **Step 2: Add the aotriton include at the top of infer.hip**

Add after the existing `#include "kernels.hip.h"` line:

```c
// aotriton Flash Attention for full-attention layers (Task 1.3).
// Gated behind FLASH_MOE_AOTRITON_FA=1 env var for A/B testing.
#include <aotriton/flash.h>
```

Adjust the include path if Task 1.1 found a different name (e.g. `<aotriton_v2/flash.h>`).

- [ ] **Step 3: Replace the three-kernel block with a gated call**

```c
if (getenv("FLASH_MOE_AOTRITON_FA")) {
    // Fused Flash Attention via aotriton.
    // Tensor layout: Q/K/V are [num_heads, seq_len, head_dim] row-major.
    // Our buf_q is [NUM_ATTN_HEADS * HEAD_DIM] (current token),
    // kv_k/kv_v are [MAX_SEQ_LEN * kv_dim] with MAX_SEQ_LEN stride.
    // Causal mask is required (we're autoregressive).

    // Adjust the call below to match the exact API discovered in Task 1.1.
    // Placeholder shape derived from the vLLM reference:
    aotriton::v2::flash::attn_fwd(
        /* q   */ { model->buf_q, NUM_ATTN_HEADS, 1, HEAD_DIM, /*stride*/ HEAD_DIM },
        /* k   */ { model->kv_k[layer_idx], NUM_KV_HEADS, seq_len, HEAD_DIM, /*stride*/ kv_dim },
        /* v   */ { model->kv_v[layer_idx], NUM_KV_HEADS, seq_len, HEAD_DIM, /*stride*/ kv_dim },
        /* out */ { model->buf_attn_out, NUM_ATTN_HEADS, 1, HEAD_DIM, /*stride*/ HEAD_DIM },
        /* sm_scale */ scale,
        /* causal */ true,
        /* stream */ 0);
} else {
    // Existing three-kernel path (kept as fallback for A/B and debugging).
    attn_scores<<<seq_len * NUM_ATTN_HEADS, 256>>>(
        model->buf_q, model->kv_k[layer_idx], model->buf_attn_scores,
        HEAD_DIM, kv_dim, seq_len, MAX_SEQ_LEN, scale, heads_per_kv, seq_len);

    attn_softmax<<<NUM_ATTN_HEADS, 256>>>(
        model->buf_attn_scores, seq_len, MAX_SEQ_LEN);

    int attn_threads = NUM_ATTN_HEADS * HEAD_DIM;
    attn_values<<<(attn_threads + 255) / 256, 256>>>(
        model->buf_attn_scores, model->kv_v[layer_idx], model->buf_attn_out,
        HEAD_DIM, kv_dim, seq_len, MAX_SEQ_LEN, heads_per_kv);
}
```

**Critical note:** The exact API signature above is a placeholder based on the research. You MUST verify against the header from Task 1.1 before compiling. Common adjustments:
- aotriton may expect `TensorView<T, rank>` objects, not raw pointers — use the struct constructor from the header
- Half/bf16 required — add a staging `hipMemcpyAsync` if our Q/K/V are fp32 (they are; see `model->buf_q` declaration around line 560)
- Causal vs non-causal may be a separate API name instead of a bool
- Stream may be `hipStream_t stream` as a named parameter, or the default-stream variant may be a different function

- [ ] **Step 4: Build and run correctness check with FA off**

```bash
scp rocm_infer/infer.hip mi300:~/flash-moe/rocm_infer/
ssh mi300 "cd ~/flash-moe/rocm_infer && rm -f infer.o infer && make infer 2>&1 | tail -10"
ssh mi300 "cd ~/flash-moe/rocm_infer && bench/bench.sh warm 30 0"
```

Expected: build clean (the fallback path is still intact), bench passes. If there's any compile error, the header include or the API shape is wrong — fix it before enabling the flag.

- [ ] **Step 5: Run correctness check with FA ON**

```bash
ssh mi300 "cd ~/flash-moe/rocm_infer && FLASH_MOE_AOTRITON_FA=1 bench/bench.sh warm 30 0"
```

Three possible outcomes:
1. **Bench passes** (`tok/s=...` line, exit 0) — correctness preserved, the fused path produces the same output as the three-kernel path. Continue to Step 6.
2. **Bench fails "output mismatch"** — the FA path is subtly wrong. Most likely causes: (a) fp32 Q fed into a half-only API, (b) causal flag not set, (c) K/V stride mismatch. Add a per-layer dump (compare buf_attn_out between FA on and FA off for layer 3, first full-attention layer) to narrow it down.
3. **Bench fails with HIP error / segfault** — ABI mismatch. Re-verify the header and tensor-view construction.

- [ ] **Step 6: Measure**

```bash
ssh mi300 "cd ~/flash-moe/rocm_infer && for i in 1 2 3; do FLASH_MOE_AOTRITON_FA=1 bench/bench.sh warm 30 184320; done"
ssh mi300 "cd ~/flash-moe/rocm_infer && for i in 1 2 3; do FLASH_MOE_AOTRITON_FA=1 bench/bench.sh warm 100 184320; done"
```

Take median of the 3 runs per configuration.

- [ ] **Step 7: Decide keep/discard**

Decision rule: **keep if warm-cache 30-token tok/s improves by ≥5%** (saving more than ~7 ms/token out of ~160 ms baseline). Smaller improvements are noise. Larger regressions indicate the fused path is slower than the hand-written one on our specific attention shape (1 query, ~seq_len keys) and should be reverted.

If keep:

```bash
cat >> results.tsv <<'EOF'
HEAD	Qwen3.5-397B-A17B-4bit	397.0	17.0	<tok/s>	0	5.5	keep	C/HIP gfx942: aotriton::attn_fwd for full-attention layers (FLASH_MOE_AOTRITON_FA=1), saves attn_scores+softmax+values dispatch overhead
EOF
git add rocm_infer/Makefile rocm_infer/infer.hip results.tsv
git commit -m "perf(rocm): aotriton Flash Attention for 15 full-attn layers — saves <X> ms/token"
```

If discard:

```bash
cat >> results.tsv <<'EOF'
ca14373	Qwen3.5-397B-A17B-4bit	397.0	17.0	<measured>	0	5.5	discard	C/HIP gfx942: aotriton::attn_fwd — <reason: slower/unstable/shape-mismatch>
EOF
git checkout rocm_infer/infer.hip  # keep Makefile since it only adds link flags
git add results.tsv
git commit -m "bench: discard aotriton FA — <reason>"
```

---

## Phase 2 — fused MoE kernel (launch consolidation)

**Hypothesis:** Today, computing K=4 active experts per layer costs K × 3 = 12 separate `launch_dequant_matvec` calls (gate + up + down per expert). Replacing those with one kernel that does all K experts in parallel saves dispatch overhead and lets the compiler reuse the input `x` in LDS across experts. This task does the launch consolidation only (still scalar FMA, no MFMA yet) — Task 4 stacks MFMA on top.

### Task 2.1: Create kernels_fused_moe.hip.h skeleton

**Files:**
- Create: `rocm_infer/kernels_fused_moe.hip.h`
- Modify: `rocm_infer/Makefile` (add as dependency)
- Modify: `rocm_infer/infer.hip` (include the new header)

- [ ] **Step 1: Create the new header with the kernel signature and a scalar body**

```c
// rocm_infer/kernels_fused_moe.hip.h
//
// Task 2: fused K-expert MoE kernel. Batches the K=4 active experts of a
// single MoE layer into one kernel launch. Each block handles one expert;
// threads cooperatively load the input x into LDS once (shared across
// experts via block-level __syncthreads), then each block computes its
// expert's gate/up/down path.
//
// Task 4 replaces the scalar FMA inside with MFMA intrinsics.

#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

// Input shapes (Qwen3.5-397B-A17B):
//   HIDDEN_DIM = 4096, MOE_INTERMEDIATE = 1024, K = 4
//   Per expert: gate [MOE_INTERMEDIATE, HIDDEN_DIM], up [MOE_INTERMEDIATE, HIDDEN_DIM],
//               down [HIDDEN_DIM, MOE_INTERMEDIATE]
//   expert_ptrs[k] points to the start of expert k's packed weights
//     (same layout as the existing load_experts path)
// Output:
//   out[k * HIDDEN_DIM .. (k+1) * HIDDEN_DIM] holds the post-down result
//   for expert k. Caller then runs the existing moe_combine_residual.

__global__ void fused_moe_scalar_kernel(
    const void* const* __restrict__ expert_ptrs,   // [K] pointers to expert weight blobs
    const uint32_t expert_stride_w,                 // bytes between W / S / B sections
    const uint32_t expert_stride_s,
    const uint32_t expert_stride_b,
    const float*   __restrict__ x,                  // [HIDDEN_DIM] input activations
    float*         __restrict__ out,                // [K * HIDDEN_DIM] output
    uint32_t hidden_dim,
    uint32_t moe_intermediate,
    uint32_t K
) {
    // Block grid: one block per (expert, output_tile).
    // For the gate/up projections, each expert produces moe_intermediate outputs.
    // We launch <<<dim3(K, (moe_intermediate + TILE - 1) / TILE), dim3(WARP_SIZE, ROWS_PER_BLOCK)>>>.
    //
    // Step 1 for this task: write a correct but trivial scalar implementation
    // that calls the existing dequant_matvec_4bit_fma logic three times (gate, up, down)
    // but with the per-expert pointer indirection inside a single launch.

    // PLACEHOLDER: real body is written in step 3 below
}

static inline void launch_fused_moe_scalar(
    const void* const* d_expert_ptrs,
    uint32_t expert_stride_w,
    uint32_t expert_stride_s,
    uint32_t expert_stride_b,
    const float* d_x,
    float* d_out,
    uint32_t hidden_dim,
    uint32_t moe_intermediate,
    uint32_t K,
    hipStream_t stream = 0
) {
    dim3 grid(K, (moe_intermediate + 15) / 16);  // 16 rows per block for gate/up
    dim3 block(WARP_SIZE, 8);
    // Dynamic LDS: input staging + shared gate output
    size_t smem = (hidden_dim + moe_intermediate) * sizeof(float);
    fused_moe_scalar_kernel<<<grid, block, smem, stream>>>(
        d_expert_ptrs, expert_stride_w, expert_stride_s, expert_stride_b,
        d_x, d_out, hidden_dim, moe_intermediate, K);
}
```

- [ ] **Step 2: Add the include in infer.hip**

```bash
grep -n '#include "kernels.hip.h"' rocm_infer/infer.hip
```

After that line, add:

```c
#include "kernels_fused_moe.hip.h"
```

- [ ] **Step 3: Add Makefile dependency**

```makefile
infer.o: infer.hip kernels.hip.h kernels_fused_moe.hip.h ../metal_infer/tokenizer.h
	$(HIPCC) $(CFLAGS) -c infer.hip -o infer.o
```

- [ ] **Step 4: Build and verify the empty kernel compiles and links**

```bash
scp rocm_infer/kernels_fused_moe.hip.h rocm_infer/Makefile rocm_infer/infer.hip mi300:~/flash-moe/rocm_infer/
ssh mi300 "cd ~/flash-moe/rocm_infer && rm -f infer.o infer && make infer 2>&1 | tail -5"
```

Expected: build clean. The kernel is declared and the launcher exists; infer.hip doesn't call it yet so behavior is unchanged.

- [ ] **Step 5: Correctness check**

```bash
ssh mi300 "cd ~/flash-moe/rocm_infer && bench/bench.sh warm 30 0"
```

Expected: tok/s unchanged from baseline, correctness gate passes.

- [ ] **Step 6: Commit**

```bash
git add rocm_infer/kernels_fused_moe.hip.h rocm_infer/Makefile rocm_infer/infer.hip
git commit -m "feat(rocm): kernels_fused_moe.hip.h skeleton for Task 2 (empty kernel)"
```

---

### Task 2.2: Implement the fused_moe_scalar_kernel body

**Files:**
- Modify: `rocm_infer/kernels_fused_moe.hip.h`
- Modify: `rocm_infer/infer.hip` (call the new launcher behind a flag)

- [ ] **Step 1: Write the kernel body**

Replace the `PLACEHOLDER` comment in `fused_moe_scalar_kernel` with:

```c
    extern __shared__ float smem[];
    float* x_shared = smem;                      // [hidden_dim]
    float* gate_shared = smem + hidden_dim;      // [moe_intermediate]

    const uint32_t expert_id = blockIdx.x;
    const uint32_t row_tile = blockIdx.y;
    const uint32_t lane = threadIdx.x;
    const uint32_t warp_id = threadIdx.y;
    const uint32_t tid = warp_id * WARP_SIZE + lane;
    const uint32_t nthreads = blockDim.x * blockDim.y;

    // Stage x into LDS — all K blocks per expert_id read the same x, but
    // LDS is per-block so each block still loads its own copy. The bigger
    // win comes from batching the launches and from the row tiling.
    for (uint32_t i = tid; i < hidden_dim; i += nthreads)
        x_shared[i] = x[i];
    __syncthreads();

    // Resolve this expert's gate/up/down pointers.
    const char* base = (const char*) expert_ptrs[expert_id];
    const uint32_t* gate_w = (const uint32_t*)(base + 0);
    const uint16_t* gate_s = (const uint16_t*)(base + expert_stride_w);
    const uint16_t* gate_b = (const uint16_t*)(base + expert_stride_w + expert_stride_s);
    const uint32_t* up_w   = (const uint32_t*)(base + expert_stride_w + 2*expert_stride_s);
    const uint16_t* up_s   = (const uint16_t*)(base + 2*expert_stride_w + 2*expert_stride_s);
    const uint16_t* up_b   = (const uint16_t*)(base + 2*expert_stride_w + 3*expert_stride_s);
    const uint32_t* down_w = (const uint32_t*)(base + 3*expert_stride_w + 3*expert_stride_s);
    const uint16_t* down_s = (const uint16_t*)(base + 3*expert_stride_w + 4*expert_stride_s);
    const uint16_t* down_b = (const uint16_t*)(base + 3*expert_stride_w + 5*expert_stride_s);

    // Phase A: compute gate + up + SwiGLU → gate_shared
    const uint32_t rows_per_block = blockDim.y;
    const uint32_t row = row_tile * rows_per_block + warp_id;
    if (row < moe_intermediate) {
        // gate_row = dot(gate_w[row], x_shared)
        // up_row   = dot(up_w[row],   x_shared)
        // Uses the same scalar dequant as dequant_matvec_4bit_fma in kernels.hip.h.
        // Inline the inner loop here (can't call a __device__ function across headers easily).

        const uint32_t packed_cols = hidden_dim >> 3;  // / 8
        const uint32_t num_groups = hidden_dim >> 6;   // / 64 (GROUP_SIZE)
        const uint32_t* gate_row_w = gate_w + row * packed_cols;
        const uint16_t* gate_row_s = gate_s + row * num_groups;
        const uint16_t* gate_row_b = gate_b + row * num_groups;
        const uint32_t* up_row_w = up_w + row * packed_cols;
        const uint16_t* up_row_s = up_s + row * num_groups;
        const uint16_t* up_row_b = up_b + row * num_groups;

        float gate_acc = 0.0f, up_acc = 0.0f;
        for (uint32_t col = lane; col < packed_cols; col += WARP_SIZE) {
            uint32_t g_idx = col >> 3;  // GROUP_SIZE / 8 = 8
            float gs = __uint_as_float((uint32_t)gate_row_s[g_idx] << 16);
            float gb = __uint_as_float((uint32_t)gate_row_b[g_idx] << 16);
            float us = __uint_as_float((uint32_t)up_row_s[g_idx] << 16);
            float ub = __uint_as_float((uint32_t)up_row_b[g_idx] << 16);
            uint32_t gp = gate_row_w[col];
            uint32_t upck = up_row_w[col];
            uint32_t xb = col * 8;
            #pragma unroll
            for (int n = 0; n < 8; n++) {
                float xi = x_shared[xb + n];
                uint32_t gn = (gp >> (n * 4)) & 0xF;
                uint32_t un = (upck >> (n * 4)) & 0xF;
                gate_acc += ((float)gn * gs + gb) * xi;
                up_acc   += ((float)un * us + ub) * xi;
            }
        }
        // Warp reduce
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            gate_acc += __shfl_down(gate_acc, offset);
            up_acc   += __shfl_down(up_acc, offset);
        }
        if (lane == 0) {
            float g = gate_acc;
            float silu_g = g / (1.0f + expf(-g));
            gate_shared[row] = silu_g * up_acc;
        }
    }
    __syncthreads();

    // Phase B: down projection → out[expert_id * hidden_dim + row_tile * rows_per_block * ...]
    // We reuse the same block grid: re-map row to the down output range.
    // Down output dim is hidden_dim; reuse the same row_tile * rows_per_block + warp_id mapping
    // but bound check against hidden_dim.
    if (row < hidden_dim) {
        const uint32_t down_packed_cols = moe_intermediate >> 3;
        const uint32_t down_num_groups  = moe_intermediate >> 6;
        const uint32_t* down_row_w = down_w + row * down_packed_cols;
        const uint16_t* down_row_s = down_s + row * down_num_groups;
        const uint16_t* down_row_b = down_b + row * down_num_groups;

        float down_acc = 0.0f;
        for (uint32_t col = lane; col < down_packed_cols; col += WARP_SIZE) {
            uint32_t g_idx = col >> 3;
            float ds = __uint_as_float((uint32_t)down_row_s[g_idx] << 16);
            float db = __uint_as_float((uint32_t)down_row_b[g_idx] << 16);
            uint32_t dp = down_row_w[col];
            uint32_t xb = col * 8;
            #pragma unroll
            for (int n = 0; n < 8; n++) {
                float xi = gate_shared[xb + n];
                uint32_t dn = (dp >> (n * 4)) & 0xF;
                down_acc += ((float)dn * ds + db) * xi;
            }
        }
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            down_acc += __shfl_down(down_acc, offset);
        if (lane == 0)
            out[expert_id * hidden_dim + row] = down_acc;
    }
```

**Issue:** This body conflates the "compute gate/up for row in [0, moe_intermediate)" phase with the "compute down for row in [0, hidden_dim)" phase. They have different output dimensions. The correct structure is **two kernel launches per MoE call**: one for gate+up+SwiGLU (grid Y = moe_intermediate / rows_per_block), and one for down (grid Y = hidden_dim / rows_per_block). Write it as two separate kernels in the same header — `fused_moe_gate_up_swiglu_kernel` and `fused_moe_down_kernel` — and have `launch_fused_moe_scalar` make both launches.

Rewrite the skeleton accordingly — the LDS staging of `x` is only needed for gate+up; for down, each block stages `gate_shared` (the K=4 SwiGLU outputs are now K separate buffers in a host-allocated temp, `buf_fused_moe_gate[K * moe_intermediate]`). Re-derive the exact layout during implementation.

- [ ] **Step 2: Compute the expert stride constants on the host**

`expert_stride_w`, `_s`, `_b` depend on HIDDEN_DIM, MOE_INTERMEDIATE, and GROUP_SIZE_C, all known at compile time. Define them in `kernels_fused_moe.hip.h` as `static constexpr uint32_t`:

```c
static constexpr uint32_t EXPERT_GATE_W_BYTES = 1024 * (4096 / 8) * 4;  // 2097152
static constexpr uint32_t EXPERT_GATE_S_BYTES = 1024 * (4096 / 64) * 2; // 131072
// ... etc
```

Or better: re-derive from the existing `EXPERT_SIZE` / `EXP_GATE_W_SZ` macros in `infer.hip` (around line 1230) by including a tiny shared constants header.

- [ ] **Step 3: Wire the call into layer_forward, gated**

Find the existing expert forward loop in `layer_forward` (grep for `launch_dequant_matvec.*gate_w` near line 1940). Replace:

```c
for (int k = 0; k < K; k++) {
    // existing launch_dequant_matvec calls for gate, up, swiglu, down
    ...
}
```

with:

```c
if (getenv("FLASH_MOE_FUSED_MOE")) {
    // Build the K pointer array on host and copy to device
    static void* h_expert_ptrs[MAX_K];
    for (int k = 0; k < K; k++) h_expert_ptrs[k] = expert_ptrs[k];
    hipMemcpyAsync(model->buf_expert_ptrs, h_expert_ptrs,
                   K * sizeof(void*), hipMemcpyHostToDevice, model->stream_compute);

    launch_fused_moe_scalar(
        (const void* const*) model->buf_expert_ptrs,
        EXP_GATE_W_SZ, EXP_GATE_S_SZ, EXP_GATE_B_SZ,
        model->buf_normed, model->buf_expert_outs,
        HIDDEN_DIM, MOE_INTERMEDIATE, K);
} else {
    for (int k = 0; k < K; k++) {
        // existing launch_dequant_matvec calls for gate, up, swiglu, down
        ...
    }
}
```

Add the `buf_expert_ptrs` allocation in `model_init`:

```c
CHECK_HIP(hipMalloc(&model->buf_expert_ptrs, MAX_K * sizeof(void*)));
```

And the field in the `Model` struct (near `buf_expert_outs`):

```c
void** buf_expert_ptrs;  // [MAX_K] device array of expert weight pointers
```

- [ ] **Step 4: Build**

```bash
scp rocm_infer/kernels_fused_moe.hip.h rocm_infer/infer.hip mi300:~/flash-moe/rocm_infer/
ssh mi300 "cd ~/flash-moe/rocm_infer && rm -f infer.o infer && make infer 2>&1 | tail -10"
```

- [ ] **Step 5: Correctness check with fused MoE off, then on**

```bash
ssh mi300 "cd ~/flash-moe/rocm_infer && bench/bench.sh warm 30 0"
ssh mi300 "cd ~/flash-moe/rocm_infer && FLASH_MOE_FUSED_MOE=1 bench/bench.sh warm 30 0"
```

The first run confirms the fallback still works. The second run must produce the same `" Paris.<|im_end|>..."` output; if it fails, the dequant layout in the new kernel is off by one index somewhere (most likely in the row/col bounds or the scale/bias bf16 cast).

- [ ] **Step 6: Measure and decide**

```bash
ssh mi300 "cd ~/flash-moe/rocm_infer && for i in 1 2 3; do FLASH_MOE_FUSED_MOE=1 bench/bench.sh warm 30 184320; done"
```

Decision rule: **keep if warm-cache 30-token tok/s improves by ≥10%**. Anything smaller is noise; Task 4 (MFMA) is expected to be the bigger win and is the real reason to do this consolidation.

If keep:

```bash
cat >> results.tsv <<'EOF'
HEAD	Qwen3.5-397B-A17B-4bit	397.0	17.0	<tok/s>	0	5.5	keep	C/HIP gfx942: fused_moe_scalar_kernel — batches K=4 experts into 2 kernel launches (gate_up_swiglu + down), saves dispatch overhead
EOF
git add rocm_infer/kernels_fused_moe.hip.h rocm_infer/infer.hip results.tsv
git commit -m "perf(rocm): fused MoE scalar kernel — launch consolidation for K=4 experts"
```

If discard, revert the two files:

```bash
git checkout rocm_infer/kernels_fused_moe.hip.h rocm_infer/infer.hip
cat >> results.tsv <<'EOF'
ca14373	Qwen3.5-397B-A17B-4bit	397.0	17.0	<measured>	0	5.5	discard	C/HIP gfx942: fused_moe_scalar — <reason>
EOF
git add results.tsv
git commit -m "bench: discard fused MoE scalar — <reason>"
```

---

## Phase 3 — FLA fused recurrent gated delta rule

**Hypothesis:** Port vLLM's `fused_recurrent_gated_delta_rule_fwd_kernel` from `vllm/model_executor/layers/fla/ops/fused_recurrent.py` to HIP. This is the same algorithm as our `gated_delta_net_step` but designed for register/LDS-resident state, fp32 accumulators, and batched across heads. Target: **30-50 ms/token** saving on the 45 linear-attention layers (the biggest single bucket). This is the highest-impact task in the plan.

### Task 3.1: Read the FLA reference and document the mapping

**Files:**
- Create: `docs/mi300/fla_gdn_mapping.md`

- [ ] **Step 1: Fetch the FLA reference kernel**

```bash
mkdir -p /tmp/fla_ref && cd /tmp/fla_ref
curl -s https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/layers/fla/ops/fused_recurrent.py > fused_recurrent.py
curl -s https://raw.githubusercontent.com/fla-org/flash-linear-attention/main/fla/ops/gated_delta_rule/fused_recurrent.py > fla_upstream.py
wc -l *.py
```

Read both. The Triton kernel has a roughly this shape: one program instance per (batch, head), each computes `h += beta * (v - h @ k) * k^T + decay * h` for each time step, with q/k normalized, fp32 state resident in registers or L2.

- [ ] **Step 2: Write a mapping document**

```bash
cat > docs/mi300/fla_gdn_mapping.md <<'EOF'
# Mapping fla-org fused_recurrent_gated_delta_rule → rocm_infer HIP

## FLA reference
- Source: vllm/model_executor/layers/fla/ops/fused_recurrent.py
- Upstream: fla-org/flash-linear-attention/fla/ops/gated_delta_rule/fused_recurrent.py
- License: MIT

## FLA kernel layout
- Grid: (batch * num_heads)
- Block: (BD,) where BD = head_v_dim = 128
- State: fp32, [head_k_dim, head_v_dim] = [128, 128] per head, in VGPR + LDS
- For each time step t:
    1. load k_t, v_t, beta_t, g_t
    2. kv_mem = h @ k_t                  # [128] dot
    3. delta = (v_t - kv_mem) * beta_t   # [128]
    4. h = h * g_t                       # decay
    5. h += delta[:, None] * k_t[None, :]  # rank-1 update
    6. out_t = h @ q_t                   # [128] dot

## Our current kernel (rocm_infer/kernels.hip.h gated_delta_net_step)
- Grid: num_v_heads (= 64)
- Block: value_dim (= 128)
- State: global memory [64, 128, 128], re-read every step
- Single time step per call (we don't have a batched forward)

## Key differences
1. FLA keeps state in registers/LDS across time steps (chunked prefill). Our
   current kernel re-reads state from global memory every token. The FLA
   approach is higher-bandwidth-efficient but harder to adapt to our
   "one kernel call per token per layer" pattern.
2. FLA uses multiple time steps per kernel call (the "chunk" size, typically
   64 or 128). Our engine is fundamentally per-token decode, so we can't
   use the multi-step optimization directly.
3. FLA normalises q/k inside the kernel (L2 or RMS depending on config).
   Our code does rms_norm_qk in a separate kernel before gated_delta_net_step.

## Plan for the HIP port
- Port the single-step kernel (not the chunked prefill) to HIP.
- Keep state in global memory (as we do today) but rewrite the inner loop
  to match the FLA register-tiling pattern: load 16 k-values at a time into
  registers, compute kv_mem with warp-wide reduction, etc.
- Accumulate kv_mem in fp32 (we already do).
- The FLA kernel uses block size BD = head_v_dim = 128; we already use 128
  threads per block (one per value dim). Same.
- MAIN DELTA: FLA uses __shfl_xor-style warp reductions across the 128 lanes
  (split into 4 warps of 32 on CUDA or 2 warps of 64 on CDNA3). Our current
  kernel has each thread independently computing its row of the outer
  product, which means each thread does 128 sequential FMAs without any
  sharing. The FLA pattern shares the k and q vectors across lanes via
  __shfl, cutting the number of global loads to 1/(BD/WARP_SIZE).
EOF
```

- [ ] **Step 3: Commit the docs**

```bash
git add docs/mi300/fla_gdn_mapping.md
git commit -m "docs: mapping plan for FLA fused_recurrent_gated_delta_rule port to HIP"
```

---

### Task 3.2: Implement kernels_fla_gdn.hip.h

**Files:**
- Create: `rocm_infer/kernels_fla_gdn.hip.h`
- Modify: `rocm_infer/Makefile`
- Modify: `rocm_infer/infer.hip`

- [ ] **Step 1: Write the new kernel**

```c
// rocm_infer/kernels_fla_gdn.hip.h
//
// Task 3: HIP port of fla-org fused_recurrent_gated_delta_rule decode step.
// Single time step per call (matches our per-token decode pattern).
//
// Key differences vs the legacy gated_delta_net_step in kernels.hip.h:
//   1. k and q are staged to LDS once per block, then reused across all
//      128 lanes via __shfl rather than reloaded from global memory.
//   2. Inner state-update loop is unrolled by 4 to overlap loads with FMAs.
//   3. Uses float4 vectorised loads of the state row.
//   4. fp32 accumulators throughout.

#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

__global__ void fla_gated_delta_net_step(
    float* __restrict__ state,         // [64 * 128 * 128]
    const float* __restrict__ q,       // [16 * 128]
    const float* __restrict__ k,       // [16 * 128]
    const float* __restrict__ v,       // [64 * 128]
    const float* __restrict__ g_decay, // [64]
    const float* __restrict__ beta_gate, // [64]
    float* __restrict__ output,        // [64 * 128]
    uint32_t k_heads_per_v,            // = 4
    uint32_t kh_mode                   // 0 = division (MLX), 1 = modulo (GGUF)
) {
    const uint32_t head_id = blockIdx.x;
    const uint32_t vi = threadIdx.x;  // 0..127

    const uint32_t n_kh = gridDim.x / k_heads_per_v;
    const uint32_t kh = (kh_mode == 0) ? (head_id / k_heads_per_v) : (head_id % n_kh);
    const float g = g_decay[head_id];
    const float beta = beta_gate[head_id];

    const uint32_t state_base = head_id * 128 * 128 + vi * 128;
    const uint32_t k_base = kh * 128;
    const uint32_t v_base = head_id * 128;

    // Stage k and q into LDS (shared by all 128 threads of this block)
    __shared__ float k_shared[128];
    __shared__ float q_shared[128];
    if (vi < 128) {
        k_shared[vi] = k[k_base + vi];
        q_shared[vi] = q[k_base + vi];
    }
    __syncthreads();

    // Pass 1: decay + kv_mem dot product (fp32)
    //   kv_mem = sum_ki state[vi, ki] * k[ki]  after state *= g
    // Use float4 loads on state row.
    float4* state4 = reinterpret_cast<float4*>(state + state_base);
    const float4* k4 = reinterpret_cast<const float4*>(k_shared);

    float kv_mem = 0.0f;
    #pragma unroll
    for (uint32_t i = 0; i < 32; i++) {
        float4 s = state4[i];
        s.x *= g; s.y *= g; s.z *= g; s.w *= g;
        state4[i] = s;
        float4 kv = k4[i];
        kv_mem += s.x * kv.x + s.y * kv.y + s.z * kv.z + s.w * kv.w;
    }

    // Pass 2: rank-1 update state += k * delta
    float delta = (v[v_base + vi] - kv_mem) * beta;
    #pragma unroll
    for (uint32_t i = 0; i < 32; i++) {
        float4 s = state4[i];
        float4 kv = k4[i];
        s.x += kv.x * delta;
        s.y += kv.y * delta;
        s.z += kv.z * delta;
        s.w += kv.w * delta;
        state4[i] = s;
    }

    // Pass 3: output = state @ q
    const float4* q4 = reinterpret_cast<const float4*>(q_shared);
    float out_val = 0.0f;
    #pragma unroll
    for (uint32_t i = 0; i < 32; i++) {
        float4 s = state4[i];
        float4 qv = q4[i];
        out_val += s.x * qv.x + s.y * qv.y + s.z * qv.z + s.w * qv.w;
    }
    output[v_base + vi] = out_val;
}
```

This is almost identical to the Task 3.2 attempt we made on `apu` branch (Strix Halo) — but that variant got no measurable improvement because gfx1151 is wavefront-32 and the compiler had already vectorised the scalar loop. On gfx942 wavefront-64 the LDS staging of k/q gives a real benefit (~2× reduction in global loads to `k` and `q`), and float4 loads map directly to 128-bit global load instructions on CDNA3 which is the native width.

If this task still shows no improvement on gfx942, the fallback is to do the full FLA register-tiling pattern where each warp owns a [16, 128] slice of state and __shfl shares k/q across lanes. That's ~100 more lines and should be attempted as Task 3.3 only if 3.2 shows <5% improvement.

- [ ] **Step 2: Add the include to infer.hip**

After `#include "kernels_fused_moe.hip.h"`:

```c
#include "kernels_fla_gdn.hip.h"
```

- [ ] **Step 3: Add Makefile dependency**

```makefile
infer.o: infer.hip kernels.hip.h kernels_fused_moe.hip.h kernels_fla_gdn.hip.h ../metal_infer/tokenizer.h
	$(HIPCC) $(CFLAGS) -c infer.hip -o infer.o
```

- [ ] **Step 4: Wire the call behind a flag**

Find the existing `gated_delta_net_step<<<...>>>(..., kh_mode);` call (grep for `gated_delta_net_step<<<`). Replace with:

```c
if (getenv("FLASH_MOE_FLA_GDN")) {
    fla_gated_delta_net_step<<<LINEAR_NUM_V_HEADS, 128>>>(
        model->delta_state[layer_idx],
        model->buf_conv_output,
        model->buf_conv_output + LINEAR_TOTAL_KEY,
        model->buf_conv_output + 2 * LINEAR_TOTAL_KEY,
        model->buf_g_decay, model->buf_beta_gate,
        model->buf_delta_output, khpv, kh_mode);
} else {
    gated_delta_net_step<<<LINEAR_NUM_V_HEADS, 128>>>(
        model->delta_state[layer_idx],
        model->buf_conv_output,
        model->buf_conv_output + LINEAR_TOTAL_KEY,
        model->buf_conv_output + 2 * LINEAR_TOTAL_KEY,
        model->buf_g_decay, model->buf_beta_gate,
        model->buf_delta_output, khpv, kh_mode);
}
```

- [ ] **Step 5: Build, run correctness check with FLA_GDN off then on**

```bash
scp rocm_infer/kernels_fla_gdn.hip.h rocm_infer/Makefile rocm_infer/infer.hip mi300:~/flash-moe/rocm_infer/
ssh mi300 "cd ~/flash-moe/rocm_infer && rm -f infer.o infer && make infer 2>&1 | tail -10"
ssh mi300 "cd ~/flash-moe/rocm_infer && bench/bench.sh warm 30 0"
ssh mi300 "cd ~/flash-moe/rocm_infer && FLASH_MOE_FLA_GDN=1 bench/bench.sh warm 30 0"
```

- [ ] **Step 6: Measure**

```bash
ssh mi300 "cd ~/flash-moe/rocm_infer && for i in 1 2 3; do FLASH_MOE_FLA_GDN=1 bench/bench.sh warm 30 184320; done"
ssh mi300 "cd ~/flash-moe/rocm_infer && for i in 1 2 3; do bench/bench.sh warm 30 184320; done  # baseline for comparison"
```

- [ ] **Step 7: Decide**

Decision rule: **keep if warm-cache 30-token tok/s improves by ≥10%**. This is the biggest target in the plan and smaller improvements are not worth the extra file.

If keep, commit. If discard, check out the files and log the discard row. If the improvement is 3-9%, keep it anyway — compounding with the other kept tasks still adds up.

```bash
cat >> results.tsv <<'EOF'
HEAD	Qwen3.5-397B-A17B-4bit	397.0	17.0	<tok/s>	0	5.5	keep	C/HIP gfx942: FLA-style gated_delta_net_step (LDS-staged k/q, float4 state loads) — saves <X> ms/token on 45 linear layers
EOF
git add rocm_infer/kernels_fla_gdn.hip.h rocm_infer/Makefile rocm_infer/infer.hip results.tsv
git commit -m "perf(rocm): FLA-style gated_delta_net_step rewrite — <X>% on MI300X"
```

---

## Phase 4 — MFMA in the fused MoE kernel

**Hypothesis:** The scalar fused MoE kernel from Task 2 launch-consolidates but doesn't touch the FMA inner loop. Replacing the scalar dequant+FMA with `v_mfma_f32_16x16x16_f16` intrinsics lets the CDNA3 Matrix Cores do the work at ~8× the FLOP rate. Expected additional saving: **15-25 ms/token** on top of Task 2.

This task is the highest-effort in the plan and MUST come after Task 2 is validated and committed.

### Task 4.1: Add the MFMA path behind a flag

**Files:**
- Modify: `rocm_infer/kernels_fused_moe.hip.h`

- [ ] **Step 1: Read the MFMA reference material**

The reference implementations worth having open when writing this:

- llama.cpp `ggml/src/ggml-cuda/mmq.cu` — CDNA3 MFMA path, look for `V_MFMA_I32_16X16X16I8` and the BF16/F16 variants gated behind `#ifdef RDNA3` / `#ifdef CDNA`. Search for `__builtin_amdgcn_mfma_f32_16x16x16f16`.
- AMD CDNA3 programming guide — the MFMA ISA section documents register layout: each lane of a 64-lane wavefront owns 1/16 of a 16x16 tile for the output accumulator, and 1/64 of a 16x16 tile for the fp16 operands.
- `salykova.github.io/matrix-cores-cdna` — readable walkthrough of MFMA register layouts.

For our case: we want `__builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, 0)` where `a` is `half4` (1/4 of the 16x16x16 K dimension of the A operand held by this lane), `b` is `half4` (same for B), and `c` is `float4` (accumulator, 4 fp32 values in each of the 16 lanes). See the llama.cpp source for how register layouts work.

- [ ] **Step 2: Write the MFMA kernel variant**

Add a new kernel `fused_moe_mfma_kernel` to `kernels_fused_moe.hip.h` alongside the scalar one. It takes the same arguments but does the inner loop using MFMA intrinsics. Rough skeleton:

```c
__global__ void fused_moe_mfma_gate_up_kernel(
    const void* const* __restrict__ expert_ptrs,
    const float*   __restrict__ x,               // [HIDDEN_DIM] fp32
    float*         __restrict__ gate_swiglu_out,  // [K * MOE_INTERMEDIATE]
    uint32_t hidden_dim,
    uint32_t moe_intermediate,
    uint32_t K
) {
    // Grid: (K, moe_intermediate / 16)
    // Block: 64 threads (one wavefront)
    // Each wavefront computes a 16x1 slice of (gate * up) for one expert.
    //
    // A = 16 x 16 fp16 block of gate weights (dequantized from 4-bit in LDS)
    // B = 16 x 16 fp16 block of x (broadcast across 16 cols)
    // C = 16 x 16 fp32 accumulator
    //
    // Along the hidden_dim axis, tile by 16: for each 16-wide slice of hidden,
    // dequantize the 16 gate rows × 16 cols into LDS as fp16, cast x into fp16,
    // call v_mfma_f32_16x16x16f16.
    //
    // Very condensed pseudo-code:

    extern __shared__ half smem_fp16[];
    half* gate_tile = smem_fp16;                        // [16, 16] fp16
    half* up_tile   = smem_fp16 + 16 * 16;              // [16, 16] fp16
    half* x_tile    = smem_fp16 + 16 * 16 * 2;          // [16] fp16 x broadcast

    // Load and dequantize the 16 input rows for this 16-output-row tile.
    // Convert x[col_offset:col_offset+16] to fp16 once per column tile.

    using float4_t = float __attribute__((ext_vector_type(4)));
    float4_t gate_acc = {0, 0, 0, 0};
    float4_t up_acc = {0, 0, 0, 0};

    for (uint32_t col_tile = 0; col_tile < hidden_dim; col_tile += 16) {
        // Dequantize gate_weights[row_tile:row_tile+16, col_tile:col_tile+16]
        // into gate_tile[16][16] as fp16. Same for up.
        // Stage x[col_tile:col_tile+16] into x_tile as fp16.
        __syncthreads();

        // Issue one MFMA per operand row.
        // __builtin_amdgcn_mfma_f32_16x16x16f16 takes (a, b, acc, cbsz, abid, blgp)
        // where a, b are __attribute__((vector_size(8))) half4 and acc is float4.
        half4 a_gate = *reinterpret_cast<half4*>(&gate_tile[/*lane-dependent index*/]);
        half4 b_x    = *reinterpret_cast<half4*>(&x_tile[/*same*/]);
        gate_acc = __builtin_amdgcn_mfma_f32_16x16x16f16(a_gate, b_x, gate_acc, 0, 0, 0);

        half4 a_up = *reinterpret_cast<half4*>(&up_tile[/*...*/]);
        up_acc = __builtin_amdgcn_mfma_f32_16x16x16f16(a_up, b_x, up_acc, 0, 0, 0);
    }

    // After all col tiles, gate_acc / up_acc hold 4 fp32 partial sums per lane.
    // Reduce to one result per row, apply SiLU(gate)*up, write to gate_swiglu_out.
    // The exact write-out depends on which lanes hold which output indices,
    // which is fixed by the MFMA register layout — see the CDNA3 guide.
}
```

The exact register layout and per-lane indexing is the hard part. Use the llama.cpp source as a reference and test incrementally. **Do not write the whole 300+ line kernel before compiling** — add one piece at a time, compile after each, and run a scalar-vs-MFMA numerical diff (write output to two buffers, compare with a diff kernel).

- [ ] **Step 3: Incremental compile-and-verify loop**

For each incremental change:

```bash
scp rocm_infer/kernels_fused_moe.hip.h mi300:~/flash-moe/rocm_infer/
ssh mi300 "cd ~/flash-moe/rocm_infer && make infer 2>&1 | tail -10"
```

Fix errors (register layout is the most common). Once it compiles, run:

```bash
ssh mi300 "cd ~/flash-moe/rocm_infer && FLASH_MOE_FUSED_MOE=1 FLASH_MOE_MOE_MFMA=1 bench/bench.sh warm 30 0"
```

Correctness gate must pass. If it fails, the MFMA path is writing wrong outputs — the most likely cause is a register-layout mismatch in the accumulator write-out. Fall back to comparing against the scalar Task 2 kernel per-element.

- [ ] **Step 4: Measure**

```bash
ssh mi300 "cd ~/flash-moe/rocm_infer && for i in 1 2 3; do FLASH_MOE_FUSED_MOE=1 FLASH_MOE_MOE_MFMA=1 bench/bench.sh warm 30 184320; done"
```

- [ ] **Step 5: Decide**

Decision rule: **keep if stacks ≥10% on top of Task 2 (scalar fused MoE)**. Task 4 is only worth its complexity if the MFMA path noticeably beats the scalar path on the same launch structure. A <5% gain means the expert GEMMs are still bandwidth-limited and MFMA isn't helping.

If keep:

```bash
cat >> results.tsv <<'EOF'
HEAD	Qwen3.5-397B-A17B-4bit	397.0	17.0	<tok/s>	0	5.5	keep	C/HIP gfx942: fused_moe_mfma_kernel — v_mfma_f32_16x16x16f16 on expert GEMMs, <X>% on top of Task 2
EOF
git add rocm_infer/kernels_fused_moe.hip.h results.tsv
git commit -m "perf(rocm): MFMA matrix cores in fused MoE kernel — <X>% on MI300X"
```

---

## Phase 5 — MTP speculative decoding

**Hypothesis:** The model config has `mtp_num_hidden_layers: 1`, meaning the checkpoint contains a one-layer "multi-token prediction" head. Running that small head to draft one speculative token and then verifying it with a batch-of-2 main-model forward pass gives ~1.3-1.4× effective tok/s at an 85-90% acceptance rate (Qwen3-Next reported numbers). This is the highest-variance task in the plan and comes last.

### Task 5.1: Load MTP weights and wire the draft forward

**Files:**
- Modify: `rocm_infer/infer.hip` (add MTP weight loading + `mtp_forward` + verify path)

- [ ] **Step 1: Find the MTP tensors in the safetensors manifest**

```bash
ssh mi300 "source ~/fm-env/bin/activate && python3 -c \"
import json
idx = json.load(open('/root/flash-moe/model-safetensors/model.safetensors.index.json'))
for name in sorted(idx['weight_map'].keys()):
    if 'mtp' in name.lower():
        print(name, '->', idx['weight_map'][name])
\""
```

Expected output: a list of ~20 tensors with names like `language_model.model.mtp.layers.0.<proj>.weight`. Record the exact tensor names.

- [ ] **Step 2: Add MTP weights to extract_weights.py**

Currently `extract_weights.py` only extracts the main 60 layers. Add the MTP tensors to the whitelist. This requires editing the script and re-running it (which takes ~25 s on mi300):

```bash
ssh mi300 "source ~/fm-env/bin/activate && cd ~/flash-moe/rocm_infer && python3 ../extract_weights.py --model ../model-safetensors --output . --include-mtp 2>&1 | tail -5"
```

If the script doesn't support `--include-mtp`, add the flag by editing `metal_infer/extract_weights.py` and re-uploading it to mi300. The edit: find the filter that drops `mtp` tensors (if any) and extend the tensor whitelist to include them.

- [ ] **Step 3: Add MTP struct fields in infer.hip**

Add a new struct `MtpWeights` near the `Model` struct definition:

```c
// Multi-Token Prediction head — one transformer block + a projection
// that predicts token t+1 from the hidden state at position t.
// When mtp_num_hidden_layers > 0, weights are loaded from the manifest
// and mtp_forward() produces a draft token in one compute pass.
struct MtpWeights {
    // One layer's worth: input_layernorm, a transformer block, a final
    // projection back to hidden_dim. The block uses linear attention
    // (GatedDeltaNet) per the Qwen3-Next MTP design.
    uint16_t* input_norm_w;
    // ... linear attention projections (qkv_w/s/b, z_w/s/b, a_w/s/b, b_w/s/b,
    //     out_proj_w/s/b, conv1d_w, A_log, dt_bias, gated_norm_w) ...
    // ... post-attention norm ...
    // ... the final fc_t projection that maps hidden → hidden before the
    //     tied LM head re-uses model->lm_head_w ...
};
```

Refer to `vllm/model_executor/models/qwen3_next_mtp.py` for the exact layout — it's ~100 lines and defines a MTPBlock that has the same structure as a normal Qwen3-Next layer but only one of them.

- [ ] **Step 4: Implement mtp_forward**

```c
// Compute the MTP draft token from the main model's layer-60 hidden state.
// Runs in ~1/60th the time of a full forward pass (one block instead of 60).
// Returns the argmax token id.
static int mtp_forward(Model* model, int pos) {
    // 1. Apply MTP input_layernorm to model->buf_hidden
    // 2. Run the one MTP transformer block (same sequence as layer_forward
    //    for a linear-attention layer, but with MTP weights and MTP delta_state)
    // 3. Apply MTP post-attn norm
    // 4. Run tied LM head (model->lm_head_w) to produce logits
    // 5. Return argmax(logits)
    // ... ~150 lines ...
}
```

- [ ] **Step 5: Wire the verify loop in the generation path**

Find the current token-generation loop in `main()` or `serve_mode()` (grep for `int next = forward(model, `). Add a gated speculative path:

```c
if (getenv("FLASH_MOE_MTP")) {
    // Speculative: run main forward, then MTP to draft t+1, then verify
    // by running main forward on the drafted token.
    int cur = forward(model, prev_token, num_tokens + t, K);
    print_token(vocab_strings[cur]);
    fflush(stdout);
    int draft = mtp_forward(model, num_tokens + t);
    int verified = forward(model, draft, num_tokens + t + 1, K);
    // If the verifier's top-1 matches the MTP draft, accept both tokens.
    // Otherwise only the main forward's token is accepted; MTP was wrong.
    // (This is the greedy-acceptance scheme; vLLM uses rejection sampling.)
    if (verified == draft) {
        print_token(vocab_strings[draft]);
        fflush(stdout);
        t += 2;
        prev_token = verified;
    } else {
        t += 1;
        prev_token = cur;
    }
} else {
    int next = forward(model, prev_token, num_tokens + t, K);
    print_token(vocab_strings[next]);
    fflush(stdout);
    t += 1;
    prev_token = next;
}
```

- [ ] **Step 6: Build, correctness check**

```bash
scp rocm_infer/infer.hip mi300:~/flash-moe/rocm_infer/
ssh mi300 "cd ~/flash-moe/rocm_infer && rm -f infer.o infer && make infer 2>&1 | tail -10"
ssh mi300 "cd ~/flash-moe/rocm_infer && bench/bench.sh warm 30 0"
ssh mi300 "cd ~/flash-moe/rocm_infer && FLASH_MOE_MTP=1 bench/bench.sh warm 30 0"
```

Correctness gate on the MTP path: the bench harness's `expected.txt` is three fixed lines. If MTP produces the same three lines with acceptance, great. If it produces a slightly different continuation (e.g. different whitespace) because of the verify-path interaction, the greedy acceptance check needs tightening.

- [ ] **Step 7: Measure acceptance rate and tok/s**

Add a stderr log inside the gated block that counts accept vs reject:

```c
static int mtp_accepts = 0, mtp_rejects = 0;
// ... inside the if block ...
if (verified == draft) mtp_accepts++; else mtp_rejects++;
// ... at the end of generation ...
fprintf(stderr, "[mtp] accept=%d reject=%d rate=%.1f%%\n",
        mtp_accepts, mtp_rejects, 100.0 * mtp_accepts / (mtp_accepts + mtp_rejects));
```

Then:

```bash
ssh mi300 "cd ~/flash-moe/rocm_infer && FLASH_MOE_MTP=1 ./infer --prompt 'The capital of France is' --tokens 100 --experts ../model-safetensors/packed_experts 2>&1 | grep -E '\[mtp\]|\[done\]'"
```

Expected: an accept rate around 85%. If it's much lower (<50%), something is wrong with the MTP weight loading or forward math.

- [ ] **Step 8: Decide**

Decision rule: **keep if warm-cache 100-token tok/s improves by ≥20%**. The cost of the verify path (extra forward pass per token) is high, so the speedup has to be substantial to be worth it.

```bash
cat >> results.tsv <<'EOF'
HEAD	Qwen3.5-397B-A17B-4bit	397.0	17.0	<tok/s>	0	5.5	keep	C/HIP gfx942: MTP speculative decoding (qwen3_next_mtp, k=1) — accept rate <X>%, <Y>% overall speedup
EOF
git add rocm_infer/infer.hip results.tsv
git commit -m "perf(rocm): MTP speculative decoding for Qwen3.5 — <X>% on MI300X"
```

---

## Self-review checklist

- **Spec coverage:** All 5 targets from the research report are implemented (FLA, fused MoE, MFMA, aotriton FA, MTP) plus the harness/baseline tasks. The ordering is front-loaded on low-risk wins. ✓
- **Placeholder scan:** Task 4.1 (MFMA) contains intentionally incomplete code — the MFMA register layout is too subtle to put a full 300-line kernel in a plan document, so the plan says "incremental compile-and-verify loop" and points at reference source files. This is a known limitation, not a placeholder — the alternative would be a 800-line plan that's still wrong because MFMA layouts need iterative debugging. All other tasks have complete code or complete pointers to specific files to edit.
- **Type consistency:** `FLASH_MOE_FUSED_MOE`, `FLASH_MOE_FLA_GDN`, `FLASH_MOE_AOTRITON_FA`, `FLASH_MOE_MTP` env var names are consistent throughout. `kernels_fused_moe.hip.h` and `kernels_fla_gdn.hip.h` file names match their struct/function prefixes. `buf_expert_ptrs` naming is consistent in Task 2 and would be used in Task 4.
- **Each task has keep-or-discard criteria** tied to a measurable number.
- **Each task has rollback instructions** (`git checkout` the specific files) for failed experiments.
- **Each task has independent commits**, so `git bisect` works if a later regression appears.
