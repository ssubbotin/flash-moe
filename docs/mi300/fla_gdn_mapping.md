# Mapping FLA fused_recurrent_gated_delta_rule to rocm_infer HIP port

## References

- Upstream FLA kernel: fla-org/flash-linear-attention, `fla/ops/gated_delta_rule/fused_recurrent.py` (MIT)
- vLLM integration: vllm-project/vllm, `vllm/model_executor/layers/fla/ops/fused_recurrent.py` (Apache-2.0, contains MIT-licensed code from FLA)

## FLA reference kernel

### Inputs, shapes, and state layout

| Tensor | Shape | dtype |
|--------|-------|-------|
| q | [B, T, H, K] | fp16/bf16 |
| k | [B, T, H, K] | fp16/bf16 |
| v | [B, T, HV, V] | fp16/bf16 |
| g (decay) | [B, T, HV] | fp16/bf16 |
| beta | [B, T, HV] or [B, T, HV, V] | fp16/bf16 |
| state h | [N, HV, K, V] or [N, HV, V, K] (transposed) | fp32 |

The FLA kernel supports Grouped Value Attention (GVA) where `HV >= H` and
`HV % H == 0`. In the Qwen3.5 model, H=16 (K heads), HV=64 (V heads), K=V=128.
The vLLM wrapper fixes the state layout as [N, HV, V, K] (transposed relative
to FLA upstream default of [K, V]).

### Grid and block dimensions

FLA upstream grid: `(NV, N*HV)` where `NV = ceil(V / BV)`.

With V=128 and BV set to `min(8, next_pow2(V)) = 8` (when no per-element gv),
this gives `NV = 16`. So each (batch×head) pair launches 16 Triton programs
in the V dimension.

Block shape: `[BV]` threads per program, where BV=8 (fp32 state row slice).

vLLM grid: `(NK, NV, N*HV)` — adds a K-tile axis. With K=128 and BK=128,
NK=1. BV is `min(next_pow2(V), 32) = 32`, so NV=4 for V=128.

One Triton program owns a slice `[BV]` of the V dimension for one head of
one sequence and processes all T time steps.

### Per-time-step math (inside the kernel loop)

Each Triton program maintains state `b_h` of shape `[BV, BK]` (or `[BK, BV]`
without transposition) in registers across loop iterations — this is the key
optimization.

Per time step:

1. Load `b_k[BK]`, `b_q[BK]` (broadcast from the K-head mapped to this V-head)
2. Load `b_v[BV]` (this program's V slice)
3. Apply optional L2 norm to q and k
4. Load scalar `b_g` (one per V-head per timestep); apply decay: `b_h *= exp(b_g)`
5. Compute `kv_mem[BV] = sum(b_h * b_k, axis=K)` (inner product of state rows with k)
6. Load `b_beta` (scalar or vector); compute correction: `b_delta[BV] = (b_v - kv_mem) * b_beta`
7. Outer update: `b_h += b_delta[:, None] * b_k[None, :]`
8. Compute output: `b_o[BV] = sum(b_h * b_q, axis=K)`
9. Store `b_o` to output

The decay `g` arrives as log-sigmoid values from the model; the kernel applies
`exp(g)` (or `exp2(g * log2e)`) to get the multiplicative factor.

The vLLM decode-specific kernel (`fused_recurrent_gated_delta_rule_packed_decode_kernel`)
also fuses the softplus + A_log gate computation inline, since vLLM controls
those projections.

### Key properties of the FLA kernel

1. **State lives in registers.** The entire `[BV, BK]` = `[8, 128]` or `[32, 128]`
   state slice stays in Triton registers across all T time steps within one
   program. No global memory round-trip per step. For BV=8, BK=128: 1024
   float32 = 4 KB per program, well within CDNA3 register file.

2. **k/q are broadcast to all programs sharing the same K-head.** In the FLA
   kernel, each Triton program loads k/q from the same K-head index (`i_h`)
   without any cross-program coordination. The compiler can choose to cache
   them in registers across the T loop.

3. **V dimension is parallelized across programs.** With BV=8-32, multiple
   programs handle disjoint V slices of the same state independently — no
   LDS needed between programs.

4. **Chunked prefill is the same kernel.** The same kernel handles T>1 (prefill)
   and T=1 (decode) with no branching at the Python dispatch level. Decode is
   just T=1 through the T-loop.

5. **Supports optional per-element gk/gv decays** in the upstream FLA version
   (key-wise or value-wise separate decay tensors). The vLLM wrapper uses only
   the scalar-per-head `g` path.

## Our current rocm_infer kernel (gated_delta_net_step)

### Location

`rocm_infer/kernels.hip.h:744`

### Model dimensions (compile-time constants)

| Constant | Value |
|----------|-------|
| LINEAR_NUM_V_HEADS | 64 |
| LINEAR_NUM_K_HEADS | 16 |
| LINEAR_KEY_DIM (K) | 128 |
| LINEAR_VALUE_DIM (V) | 128 |
| k_heads_per_v | 4 (= 64/16) |

State shape per layer: `[64, 128, 128]` float32 = 4 MB per GDN layer.

### Launch configuration

```
gated_delta_net_step<<<LINEAR_NUM_V_HEADS, 128>>>
```

Grid: `(64,)` — one block per V-head.
Block: `(128,)` — one thread per V row of the state (thread `vi` owns `state[head, vi, 0..127]`).

### Per-token math (HIP)

```c
// Thread vi owns state[head_id, vi, 0..127] = row vi of the 128x128 state matrix

// 1. Decay + kv_mem (Pass A — fused: state *= g, dot with k)
float kv_mem = 0.0f;
for (uint32_t ki = 0; ki < 128; ki++) {
    float s = state[state_base + ki] * g;
    state[state_base + ki] = s;           // write-back each element
    kv_mem += s * k[k_base + ki];
}

// 2. Delta
float delta = (v[v_base + vi] - kv_mem) * beta;

// 3. State outer update (Pass B)
for (uint32_t ki = 0; ki < 128; ki++)
    state[state_base + ki] += k[k_base + ki] * delta;

// 4. Output dot product (Pass C)
float out_val = 0.0f;
for (uint32_t ki = 0; ki < 128; ki++)
    out_val += state[state_base + ki] * q[k_base + ki];
output[v_base + vi] = out_val;
```

The kernel touches `state[state_base + ki]` three times across three loops:
once in pass A (read+write), once in pass B (read+write), once in pass C (read).
Each pass is a full 128-element scan of the state row with global memory traffic.

### Current performance on MI300X

From `rocm_infer/bench/profile_baseline.txt` (averaged over last 10 tokens):

- `attn` phase: 0.350 ms/layer, 21.00 ms per token across 60 layers
- `gated_delta_net_step` runs in 45 of the 60 layers (all GDN layers)
- The `attn` phase at 0.350 ms/layer covers delta-net recurrence plus the
  associated norm and projection kernels — not exclusively this kernel

### kh_mode parameter

The caller passes `kh_mode = (g_quant_format == 1) ? 1u : 0u`.

- `kh_mode = 0` (MLX / default): `kh = head_id / k_heads_per_v`
  V heads 0–3 share K head 0, V heads 4–7 share K head 1, etc. (chunked)
- `kh_mode = 1` (GGUF / llama.cpp): `kh = head_id % num_k_heads`
  V heads 0, 16, 32, 48 share K head 0, etc. (interleaved)

This must be preserved in any rewrite.

## Key differences vs FLA

| Aspect | FLA Triton | rocm_infer current | Implication for port |
|--------|-----------|-------------------|----------------------|
| State residence | Registers (entire BV×BK slice per program) | Global memory, re-read 3 times per token | Cannot replicate register storage for full 128×128 row — too large. But can reorganize passes to reduce global reads from 3 to 1. |
| k/q loading | Loaded once per time step from global, cached in registers for that step | Each thread reads k[ki] 3 times (once per loop pass) | Stage k[] and q[] into LDS once per kernel call; all 128 threads read from LDS. |
| V parallelism | Multiple programs handle disjoint BV slices; no cross-program coordination | One block = one full V-head (128 threads, each owns one row). Same net parallelism. | Structure is equivalent — just different BV tile size. |
| Passes over state | One fused loop per time step (all math in one pass over h) | Three separate loops per token | Merge into 2 passes: (A+B) fused, (C) separate — or 1 pass if state is reread only once. |
| Vectorized loads | Triton block pointers (compiler vectorizes) | Scalar float loads | Explicit float4 on state row: each thread reads 32 float4s = 128 floats. |
| Accumulator precision | fp32 for state, fp32 for kv_mem and out | fp32 throughout | Same |
| Warp width | Triton: 32 (NVIDIA) | CDNA3: 64 | Warp reductions for kv_mem/out_val need 64-lane `__shfl_down` (not 32) |
| Time steps per launch | All T in one kernel (passes T through loop) | 1 (called once per token) | We stay at 1 — decode-only. Cannot use multi-step optimization. |
| State layout (transpose) | Configurable [K,V] or [V,K] | [V,K] = state[head, vi, ki] | Match current layout; use float4 on ki (innermost) dimension. |
| g decay input | log-sigmoid, kernel calls exp() | Pre-computed float g (already sigmoid-passed) | Caller already computes g in float; no change needed in kernel. |
| k/q scale | scale = 1/sqrt(K), applied to q inside kernel | scale applied to q buffer by caller before launch (GGUF only) | No change — keep caller-side scaling. |

## Plan for the HIP port (Task 3.2 will implement)

1. **Keep state in global memory.** The state is 128×128 fp32 per head = 64 KB
   per head, 4 MB per layer. Registers and LDS are nowhere near enough.
   State will continue to live in the `delta_state` global buffer.

2. **Stage k[] and q[] in LDS once per block.** Each block processes one V-head.
   The 128 threads cooperatively load k[0..127] and q[0..127] into `__shared__`
   at kernel entry (one load per thread), then `__syncthreads()`. All inner loop
   iterations read from LDS, eliminating 255 redundant global k/q reads per
   token per head.

3. **Merge passes A and B.** The current code does:
   - Pass A: `state[vi][ki] *= g; kv_mem += state[vi][ki] * k[ki]`
   - Pass B: `state[vi][ki] += k[ki] * delta`
   These can run as a single loop since `delta` is known after Pass A completes
   (it depends on kv_mem which is a reduction across all vi threads). The
   sequence is: each thread computes kv_mem for its row, then a warp reduction
   gives per-thread delta, then a single merged loop writes `state[vi][ki] =
   (old * g + k[ki] * delta)`. This reduces state reads from 2×128 to 1×128
   for passes A+B.

4. **Use float4 vectorized loads on the state row.** Thread `vi` owns
   `state[head, vi, 0..127]`. Read as 32 float4s (`__builtin_amdgcn_raw_buffer_load`
   or plain `float4 *` cast). Write back as 32 float4s. Reduces load/store
   instruction count by 4×.

5. **Warp reductions for kv_mem and out_val must use 64-lane shuffles.**
   `__shfl_down_sync(0xffffffffffffffff, val, offset, 64)` — the mask is 64 bits
   wide on CDNA3. Each thread holds its own partial sum; the warp-level reduction
   produces one scalar (kv_mem or out_val) per warp. With 128 threads / 64-lane
   wavefronts, there are 2 warps per block; a shared memory accumulation step
   follows to sum across warps.

6. **Preserve kh_mode.** The key-head mapping logic (`kh = head_id / khpv` vs
   `kh = head_id % n_kh`) is a single expression computed once per block and
   determines which LDS slice of k/q to load. Preserve it unchanged.

## Open questions for Task 3.2

- [ ] Can passes A+B+C all be fused into a single pass over state rows? The
      challenge: kv_mem is a sum over vi for a fixed ki (it requires all threads'
      contributions), while the pass loop index is ki. kv_mem is actually a sum
      over ki for a fixed vi — but delta depends on kv_mem which is already
      per-thread (each thread vi has its own kv_mem). So yes: passes A and B
      can fuse per-thread. Pass C is a separate dot product after the outer
      update. Verify: can pass C be overlapped with LDS q broadcast in the same
      loop as pass A+B to save one state re-read?

- [ ] Does the AMD compiler auto-vectorize scalar state loads to BUFFER_LOAD_DWORDX4
      if we write plain float[] loops, or is explicit `float4` required to get
      vectorized instructions in the ISA?

- [ ] Should the state loop be unrolled by 4 (32 iterations of float4) or 8?
      CDNA3 has deep wavefront pipelines; unrolling might help hide L2 latency.
      Empirically measure.

- [ ] The kv_mem reduction crosses 128 threads using 2 wavefronts. Is
      warp-level `__shfl_down` + 2-element shared-mem accumulation the fastest
      path, or is a single `__syncthreads` + sequential reduction in shared
      memory preferable given CDNA3's LDS bandwidth?

- [ ] The FLA vLLM wrapper stores state in [N, HV, V, K] layout (V-major). Our
      current state buffer is `state[head_id * 128 * 128 + vi * 128 + ki]` which
      is the same layout: [HV, V, K]. Confirm both match before any layout
      change in Task 3.2.

- [ ] Should we add an optional `use_qk_l2norm_in_kernel` path (as FLA does)
      to save the separate `l2_norm_qk` kernel launch for GGUF models? Fusing
      would eliminate one kernel dispatch per GDN layer per token.
