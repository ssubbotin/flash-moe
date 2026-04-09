# Kernel-level profile derivation (Task 3.1)

Fedora 43 ROCm 6.4 ships no rocprof, so we derive kernel costs from the
engine's `--timing` phase timers plus static analysis of each kernel's
memory-bandwidth / FLOP intensity.

## Phase breakdown (Task 1.1 baseline, gfx1151, warm 20 GB VRAM cache)

```
phase     ms/layer   x60 tokens   % of 263 ms
attn         2.12     127.2        48%
expert       1.00      60.0        23%
io           0.58      34.8        13%
shared       0.26      15.6         6%
route        0.07       4.2         2%
oproj        0.02       1.2        <1%
norm         0.02       1.2        <1%
combine      0.01       0.6        <1%
```

## Drill-down on `attn` (127 ms/token, 48%)

`attn` aggregates all attention compute for both layer types:

- **45 linear-attention layers** (GatedDeltaNet):
  rms_norm → qkv_proj → z/a/b proj → conv1d_step → rms_norm_qk →
  compute_decay_beta → **gated_delta_net_step** → gated_rms_norm → o_proj
- **15 full-attention layers**:
  rms_norm → q/k/v proj → CPU deinterleave/Q-K norm/RoPE → attn_scores →
  attn_softmax → attn_values → sigmoid_gate → o_proj

Rough split based on kernel count and memory traffic:

- Linear-attn kernels (× 45 layers): ~1.5 ms/layer = **~68 ms/token**
  - dequant_matvec (qkv/z/a/b/out_proj): ~1.0 ms/layer (5 matvecs)
  - gated_delta_net_step: ~0.3 ms/layer (memory-bound 128×128 recurrence)
  - conv1d + rms_norm_qk + compute_decay_beta + gated_rms_norm: ~0.2 ms/layer
- Full-attn (× 15 layers): ~4 ms/layer = **~60 ms/token**
  - dequant_matvec (q/k/v/o proj): ~1.5 ms/layer
  - attn_scores + softmax + values: ~1 ms/layer
  - CPU deinterleave + RoPE + Q/K norm: ~0.5 ms/layer
  - Misc GPU dispatch + sync: ~1 ms/layer

Dominant cost: **dequant_matvec_4bit_fma_vec4** runs ~5 × 45 + 4 × 15 =
285 times per token across all attention projections.

## Drill-down on `expert` (60 ms/token, 23%)

Per layer, K=4 experts each run gate_proj + up_proj + down_proj =
3 matvecs. 60 layers × 4 experts × 3 matvecs = **720 matvec calls/token**
for MoE experts alone.

At ~80 µs/matvec the budget adds up: 720 × 80 µs = 57.6 ms/token. Matches
the measured 60 ms.

## Ranking optimization targets

| Target | Potential savings | Complexity |
|---|---|---|
| **dequant_matvec_4bit_fma_vec4 → WMMA** | Hits ~1005 calls/token (expert + attn projections). 30% speedup → ~20 ms/token | High |
| **gated_delta_net_step → WMMA** | Hits 45 calls/token, ~14 ms/token today. 50% speedup → ~7 ms/token | High |
| **`attn` full-attention CPU code** | ~7 ms/token, small saving (Task 2.2 — already skipped) | Low |
| **LDS bank conflicts in matvec** | If measurable, 3-5% speedup. Needs rocprof counters we don't have | Medium |
| **Larger LDS tiling (Task 3.5)** | 2x scales/biases bandwidth save on dequant matvec. Maybe 10% | Medium |

Task 3.3 (dequant matvec → WMMA) is the highest-value target because it
affects the most per-token kernel invocations. Task 3.2 (GDN → WMMA) is a
smaller but cleaner rewrite target. Will try 3.3 first.
