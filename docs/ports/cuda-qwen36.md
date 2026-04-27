# Port: cuda-qwen36

> **Branch:** `cuda-qwen36`
> **Status:** working at ~7-8 tok/s on RTX 4090 with VRAM LRU expert cache.
> **Coarse summary:** see [`../STATUS.md`](../STATUS.md).
> **Source notes:** local working notes (Obsidian "Qwen3.6 Port" + `~/.claude/projects/-home-sergey-flash-moe/memory/project_qwen36_port.md`).

## Capability matrix (detailed) — model: Qwen3.6-35B-A3B-FP8

Cell values: ✅ verified  ·  🟡 partial / unstable  ·  ❌ broken / missing  ·  `—` N/A.

| # | Capability | Status | Notes |
|---|---|---|---|
| 1 | Model loads end-to-end | ✅ | 40 layers + globals; `~3.7 GB` persistent + `~1 GB` scratch |
| 2 | Greedy sampling | ✅ | `--greedy` flag |
| 3 | Top-p / temperature sampling | ✅ | `--temp 0.7 --top-p 0.9` |
| 4 | Tokenizer subprocess | ✅ | `kimi_tokenize.py serve --model-dir <path>` (generic across HF tokenizers; KIMI prefix historical) |
| 5 | Tool calling (JSON-stable) | ❌ | not implemented |
| 6 | Multi-turn agent loop | ❌ | not implemented |
| 7 | HTTP/SSE serve mode | ❌ | not ported (Qwen3.5-397B `infer.cu` has the OpenAI-compatible server — porting is on the wishlist) |
| 8 | Streaming SSD experts | ✅ | one 3.1 MB pread per active expert from `packed_experts/layer_N.bin` |
| 9 | VRAM/RAM LRU expert cache | ✅ | `q36::ExpertCache`, capacity × 3.1 MB pool, O(1) LRU; 6000 slots = 18.9 GB; 74% hit at warm steady-state |
| 10 | int4 quant (custom pack) | — | Qwen3.6-FP8 doesn't use it |
| 11 | FP8 e4m3 block-128 quant | ✅ | `qwen36::fp8_block_matvec`, max_abs `~1e-7` vs oracle |
| 12 | sym-int4 (compressed-tensors) | — | Kimi-only |
| 13 | RoPE (full) | — | Qwen3.6 uses partial only |
| 14 | RoPE (partial) | ✅ | `rope_partial_inplace` on first 64 of 256 dims (`partial_rotary_factor=0.25`, `rope_theta=1e7`) |
| 15 | RMSNorm (standard) | ✅ | gated variant uses `weight*normalized*silu(z)` |
| 16 | RMSNorm-plus-one variant | ✅ | `qwen36::rms_norm_bf16_plus_one` and `rms_norm_per_row_plus_one` for `(1+w)*normalized` semantics |
| 17 | GatedDeltaNet (linear attention) | ✅ | layers 0,1,2 then 4,5,6 …; chunked Q/K replicate 16→32 heads, k_heads_per_v=1 to avoid double-divide |
| 18 | Full attention with output-gate | ✅ | layers 3,7,11,…,39; `attn_output_gate=true`, sigmoid(gate) on output |
| 19 | MLA attention | — | Kimi-only |
| 20 | MoE routing (softmax+topK) | ✅ | router bf16 [256,2048] → softmax → top-K=8 → renorm |
| 21 | MoE routing (noaux_tc sigmoid+bias) | — | Kimi-only |
| 22 | Shared expert | ✅ | `sigmoid(shared_expert_gate(h)) * shared_expert_out` |
| 23 | Embed tensor `>2GB` load (chunked pread) | 🟡 | not verified; Qwen3.6 embed is `~1 GB` so the bug latent in Kimi (fixed there) doesn't bite. Worth verifying defensively — see seed issue. |
| 24 | Per-substep oracle validation | ✅ | `test_qwen36_linear_attn`, `test_qwen36_moe`, `test_qwen36_full_attn` |
| 25 | Per-layer L2 sanity vs oracle | ✅ | `test_qwen36_layer` at L2rel `~1-2%` |
| 26 | End-to-end top-1 bit-match vs oracle | ✅ | `test_qwen36_chain` — top-1 next-token bit-exact for all 5 prompt positions; final logits L2rel 2-6% |
| 27 | K-head replication correctness | ✅ | host-side repeat_interleave 16→32, downstream `k_heads_per_v=1` (bug discovered + fixed during port) |
| 28 | Conv1d output position handling | ✅ | `F.conv1d(padding=K-1=3)` → take `[:seq_len]` (positions 0..4, NOT 3..7) |
| 29 | Weight absorption | — | Kimi-only |
| 30 | Build target in Makefile | ✅ | `make infer_qwen36` (`-arch=sm_89`) |

## Models

| Model | Status | tok/s (warm) | Last verified |
|---|---|---|---|
| Qwen3.6-35B-A3B-FP8 | ✅ working | 7-8 (warm + VRAM cache 6000 slots) | 2026-04-24 (per source memory) |

## Known-good commit

`5198058 — qwen3.6: shared LRU expert cache in VRAM (--cache-experts N)`

## Run

```bash
./infer_qwen36 \
  --model-dir /home/user1/qwen36-fp8 \
  --packed-dir /home/user1/qwen36-fp8/packed_experts \
  --cache-experts 6000 \
  --prompt "..." --max-tokens 64 [--greedy | --temp 0.7 --top-p 0.9]
```

## Open work (from spec North Star — reliability before throughput)

The "reliability" bar for this port (multi-turn agent loop + small fixed eval) is **not yet met**. Cells 5-7 are the gap:
- HTTP/SSE serve mode (port from Qwen3.5-397B's `infer.cu`)
- Tool calling (JSON-stable)
- Multi-turn agent loop

Throughput optimization plans (CUDA stream overlap, prefill batching) are paused per the design spec until cells 5, 6, 7 turn ✅.

## Backlinks

- [`../STATUS.md`](../STATUS.md) — coarse project capability matrix
