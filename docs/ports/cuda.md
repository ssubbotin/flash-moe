# Port: cuda

> **Branch:** `cuda`
> **Status:** stub — fill cells as capabilities are verified.
> **Coarse summary:** see [`../STATUS.md`](../STATUS.md).

## Capability matrix (detailed)

Cell values: ✅ verified  ·  🟡 partial / unstable  ·  ❌ broken / missing  ·  `—` N/A.

| # | Capability | Per-model status | Notes |
|---|---|---|---|
| 1 | Model loads end-to-end | | |
| 2 | Greedy sampling | | |
| 3 | Top-p / temperature sampling | | |
| 4 | Tokenizer subprocess | | |
| 5 | Tool calling (JSON-stable) | | |
| 6 | Multi-turn agent loop | | |
| 7 | HTTP/SSE serve mode | | |
| 8 | Streaming SSD experts | | |
| 9 | VRAM/RAM LRU expert cache | | |
| 10 | int4 quant (custom pack) | | |
| 11 | FP8 e4m3 block-128 quant | | |
| 12 | sym-int4 (compressed-tensors) | | |
| 13 | RoPE (full) | | |
| 14 | RoPE (partial) | | |
| 15 | RMSNorm (standard) | | |
| 16 | RMSNorm-plus-one variant | | |
| 17 | GatedDeltaNet (linear attention) | | |
| 18 | Full attention with output-gate | | |
| 19 | MLA attention | | |
| 20 | MoE routing (softmax+topK) | | |
| 21 | MoE routing (noaux_tc sigmoid+bias) | | |
| 22 | Shared expert | | |
| 23 | Embed tensor `>2GB` load (chunked pread) | | |
| 24 | Per-substep oracle validation | | |
| 25 | Per-layer L2 sanity vs oracle | | |
| 26 | End-to-end top-1 bit-match vs oracle | | |
| 27 | K-head replication correctness | | |
| 28 | Conv1d output position handling | | |
| 29 | Weight absorption (`W_Q_abs`, `W_O_abs`) | | |
| 30 | Build target in Makefile | | |

## Models

List models known to run on this port. Empty = none verified.

| Model | Status | tok/s (warm) | Last verified |
|---|---|---|---|
| | | | |

## Known-good commit

`<commit-hash> — short description of last working state>`

## Backlinks

- [`../STATUS.md`](../STATUS.md) — coarse project capability matrix
