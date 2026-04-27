# Flash-MoE — Project Status

> **Source of truth:** "what can this project do, on which backend, with which model."
> **Updated by:** PRs that change a capability cell.
> **Detail per port:** see `docs/ports/<port>.md`.

## Cell legend

| Symbol | Meaning |
|---|---|
| ✅ | shipped & tool-call-tested |
| 🟡 | works but partial / unstable |
| ❌ | broken / missing |
| `—` | not applicable for this combination |

## Coarse capability matrix

Columns are `<backend> × <model>`. Rows are user-visible capabilities.

| Capability | Metal × Qwen3.5-397B | Metal × Qwen3.6 | Metal × Kimi | CUDA × Qwen3.5-397B | CUDA × Qwen3.6 | CUDA × Kimi | ROCm × Qwen3.5-397B | ROCm × Qwen3.6 | ROCm × Kimi | APU × Qwen3.5-397B | APU × Qwen3.6 | APU × Kimi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Run model end-to-end | ✅ | — | — | ✅ | ✅ | ❌ | ✅ | — | — | ❌ | — | — |
| Tool calling (JSON-stable) | ✅ | — | — | ❌ | ❌ | ❌ | ❌ | — | — | ❌ | — | — |
| Multi-turn agent loop | ❌ | — | — | ❌ | ❌ | ❌ | ❌ | — | — | ❌ | — | — |
| HTTP/SSE serve mode (OpenAI-compatible) | ✅ | — | — | ✅ | ❌ | ❌ | ❌ | — | — | ❌ | — | — |
| Streaming SSD experts | ✅ | — | — | ✅ | ✅ | ✅ | ✅ | — | — | 🟡 | — | — |
| VRAM/RAM LRU expert cache | — | — | — | ❌ | ✅ | 🟡 | ✅ | — | — | ❌ | — | — |
| FP8 quant (e4m3 block-128) | — | — | — | — | ✅ | — | — | — | — | — | — | — |
| int4 quant (custom pack) | ✅ | — | — | ✅ | — | — | ✅ | — | — | 🟡 | — | — |
| sym-int4 quant (compressed-tensors) | — | — | — | — | — | ✅ | — | — | — | — | — | — |
| MLA attention | — | — | — | — | — | ✅ | — | — | — | — | — | — |
| GatedDeltaNet (linear attention) | ✅ | — | — | ✅ | ✅ | — | ✅ | — | — | 🟡 | — | — |
| Full attention with output-gate | — | — | — | — | ✅ | — | — | — | — | — | — | — |
| RoPE (full + partial) | ✅ | — | — | ✅ | ✅ | 🟡 | ✅ | — | — | 🟡 | — | — |
| Tokenizer subprocess | — | — | — | ✅ | ✅ | ✅ | ✅ | — | — | 🟡 | — | — |
| Greedy + top-p sampling | ✅ | — | — | ✅ | ✅ | ❌ | ✅ | — | — | ❌ | — | — |

## Reliability bar

A cell is allowed to be ✅ only when the model on that backend has cleared the project bar from the design spec:
1. Multi-turn agent loop (5+ tool calls, file edits, no drift)
2. Small fixed eval pass (~10s of prompts; benchmark vs reference vLLM/HF)

If a capability works but the model has *not* cleared the reliability bar, downgrade the cell to 🟡 even if the underlying mechanism is solid.

## How to update this file

1. PR that changes a cell must justify the change in the PR description.
2. PR template (`.github/PULL_REQUEST_TEMPLATE.md`) reminds you to check this matrix before merging.
3. If a cell's status changes due to a fix on one branch, open a follow-up issue with `propagate:*` labels for the other ports that need the same fix.

## Per-port detail

| Port (branch) | Doc |
|---|---|
| Metal (`main`) | [`ports/metal.md`](ports/metal.md) |
| CUDA (`cuda`) | [`ports/cuda.md`](ports/cuda.md) |
| CUDA Qwen3.6 (`cuda-qwen36`) — **priority port** | [`ports/cuda-qwen36.md`](ports/cuda-qwen36.md) |
| CUDA Kimi (`cuda-kimi`) | [`ports/cuda-kimi.md`](ports/cuda-kimi.md) |
| ROCm (`rocm`) | [`ports/rocm.md`](ports/rocm.md) |
| MI300 optimization (`mi300-opt`) | [`ports/mi300-opt.md`](ports/mi300-opt.md) |
| APU (`apu`) | [`ports/apu.md`](ports/apu.md) |

## Provenance

This file is the public canonical capability matrix. Underlying design rationale and implementation plan are kept as private working notes outside the repo (see `.gitignore`).
