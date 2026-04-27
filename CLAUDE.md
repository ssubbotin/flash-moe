# Flash-MoE: Running a 397B Parameter Model on a Laptop

> **[Read the paper](paper/flash_moe.pdf)** — Full technical details, 90+ experiments, and the story of how an AI and a human built this in 24 hours.

Pure C/Metal inference engine that runs **Qwen3.5-397B-A17B** (a 397 billion parameter Mixture-of-Experts model) on a MacBook Pro with 48GB RAM at **4.4+ tokens/second** with production-quality output including tool calling.

The entire 209GB model streams from SSD through a custom Metal compute pipeline. No Python. No frameworks. Just C, Objective-C, and hand-tuned Metal shaders.

The project has since expanded to a CUDA backend (RTX 4090 / 5090 / PRO 6000), a ROCm backend (MI300X), and additional MoE families (Qwen3.6-35B-A3B-FP8, Kimi K2.6). See [Backends & ports](#backends--ports) below for what runs where.

## Project status

**What works on which backend, with which model:** see [`docs/STATUS.md`](docs/STATUS.md).

Per-port detail in [`docs/ports/<port>.md`](docs/ports/).

A cell is allowed to be ✅ only when the model on that backend has cleared the project reliability bar: a multi-turn agent loop (5+ tool calls, file edits, no drift) plus a small fixed eval. Working mechanism without that test stays 🟡.

## Backends & ports

| Backend | Branch | Hardware | Model | tok/s (warm) | Notes |
|---|---|---|---|---|---|
| Metal | `main` | M3 Max 48GB | Qwen3.5-397B-A17B | 4.36 (4-bit) | Production reference. Tool calling. The paper. |
| CUDA | `cuda` | RTX 4090 / 5090 / PRO 6000 | Qwen3.5-397B-A17B | 1.75 (4090) → 10.57 peak (5090) → 43.2 cached (PRO 6000 96 GB) | Multi-hardware benchmarks. |
| CUDA | `cuda-qwen36` | RTX 4090 | Qwen3.6-35B-A3B-FP8 | ~7-8 | FP8 e4m3 + VRAM LRU expert cache. |
| CUDA | `cuda-kimi` | RTX PRO 6000 (96GB) | Kimi K2.6 (1T MoE) | — | Active development; hidden-state collapse between L1 and L5 under investigation ([#1](https://github.com/ssubbotin/flash-moe/issues/1)). |
| ROCm | `rocm`, `mi300-opt` | MI300X 192GB | Qwen3-235B-A22B | 7.13 (30 tok) → 8.99 (100 tok) | Optimization targeting AMD Developer Hackathon (May 2026). |
| APU | `apu` | Strix Halo (exploratory) | — | — | Unified-memory APU evaluation. |

**Open work:** see [open issues](https://github.com/ssubbotin/flash-moe/issues) (filter by `reliability`, `correctness`, or `port:*` labels). The current priority is getting Qwen3.6 on `cuda-qwen36` to clear the reliability bar — multi-turn agent loop + small fixed eval ([#5](https://github.com/ssubbotin/flash-moe/issues/5)).

**Branch lifecycle:** the model branches above (`cuda-qwen36`, `cuda-kimi`, `mi300-opt`, `apu`) carry their per-port working state and have not been merged into `main`. `main` carries the Metal reference engine and the project-wide capability matrix; per-port code is read directly off its branch. Cross-branch fixes propagate via labeled issues — see the [PR template](.github/PULL_REQUEST_TEMPLATE.md) and the `port:*` / `propagate:*` labels.

## Results — Metal × Qwen3.5-397B (M3 Max, 4-bit production config)

![Progress](progress.png)

| Configuration | tok/s | Quality | Notes |
|--------------|-------|---------|-------|
| 4-bit experts, FMA kernel | **4.36** | Excellent | Current best. Full tool calling. 209GB on disk. |
| 4-bit experts, baseline | 3.90 | Excellent | Before FMA kernel optimization. |
| 2-bit experts, trust OS | 5.74 | Good* | 120GB on disk. *Breaks JSON/tool calling. |
| 2-bit peak single token | 7.05 | Good* | Warm cache burst. *Not suitable for tool use. |

*2-bit quantization produces `\name\` instead of `"name"` in JSON output, making tool calling unreliable. 4-bit is the production configuration.

## Hardware (Metal reference)

- **Machine**: MacBook Pro, Apple M3 Max
- **Chip**: 16-core CPU (12P + 4E), 40-core GPU, 16-core ANE
- **Memory**: 48 GB unified (~400 GB/s bandwidth)
- **SSD**: 1TB Apple Fabric, **17.5 GB/s sequential read** (measured)
- **macOS**: 26.2 (Darwin 25.2.0)

For other hardware configurations (RTX 4090 / 5090 / PRO 6000 / MI300X), see the per-port docs under [`docs/ports/`](docs/ports/).

## Architecture

The model has 60 transformer layers: 45 GatedDeltaNet (linear attention) + 15 standard full attention. Each layer has 512 experts, of which K=4 are activated per token (plus one shared expert). Hidden dimension is 4096.

### Key Techniques

1. **SSD Expert Streaming** — Expert weights (209GB at 4-bit) are read from NVMe SSD on demand via parallel `pread()` with GCD dispatch groups. Only the K=4 active experts per layer are loaded (~6.75MB each). The OS page cache manages caching — no custom cache needed ("Trust the OS" principle). Inspired by Apple's "LLM in a Flash" paper.

2. **FMA-Optimized Dequant Kernel** — The inner loop of the 4-bit dequantized matrix-vector multiply rearranges the math from `(nibble * scale + bias) * x` to `fma(nibble, scale*x, bias*x)`. Pre-computing `scale*x` and `bias*x` lets the GPU fused multiply-add unit do dequant+multiply in one instruction. 12% faster than the naive formulation.

3. **Metal Compute Shaders** — Hand-written Metal kernels for:
   - 4-bit and 2-bit dequantized matrix-vector multiply (tiled, SIMD-reduced, shared input cache, FMA-optimized)
   - Fused SwiGLU activation
   - RMS normalization (two-pass: sum-of-squares reduction + apply)
   - Batched GPU attention (Q@K^T, softmax, scores@V) for full attention layers
   - GPU RoPE (fused with Q deinterleave and K normalization)
   - MoE combine + residual + sigmoid gate (fused kernel)

4. **Deferred GPU Expert Compute** — CMD3 (expert forward pass) is submitted without waiting. The GPU executes it while the CPU prepares the next layer. The combine + residual + norm are also on GPU, feeding directly into the next layer's attention projections.

5. **Accelerate BLAS for Linear Attention** — The GatedDeltaNet recurrence uses `cblas_sscal`, `cblas_sgemv`, and `cblas_sger` for the 64-head × 128×128 state matrix update. 64% faster than scalar code.

6. **Trust the OS** — No custom expert cache. The OS page cache (~35GB) manages expert data caching via standard LRU. Every custom caching approach we tested (Metal LRU, malloc cache, LZ4 compressed cache) was slower due to GPU memory pressure or overhead. The page cache achieves ~71% hit rate naturally.

### Pipeline Per Layer (4.28ms average at 4-bit)

```
CMD3(prev) → CMD1: attention projections + delta-net  [1.22ms GPU]
           → CPU: flush results                       [0.01ms CPU]
           → CMD2: o_proj + norm + routing + shared    [0.55ms GPU]
           → CPU: softmax + topK routing               [0.003ms]
           → I/O: parallel pread K=4 experts           [2.41ms SSD]
           → CMD3: expert forward + combine + norm     [0.04ms encode, DEFERRED]
```

### Unified Memory Constraint

On Apple Silicon, SSD DMA and GPU compute share the same memory controller and cannot be profitably overlapped. The GPU's dequant kernels are bandwidth-saturated at ~418 GiB/s. Even small background SSD DMA causes disproportionate GPU latency spikes through memory controller arbitration. The serial pipeline (GPU → SSD → GPU) is hardware-optimal.

## Quick Start

### Metal (M3 Max → Qwen3.5-397B)

```bash
cd metal_infer
make
# 4-bit inference (needs packed_experts/ directory)
./infer --prompt "Explain quantum computing" --tokens 100

# 2-bit inference (faster but breaks tool calling)
./infer --prompt "Explain quantum computing" --tokens 100 --2bit

# Interactive chat with tool calling
./chat

# Per-layer timing breakdown
./infer --prompt "Hello" --tokens 20 --timing
```

### CUDA (NVIDIA GPUs)

The CUDA backend lives in `cuda_infer/` on branches `cuda` (Qwen3.5-397B), `cuda-qwen36` (Qwen3.6-35B-A3B-FP8), and `cuda-kimi` (Kimi K2.6 — under active development). Build with `make` from `cuda_infer/`; requires CUDA 12.8+ and libcufile. Per-branch run commands live in [`docs/ports/cuda*.md`](docs/ports/) and `cuda_infer/README.md`.

### ROCm (AMD GPUs)

The ROCm backend lives in `rocm_infer/` on branches `rocm` and `mi300-opt`. Build with `make` from `rocm_infer/` on MI300X (gfx942, ROCm 7.2+). Run commands in [`docs/ports/rocm.md`](docs/ports/rocm.md) and [`docs/ports/mi300-opt.md`](docs/ports/mi300-opt.md).

## Project Structure

```
metal_infer/             # Metal backend — main branch
  infer.m                # Complete inference engine (~7000 lines)
  shaders.metal          # Metal compute kernels (~1200 lines)
  chat.m                 # Interactive chat TUI with tool calling
  tokenizer.h            # C BPE tokenizer (single-header, 449 lines)
  main.m                 # MoE-only benchmark
  Makefile               # Build system
  extract_weights.py     # Creates model_weights.bin from safetensors
  repack_experts_2bit.py # 4-bit → 2-bit expert requantization
  train_predictor.py     # Expert routing prediction analysis

cuda_infer/              # CUDA backend — branches cuda, cuda-qwen36, cuda-kimi
                         # Per-model engines and FP8 / sym-int4 dequant kernels.

rocm_infer/              # ROCm/HIP backend — branches rocm, mi300-opt
                         # MI300X port; aotriton attention, FLA mapping.

apu_infer/               # APU exploration — branch apu

docs/
  STATUS.md              # Coarse capability matrix (canonical "what works")
  ports/<port>.md        # Per-port detailed matrix and run commands
  mi300/                 # MI300-specific notes (aotriton API, FLA mapping)

paper/                   # Flash-MoE paper sources (LaTeX, arXiv, TMLR variants)
progress.py              # Results visualization (Q2/Q4 tracks)
results.tsv              # Experiment log (Metal track)
repack_experts.py        # 4-bit expert packing from safetensors (Metal)
repack_experts_kimi.py   # sym-int4 expert packing for Kimi
repack_experts_qwen36.py # FP8 expert packing for Qwen3.6
```

## What We Tried (and What Worked) — Metal track

The tables below cover the Metal × Qwen3.5-397B optimization track (the paper's subject). CUDA, ROCm, and per-port-model experiment logs live in their respective branches and per-port docs.

### Kept
| Approach | Result | Impact |
|----------|--------|--------|
| FMA dequant kernel | GPU compute -12% | **+12% tok/s** |
| Trust OS page cache | Deleted Metal LRU → +38% | **Foundational** |
| GPU combine+norm in CMD3 | Eliminates CPU round-trip | **Pipeline** |
| BLAS delta-net (Accelerate) | cpu_attn 0.78→0.28ms | **+64% attn** |
| F_NOCACHE for 2-bit | +3% from avoiding page thrash | **2-bit only** |
| GPU fused attention (RoPE) | +2% for full-attn layers | **Small** |
| C BPE tokenizer | 180ms vs 3500ms startup | **20x startup** |
| Deferred CMD3 execution | GPU/CPU overlap | **Pipeline** |

### Discarded (58 experiments, highlights)
| Approach | Result | Why |
|----------|--------|-----|
| LZ4 expert compression | -13% | Decompress overhead > warm cache savings |
| F_RDADVISE prefetch | net 0% | Unified memory: SSD DMA slows GPU -73% |
| Temporal expert prediction | -18% | 25% hit rate, SSD bandwidth waste |
| MLP routing predictor | 31% accuracy | Worse than temporal baseline |
| GPU LUT dequant kernel | -2% | Indirect register access serializes |
| GPU private buffer compression | -20% pipeline | Blit cost 4×7MB > matvec savings |
| Spin-poll GPU wait | -23% | CPU thermal competes with GPU |
| Expert file clustering | 0% | NVMe ignores scatter at 7MB granularity |
| dispatch_io | -70% | dispatch_data management overhead |
| mmap expert files | -5x | Per-page fault overhead on cold data |
| Speculative early routing | -38% | Cache pollution + overhead |
| MTP speculative decoding | break-even | MoE I/O scales per-token (unlike dense) |

## Contributors

**Daniel Woods** ([@danveloper](https://github.com/danveloper)) — original Flash-MoE author. Designed and implemented the Metal inference engine, the SSD-streamed expert pipeline, the Q4/Q2 quantization tracks, and the multi-hardware foundations. Co-author of the [paper](paper/flash_moe.pdf).

**Sergey Subbotin** ([@ssubbotin](https://github.com/ssubbotin)) — fork maintainer. CUDA backend (RTX 4090 / 5090 / PRO 6000), ROCm/HIP backend (MI300X, gfx942), MI300 optimization plan, additional MoE family ports (Qwen3.6-35B-A3B-FP8, Kimi K2.6), multi-hardware benchmarks, and the project coherence system (`docs/STATUS.md`, per-port docs, propagation labels).

The original 397B-on-MacBook story and its 24-hour collaboration narrative are documented in the [paper](paper/flash_moe.pdf).

## Safety

This is a primary development machine. The engine explicitly controls memory:
- Non-expert weights: 5.5GB (mmap'd, read-only)
- Metal scratch buffers: ~200MB
- Total: ~6GB, leaving 42GB for OS + page cache
- No OOM risk. Expert data streams from SSD on demand.
- No custom caches. Trust the OS.
