# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**NOTE:** README.md is a symlink to this file. Keep content useful for both Claude Code and GitHub readers.

## Build & Run

Three backends: `metal_infer/` for Apple Silicon, `cuda_infer/` for NVIDIA GPUs, `apu_infer/` for AMD APUs.

### CUDA backend (NVIDIA GPUs)

See [`cuda_infer/README.md`](cuda_infer/README.md) for full documentation. Quick start:

```bash
cd cuda_infer
make                  # requires CUDA 12.8+ and libcufile
./infer --prompt "Hello" --tokens 20
```

For non-default models, use `configure.py` to generate compile flags from `model_weights.json`:

```bash
python3 configure.py --manifest model_weights.json --print-cmd
```

### Metal backend (Apple Silicon)

All binaries are built from `metal_infer/`. Metal shaders compile at runtime (no offline metal compiler needed).

```bash
cd metal_infer
make              # builds metal_infer (benchmark) + infer (inference engine)
make chat         # builds chat TUI client (separate target)
make clean        # remove build artifacts
```

### APU backend (AMD Ryzen AI / Strix Halo)

Built from `apu_infer/`. Targets AMD APUs with unified memory (RDNA 3.5, gfx1151). Uses HIP/ROCm with a dp4a (int8 dot product) kernel optimized for the shared CPU-GPU memory architecture.

```bash
cd apu_infer
make                  # requires ROCm 6.4+ and hipcc
./infer --prompt "Hello" --tokens 20 --experts ../model-safetensors/packed_experts
USE_DP4A=1 ./infer --prompt "Hello" --tokens 20 --experts ../model-safetensors/packed_experts  # dp4a mode (+78%)
```

### Inference engine (`infer`)

All three backends produce an `infer` binary with the same CLI interface:

```bash
./infer --prompt "Hello" --tokens 50              # basic generation
./infer --prompt "Hello" --tokens 50 --2bit       # 2-bit mode (Metal only, breaks JSON)
./infer --prompt "Hello" --tokens 20 --timing     # per-layer timing breakdown
./infer --serve 8080                              # HTTP server (OpenAI-compatible API)
./infer --prompt "Hello" --tokens 20 --freq       # expert frequency tracking
./infer --prompt "Hello" --tokens 20 --cache-telemetry  # cold vs eviction miss analysis
```

### Chat TUI (`chat`)

Metal backend only. Thin HTTP/SSE client that connects to the inference server. Sessions persist to `~/.flash-moe/sessions/<id>.jsonl`.

```bash
./chat                          # connect to default port
./chat --port 8000              # specify server port
./chat --show-think             # show thinking tokens
./chat --resume <session_id>    # resume previous session
```

### Custom system prompt

Place a file at `~/.flash-moe/system.md` to override the default system prompt used by the serve mode.

### MoE benchmark (Metal only)

```bash
make run           # single expert forward pass
make verify        # Metal vs CPU reference verification
make bench         # benchmark single expert (10 iterations)
make moe           # full MoE forward pass (K experts, single layer)
make full          # full 60-layer forward pass (K=4)
make fullbench     # benchmark full 60-layer forward (3 iterations)
```

## Code Architecture

Three parallel single-file inference engines targeting different hardware, sharing the same tokenizer and model format.

### Metal backend (`metal_infer/`)

Three Objective-C files, one Metal shader file, one C header — no frameworks, no dependencies beyond Apple system libraries.

- **`infer.m`** (~7000 lines) — The entire inference engine: model loading, Metal pipeline setup, all 60-layer forward pass, tokenization, sampling, HTTP server (OpenAI-compatible SSE), tool calling, KV cache management.
- **`shaders.metal`** (~1300 lines) — All Metal compute kernels: 4-bit/2-bit dequant matvec, SwiGLU, RMS norm, attention, RoPE, MoE combine+residual.
- **`chat.m`** — Thin HTTP/SSE client with linenoise line editing. No model logic.
- **`main.m`** — Standalone MoE benchmark. Verifies Metal vs CPU.
- **`tokenizer.h`** — Single-header C BPE tokenizer (449 lines). Shared with CUDA backend.

### CUDA backend (`cuda_infer/`)

- **`infer.cu`** (~3600 lines) — Port of `infer.m` for NVIDIA GPUs. Same structure: model loading, forward pass, HTTP server, tool calling. Adds VRAM expert cache (~17GB LRU) and GDS support.
- **`kernels.cuh`** (~1200 lines) — 15 CUDA kernels, ported from `shaders.metal`. Same algorithms, adapted for warp-level CUDA primitives (`__shfl_down_sync`, shared memory).
- **`tokenizer_impl.c`** — Thin wrapper to compile `tokenizer.h` as C and link into the CUDA binary.
- **`configure.py`** — Generates compile-time `-D` flags from `model_weights.json` for non-default models.

### APU backend (`apu_infer/`)

- **`infer.hip`** (~4000 lines) — Port of `infer.cu` for AMD APUs via HIP/ROCm. Tuned for unified-memory RDNA 3.5 (gfx1151). Uses `hipMalloc` for expert staging (CPU pread + GPU read share the same physical DRAM on APU — 27% more bandwidth than `hipMallocManaged`).
- **`kernels.hip.h`** (~1400 lines) — HIP compute kernels including dp4a-optimized dequant matvec using `__builtin_amdgcn_sudot4` (4 int8×int8 MACs per instruction). Also has FMA fallback and GGUF Q4_K/Q5_K/Q6_K format support.
- **`tokenizer_impl.c`** — Same as CUDA backend.

### Key design constraints

- **Single-file engines**: All inference logic in one file per backend (`infer.m` / `infer.cu` / `infer.hip`). Intentional for simplicity.
- **Expert caching differs by platform**:
  - Metal: No custom cache — relies entirely on OS page cache ("Trust the OS"). Every custom cache we tried was slower on unified memory.
  - CUDA: VRAM LRU cache (~17GB, ~2500 experts) because discrete GPUs need data in VRAM. OS page cache still used as second tier.
  - APU: OS page cache + optional VRAM LRU. `hipMalloc` expert staging gives full GPU bandwidth because CPU and GPU share physical LPDDR5X. XNACK (direct mmap GPU access) is a hardware limitation on RDNA — not supported.
- **Pipeline differs by platform**:
  - Metal: Serial GPU→SSD→GPU (SSD DMA and GPU share memory controller, overlapping causes latency spikes).
  - CUDA: I/O and compute can overlap via separate PCIe buses, but expert loads use CUDA streams for async transfer.
  - APU: Serial pipeline like Metal (CPU, GPU, and NPU share the same LPDDR5X bus — overlapping causes contention, not parallelism).
- **Metal shaders compile at runtime** via `MTLDevice newLibraryWithSource:`. No offline `.metallib` needed (though `make metallib` exists as an option).

### Per-layer pipeline — Metal (3 command buffers)

```
CMD3(prev) → CMD1: attention projections + delta-net  [GPU]
           → CPU: flush results
           → CMD2: o_proj + norm + routing + shared    [GPU]
           → CPU: softmax + topK routing
           → I/O: parallel pread K=4 experts           [SSD]
           → CMD3: expert forward + combine + norm     [GPU, DEFERRED]
```

CMD3 is submitted without waiting (deferred). The GPU serializes CMD3(N-1) then CMD1(N) via queue ordering.

### Per-layer pipeline — CUDA

```
1. RMS norm → attention projections (dequant matvec)     [GPU]
2. Attention: GatedDeltaNet (45 layers) or full (15)     [GPU]
3. o_proj → residual → post-attn norm → routing + shared [GPU]
4. CPU: softmax → topK routing decision
5. Expert load: VRAM cache hit → page cache → SSD pread  [I/O]
6. Expert forward: gate+up → SwiGLU → down × K           [GPU]
7. MoE combine + residual                                [GPU]
```

### Per-layer pipeline — APU (HIP/ROCm, gfx1151)

```
1. quantize_x_q8: float→int8 activation quantization      [GPU, ~1µs]
2. RMS norm → attention projections (dp4a dequant matvec)   [GPU]
3. Attention: GatedDeltaNet (45 layers) or full (15)        [GPU]
4. o_proj → residual → post-attn norm → routing + shared    [GPU]
5. CPU: softmax → topK routing decision
6. Expert load: page cache hit → SSD pread → hipMalloc buf  [I/O]
7. Expert forward: dp4a gate+up → SwiGLU → dp4a down × K    [GPU]
8. MoE combine + residual                                   [GPU]
```

Key APU technique: dp4a (`__builtin_amdgcn_sudot4`) replaces FMA in the dequant matvec kernel. Pre-quantizes activations to int8, then uses 4-way integer dot product — 8x fewer instructions per weight byte. Expert buffers use `hipMalloc` (not `hipMallocManaged`) because on APU the CPU can pread directly into device memory (shared physical DRAM) and the GPU gets 27% more bandwidth from better TLB coverage.

---

# Flash-MoE: Running a 397B Parameter Model on Consumer Hardware

> **[Read the paper](paper/flash_moe.pdf)** — Full technical details, 90+ experiments, and the story of how an AI and a human built this in 24 hours.

Pure C/Metal/HIP inference engine that runs **Qwen3.5-397B-A17B** (a 397 billion parameter Mixture-of-Experts model) on consumer hardware at **4.4–6.1 tokens/second** with production-quality output including tool calling.

The entire 209GB model streams from SSD through custom GPU compute pipelines. No Python. No frameworks. Just C, Objective-C, Metal shaders, and HIP kernels.

## Results

![Progress](progress.png)

### Apple Silicon (M3 Max, 48GB, 17.5 GB/s SSD)

| Configuration | tok/s | Quality | Notes |
|--------------|-------|---------|-------|
| 4-bit experts, FMA kernel | **4.36** | Excellent | Current best. Full tool calling. 209GB on disk. |
| 4-bit experts, baseline | 3.90 | Excellent | Before FMA kernel optimization. |
| 2-bit experts, trust OS | 5.74 | Good* | 120GB on disk. *Breaks JSON/tool calling. |
| 2-bit peak single token | 7.05 | Good* | Warm cache burst. *Not suitable for tool use. |

*2-bit quantization produces `\name\` instead of `"name"` in JSON output, making tool calling unreliable. 4-bit is the production configuration.

### AMD APU (Ryzen AI MAX+ 395, 64GB, 3.4 GB/s SSD)

| Configuration | tok/s | Quality | Notes |
|--------------|-------|---------|-------|
| 4-bit, dp4a + hipMalloc | **6.14** | Excellent | sudot4 int8 dot product + hipMalloc expert staging |
| 4-bit, dp4a only | 5.55 | Excellent | Before hipMalloc optimization |
| 4-bit, FMA baseline | 3.45 | Excellent | Before dp4a kernel |

The APU achieves higher tok/s than M3 Max despite a 5x slower SSD (3.4 vs 17.5 GB/s) because the dp4a kernel saturates more of the 215 GB/s shared memory bandwidth. Expert I/O (59% of time) is now the bottleneck — a faster NVMe or BIOS VRAM reduction (freeing RAM for page cache) would help further.

## Hardware

### Apple Silicon
- **Machine**: MacBook Pro, Apple M3 Max
- **Chip**: 16-core CPU (12P + 4E), 40-core GPU, 16-core ANE
- **Memory**: 48 GB unified (~400 GB/s bandwidth)
- **SSD**: 1TB Apple Fabric, **17.5 GB/s sequential read** (measured)
- **macOS**: 26.2 (Darwin 25.2.0)

### AMD APU
- **Machine**: Desktop, AMD Ryzen AI MAX+ 395
- **Chip**: 16-core CPU (Zen 5), 40 CU GPU (RDNA 3.5, gfx1151), XDNA2 NPU
- **Memory**: 64 GB unified LPDDR5X (~215 GB/s bandwidth, shared across CPU/GPU/NPU)
- **SSD**: 1TB Crucial T500, **3.4 GB/s sequential read** (measured)
- **OS**: Fedora 43 (kernel 6.19.11), ROCm 6.4

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

### APU-Specific Techniques (AMD Strix Halo)

7. **dp4a Int8 Dot Product Kernel** — Pre-quantize the activation vector to int8 (Q8_1 format: per-32-element scale and sum). The inner loop uses `__builtin_amdgcn_sudot4` to do 4 int8×int8 multiply-accumulates per instruction instead of 8 separate FMAs. 8x fewer instructions per weight byte. Technique adapted from llama.cpp's RDNA kernel but applied to our custom affine quantization format (scale+bias per 64 elements). **3.7x faster attention phase.**

8. **hipMalloc for Expert Staging on APU** — On AMD APUs, CPU and GPU share the same physical LPDDR5X. We discovered that CPU can `pread()` directly into `hipMalloc`'d buffers (not documented — verified experimentally). `hipMalloc` gives 220 GB/s GPU read bandwidth vs 173 GB/s for `hipMallocManaged` (+27%) due to better TLB coverage (larger page table entries). No existing inference engine uses this technique. **2.2x faster expert phase.**

9. **XNACK Is a Hardware Limitation** — We investigated whether the GPU could read directly from mmap'd file pages (the "Trust the OS" dream from Metal). XNACK (GPU page fault retry) is physically absent from all RDNA silicon starting with RDNA 2 — the retry circuit was removed from the Shader Sequencer. Not a driver bug, not fixable by patching. The `pread → hipMalloc` path is the correct workaround for RDNA APUs.

### Unified Memory Constraints

On Apple Silicon, SSD DMA and GPU compute share the same memory controller and cannot be profitably overlapped. The GPU's dequant kernels are bandwidth-saturated at ~418 GiB/s. Even small background SSD DMA causes disproportionate GPU latency spikes through memory controller arbitration. The serial pipeline (GPU → SSD → GPU) is hardware-optimal.

On AMD Strix Halo, the same constraint applies — CPU, GPU, and NPU share a single 256-bit LPDDR5X bus (~215 GB/s). We investigated NPU offload for expert computation but confirmed it would cause bandwidth contention (same failure mode as Apple Silicon SSD DMA). The NPU's 84µs dispatch latency also makes per-layer dispatch impractical (720 expert matvecs × 84µs = 60ms overhead, equaling the entire expert compute budget).

## Quick Start

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

## Project Structure

```
metal_infer/
  infer.m              # Complete inference engine (~7000 lines)
  shaders.metal        # Metal compute kernels (~1200 lines)
  chat.m               # Interactive chat TUI with tool calling
  tokenizer.h          # C BPE tokenizer (single-header, 449 lines)
  main.m               # MoE-only benchmark
  Makefile             # Build system
  extract_weights.py   # Creates model_weights.bin from safetensors
  repack_experts_2bit.py  # 4-bit → 2-bit expert requantization
  train_predictor.py   # Expert routing prediction analysis
  model_weights.bin    # Non-expert weights (5.5GB, mmap'd)
  model_weights.json   # Tensor manifest
  vocab.bin            # Vocabulary for token decoding
  tokenizer.bin        # Pre-exported BPE tokenizer data

repack_experts.py      # 4-bit expert packing from safetensors
progress.py            # Results visualization (Q2/Q4 tracks)
results.tsv            # Experiment log (58 experiments)
```

## What We Tried (and What Worked)

### Kept
| Approach | Platform | Result | Impact |
|----------|----------|--------|--------|
| dp4a int8 dot product kernel | APU | attn 3.7x, total +61% | **Largest single win** |
| hipMalloc expert staging | APU | expert 2.2x, +27% BW | **APU-specific discovery** |
| FMA dequant kernel | Metal | GPU compute -12% | **+12% tok/s** |
| Trust OS page cache | Metal | Deleted Metal LRU → +38% | **Foundational** |
| GPU combine+norm in CMD3 | Metal | Eliminates CPU round-trip | **Pipeline** |
| BLAS delta-net (Accelerate) | Metal | cpu_attn 0.78→0.28ms | **+64% attn** |
| F_NOCACHE for 2-bit | Metal | +3% from avoiding page thrash | **2-bit only** |
| GPU fused attention (RoPE) | Metal | +2% for full-attn layers | **Small** |
| C BPE tokenizer | All | 180ms vs 3500ms startup | **20x startup** |
| Deferred CMD3 execution | Metal | GPU/CPU overlap | **Pipeline** |

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
| Nontemporal weight loads (APU) | 0% | Infinity Cache not the bottleneck on APU |
| XNACK mmap GPU access (APU) | N/A | Hardware absent on all RDNA silicon |
| NPU expert offload (APU) | N/A | Shared LPDDR5X bus = contention, +84µs/dispatch |
| hipMallocManaged (APU) | -27% BW | hipMalloc gives better TLB coverage on APU |

## Safety

This is a primary development machine. The engine explicitly controls memory:
- Non-expert weights: 5.5GB (mmap'd, read-only)
- Metal scratch buffers: ~200MB
- Total: ~6GB, leaving 42GB for OS + page cache
- No OOM risk. Expert data streams from SSD on demand.
- No custom caches. Trust the OS.
