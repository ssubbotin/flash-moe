# Running 397B Parameters on an AMD APU: dp4a Tricks and Undocumented Memory APIs

## How we got 6.14 tok/s on a Ryzen AI MAX+ 395 by exploiting hardware int8 dot products and a memory allocation technique no inference engine uses

---

Two months ago, [Flash-MoE](https://github.com/danveloper/flash-moe) showed that a 397-billion-parameter model could run on a MacBook Pro at 4.4 tokens/second. Then we [ported it to NVIDIA](article_medium.md) and hit 5.57 tok/s on an RTX 4090. Both versions use the same core idea: stream 209GB of expert weights from NVMe SSD through custom GPU kernels, caching what fits in fast memory.

We asked: **what about AMD's new APUs, where CPU and GPU share the same physical memory?**

The answer turned out to be more interesting than we expected. After a straightforward HIP port that gave us a baseline of 3.45 tok/s, two optimizations — one exploiting a hardware dot-product instruction, the other exploiting an undocumented property of `hipMalloc` on unified memory — pushed performance to **6.14 tok/s**. A 78% improvement, and techniques that no existing inference engine uses.

---

## The Hardware

AMD Ryzen AI MAX+ 395. Codename Strix Halo. The chip that made us rethink several assumptions about GPU memory.

| Component | Specs |
|-----------|-------|
| CPU | 16 Zen 5 cores (32 threads), up to 5.1 GHz |
| GPU | 40 RDNA 3.5 CUs (2560 shaders), ~22 TFLOPS FP32 |
| Memory | 64 GB LPDDR5X unified, ~215 GB/s bandwidth |
| SSD | Crucial T500 2TB, PCIe 4.0 x4, 3.4 GB/s sequential |
| OS | Fedora 43, ROCm 6.4 |

The key word is *unified*. Like Apple Silicon, the CPU and GPU share the same physical LPDDR5X pool. There is no PCIe bus between them. There is no separate VRAM. The BIOS carves the 64GB into a "GPU" region and a "system" region (32/32 in our configuration), but both map to the same DIMMs.

This is architecturally similar to the M3 Max we started with, but with a fundamentally different GPU: RDNA 3.5 instead of Apple's custom design. Same unified memory concept, different instruction set, different performance characteristics.

## The Model

Same model as always: Qwen3.5-397B-A17B. 397 billion parameters, of which only 17 billion activate per token. 60 transformer layers with 512 experts each, K=4 activated per token. 209GB at 4-bit quantization.

Each token touches about 5.3GB of weight data: 240 expert reads at ~6.75MB each, plus attention projections and shared experts. The question for any hardware is: how fast can you get those bytes to the ALU?

## The Starting Point: 3.45 tok/s

Porting the CUDA backend to HIP was mechanically straightforward. HIP is designed as a thin translation layer over CUDA — most of the code changes were `cuda` → `hip` in function names. The 15 compute kernels ported without algorithm changes. We had a working `apu_infer/` backend within a day.

But we also had to make a design decision. On the RTX 4090, we use a three-tier caching hierarchy: VRAM cache (17GB LRU) → OS page cache → SSD. On an APU, the VRAM cache doesn't make sense. Every byte you allocate as "VRAM expert cache" is a byte stolen from the OS page cache, which would have served the same purpose. It's unified memory — there is no fast tier.

So we took the Apple Silicon approach: trust the OS page cache. No custom VRAM cache. `pread()` experts from SSD, let the kernel manage the LRU.

The baseline: **3.45 tok/s**.

For context:

| Hardware | tok/s |
|----------|------:|
| Apple M3 Max (48 GB unified) | 4.36 |
| **Strix Halo baseline** | **3.45** |
| RTX 3060 (12 GB VRAM) | 2.92 |

Not bad for a first port. But also not great. The M3 Max was 26% faster, and that machine has less memory bandwidth (400 GB/s vs 215 GB/s — wait, Apple is faster *and* has more bandwidth?). Something was off.

## The Roofline Analysis

Time to check whether we're using the hardware well.

The bandwidth roofline for this workload:
- 5.3 GB of weight data per token
- 215 GB/s memory bandwidth
- **Theoretical maximum: 40.6 tok/s**

We measured 3.45 tok/s. That's **8.5% bandwidth utilization**. On the M3 Max, we achieve about 19% utilization. Something on the AMD side was leaving 90% of the bandwidth on the table.

The `--timing` breakdown told us where:

```
phase     ms/layer   x60/token   share
attn         2.12     127.2       48%
expert       1.00      60.0       23%
io           0.58      34.8       13%
shared       0.26      15.6        6%
route        0.07       4.2        2%
```

**48% of time was in attention compute.** Not I/O. Not memory bandwidth. The GPU was spending half its time doing arithmetic in the dequant matvec kernel, and it was doing it slowly.

The dequant matvec kernel runs 1005 times per token (285 attention projections + 720 expert matvecs). It's the innermost loop of the entire engine. And our profiling told us it was instruction-throughput limited, not bandwidth limited — the kernel had 100% occupancy with only 39 VGPRs, but the ALU was saturated. The GPU couldn't issue memory loads fast enough because it was too busy doing arithmetic.

## Research: What llama.cpp Knows About RDNA

We studied llama.cpp's RDNA kernel implementations to understand what the community had already figured out about AMD GPU performance. One pattern stood out: their quantized matmul kernels use **dp4a** — the `v_dot4_i32_iu8` instruction, exposed through `__builtin_amdgcn_sudot4`.

dp4a does 4 int8-by-int8 multiply-accumulates in a single instruction. Four MACs, one cycle. Compare that to our FMA kernel, which processes 8 nibbles from a packed uint32 like this:

```c
// FMA kernel: 8 nibbles = ~32 ALU instructions
acc += __fmaf_rn((float)((packed >>  0) & 0xF), sx0, bx0);
acc += __fmaf_rn((float)((packed >>  4) & 0xF), sx1, bx1);
acc += __fmaf_rn((float)((packed >>  8) & 0xF), sx2, bx2);
// ... 8 times, each needing shift + mask + float convert + FMA
```

Each packed uint32 of 8 nibbles required ~32 ALU instructions: 8 shifts, 8 masks, 8 int-to-float conversions, and 8 FMAs. The dp4a approach needs ~4 instructions for the same 8 nibbles: 2 nibble repacks and 2 `sudot4` calls. That's **8x fewer instructions** for the same amount of weight data.

The catch: dp4a operates on int8 inputs. Our activation vector is float32. We'd need to quantize it first.

## The dp4a Kernel

The idea is simple: before each batch of matvecs that share the same input vector, quantize that vector from float32 to int8 once. Then use integer dot products in the inner loop.

Our affine quantization format stores weights as `value = nibble * scale + bias` per group of 64 elements. With the activation pre-quantized to Q8_1 format (int8 values with a per-32-element scale `d` and sum), the dot product becomes:

```
dot(W_row, x) ≈ w_scale * d_q8 * Σ(nibble_i × q8_i) + w_bias * d_q8 * Σ(q8_i)
```

The expensive part — `Σ(nibble_i × q8_i)` — is now an integer dot product computed via chained `sudot4` calls. The `Σ(q8_i)` sums are precomputed during quantization.

The inner loop:

```c
__device__ __forceinline__ int sudot4_ui8(int a_u4x4, int b_i8x4, int c) {
    return __builtin_amdgcn_sudot4(false, a_u4x4, true, b_i8x4, c, false);
}

// Per group of 64 elements (8 packed uint32, split into two Q8 blocks):
int idot_a = 0;
const int32_t* q8_row = (const int32_t*)(q8 + elem_base);
#pragma unroll
for (uint32_t p = 0; p < 4; p++) {
    uint32_t packed = w_row[col_base + p];
    // Repack nibbles to sequential bytes for sudot4
    uint32_t lo_nib = packed & 0xFFFF;
    uint32_t hi_nib = packed >> 16;
    int seq0 = (int)((lo_nib & 0xF) | ((lo_nib & 0xF0) << 4) |
                     ((lo_nib & 0xF00) << 8) | ((lo_nib & 0xF000) << 12));
    int seq1 = (int)((hi_nib & 0xF) | ((hi_nib & 0xF0) << 4) |
                     ((hi_nib & 0xF00) << 8) | ((hi_nib & 0xF000) << 12));

    idot_a = sudot4_ui8(seq0, q8_row[p * 2 + 0], idot_a);
    idot_a = sudot4_ui8(seq1, q8_row[p * 2 + 1], idot_a);
}

// Reconstruct float result
acc += w_scale * (d_a * (float)idot_a + d_b * (float)idot_b)
     + w_bias  * (d_a * sum_a + d_b * sum_b);
```

Two `sudot4` calls per packed uint32 instead of 8 FMAs plus 24 supporting instructions. The nibble repacking is pure bitwise logic that the RDNA SALU handles in parallel with the VALU dot products.

### The Warp Reduce Bug

This is the kind of bug that costs you a day and teaches you something fundamental about GPU programming.

The activation quantization kernel needs to find the maximum absolute value across each 32-element block (one warp on RDNA) to compute the quantization scale. Standard pattern:

```c
float amax = warp_reduce_max(fabsf(val));
```

`warp_reduce_max` uses `__shfl_down` to tree-reduce across the warp. After the reduction, **only lane 0 has the correct maximum**. All other lanes have partial results from intermediate reduction steps.

Our first version then did:

```c
float d = amax / 127.0f;
float inv_d = (d > 0.0f) ? (1.0f / d) : 0.0f;
int q = __float2int_rn(val * inv_d);
```

Every lane used its own `amax` — which was wrong for 31 out of 32 lanes. The quantized int8 values were systematically wrong. We measured: **43% of q8 values had incorrect magnitude**, producing accumulation errors in the dot product that made the output subtly but consistently degraded.

The fix is one line:

```c
float amax = warp_reduce_max(fabsf(val));
amax = __shfl(amax, 0);  // broadcast lane 0's result to all lanes
```

A classic warp-reduce mistake. The reduction gives you the answer in one lane; if everyone needs it, you have to broadcast. This pattern is documented, it's well-known, and we still wrote the bug. The GPU doesn't warn you — it happily computes wrong answers at full speed.

### dp4a Results

With the corrected quantization and the dp4a inner loop:

| Phase | Before | After | Speedup |
|-------|-------:|------:|--------:|
| Attention (per layer) | 2.12 ms | 0.58 ms | **3.7x** |
| Expert compute (per layer) | 1.00 ms | 0.42 ms | **2.4x** |
| **Total tok/s** | **3.45** | **5.55** | **+61%** |

The attention phase went from 48% of total time to 22%. The bottleneck shifted from compute to I/O — exactly what we wanted.

---

## The hipMalloc Discovery

After dp4a, the profile looked like this:

```
phase     ms/layer   x60/token   share
io           2.28     136.8       53%
attn         0.61      36.6       22%
expert       0.93      55.8       20%
```

Expert compute was still at 0.93 ms/layer. We were using `hipMallocManaged` for the expert staging buffers — the same approach that worked for zero-copy access in the initial port. The GPU could read the buffer directly after `pread()` filled it, no explicit copy needed. But was `hipMallocManaged` actually giving us the best bandwidth?

We ran a test. Allocated the same buffer three ways, wrote data from the CPU, and measured GPU read bandwidth:

| Allocation | GPU Read Bandwidth | CPU Write? |
|------------|------------------:|:----------:|
| `hipMallocManaged` | 173 GB/s | Yes |
| `hipMalloc` | **220 GB/s** | ??? |
| `hipHostMallocNonCoherent` | 160 GB/s | Yes |

`hipMalloc` was 27% faster for GPU reads. But `hipMalloc` allocates *device* memory. On a discrete GPU, the CPU can't touch it — it's on the other side of a PCIe bus. On an APU... there is no other side. It's the same LPDDR5X.

The question: can the CPU `pread()` directly into a `hipMalloc`'d buffer?

```c
// The test that shouldn't work (but does)
void *d_buf;
hipMalloc(&d_buf, size);

// CPU writes directly to "device" memory
pread(fd, d_buf, size, offset);

// GPU kernel reads it
my_kernel<<<grid, block>>>(d_buf);
hipDeviceSynchronize();

// Verify: does the GPU see what the CPU wrote?
// YES.
```

It works. On an APU, `hipMalloc` returns a pointer that both the CPU and GPU can access, because they share the same physical address space. The reason it's faster than `hipMallocManaged` is **TLB coverage**: `hipMalloc` uses larger page table entries (likely 2MB huge pages vs 4KB for managed memory), so the GPU's TLB can cover more memory per entry, resulting in fewer TLB misses during the bandwidth-intensive dequant matvec kernel.

This is not documented anywhere in AMD's HIP programming guide. The guide says `hipMalloc` allocates device memory and the host cannot access it. That's true for discrete GPUs. On APUs, the implementation detail leaks through: "device memory" is just LPDDR5X with a different page table mapping.

We verified it three ways:
1. CPU `memset` to `hipMalloc` buffer, GPU reads correct values
2. CPU `pread()` to `hipMalloc` buffer, GPU reads correct values
3. GPU bandwidth test: 220 GB/s read vs 173 GB/s for managed

No existing inference engine does this. llama.cpp, vLLM, Ollama — they all use either `hipMallocManaged` or `hipHostMalloc` on AMD APUs. Nobody uses `hipMalloc` for CPU-written, GPU-read buffers on unified memory, because the documentation says you can't.

### hipMalloc Results

Switching expert staging from `hipMallocManaged` to `hipMalloc`:

```c
// Before: hipMallocManaged for expert staging
CHECK_HIP(hipMallocManaged(&model->h_expert_buf[i], g_expert_size));

// After: hipMalloc — 27% higher GPU bandwidth on APU
// CPU pread() works because CPU and GPU share physical LPDDR5X
CHECK_HIP(hipMalloc(&model->h_expert_buf[i], g_expert_size));
```

| Phase | Before | After | Speedup |
|-------|-------:|------:|--------:|
| Expert compute (per layer) | 0.93 ms | 0.42 ms | **2.2x** |
| **Total tok/s** | **5.55** | **6.14** | **+11%** |

---

## The Dead Ends

Not everything worked. Three promising ideas turned out to be dead on arrival:

### XNACK: Direct mmap GPU Access

The dream: `mmap()` the 209GB of expert files and let the GPU access them directly through page faults. No `pread()`, no staging buffers, no copies. The OS handles everything.

AMD's documentation mentions XNACK (eXtended NACK) as the mechanism for this — the GPU page-faults on a missing page, the Shader Sequencer retries the access after the kernel fills the page. It's how HMM (Heterogeneous Memory Management) is supposed to work.

The reality: **XNACK hardware was removed from all RDNA silicon starting with RDNA 2.** The retry circuit in the Shader Sequencer was dropped to save die area. It exists on CDNA (MI300X has it), but not on any consumer GPU or APU. When our kernel touched an unfaulted mmap'd page on gfx1151, we got a hard fault:

```
Memory access fault ... Reason: Page not present or supervisor privilege
```

This is not a driver bug. This is not a ROCm version issue. The hardware physically cannot do it. Any future RDNA consumer part would need to re-add the retry circuit, which AMD has shown no indication of doing.

### NPU Offload (XDNA2)

The Ryzen AI MAX+ 395 has an XDNA2 NPU rated at 50 TOPS int8. Could we offload some of the int8 dot products to it?

Two problems killed this:

1. **Shared bus.** The NPU reads from the same LPDDR5X as the GPU. Running both simultaneously causes memory controller contention, not parallelism. We'd trade GPU bandwidth for NPU bandwidth, not add to it.

2. **Dispatch overhead.** The XDNA2 dispatch latency is ~84 microseconds per call. Our engine makes 720 expert matvec calls per token. That's 720 x 84us = **60ms of pure dispatch overhead** — more than the entire expert compute phase.

The NPU might help for batch inference where you can amortize dispatch across many tokens, but for single-token autoregressive generation, it's a non-starter.

### Nontemporal Loads

RDNA 3.5 supports nontemporal load hints (`__builtin_nontemporal_load`) that bypass the Infinity Cache and go straight to DRAM. The theory: our dequant kernel streams weight data sequentially and never reuses it, so caching is waste. Bypassing the cache frees it for the activation vector and metadata.

The result: **0% improvement.** The Infinity Cache (32MB on Strix Halo) is not the bottleneck. The weights already stream efficiently through the cache hierarchy, and the working set of activations/scales/biases is small enough to fit regardless. The nontemporal hint changes nothing measurable.

---

## Final Results

Starting from 3.45 tok/s, two optimizations brought us to 6.14 tok/s:

| Optimization | tok/s | Improvement |
|-------------|------:|:----------:|
| Baseline (FMA kernel, hipMallocManaged) | 3.45 | — |
| + dp4a kernel (sudot4 int8 dot products) | 5.55 | +61% |
| + hipMalloc expert staging | **6.14** | **+78% total** |

The final per-phase breakdown:

```
phase        ms/layer   x60/token   %
io              2.28     136.8     59%  <-- SSD/page cache bottleneck
attn            0.61      36.6     22%
expert          0.42      25.2     15%
shared          0.08       4.8      3%
route           0.03       1.8      1%
total          ~2.72     163.0    6.14 tok/s
```

The engine is now **I/O dominated** at 59% — exactly where we want it. The GPU compute phases (attention + expert + shared) total 67ms of the 163ms per token. The remaining 136ms is waiting for data from SSD or page cache. Faster storage or a larger page cache (the 128GB Strix Halo SKU) would directly translate to higher tok/s.

### Cross-Hardware Comparison

| Hardware | Memory | tok/s | Notes |
|----------|--------|------:|-------|
| MI300X (192 GB VRAM) | Discrete HBM3 | 6.27 | 91% of experts cached in VRAM |
| **Strix Halo (64 GB)** | **Unified LPDDR5X** | **6.14** | **dp4a + hipMalloc** |
| RTX 4090 (24 GB VRAM) | Discrete GDDR6X | 5.57 | 17GB VRAM cache, PCIe 4.0 SSD |
| Apple M3 Max (48 GB) | Unified LPDDR5 | 4.36 | Trust the OS page cache |
| RTX 3060 (12 GB VRAM) | Discrete GDDR6 | 2.92 | Small VRAM cache |

A $2000 laptop APU matching a $15,000 datacenter GPU. The MI300X has 192GB of HBM3 at 5.3 TB/s bandwidth and barely edges out a chip that reads from LPDDR5X at 215 GB/s. The difference: MI300X has 91% of experts cached in VRAM (near-zero I/O), while Strix Halo has to read them from SSD every time. The dp4a kernel closed the compute gap; only I/O separates them now.

---

## What We Learned

**Instruction throughput matters more than bandwidth on RDNA.** Our original FMA kernel had 100% occupancy and only 39 VGPRs — every metric said it was healthy. But the ALU was doing 32 instructions per 8 nibbles when the hardware has a 4-for-1 instruction available. The roofline model said we should be 8.5% bandwidth-utilized; dp4a brought us to ~14%. Still not great, but the bottleneck moved to I/O, which is the correct place for it to be.

**Undocumented behavior is still behavior.** `hipMalloc` on APU giving CPU-accessible memory is a real, measurable, reproducible property of the hardware. We verified it three different ways. The documentation is wrong (or rather, written for discrete GPUs and never updated for APUs). Relying on undocumented behavior is risky for production systems, but for a research engine exploring the hardware's limits, it's exactly the kind of thing worth finding.

**Warp reduce needs a broadcast.** This is the most basic GPU programming mistake, and we made it. `warp_reduce_max` gives you the answer in lane 0. If all lanes need it, you must `__shfl(result, 0)`. The GPU will happily run at full speed with 97% of lanes using wrong values. Write a correctness test before you write a benchmark.

**Unified memory doesn't mean uniform performance.** `hipMalloc`, `hipMallocManaged`, and `hipHostMalloc` all allocate from the same physical LPDDR5X on an APU, but the page table mapping determines GPU bandwidth: 220 vs 173 vs 160 GB/s. The allocation API is an implicit choice about TLB behavior, not just about which processor can access the memory.

**The OS page cache is the right default on unified memory.** We tried custom VRAM caches, CPU offloading, NPU offloading, mmap with HMM page faults. The simplest approach — `pread()` into a device buffer and let the OS manage caching — gave the best results. Same lesson we learned on Apple Silicon, reconfirmed on AMD.

---

## The Code

Two files, no frameworks:

| File | Lines | Contents |
|------|------:|----------|
| `apu_infer/infer.hip` | ~3600 | Model loading, 60-layer forward pass, HTTP server, tool calling |
| `apu_infer/kernels.hip.h` | ~1200 | 17 HIP kernels including dp4a dequant matvec and Q8 quantization |

Build:

```bash
cd apu_infer && make
./infer --prompt "Explain quantum computing" --tokens 50
./infer --serve 8080  # OpenAI-compatible HTTP server
```

Requires: AMD APU with RDNA 3/3.5 (Strix Halo tested), ROCm 6.4+, 64GB+ unified memory, NVMe SSD with 250GB+ free. Model download ~209GB.

---

## What's Next

The engine is I/O bound at 59%. The path to faster inference on this hardware is faster storage:

- **PCIe 5.0 NVMe** (Samsung 990 Pro or similar): 7 GB/s → potentially 2x the I/O phase, pushing toward 9-10 tok/s
- **128GB Strix Halo SKU**: double the page cache capacity, dramatically reducing SSD reads after warm-up
- **BIOS VRAM split tuning**: our 32/32 split may not be optimal. 16/48 (more page cache) or 48/16 (VRAM cache enabled) could each win for different access patterns

The dp4a technique also applies to discrete AMD GPUs (RDNA 3, RDNA 4) and would improve their performance on this workload. And the hipMalloc discovery is relevant to any AMD APU inference engine — someone should probably tell the llama.cpp team.

---

*397 billion parameters. One APU chip. 64 GB of shared memory. 6.14 tokens per second. The line between "laptop" and "datacenter" hardware keeps getting thinner.*

---

*This work is part of the [Flash-MoE project](https://github.com/danveloper/flash-moe). The [paper](paper/flash_moe.pdf) covers the original Apple Silicon implementation, and the [CUDA article](article_medium.md) covers the NVIDIA port. The AMD APU backend lives on the `apu` branch.*
