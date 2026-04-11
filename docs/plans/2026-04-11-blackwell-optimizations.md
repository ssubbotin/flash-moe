# Flash-MoE Blackwell Optimizations — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Maximize Qwen3.5-397B tok/s on RTX PRO 6000 Blackwell (SM 12.0). Baseline: 8.30 tok/s warm.

**Architecture:** Five independent optimizations applied sequentially to `cuda_infer/`. Each optimization is measured against the previous baseline with a correctness gate — only changes that pass verification AND improve speed are kept. The key bottleneck is the matvec kernel (50% of compute time) and CPU sync for MoE routing (60 syncs/token).

**Tech Stack:** CUDA 13.1, SM 12.0 (Blackwell), C/CUDA single-file engine

**Target host:** aisrv (10.10.10.138, user1), RTX PRO 6000, 96 GB VRAM

---

## File Map

All changes in `cuda_infer/` on the `cuda` branch of `~/flash-moe`:

| Action | Path | Purpose |
|---|---|---|
| Modify | `cuda_infer/kernels.cuh` | Add INT8 kernels, GPU softmax/topK |
| Modify | `cuda_infer/infer.cu` | Wire new kernels, add benchmark harness |
| Modify | `cuda_infer/Makefile` | Update compiler flags |

---

### Task 1: Benchmark harness with correctness gate

**Files:**
- Modify: `cuda_infer/infer.cu` (add `--bench` flag)

The harness runs N tokens, captures output text, measures tok/s. Correctness: compare output token IDs against a reference run. Every subsequent task uses this harness.

- [ ] **Step 1: Add `--bench` CLI flag**

In `cuda_infer/infer.cu`, in the argument parsing section (around line 1430), add:

```c
int g_bench_mode = 0;
int g_bench_tokens = 20;

// In arg parsing:
if (strcmp(argv[i], "--bench") == 0) {
    g_bench_mode = 1;
    if (i + 1 < argc) g_bench_tokens = atoi(argv[++i]);
}
```

- [ ] **Step 2: Add benchmark runner after the main generation loop**

After the existing generation code (around line 2100), add a benchmark block that:
1. Runs `g_bench_tokens` tokens 3 times
2. Records tok/s for each run
3. Prints min/max/avg
4. Outputs the generated token IDs for correctness comparison

```c
if (g_bench_mode) {
    printf("\n[bench] Running %d tokens x 3 iterations...\n", g_bench_tokens);
    // Save first run's token IDs as reference
    int ref_tokens[512];
    double times[3];
    for (int iter = 0; iter < 3; iter++) {
        // Reset model state, run generation, time it
        // Compare token IDs with ref_tokens on iter > 0
        // Print: "[bench] iter N: X.XX tok/s [MATCH|MISMATCH]"
    }
    printf("[bench] avg: %.2f tok/s (min=%.2f max=%.2f)\n",
           avg, min, max);
}
```

- [ ] **Step 3: Build and establish baseline**

```bash
ssh user1@10.10.10.138 'cd ~/flash-moe && make -C cuda_infer clean && make -C cuda_infer NVCC=/usr/local/cuda-13.1/bin/nvcc CFLAGS="-O2 -arch=sm_120"'
```

Run baseline:
```bash
ssh user1@10.10.10.138 'cd ~/flash-moe && sudo systemctl stop vllm-qwen-coder; cuda_infer/infer --bench 50'
```

Expected: ~8.3 tok/s, all 3 iterations MATCH.

- [ ] **Step 4: Commit**

```bash
cd ~/flash-moe && git add cuda_infer/ && git commit -m "bench: add --bench harness with correctness gate"
```

---

### Task 2a: INT8 dp4a matvec kernel

Port the APU's `quantize_x_q8` + `dequant_matvec_4bit_dp4a` to CUDA using `__dp4a()` intrinsic.

**Files:**
- Modify: `cuda_infer/kernels.cuh` (add 2 new kernels + launch wrappers)
- Modify: `cuda_infer/infer.cu` (add INT8 buffers, add `--int8` flag, wire into `do_matvec`)

- [ ] **Step 1: Add `quantize_x_q8` kernel to `kernels.cuh`**

Port from APU's `kernels.hip.h`. Replace `__shfl` with `__shfl_sync`, `warp_reduce_max` with CUDA warp reduce. WARP_SIZE=32 (same on CUDA). Add after the existing kernel #15.

```cuda
// 16. Activation quantization: float32 → Q8_1 (int8 + per-block scale/sum)
#define Q8_BLOCK_SIZE 32

__global__ void quantize_x_q8(
    const float* __restrict__ x,
    int8_t*      __restrict__ q8,
    float*       __restrict__ q8_scales,
    float*       __restrict__ q8_sums,
    uint32_t dim
) {
    uint32_t block_id = blockIdx.x;
    uint32_t lane = threadIdx.x;
    uint32_t base = block_id * Q8_BLOCK_SIZE;
    if (base >= dim) return;

    uint32_t idx = base + lane;
    float val = (idx < dim) ? x[idx] : 0.0f;

    // Warp max
    float amax = fabsf(val);
    for (int offset = 16; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_down_sync(0xffffffff, amax, offset));
    amax = __shfl_sync(0xffffffff, amax, 0);

    float d = amax / 127.0f;
    float inv_d = (d > 0.0f) ? (1.0f / d) : 0.0f;
    int q = __float2int_rn(val * inv_d);
    q = max(-128, min(127, q));

    if (idx < dim) q8[idx] = (int8_t)q;

    // Warp sum for bias correction
    float q_sum = (float)q;
    for (int offset = 16; offset > 0; offset >>= 1)
        q_sum += __shfl_down_sync(0xffffffff, q_sum, offset);

    if (lane == 0) {
        q8_scales[block_id] = d;
        q8_sums[block_id] = q_sum;
    }
}
```

- [ ] **Step 2: Add `dequant_matvec_4bit_dp4a` kernel to `kernels.cuh`**

Port from APU. Replace `__builtin_amdgcn_sudot4` with `__dp4a`. The math is identical:

```cuda
// 17. dp4a int8 dequant matvec — uses pre-quantized int8 activations
#define DP4A_ROWS_PER_BLOCK 4

__global__ void dequant_matvec_4bit_dp4a(
    const uint32_t* __restrict__ W_packed,
    const uint16_t* __restrict__ scales,
    const uint16_t* __restrict__ biases,
    const int8_t*   __restrict__ q8,
    const float*    __restrict__ q8_scales,
    const float*    __restrict__ q8_sums,
    float*          __restrict__ out,
    uint32_t out_dim,
    uint32_t in_dim
) {
    // Same structure as APU kernel but using __dp4a()
    // __dp4a(a, b, c) computes: c + dot4(a_bytes, b_bytes)
    // where a has 4 unsigned 8-bit values, b has 4 signed 8-bit values
    // ...
}
```

Full kernel: port line-by-line from `git show apu:apu_infer/kernels.hip.h` replacing `sudot4_ui8(a, b, c)` with `__dp4a(a, b, c)`.

- [ ] **Step 3: Add INT8 buffers to Model struct in `infer.cu`**

Around line 200 (Model struct):
```c
// INT8 activation buffers
int8_t *buf_q8;        // [max_dim] quantized activations
float *buf_q8_scales;  // [max_dim/32] per-block scales
float *buf_q8_sums;    // [max_dim/32] per-block sums
```

Allocate in `init_model()` (around line 400):
```c
CHECK_CUDA(cudaMalloc(&model->buf_q8, HIDDEN_DIM * sizeof(int8_t)));
CHECK_CUDA(cudaMalloc(&model->buf_q8_scales, (HIDDEN_DIM / Q8_BLOCK_SIZE) * sizeof(float)));
CHECK_CUDA(cudaMalloc(&model->buf_q8_sums, (HIDDEN_DIM / Q8_BLOCK_SIZE) * sizeof(float)));
```

- [ ] **Step 4: Add `--int8` flag and wire into `do_matvec`**

Add `g_use_int8` global flag. When enabled, `do_matvec` first quantizes x to Q8, then calls `dequant_matvec_4bit_dp4a` instead of `dequant_matvec_4bit_fma`.

```c
static int g_use_int8 = 0;

static inline void do_matvec(
    const uint32_t *W, const uint16_t *S, const uint16_t *B,
    const float *x, float *out, uint32_t out_dim, uint32_t in_dim,
    int gguf_type, cudaStream_t stream = 0
) {
    if (g_quant_format == 1) {
        launch_dequant_matvec_gguf((const void *)W, x, out, out_dim, in_dim, gguf_type, stream);
    } else if (g_use_int8) {
        // Quantize activation to INT8
        launch_quantize_x_q8(x, g_model->buf_q8, g_model->buf_q8_scales,
                              g_model->buf_q8_sums, in_dim, stream);
        // dp4a matvec
        launch_dequant_matvec_dp4a(W, S, B, g_model->buf_q8,
                                    g_model->buf_q8_scales, g_model->buf_q8_sums,
                                    out, out_dim, in_dim, stream);
    } else {
        launch_dequant_matvec(W, S, B, x, out, out_dim, in_dim, stream);
    }
}
```

- [ ] **Step 5: Build and benchmark**

```bash
ssh user1@10.10.10.138 'cd ~/flash-moe && make -C cuda_infer clean && make -C cuda_infer NVCC=/usr/local/cuda-13.1/bin/nvcc CFLAGS="-O2 -arch=sm_120"'
ssh user1@10.10.10.138 'cd ~/flash-moe && cuda_infer/infer --bench 50 --int8'
```

Expected: correctness MATCH, tok/s improvement measurable.

- [ ] **Step 6: Commit**

```bash
git add cuda_infer/ && git commit -m "perf(cuda): dp4a int8 dequant matvec kernel — port from APU"
```

---

### Task 2b: Tensor core WMMA INT8 matvec

Alternative INT8 implementation using `wmma::mma_sync` for higher throughput.

**Files:**
- Modify: `cuda_infer/kernels.cuh` (add WMMA kernel)
- Modify: `cuda_infer/infer.cu` (add `--wmma` flag)

- [ ] **Step 1: Add WMMA INT8 kernel to `kernels.cuh`**

Use `nvcuda::wmma` with 16×16×16 int8 tiles. The matvec M=1 case needs special handling — pad to M=16 or use a row-tile approach.

```cuda
#include <mma.h>
using namespace nvcuda;

// 18. Tensor core WMMA int8 matvec
__global__ void dequant_matvec_4bit_wmma(
    const uint32_t* __restrict__ W_packed,
    const uint16_t* __restrict__ scales,
    const uint16_t* __restrict__ biases,
    const int8_t*   __restrict__ q8,
    const float*    __restrict__ q8_scales,
    float*          __restrict__ out,
    uint32_t out_dim,
    uint32_t in_dim
) {
    // Load weight tile as int8 (dequant 4-bit → int8 on the fly)
    // Load activation tile from q8
    // wmma::mma_sync with int8 accumulating to int32
    // Apply scales post-multiply
}
```

- [ ] **Step 2: Add `--wmma` flag, wire into do_matvec, build, benchmark**

Same pattern as Task 2a but with `g_use_wmma` flag.

- [ ] **Step 3: Commit**

```bash
git commit -m "perf(cuda): WMMA tensor core int8 matvec kernel"
```

---

### Task 2c: cuBLAS INT8 GEMV

**Files:**
- Modify: `cuda_infer/infer.cu` (add `--cublas` flag)
- Modify: `cuda_infer/Makefile` (add `-lcublas`)

- [ ] **Step 1: Add cuBLAS INT8 GEMV path**

Use `cublasLtMatmul` with INT8 compute type. Requires weight repacking to INT8 column-major format at load time.

```c
#include <cublasLt.h>

// Pre-convert weights from 4-bit → INT8 at model load
// Use cublasLtMatmul with CUDA_R_8I compute for the matvec
```

- [ ] **Step 2: Build with `-lcublas -lcublasLt`, benchmark**

- [ ] **Step 3: Commit**

```bash
git commit -m "perf(cuda): cuBLAS INT8 GEMV path"
```

---

### Task 2d: Compare all three INT8 approaches

- [ ] **Step 1: Run all three on the same prompt**

```bash
# Baseline (FMA)
cuda_infer/infer --bench 50
# dp4a
cuda_infer/infer --bench 50 --int8
# WMMA
cuda_infer/infer --bench 50 --wmma
# cuBLAS
cuda_infer/infer --bench 50 --cublas
```

- [ ] **Step 2: Record results and keep the winner as default**

Update `do_matvec` to use the fastest INT8 path by default when SM >= 12.0 is detected.

- [ ] **Step 3: Commit**

```bash
git commit -m "perf(cuda): select best INT8 kernel — [winner] is N% faster"
```

---

### Task 3: GPU-side softmax + topK (eliminate CPU sync)

Move MoE routing from CPU to GPU to remove the forced `cudaDeviceSynchronize()` at line 1850.

**Files:**
- Modify: `cuda_infer/kernels.cuh` (add `gpu_softmax_topk` kernel)
- Modify: `cuda_infer/infer.cu` (replace CPU routing with GPU kernel)

- [ ] **Step 1: Add `gpu_softmax_topk` kernel**

```cuda
// 19. GPU-side softmax + topK for MoE routing
// Single block, NUM_EXPERTS threads (512 for Qwen3.5)
__global__ void gpu_softmax_topk(
    const float* __restrict__ scores,   // [NUM_EXPERTS]
    float* __restrict__ out_weights,     // [K]
    int* __restrict__ out_indices,       // [K]
    int num_experts,
    int k
) {
    // Shared memory softmax
    // Parallel topK via k-pass argmax
    // Write top-K indices + weights to device memory
}
```

- [ ] **Step 2: Replace CPU routing in `infer.cu`**

Replace lines 1850-1858:
```c
// OLD: sync + memcpy + CPU softmax + CPU topK
// NEW: launch GPU kernel, sync only once for the expert IDs
launch_gpu_softmax_topk(model->buf_gate_scores, model->buf_expert_weights,
                        model->buf_expert_ids, NUM_EXPERTS, K);
CHECK_CUDA(cudaDeviceSynchronize());
CHECK_CUDA(cudaMemcpy(expert_ids, model->buf_expert_ids, K * sizeof(int), cudaMemcpyDeviceToHost));
CHECK_CUDA(cudaMemcpy(expert_weights, model->buf_expert_weights, K * sizeof(float), cudaMemcpyDeviceToHost));
```

This still needs one sync (CPU must know which experts to load from SSD), but removes the separate gate matvec sync.

- [ ] **Step 3: Build, verify correctness (same expert choices), benchmark**

Expected: same token output, ~10-20% faster from reduced sync overhead.

- [ ] **Step 4: Commit**

```bash
git commit -m "perf(cuda): GPU-side softmax+topK — remove routing sync"
```

---

### Task 4: Compiler flags optimization

**Files:**
- Modify: `cuda_infer/Makefile`

- [ ] **Step 1: Update Makefile**

```makefile
NVCC = /usr/local/cuda-13.1/bin/nvcc
CFLAGS = -O3 --use_fast_math -arch=sm_120 -Wno-deprecated-gpu-targets
```

- [ ] **Step 2: Build, verify correctness, benchmark**

```bash
make -C cuda_infer clean && make -C cuda_infer
cuda_infer/infer --bench 50
```

Expected: same output, +3-5% from compiler optimizations.

- [ ] **Step 3: Commit**

```bash
git commit -m "perf(cuda): -O3 --use_fast_math for Blackwell"
```

---

### Task 5: CUDA streams for I/O overlap

Pipeline expert SSD reads with shared expert GPU compute.

**Files:**
- Modify: `cuda_infer/infer.cu` (add stream for expert I/O)

- [ ] **Step 1: Create a compute stream for shared expert**

```c
cudaStream_t stream_compute, stream_io;
cudaStreamCreate(&stream_compute);
cudaStreamCreate(&stream_io);
```

- [ ] **Step 2: Run shared expert on `stream_compute` while expert I/O happens on CPU**

The existing code already does shared expert after routing (lines 1867-1878). Move shared expert launches to `stream_compute`, expert pread to overlap.

- [ ] **Step 3: Benchmark cold and warm**

Cold (I/O heavy): should see improvement.
Warm (no I/O): should be neutral.

- [ ] **Step 4: Commit**

```bash
git commit -m "perf(cuda): CUDA streams — overlap shared expert with I/O"
```

---

## Success Criteria

| Task | Metric | Target |
|---|---|---|
| 1 (Harness) | Builds, baseline established | 8.3 tok/s reference |
| 2 (INT8) | Best of 3 approaches wins | +30-60% on matvec |
| 3 (GPU routing) | Same expert choices | +10-20% total |
| 4 (Flags) | Same output | +3-5% |
| 5 (Streams) | Cold start improvement | +5-10% cold |
| **Combined** | **All correctness gates pass** | **12+ tok/s** |
