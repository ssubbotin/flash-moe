# aotriton Flash Attention API (as observed on mi300, ROCm 7.2)

## Install location
- Library: `/opt/aotriton/lib/libaotriton_v2.so` (after source build completes)
- Headers: `/opt/aotriton/include/aotriton/`
- Version: `5b05b446e38d4db3f5c68b2643a2f71d2ed2a8af` (git SHA, commit: "Migrate the Tuner to V3 API (#164)")
- aotriton package version: `0.12.0`
- Installed via: source build from `https://github.com/ROCm/aotriton.git`
- Target arch: `gfx942` (MI300X)

Headers were inspected from the source tree at `/tmp/aotriton/include/aotriton/` before build completion.
Not available via `apt-get` and not bundled in `/opt/rocm*`.

## Forward attention function

### V3 API (current, use this one)

#### Params struct
```c++
// aotriton/flash.h — namespace aotriton::v3::flash

struct AOTRITON_API attn_fwd_params {
  TensorView<4> Q;           // query tensor
  TensorView<4> K;           // key tensor
  TensorView<4> V;           // value tensor
  TensorView<4> B;           // attention bias (set to null tensor if unused)
  TensorView<2> A;           // alibi slopes (set to null tensor if unused)
  float         Sm_scale;    // softmax scale = 1/sqrt(head_dim)
  TensorView<2> L;           // log-sum-exp output (can be null tensor)
  TensorView<4> Out;         // output tensor
  TensorView<1> cu_seqlens_q;   // cumulative seqlens for Q (null if not varlen)
  TensorView<1> cu_seqlens_k;   // cumulative seqlens for K (null if not varlen)
  int32_t       Max_seqlen_q;   // unused if cu_seqlens_q is empty
  int32_t       Max_seqlen_k;   // unused if cu_seqlens_k is empty
  TensorView<1> seq_strides_q;  // padded varlen strides (null if not used)
  TensorView<1> seq_strides_k;
  float         dropout_p;      // dropout probability (set to 0.0f to disable)
  TensorView<0> philox_seed_ptr;
  TensorView<0> philox_offset1;
  uint64_t      philox_offset2;
  TensorView<0> philox_seed_output;
  TensorView<0> philox_offset_output;
  TensorView<4> encoded_softmax;   // debug output (null tensor normally)
  TensorView<0> persistent_atomic_counter; // for causal persistent kernels
  int8_t        causal_type;    // CausalType::None (0) or CausalType::WindowedAttention (3)
  int8_t        varlen_type;    // VarlenType::None (0) for fixed-length batches
  int32_t       window_left;    // sliding window left bound (WindowedAttention only)
  int32_t       window_right;   // sliding window right bound (WindowedAttention only)

  static constexpr int32_t kVersion = 3;
  attn_fwd_params();  // default-constructs all null tensors
};
```

#### Call signature
```c++
hipError_t AOTRITON_API
aotriton::v3::flash::attn_fwd(
    const attn_fwd_params& params,
    int32_t                params_version,   // pass attn_fwd_params::kVersion (= 3)
    aotriton::Stream       stream,           // wraps hipStream_t
    const attn_options*    options = nullptr // nullptr = use default kernel selection
);
```

### V2 API (deprecated, kept for reference)

The V2 API exposes flat function arguments instead of a params struct:

```c++
// aotriton/v2/flash.h — namespace aotriton::v2::flash  (DEPRECATED)
[[deprecated]]
hipError_t AOTRITON_API
attn_fwd(TensorView<4> q,          // batch_size x num_heads x seqlen_q x hdim_qk
         TensorView<4> k,          // batch_size x num_heads x seqlen_k x hdim_qk
         TensorView<4> v,          // batch_size x num_heads x seqlen_k x hdim_vo
         TensorView<4> b,          // batch_size x num_heads x seqlen_q x seqlen_k (bias)
         float         sm_scale,
         TensorView<2> softmax_lse,
         TensorView<4> Out,        // batch_size x num_heads x seqlen_q x hdim_vo
         float         dropout_p,
         TensorView<0> philox_seed,
         TensorView<0> philox_offset1,
         int64_t       philox_offset2,
         TensorView<0> philox_seed_output,
         TensorView<0> philox_offset_output,
         TensorView<4> encoded_softmax,
         bool          is_causal,
         TensorView<0> atomic_for_causal,
         aotriton::Stream stream,
         FwdExtraArguments* extargs = nullptr);
```

V2 is marked `[[deprecated]]` everywhere in the headers. Use V3.

## TensorView wrapper

`TensorView<Rank>` is a thin C++ wrapper around a raw pointer + sizes + strides + dtype.
It is **not** a managed tensor — no ownership, no device allocation.

```c++
// Construction (rank-N):
TensorView<4> tv(
    reinterpret_cast<intptr_t>(device_ptr),   // raw GPU pointer cast to intptr_t
    {batch, heads, seqlen, head_dim},         // std::array<uint64_t, 4> sizes
    {heads*seqlen*hdim, seqlen*hdim, hdim, 1}, // std::array<uint64_t, 4> strides (in elements)
    aotriton::DType::kFloat16                 // dtype
);

// Null tensor (for optional fields like B, A, cu_seqlens_q when not needed):
auto null4 = TensorView<4>::get_null_tensor(aotriton::DType::kFloat16);
auto null1 = TensorView<1>::get_null_tensor(aotriton::DType::kFloat16);
auto null0 = TensorView<0>::get_null_tensor(aotriton::DType::kFloat16);
```

**Strides are in units of elements, not bytes.**

## DType enum

```c++
// aotriton/dtypes.h
namespace aotriton {
  enum DType : int32_t {
    kUnknown  = 0,
    kFloat32  = 1,
    kFloat16  = 2,
    kBFloat16 = 3,
    kInt8     = 10,
    // ...
  };
}
```

## Stream wrapper

```c++
// aotriton/runtime.h
aotriton::Stream stream(hip_stream_t_value);  // wraps hipStream_t
// Or use default stream:
aotriton::Stream stream(nullptr);
```

## Parameters table (V3 attn_fwd_params)

| Field | Type | Meaning |
|-------|------|---------|
| Q | TensorView<4> | Query: `[batch, num_heads_q, seqlen_q, head_dim]` |
| K | TensorView<4> | Key: `[batch, num_heads_k, seqlen_k, head_dim]` |
| V | TensorView<4> | Value: `[batch, num_heads_k, seqlen_k, head_dim_v]` |
| B | TensorView<4> | Attention bias (null tensor if none) |
| A | TensorView<2> | ALiBi slopes (null tensor if none) |
| Sm_scale | float | `1.0f / sqrtf(head_dim)` |
| L | TensorView<2> | log-sum-exp output `[batch, num_heads_q, seqlen_q]` — can be null |
| Out | TensorView<4> | Output: `[batch, num_heads_q, seqlen_q, head_dim_v]` |
| cu_seqlens_q | TensorView<1> | Varlen cumulative seqlens for Q (null for fixed-length) |
| cu_seqlens_k | TensorView<1> | Varlen cumulative seqlens for K (null for fixed-length) |
| Max_seqlen_q | int32_t | Max seqlen among batch items (0 if not varlen) |
| Max_seqlen_k | int32_t | Max seqlen among batch items (0 if not varlen) |
| dropout_p | float | Dropout probability (0.0f to disable) |
| causal_type | int8_t | `CausalType::None=0`, `CausalType::WindowedAttention=3` |
| varlen_type | int8_t | `VarlenType::None=0` for standard fixed-length |
| window_left/right | int32_t | Sliding window bounds (WindowedAttention only) |
| philox_* | TensorView<0> | RNG state for dropout; use null tensors when dropout_p=0 |
| encoded_softmax | TensorView<4> | Debug output (null tensor in production) |
| persistent_atomic_counter | TensorView<0> | Required for causal persistent kernels; can be null for non-causal |

## Tensor layout expectations

From the V2 comments (V3 uses same layout, just passed via TensorView):
- Q shape: `[batch_size, num_heads, seqlen_q, hdim_qk]` — row-major BHSD
- K shape: `[batch_size, num_heads, seqlen_k, hdim_qk]` — row-major BHSD
- V shape: `[batch_size, num_heads, seqlen_k, hdim_vo]` — row-major BHSD
- Out shape: `[batch_size, num_heads, seqlen_q, hdim_vo]` — row-major BHSD
- Dtype: fp16 (`kFloat16`) or bf16 (`kBFloat16`). **fp32 is NOT supported for Q/K/V/Out.**
  - `AOTRITON_ENABLE_FP32=1` is set during cmake configure but this controls internal
    accumulation precision, not the user-facing tensor dtype.
- Strides: must be row-major (contiguous or strided, but strides in elements).

## Flags summary

- **Causal**: controlled by `causal_type` field. `CausalType::None` = 0 (no mask). No `bool is_causal` — use the enum field.
- **Dropout**: `dropout_p = 0.0f` disables dropout; set all philox TensorView<0> fields to null tensors.
- **Stream**: `aotriton::Stream stream(hipStream_t_value)` — explicit HIP stream, not optional.
- **Varlen**: `varlen_type = VarlenType::None` and null `cu_seqlens_q/k` for standard fixed-length batches.

## Return type

Returns `hipError_t`. `hipSuccess` (0) on success.

## Example call (decode step, seqlen_q=1, no dropout, no bias, causal)

```c++
#include <aotriton/flash.h>
#include <aotriton/util.h>

using namespace aotriton;
using namespace aotriton::v3::flash;

// Assume fp16 device buffers are already allocated:
//   __half* d_q   — shape [1, num_heads_q, 1, head_dim]
//   __half* d_k   — shape [1, num_heads_k, current_pos+1, head_dim]
//   __half* d_v   — shape [1, num_heads_k, current_pos+1, head_dim]
//   __half* d_out — shape [1, num_heads_q, 1, head_dim]

int batch    = 1;
int nheads_q = 64;   // e.g., Qwen3-235B: 64 heads
int nheads_k = 8;    // GQA: 8 KV heads
int seqlen_q = 1;    // decode step
int seqlen_k = current_pos + 1;
int hdim     = 128;

float sm_scale = 1.0f / sqrtf((float)hdim);

// Build TensorViews (strides in elements, row-major)
TensorView<4> Q(
    (intptr_t)d_q,
    {(uint64_t)batch, (uint64_t)nheads_q, (uint64_t)seqlen_q, (uint64_t)hdim},
    {(uint64_t)nheads_q*seqlen_q*hdim, (uint64_t)seqlen_q*hdim, (uint64_t)hdim, 1ULL},
    DType::kFloat16
);
TensorView<4> K(
    (intptr_t)d_k,
    {(uint64_t)batch, (uint64_t)nheads_k, (uint64_t)seqlen_k, (uint64_t)hdim},
    {(uint64_t)nheads_k*seqlen_k*hdim, (uint64_t)seqlen_k*hdim, (uint64_t)hdim, 1ULL},
    DType::kFloat16
);
TensorView<4> V(
    (intptr_t)d_v,
    {(uint64_t)batch, (uint64_t)nheads_k, (uint64_t)seqlen_k, (uint64_t)hdim},
    {(uint64_t)nheads_k*seqlen_k*hdim, (uint64_t)seqlen_k*hdim, (uint64_t)hdim, 1ULL},
    DType::kFloat16
);
TensorView<4> Out(
    (intptr_t)d_out,
    {(uint64_t)batch, (uint64_t)nheads_q, (uint64_t)seqlen_q, (uint64_t)hdim},
    {(uint64_t)nheads_q*seqlen_q*hdim, (uint64_t)seqlen_q*hdim, (uint64_t)hdim, 1ULL},
    DType::kFloat16
);

auto null4 = TensorView<4>::get_null_tensor(DType::kFloat16);
auto null2 = TensorView<2>::get_null_tensor(DType::kFloat16);
auto null1 = TensorView<1>::get_null_tensor(DType::kFloat16);
auto null0 = TensorView<0>::get_null_tensor(DType::kFloat16);

attn_fwd_params params;  // default constructor zeroes everything
params.Q            = Q;
params.K            = K;
params.V            = V;
params.B            = null4;
params.A            = null2;  // TensorView<2>
params.Sm_scale     = sm_scale;
params.L            = null2;
params.Out          = Out;
params.cu_seqlens_q = null1;
params.cu_seqlens_k = null1;
params.Max_seqlen_q = 0;
params.Max_seqlen_k = 0;
params.seq_strides_q = null1;
params.seq_strides_k = null1;
params.dropout_p    = 0.0f;
params.philox_seed_ptr         = null0;
params.philox_offset1          = null0;
params.philox_offset2          = 0;
params.philox_seed_output      = null0;
params.philox_offset_output    = null0;
params.encoded_softmax         = null4;
params.persistent_atomic_counter = null0;
params.causal_type  = CausalType::None;  // or CausalType::WindowedAttention with window_left/right
params.varlen_type  = VarlenType::None;
params.window_left  = 0;
params.window_right = 0;

aotriton::Stream aotriton_stream(hip_stream);

hipError_t err = attn_fwd(params, attn_fwd_params::kVersion, aotriton_stream);
if (err != hipSuccess) { /* handle */ }
```

## Library name for linker

The cmake install layout under `/opt/aotriton`:
- `lib/libaotriton_v2.so` — main library
- `include/aotriton/` — public headers

Link flags: `-L/opt/aotriton/lib -laotriton_v2`

Soname: check with `readelf -d /opt/aotriton/lib/libaotriton_v2.so | grep SONAME` after build.

## Notes on integrating with rocm_infer

### fp32 problem (CRITICAL)

`infer.hip` accumulates attention in `float*` buffers. aotriton's kernel only supports
fp16 and bf16 for Q/K/V/Out — **not fp32**. The `AOTRITON_ENABLE_FP32=1` cmake flag
enables fp32 *accumulation precision inside the kernel*, not fp32 user tensors.

**Mitigation options (in order of preference):**
1. Add a `float→half` staging kernel before the aotriton call and `half→float` after.
   This costs one extra pass over Q and one over Out per attention layer — acceptable
   since attention is 15/60 layers and the conversion is memory-bandwidth-limited.
2. Change `buf_q`, `buf_k`, `buf_v`, `buf_out_attn` to `__half*` for the 15 full-attention
   layers. Requires surgery in infer.hip but avoids the staging copies.

### KV cache layout mismatch

Our KV cache is stored as `float*` at `[MAX_SEQ_LEN * kv_dim]` (flattened, one buffer per layer).
aotriton expects `[batch=1, num_kv_heads, seqlen_k, head_dim]` row-major with explicit strides.

**Reinterpretation:** if `kv_dim = num_kv_heads * head_dim` and the cache is contiguous,
the existing layout can be reinterpreted as `[1, num_kv_heads, seqlen_k, head_dim]` with strides
`[num_kv_heads*seqlen_k*head_dim, seqlen_k*head_dim, head_dim, 1]` — **this matches exactly**
if the cache is stored as `[seqlen_k, num_kv_heads, head_dim]` (NHD/time-first). Verify the
actual storage order in `infer.hip` before assuming.

### Decode-step (seqlen_q=1) support

aotriton v3 supports decode-step (seqlen_q=1, batch=1) via the standard `attn_fwd` call —
no special "decode-only" function required. The kernel selects an appropriate tiling at
dispatch time based on seqlen_q.

### GQA (grouped-query attention)

Qwen3.5-397B uses GQA (64 Q heads, 8 KV heads on some configs). aotriton supports GQA:
`Q.size(1) != K.size(1)` is valid as long as `Q.size(1)` is a multiple of `K.size(1)`.
The `num_head_q` and `num_head_k` are inferred from the TensorView sizes — no explicit
parameter needed.

### params_version ABI stability

The `params_version` argument (pass `attn_fwd_params::kVersion = 3`) is a version sentinel.
If we link against a future libaotriton_v2.so with a different struct layout, the call will
return an error rather than silently corrupt data. **Always pass `attn_fwd_params::kVersion`.**
Do not hardcode `3`.

### causal vs windowed

Our 15 full-attention layers use standard causal attention (lower-triangular mask).
Use `CausalType::None` with `window_right=0` and `window_left = -1` (full left context),
or use `CausalType::WindowedAttention` with `window_left = -1` and
`window_right = 0` which is equivalent to standard causal in this API.

Actually: `CausalType::None` = no mask (bidirectional). For causal, check whether the
header means "no causal type" or "fully unmasked". The V2 API had a `bool is_causal`
parameter; the V3 API replaces it with the `causal_type` field. Recommend testing
both `CausalType::None` with no mask and `CausalType::WindowedAttention` with a large
`window_left` to verify which produces correct outputs.
