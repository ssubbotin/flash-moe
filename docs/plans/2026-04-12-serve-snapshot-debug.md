# Serve Mode Snapshot Debug — Plan

## Bug

`--prompt` mode with ChatML produces correct output (`<think>...Paris...`).
Serve mode with identical tokens produces wrong output (`<|im_end|>` after 1 token).
Both paths call `model_reset_state()` + `forward()` with the same token IDs.

## Known facts

- Token IDs are identical (verified with debug prints)
- `model_reset_state()` zeros kv_k/kv_v/kv_len/delta_state/conv_state
- The `forward()` function is the same code
- The bug exists on BOTH RTX 4090 and RTX PRO 6000
- Metal backend's serve mode works with the same weights

## Hypothesis

Something in the serve code path changes model state BEFORE `model_reset_state()` in a way that `model_reset_state()` doesn't fully clean up. Candidates:

1. **Startup system prompt prefill** — runs 14 tokens through `forward()` at startup, modifying state. Then `model_reset_state()` zeros state for the request. But maybe a buffer NOT listed in `model_reset_state()` retains stale values.

2. **Missing buffer in `model_reset_state()`** — there might be a hidden state buffer (conv_state layout, attention bias, or scratch buffer) that accumulates state across tokens and isn't zeroed.

3. **Global variable pollution** — a global counter, position tracker, or flag set during startup prefill that affects subsequent `forward()` calls.

## Debug plan

### Step 0: Verify Metal serve actually works

Before using `mini` as reference, confirm the Metal serve mode produces correct chat responses:

```bash
ssh mini 'cd ~/flash-moe/metal_infer && ./infer --serve 8080 &'
sleep 15
curl -s -N http://mini:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":200}' \
  | head -20
# Expected: streamed response with "4" in the content
ssh mini 'pkill -f "infer --serve"'
```

If Metal serve also produces `<|im_end|>` after 1 token, the bug is in the shared prompt building code (`build_chat_prompt`), not the snapshot restore. If Metal works correctly, the bug is CUDA-specific state management.

### Step 1: Identify all state-carrying buffers

Read `forward()` in `infer.cu` line by line. For every buffer that's READ from and also WRITTEN to across tokens, verify it's zeroed in `model_reset_state()`.

Checklist:
- [ ] kv_k, kv_v, kv_len (full attention layers)
- [ ] delta_state (GDN layers)
- [ ] conv_state (Conv1D layers)
- [ ] Any other buffer with cross-token state

### Step 2: Compare layer-0 hidden state

After the FIRST token prefill, dump `model->buf_hidden` (4096 floats) for:
- A: `--prompt` mode (fresh model, no prior state)
- B: serve mode after `model_reset_state()` (model ran startup prefill, then zeroed)

If A != B after processing the same first token, a stale buffer is the cause.

### Step 3: Binary search for the stale buffer

If Step 2 shows divergence, add `cudaMemset(0)` calls one-by-one for every scratch buffer in `forward()` between `model_reset_state()` and the first `forward()` call. When A == B, the culprit is found.

### Step 4: Fix

Add the missing buffer to `model_reset_state()`, or restructure to avoid the stale state.

## Available hosts

| Host | GPU | Backend | Serve works? |
|---|---|---|---|
| `ssh user1@10.10.10.138` (aisrv) | RTX PRO 6000 Blackwell | cuda_infer | **No** — snapshot bug |
| `ssh user1@10.10.10.139` (aeronav-llm) | RTX 4090 Ada | cuda_infer | **No** — same bug |
| `ssh max395` | AMD Strix Halo APU | apu_infer | Check |
| `ssh mini` | Apple M4 Pro | metal_infer | **Yes** — reference |

Strategy: dump layer-0 hidden state on **mini** (known good) and **aisrv** (broken), compare byte-for-byte. The APU host can validate whether the bug is CUDA-specific or also affects HIP.

## How to run

Build with debug dumps:
```bash
# On aisrv:
cd ~/flash-moe/cuda_infer
/usr/local/cuda-13.1/bin/nvcc -O2 -arch=sm_120 -o infer infer.cu tokenizer_impl.o \
  -lpthread -L/usr/local/cuda-13.1/targets/x86_64-linux/lib -lcufile -lcublas -lcublasLt

# Capture --prompt hidden state after first token:
./infer --prompt "<ChatML prompt>" --tokens 1 --dump-hidden /tmp/hidden_prompt.bin

# Capture serve hidden state after first token:
./infer --serve 8000 --dump-hidden /tmp/hidden_serve.bin
# Then send a request

# Compare:
python3 -c "
import numpy as np
a = np.fromfile('/tmp/hidden_prompt.bin', dtype=np.float32)
b = np.fromfile('/tmp/hidden_serve.bin', dtype=np.float32)
diff = np.abs(a - b)
print(f'Max diff: {diff.max():.6e}')
print(f'Mean diff: {diff.mean():.6e}')
print(f'Nonzero: {(diff > 1e-6).sum()} / {len(diff)}')
if diff.max() > 1e-4:
    idx = np.argmax(diff)
    print(f'Worst at [{idx}]: prompt={a[idx]:.6f} serve={b[idx]:.6f}')
"
```

## Estimated time

- Step 1: 30 min (read forward(), catalog state buffers)
- Step 2: 30 min (add dump, build, run, compare)
- Step 3: 30 min (binary search)
- Step 4: 10 min (fix)

Total: ~2 hours
