#!/usr/bin/env python3
"""
Reference layer-0 forward pass for Qwen3.5 MLX 4-bit model.
Compares step-by-step with CUDA dump files in /tmp/cuda_L0_*.bin.
"""
import json
import sys
import numpy as np
from pathlib import Path


def bf16_to_f32(arr):
    return np.frombuffer(
        np.left_shift(arr.astype(np.uint32), 16).tobytes(), dtype=np.float32
    ).reshape(arr.shape)


def dequant_4bit(weight_u32, scales_bf16, biases_bf16, group_size=64):
    scales = bf16_to_f32(scales_bf16)
    biases = bf16_to_f32(biases_bf16)
    out_dim, packed_dim = weight_u32.shape
    in_dim = packed_dim * 8
    result = np.zeros((out_dim, in_dim), dtype=np.float32)
    for i in range(8):
        nibbles = (weight_u32 >> (i * 4)) & 0xF
        result[:, i::8] = nibbles.astype(np.float32)
    num_groups = in_dim // group_size
    for g in range(num_groups):
        s, e = g * group_size, (g + 1) * group_size
        result[:, s:e] = result[:, s:e] * scales[:, g:g+1] + biases[:, g:g+1]
    return result


def rms_norm(x, weight_bf16, eps=1e-6):
    w = bf16_to_f32(weight_bf16)
    rms = np.sqrt(np.mean(x ** 2) + eps)
    return (x / rms) * w


def matvec(x, w_u32, s_bf16, b_bf16, group_size=64):
    W = dequant_4bit(w_u32, s_bf16, b_bf16, group_size)
    return W @ x


def silu(x):
    return x / (1.0 + np.exp(-x))


def compare(label, ref, cuda_file):
    if cuda_file and Path(cuda_file).exists():
        cuda = np.fromfile(cuda_file, dtype=np.float32)
        n = min(len(cuda), len(ref))
        if len(cuda) != len(ref):
            print(f"  {label:20s} SIZE MISMATCH ref={len(ref)} cuda={len(cuda)}, comparing first {n}")
        diff = np.abs(ref[:n] - cuda[:n])
        print(f"  {label:20s} ref={ref[:5]}  max_diff={diff.max():.2e} mean={diff.mean():.2e}")
        if diff.max() > 0.01:
            idx = np.argmax(diff)
            print(f"    DIVERGENCE at [{idx}]: ref={ref[idx]:.6f} cuda={cuda[idx]:.6f}")
        return diff.max()
    else:
        print(f"  {label:20s} ref[0:5]={ref[:5]}")
    return None


def main():
    model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/work/weights_35b")
    token_id = int(sys.argv[2]) if len(sys.argv) > 2 else 9419

    with open(model_path / 'model_weights.json') as f:
        manifest = json.load(f)
    config = manifest['config']
    hidden_dim = config['hidden_size']

    bin_data = np.memmap(str(model_path / 'model_weights.bin'), dtype=np.uint8, mode='r')

    dtype_map = {'bf16': np.uint16, 'BF16': np.uint16,
                 'f32': np.float32, 'F32': np.float32,
                 'u32': np.uint32, 'U32': np.uint32}

    def T(name):
        info = manifest['tensors'][name]
        dtype = dtype_map[info['dtype']]
        return np.frombuffer(bin_data[info['offset']:info['offset']+info['size']].copy(),
                             dtype=dtype).reshape(info['shape'])

    print(f"Model: {model_path}, hidden={hidden_dim}, token={token_id}")
    print(f"Config: layers={config['num_hidden_layers']}, "
          f"v_heads={config['linear_num_value_heads']}, k_heads={config['linear_num_key_heads']}")

    # Embedding
    embed_W = dequant_4bit(T('model.embed_tokens.weight'),
                           T('model.embed_tokens.scales'),
                           T('model.embed_tokens.biases'))
    hidden = embed_W[token_id].astype(np.float32)
    compare("embedding", hidden, "/tmp/cuda_embed.bin")

    # Layer 0: Input RMS norm
    normed = rms_norm(hidden, T('model.layers.0.input_layernorm.weight'))
    compare("input_normed", normed, "/tmp/cuda_L0_normed.bin")

    # QKV projection
    qkv = matvec(normed,
                 T('model.layers.0.linear_attn.in_proj_qkv.weight'),
                 T('model.layers.0.linear_attn.in_proj_qkv.scales'),
                 T('model.layers.0.linear_attn.in_proj_qkv.biases'))
    compare("qkv_proj", qkv, "/tmp/cuda_L0_qkv.bin")

    # Z projection
    z = matvec(normed,
               T('model.layers.0.linear_attn.in_proj_z.weight'),
               T('model.layers.0.linear_attn.in_proj_z.scales'),
               T('model.layers.0.linear_attn.in_proj_z.biases'))
    compare("z_proj", z, None)

    # Alpha and Beta projections
    alpha = matvec(normed,
                   T('model.layers.0.linear_attn.in_proj_a.weight'),
                   T('model.layers.0.linear_attn.in_proj_a.scales'),
                   T('model.layers.0.linear_attn.in_proj_a.biases'))
    compare("alpha_proj", alpha, None)

    beta = matvec(normed,
                  T('model.layers.0.linear_attn.in_proj_b.weight'),
                  T('model.layers.0.linear_attn.in_proj_b.scales'),
                  T('model.layers.0.linear_attn.in_proj_b.biases'))
    compare("beta_proj", beta, None)

    # Conv1d step (kernel=4, first token: state is zeros, only current input matters)
    # conv_weights: [conv_dim, 4] in bf16
    conv_w_raw = T('model.layers.0.linear_attn.conv1d.weight')
    conv_w = bf16_to_f32(conv_w_raw).reshape(-1, 4)  # [conv_dim, 4]
    # For first token (state=0), output = input * weight[3] then SiLU
    conv_out = qkv * conv_w[:, 3]
    conv_out = silu(conv_out)
    compare("conv1d", conv_out, "/tmp/cuda_L0_conv.bin")

    # Split Q, K, V from conv output
    k_heads = config['linear_num_key_heads']
    k_dim = config['linear_key_head_dim']
    v_heads = config['linear_num_value_heads']
    v_dim = config['linear_value_head_dim']
    total_k = k_heads * k_dim
    total_v = v_heads * v_dim

    Q = conv_out[:total_k]
    K = conv_out[total_k:2*total_k]
    V = conv_out[2*total_k:]
    print(f"\n  Q shape: {Q.shape}, K shape: {K.shape}, V shape: {V.shape}")

    # RMS norm Q and K (per-head) — Q gets inv_scale^2, K gets inv_scale
    inv_scale = 1.0 / np.sqrt(k_dim)
    for h in range(k_heads):
        s, e = h * k_dim, (h + 1) * k_dim
        q_rms = np.sqrt(np.mean(Q[s:e] ** 2) + 1e-6)
        Q[s:e] = Q[s:e] / q_rms * inv_scale * inv_scale  # matches CUDA
        k_rms = np.sqrt(np.mean(K[s:e] ** 2) + 1e-6)
        K[s:e] = K[s:e] / k_rms * inv_scale
    Q.tofile('/tmp/ref_Q.bin')
    K.tofile('/tmp/ref_K.bin')
    V.tofile('/tmp/ref_V.bin')
    compare("Q_normed", Q, "/tmp/cuda_L0_Q.bin")
    compare("K_normed", K, "/tmp/cuda_L0_K.bin")
    compare("V_raw", V, "/tmp/cuda_L0_V.bin")

    # Decay and beta gate
    A_log = np.fromfile(str(model_path / 'model_weights.bin'),
                        dtype=np.float32, count=0)  # need from manifest
    A_log_data = T('model.layers.0.linear_attn.A_log')
    dt_bias_data = T('model.layers.0.linear_attn.dt_bias')

    # A_log is float32, dt_bias is bf16
    A_log_f = A_log_data.astype(np.float32) if A_log_data.dtype == np.float32 else bf16_to_f32(A_log_data)
    dt_bias_f = bf16_to_f32(dt_bias_data) if dt_bias_data.dtype == np.uint16 else dt_bias_data.astype(np.float32)

    # GatedDeltaNet formulas (matching CUDA compute_decay_beta kernel):
    # decay = exp(-exp(A_log) * softplus(alpha + dt_bias))
    # beta_gate = sigmoid(beta)
    A_val = np.exp(A_log_f.flatten()[:v_heads])
    sp = np.log(1.0 + np.exp(alpha + dt_bias_f.flatten()[:v_heads]))
    decay = np.exp(-A_val * sp)
    beta_gate_val = 1.0 / (1.0 + np.exp(-beta))
    decay.astype(np.float32).tofile('/tmp/ref_decay.bin')
    beta_gate_val.astype(np.float32).tofile('/tmp/ref_beta.bin')
    compare("decay", decay.astype(np.float32), "/tmp/cuda_L0_decay.bin")
    compare("beta_gate", beta_gate_val.astype(np.float32), "/tmp/cuda_L0_beta.bin")

    # GDN step (state is zeros for first token)
    khpv = v_heads // k_heads
    gdn_out = np.zeros(total_v, dtype=np.float32)
    state = np.zeros((v_heads, v_dim, k_dim), dtype=np.float32)

    for vh in range(v_heads):
        kh = vh // khpv
        g = decay[vh]
        b = beta_gate_val[vh]
        q_h = Q[kh * k_dim:(kh + 1) * k_dim]
        k_h = K[kh * k_dim:(kh + 1) * k_dim]

        # state starts at zero, decay doesn't matter
        # kv_mem = state @ k = 0
        kv_mem = state[vh] @ k_h  # [v_dim]

        # delta update
        for vi in range(v_dim):
            delta = (V[vh * v_dim + vi] - kv_mem[vi]) * b
            state[vh, vi, :] += k_h * delta

        # output = state @ q
        gdn_out[vh * v_dim:(vh + 1) * v_dim] = state[vh] @ q_h

    gdn_out.tofile('/tmp/ref_gdn.bin')
    compare("gdn_output", gdn_out, "/tmp/cuda_L0_delta.bin")

    # Gated RMS norm — norm_w is [v_dim], shared across all v_heads
    norm_w = bf16_to_f32(T('model.layers.0.linear_attn.norm.weight')).flatten()
    gated = np.zeros(total_v, dtype=np.float32)
    for vh in range(v_heads):
        s, e = vh * v_dim, (vh + 1) * v_dim
        rms_val = np.sqrt(np.mean(gdn_out[s:e] ** 2) + 1e-6)
        normed_h = gdn_out[s:e] / rms_val * norm_w
        # Gate with z (SiLU = z * sigmoid(z))
        z_h = z[s:e]
        gate = z_h / (1.0 + np.exp(-z_h))  # SiLU
        gated[s:e] = normed_h * gate
    gated.tofile('/tmp/ref_gated.bin')
    compare("gated_norm", gated, "/tmp/cuda_L0_gated.bin")

    # Output projection
    oproj = matvec(gated,
                   T('model.layers.0.linear_attn.out_proj.weight'),
                   T('model.layers.0.linear_attn.out_proj.scales'),
                   T('model.layers.0.linear_attn.out_proj.biases'))
    compare("out_proj", oproj, "/tmp/cuda_L0_oproj.bin")

    # Residual
    h_mid = hidden + oproj
    print(f"\n  attn+residual[0:5]: {h_mid[:5]}")

    print("\nDone. Check DIVERGENCE lines above to find the bug.")


if __name__ == '__main__':
    main()
