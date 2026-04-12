#!/usr/bin/env python3
"""
Reference implementation: run one token through the Qwen3.5 MLX model
and dump hidden states after each layer for comparison with flash-moe.

Uses the MLX safetensors directly with numpy — no MLX/torch dependency.
"""
import json
import struct
import sys
import numpy as np
from pathlib import Path


def load_safetensors_tensor(filepath, tensor_name):
    """Load a single tensor from a safetensors file."""
    with open(filepath, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_len))
        data_start = 8 + header_len

    if tensor_name not in header:
        return None

    info = header[tensor_name]
    dtype_map = {'BF16': np.uint16, 'F32': np.float32, 'U32': np.uint32,
                 'F16': np.float16, 'I32': np.int32}
    dtype = dtype_map.get(info['dtype'], np.uint8)
    begin, end = info['data_offsets']

    with open(filepath, 'rb') as f:
        f.seek(data_start + begin)
        data = np.frombuffer(f.read(end - begin), dtype=dtype)

    return data.reshape(info['shape'])


def bf16_to_f32(arr):
    """Convert BF16 (stored as uint16) to float32."""
    return np.frombuffer(
        np.left_shift(arr.astype(np.uint32), 16).tobytes(),
        dtype=np.float32
    ).reshape(arr.shape)


def dequant_4bit(weight_u32, scales_bf16, biases_bf16, group_size=64):
    """Dequantize 4-bit packed weights: val = nibble * scale + bias."""
    scales = bf16_to_f32(scales_bf16)
    biases = bf16_to_f32(biases_bf16)

    out_dim = weight_u32.shape[0]
    packed_dim = weight_u32.shape[1]
    in_dim = packed_dim * 8  # 8 nibbles per uint32

    # Unpack nibbles
    result = np.zeros((out_dim, in_dim), dtype=np.float32)
    for i in range(8):
        nibbles = (weight_u32 >> (i * 4)) & 0xF
        result[:, i::8] = nibbles.astype(np.float32)

    # Apply scales and biases per group
    num_groups = in_dim // group_size
    for g in range(num_groups):
        start = g * group_size
        end = start + group_size
        result[:, start:end] = result[:, start:end] * scales[:, g:g+1] + biases[:, g:g+1]

    return result


def rms_norm(x, weight_bf16, eps=1e-6):
    """RMS normalization."""
    w = bf16_to_f32(weight_bf16)
    rms = np.sqrt(np.mean(x ** 2) + eps)
    return (x / rms) * w


def matvec_4bit(x, weight_u32, scales_bf16, biases_bf16, group_size=64):
    """Matrix-vector multiply with 4-bit quantized weights."""
    W = dequant_4bit(weight_u32, scales_bf16, biases_bf16, group_size)
    return W @ x


def main():
    if len(sys.argv) < 3:
        print("Usage: dump_hidden.py <model_path> <token_id>")
        sys.exit(1)

    model_path = Path(sys.argv[1])
    token_id = int(sys.argv[2])

    # Load manifest
    with open(model_path / 'model_weights.json') as f:
        manifest = json.load(f)

    config = manifest['config']
    hidden_dim = config['hidden_size']
    num_layers = config['num_hidden_layers']

    print(f"Model: {model_path}")
    print(f"Hidden: {hidden_dim}, Layers: {num_layers}")
    print(f"Token ID: {token_id}")

    # Load weights from binary
    bin_path = model_path / 'model_weights.bin'
    weights_data = np.memmap(str(bin_path), dtype=np.uint8, mode='r')

    def get_tensor(name):
        info = manifest['tensors'][name]
        offset = info['offset']
        size = info['size']
        shape = info['shape']
        dtype_map = {'bf16': np.uint16, 'BF16': np.uint16,
                     'f32': np.float32, 'F32': np.float32,
                     'u32': np.uint32, 'U32': np.uint32}
        dtype = dtype_map.get(info['dtype'], np.uint8)
        return np.frombuffer(weights_data[offset:offset+size].copy(), dtype=dtype).reshape(shape)

    # Embedding lookup
    embed_w = get_tensor('model.embed_tokens.weight')
    embed_s = get_tensor('model.embed_tokens.scales')
    embed_b = get_tensor('model.embed_tokens.biases')
    W_embed = dequant_4bit(embed_w, embed_s, embed_b)
    hidden = W_embed[token_id].astype(np.float32)

    print(f"\nAfter embedding: hidden[0:5] = {hidden[:5]}")
    hidden.tofile('/tmp/ref_embed.bin')

    # Run through first few layers and dump
    for layer_idx in range(min(3, num_layers)):
        layer_types = config.get('layer_types', [])
        is_full = layer_types[layer_idx] == 'full_attention' if layer_idx < len(layer_types) else False

        prefix = f"model.layers.{layer_idx}"

        # Input norm
        norm_w = get_tensor(f"{prefix}.input_layernorm.weight")
        normed = rms_norm(hidden, norm_w)

        print(f"\nLayer {layer_idx} ({'full' if is_full else 'linear'}):")
        print(f"  normed[0:5] = {normed[:5]}")
        normed.tofile(f'/tmp/ref_layer{layer_idx}_normed.bin')
        hidden.tofile(f'/tmp/ref_layer{layer_idx}_hidden.bin')

    print("\nDumped reference hidden states to /tmp/ref_*.bin")


if __name__ == '__main__':
    main()
