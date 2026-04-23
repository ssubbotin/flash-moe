#!/usr/bin/env python3
"""Compare my manual sym-int4 dequant + matvec against torch's loading of the
same weights via the compressed-tensors path, on a single expert.

If they disagree, we have a dequant/pack bug.
If they agree, the math is right and the explosion is real per the weights.
"""
import argparse, json, os, struct, sys
import numpy as np
import torch
import torch.nn.functional as F

def read_st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return hdr, 8 + n

def read_tensor(path, hdr, base, name):
    info = hdr[name]; beg, end = info["data_offsets"]; dt = info["dtype"]
    shape = tuple(info["shape"])
    nb = end - beg
    with open(path, "rb") as f:
        f.seek(base + beg); raw = f.read(nb)
    if dt == "I32":
        return np.frombuffer(raw, dtype=np.int32).reshape(shape)
    if dt == "BF16":
        return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(shape)
    raise ValueError(dt)

def load_all(model_dir):
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    hdrs = {}
    def pull(name):
        shard = idx[name]
        if shard not in hdrs:
            hdrs[shard] = read_st_header(os.path.join(model_dir, shard))
        hdr, base = hdrs[shard]
        return read_tensor(os.path.join(model_dir, shard), hdr, base, name)
    return pull

def compressed_tensors_dequant_sym_int4(packed_i32: np.ndarray, scale_bf16: torch.Tensor) -> torch.Tensor:
    """Match the compressed-tensors library's actual dequant path (pack-quantized,
    symmetric int4, group=32). Used for spot verification vs my diag implementation.

    Reference (compressed-tensors/src/compressed_tensors/compressors/quantized_compressors/pack_quantized.py):
        unpacked: reinterpret int32 little-endian as bitstream, split into 4-bit
                  signed groups (LSB-first element 0 in bits [0..4), element 1 [4..8), ...)
        dequant:  float_val = signed_int4 * scale
    """
    out_dim, packed_cols = packed_i32.shape
    in_dim = packed_cols * 8
    # Unpack nibbles
    shifts = np.arange(8, dtype=np.uint32) * 4
    nib_u = (packed_i32.astype(np.uint32)[:, :, None] >> shifts) & 0xF
    nib_s = np.where(nib_u >= 8, nib_u.astype(np.int32) - 16, nib_u.astype(np.int32))
    nib_s = nib_s.reshape(out_dim, in_dim)
    W = torch.from_numpy(nib_s).to(torch.float32)
    # Per-group scale: scale_bf16 shape [out_dim, in_dim/32]
    scale_f32 = scale_bf16.to(torch.float32)
    # Expand to [out_dim, in_dim]
    scale_full = scale_f32.repeat_interleave(32, dim=1)
    return W * scale_full

def silu(x): return x * torch.sigmoid(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--expert", type=int, required=True)
    ap.add_argument("--hidden", required=True)
    args = ap.parse_args()

    pull = load_all(args.model_dir)
    base = f"language_model.model.layers.{args.layer}.mlp.experts.{args.expert}"

    gp = pull(f"{base}.gate_proj.weight_packed")      # int32 np
    gs = pull(f"{base}.gate_proj.weight_scale")       # bf16 torch
    up = pull(f"{base}.up_proj.weight_packed")
    us = pull(f"{base}.up_proj.weight_scale")
    dp = pull(f"{base}.down_proj.weight_packed")
    ds = pull(f"{base}.down_proj.weight_scale")

    Wg = compressed_tensors_dequant_sym_int4(gp, gs)
    Wu = compressed_tensors_dequant_sym_int4(up, us)
    Wd = compressed_tensors_dequant_sym_int4(dp, ds)

    x_np = np.fromfile(args.hidden, dtype=np.float32)
    x = torch.from_numpy(x_np).to(torch.float32)
    H = x.shape[0]
    print(f"input: L2={x.norm().item():.4g} RMS={x.std().item():.4g} (mean={x.mean().item():.4g})")

    gate_out = Wg @ x
    up_out   = Wu @ x
    print(f"gate_out: L2={gate_out.norm().item():.4g} RMS={gate_out.std().item():.4g} first6={gate_out[:6].tolist()}")
    print(f"up_out:   L2={up_out.norm().item():.4g} RMS={up_out.std().item():.4g} first6={up_out[:6].tolist()}")

    glu = silu(gate_out) * up_out
    print(f"silu(gate)*up: L2={glu.norm().item():.4g} RMS={glu.std().item():.4g} first6={glu[:6].tolist()}")

    down_out = Wd @ glu
    print(f"down_out: L2={down_out.norm().item():.4g} RMS={down_out.std().item():.4g} first6={down_out[:6].tolist()}")

    # Also check everything in bf16 to simulate native inference dtype
    print("\n--- same thing in bf16 activations (matches transformers native) ---")
    x_bf = x.to(torch.bfloat16)
    Wg_bf = Wg.to(torch.bfloat16)
    Wu_bf = Wu.to(torch.bfloat16)
    Wd_bf = Wd.to(torch.bfloat16)
    g = Wg_bf @ x_bf; u = Wu_bf @ x_bf
    gl = silu(g) * u
    d = Wd_bf @ gl
    print(f"gate_out(bf16): L2={g.float().norm().item():.4g}")
    print(f"up_out(bf16):   L2={u.float().norm().item():.4g}")
    print(f"silu*up(bf16):  L2={gl.float().norm().item():.4g}")
    print(f"down_out(bf16): L2={d.float().norm().item():.4g}")

if __name__ == "__main__":
    main()
