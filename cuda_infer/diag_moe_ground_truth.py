#!/usr/bin/env python3
"""Cross-check whether Kimi K2.6 expert outputs really should be huge at layer N,
or if it's a bug in my dequant. Uses compressed-tensors' OWN unpack/dequant
library as the ground-truth reference, independent of my numpy implementation.

Usage:
    python3 diag_moe_ground_truth.py --model-dir kimi-k2.6 --layer 4 \
        --hidden /tmp/kimi_L4_hnorm.bin
"""
import argparse, json, struct, os
import numpy as np
import torch
from compressed_tensors.compressors import unpack_from_int32

def read_hdr(p):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n

def pull(model_dir, name, idx):
    shard = idx[name]
    hdr, base = read_hdr(os.path.join(model_dir, shard))
    info = hdr[name]
    beg, end = info["data_offsets"]
    with open(os.path.join(model_dir, shard), "rb") as f:
        f.seek(base + beg)
        raw = f.read(end - beg)
    if info["dtype"] == "I32":
        return torch.from_numpy(np.frombuffer(raw, dtype=np.int32).reshape(info["shape"]).copy())
    if info["dtype"] == "BF16":
        return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(info["shape"]).clone()
    if info["dtype"] == "F32":
        return torch.from_numpy(np.frombuffer(raw, dtype=np.float32).reshape(info["shape"]).copy())
    raise ValueError(info["dtype"])

def silu(x): return x * torch.sigmoid(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--hidden", required=True, help="f32 [H] post-attn-norm dump")
    args = ap.parse_args()

    idx = json.load(open(os.path.join(args.model_dir, "model.safetensors.index.json")))["weight_map"]
    x = torch.from_numpy(np.fromfile(args.hidden, dtype=np.float32))
    H = x.shape[0]
    print(f"input L2={x.norm():.4g} RMS={x.std():.4g} mean={x.mean():.4g}")

    prefix = f"language_model.model.layers.{args.layer}.mlp"
    W_gate = pull(args.model_dir, f"{prefix}.gate.weight", idx).to(torch.float32)
    b_corr = pull(args.model_dir, f"{prefix}.gate.e_score_correction_bias", idx) \
             if idx.get(f"{prefix}.gate.e_score_correction_bias") else None
    b_corr = b_corr.to(torch.float32) if b_corr is not None else 0

    # Routing via DeepSeek formula
    logits = (W_gate @ x).float()
    scores = torch.sigmoid(logits)
    biased = scores + b_corr
    topk = torch.topk(biased, 8).indices
    topk_w = scores[topk] / scores[topk].sum()
    topk_w = topk_w * 2.827
    print(f"topk_idx = {topk.tolist()}")
    print(f"topk_w   = {topk_w.tolist()}")

    # Ground-truth dequant via compressed-tensors library's own unpack
    def dq_layer(expert, proj):
        base = f"{prefix}.experts.{expert}.{proj}"
        packed = pull(args.model_dir, f"{base}.weight_packed", idx)
        scale  = pull(args.model_dir, f"{base}.weight_scale", idx)
        shape  = pull(args.model_dir, f"{base}.weight_shape", idx)  # [2] int32
        orig = torch.Size(shape.tolist())
        unpacked = unpack_from_int32(packed, 4, orig)   # int8 signed in [-8, 7]
        # dequant = unpacked * scale (group_size=32 along last dim)
        scale_f = scale.to(torch.float32)
        scale_full = scale_f.repeat_interleave(32, dim=1)
        return unpacked.to(torch.float32) * scale_full

    moe_accum = torch.zeros(H)
    for k in range(8):
        e = int(topk[k])
        Wg = dq_layer(e, "gate_proj")   # [2048, 7168]
        Wu = dq_layer(e, "up_proj")
        Wd = dq_layer(e, "down_proj")   # [7168, 2048]
        g = Wg @ x
        u = Wu @ x
        glu = silu(g) * u
        eo = (Wd @ glu).float()
        if k == 0:
            print(f"expert{e} (k=0): "
                  f"gate_out L2={g.norm():.4g}  up_out L2={u.norm():.4g}  "
                  f"glu L2={glu.norm():.4g}  eo L2={eo.norm():.4g} first6={eo[:6].tolist()}")
        moe_accum += topk_w[k] * eo

    # Shared
    Wsg = pull(args.model_dir, f"{prefix}.shared_experts.gate_proj.weight", idx).to(torch.float32)
    Wsu = pull(args.model_dir, f"{prefix}.shared_experts.up_proj.weight",   idx).to(torch.float32)
    Wsd = pull(args.model_dir, f"{prefix}.shared_experts.down_proj.weight", idx).to(torch.float32)
    sg = silu(Wsg @ x) * (Wsu @ x)
    shared = (Wsd @ sg).float()

    print(f"\n=== layer {args.layer} ground-truth MoE forward ===")
    print(f"  moe_accum L2={moe_accum.norm():.4g}  mean={moe_accum.mean():.4g}")
    print(f"  shared    L2={shared.norm():.4g}     mean={shared.mean():.4g}")
    print(f"  total     L2={(moe_accum+shared).norm():.4g}")

if __name__ == "__main__":
    main()
