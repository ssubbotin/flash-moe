#!/usr/bin/env python3
"""Reference for one Kimi K2.6 MoE layer (routing + K experts + shared + combine).

Produces dumps consumed by test_kimi_moe.cu:

    hidden_in.bin           float32 [H=7168]     input (pre-MoE, already normed)
    gate_logits.bin         float32 [num_experts=384]
    gate_bias.bin           float32 [num_experts]   (e_score_correction_bias)
    expert_out_each.bin     float32 [K, H]       per-selected-expert output (post-down)
    topk_indices.bin        int32   [K]
    topk_weights.bin        float32 [K]          final weights (post-scale)
    shared_out.bin          float32 [H]
    moe_accum_ref.bin       float32 [H]          Σ_k w_k · expert_k
    layer_out_ref.bin       float32 [H]          h_in + shared + scaling * moe_accum

Also dumps one expert's weights dequantized to f32 for debugging:
    expert_dequant.bin      float32 [3, (2048 or 7168) * (7168 or 2048)]  (gate, up, down row-major)
"""
import argparse, json, os, struct, sys
import numpy as np
import torch

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
        u16 = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
        return (u16.astype(np.uint32) << 16).view(np.float32)
    if dt == "F32":
        return np.frombuffer(raw, dtype=np.float32).reshape(shape)
    raise ValueError(dt)

def dequant_sym4(packed_i32, scale_f32):
    out_dim, packed_cols = packed_i32.shape
    in_dim = packed_cols * 8
    shifts = np.arange(8, dtype=np.uint32) * 4
    nib_u = (packed_i32.astype(np.uint32)[:, :, None] >> shifts) & 0xF
    # compressed-tensors sym-int4 uses biased representation: signed = unsigned - 8.
    # (NOT two's complement — see pack_quantized.py in compressed_tensors library.)
    nib_s = nib_u.astype(np.int32) - 8
    nib_s = nib_s.reshape(out_dim, in_dim).astype(np.float32)
    scale_rep = np.repeat(scale_f32, 32, axis=1)
    return nib_s * scale_rep

def silu(x): return x / (1.0 + np.exp(-x))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Read shard map
    idx = json.load(open(os.path.join(args.model_dir, "model.safetensors.index.json")))["weight_map"]
    open_shards = {}
    def pull(name):
        shard = idx[name]
        if shard not in open_shards:
            hdr, base = read_st_header(os.path.join(args.model_dir, shard))
            open_shards[shard] = (hdr, base)
        hdr, base = open_shards[shard]
        return read_tensor(os.path.join(args.model_dir, shard), hdr, base, name)

    H = 7168
    moe_int = 2048
    num_experts = 384
    K = 8
    scaling_factor = 2.827

    # Routing
    prefix = f"language_model.model.layers.{args.layer}.mlp"
    W_gate = pull(f"{prefix}.gate.weight")                     # [num_experts, H] bf16→f32
    b_corr = pull(f"{prefix}.gate.e_score_correction_bias")    # [num_experts] f32

    # Shared expert
    W_shared_gate = pull(f"{prefix}.shared_experts.gate_proj.weight")   # [2048, H]
    W_shared_up   = pull(f"{prefix}.shared_experts.up_proj.weight")     # [2048, H]
    W_shared_down = pull(f"{prefix}.shared_experts.down_proj.weight")   # [H, 2048]

    # Seeded hidden_in
    g = torch.Generator().manual_seed(args.seed)
    x = (torch.randn(H, generator=g) * 0.05).numpy().astype(np.float32)

    # Route
    logits = W_gate @ x                                        # [num_experts]
    scores = 1.0 / (1.0 + np.exp(-logits))
    scores_biased = scores + b_corr
    topk_idx = np.argsort(-scores_biased)[:K].astype(np.int32)
    topk_w   = scores[topk_idx]
    # norm_topk_prob=True for Kimi
    topk_w = topk_w / topk_w.sum()
    topk_w = (topk_w * scaling_factor).astype(np.float32)

    # Shared forward (bf16 MLP)
    sg_out = silu(W_shared_gate @ x) * (W_shared_up @ x)       # [2048]
    shared_out = (W_shared_down @ sg_out).astype(np.float32)   # [H]

    # Per-selected-expert forward (sym-int4)
    expert_out = np.zeros((K, H), dtype=np.float32)
    moe_accum = np.zeros(H, dtype=np.float32)
    for k in range(K):
        e = int(topk_idx[k])
        base = f"{prefix}.experts.{e}"
        g_pack = pull(f"{base}.gate_proj.weight_packed")
        g_sc   = pull(f"{base}.gate_proj.weight_scale")
        u_pack = pull(f"{base}.up_proj.weight_packed")
        u_sc   = pull(f"{base}.up_proj.weight_scale")
        d_pack = pull(f"{base}.down_proj.weight_packed")
        d_sc   = pull(f"{base}.down_proj.weight_scale")
        Wg = dequant_sym4(g_pack, g_sc)        # [2048, H]
        Wu = dequant_sym4(u_pack, u_sc)        # [2048, H]
        Wd = dequant_sym4(d_pack, d_sc)        # [H, 2048]

        gv = Wg @ x
        uv = Wu @ x
        glu = silu(gv) * uv
        eo  = (Wd @ glu).astype(np.float32)    # [H]

        expert_out[k] = eo
        moe_accum += topk_w[k] * eo

    layer_out = x + shared_out + moe_accum     # scaling already baked into topk_w above? No —
    # Actually in combine:  h_out = h_in + shared + scaling * Σ_k w_k · e_k
    # Our topk_w already includes scaling_factor, so moe_accum = scaling * Σ_k (w_k/scaling) · e_k = scaling * proper
    # But we multiplied topk_w by scaling already — that means moe_accum IS scaling * Σ proper.
    # So layer_out = x + shared + moe_accum (no extra scaling).

    # Dump
    os.makedirs(args.out, exist_ok=True)
    def dump(a, name):
        a = np.ascontiguousarray(a)
        a.tofile(os.path.join(args.out, name))
        print(f"  {name:24s} shape={a.shape} dtype={a.dtype}")

    dump(x,                              "hidden_in.bin")
    dump(logits.astype(np.float32),      "gate_logits.bin")
    dump(b_corr.astype(np.float32),      "gate_bias.bin")
    dump(expert_out,                     "expert_out_each.bin")
    dump(topk_idx,                       "topk_indices.bin")
    dump(topk_w,                         "topk_weights.bin")
    dump(shared_out,                     "shared_out.bin")
    dump(moe_accum,                      "moe_accum_ref.bin")
    dump(layer_out,                      "layer_out_ref.bin")

    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({
            "H": H, "moe_int": moe_int, "num_experts": num_experts,
            "K": K, "scaling_factor": scaling_factor,
            "layer": args.layer, "seed": args.seed,
            "norm_topk_prob": True,
        }, f, indent=2)

    print(f"done. topk_idx = {topk_idx.tolist()}  topk_w = {topk_w.tolist()}")

if __name__ == "__main__":
    main()
