#!/usr/bin/env python3
"""Run the MoE forward (routing + K experts + shared + combine) in python for
a specific layer using a hidden-state dump from infer_kimi. Prints L2 norms at
each stage so we can compare against infer_kimi's debug output.

If this matches infer_kimi's numbers, the GPU path is numerically correct and
the explosion is real (model-architecture issue we're misinterpreting).
If it doesn't match, there's a bug in the GPU path that's hidden from the
random-input test.
"""
import argparse, json, os, struct, sys, numpy as np, torch

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

def load_all_tensors(model_dir):
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    hdrs = {}
    def pull(name):
        shard = idx[name]
        if shard not in hdrs:
            hdrs[shard] = read_st_header(os.path.join(model_dir, shard))
        hdr, base = hdrs[shard]
        return read_tensor(os.path.join(model_dir, shard), hdr, base, name)
    return pull

def silu(x): return x / (1.0 + np.exp(-x))

def dequant_sym4(packed_i32, scale_f32):
    out_dim, packed_cols = packed_i32.shape
    in_dim = packed_cols * 8
    shifts = np.arange(8, dtype=np.uint32) * 4
    nib_u = (packed_i32.astype(np.uint32)[:, :, None] >> shifts) & 0xF
    nib_s = np.where(nib_u >= 8, nib_u.astype(np.int32) - 16, nib_u.astype(np.int32))
    nib_s = nib_s.reshape(out_dim, in_dim).astype(np.float32)
    scale_rep = np.repeat(scale_f32, 32, axis=1)
    return nib_s * scale_rep

def l2(x, tag=""):
    v = np.linalg.norm(x.ravel())
    nan = np.isnan(x).sum()
    inf = np.isinf(x).sum()
    print(f"  [{tag:22}] L2={v:.4g} nan={nan} inf={inf} first6={x.ravel()[:6]}")
    return v

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--hidden", required=True, help="f32 [H] binary dump from infer_kimi")
    ap.add_argument("--topk-idx", default=None,
        help="override topk indices (csv); if absent, compute via the routing kernel")
    args = ap.parse_args()

    pull = load_all_tensors(args.model_dir)
    H = 7168
    K = 8
    scaling = 2.827

    x = np.fromfile(args.hidden, dtype=np.float32)
    assert x.size == H, x.size
    l2(x, "hidden_in (post-norm)")

    prefix = f"language_model.model.layers.{args.layer}.mlp"
    W_gate = pull(f"{prefix}.gate.weight")                 # [ne, H] bf16→f32
    b_corr = pull(f"{prefix}.gate.e_score_correction_bias").astype(np.float32)
    Wsg = pull(f"{prefix}.shared_experts.gate_proj.weight")
    Wsu = pull(f"{prefix}.shared_experts.up_proj.weight")
    Wsd = pull(f"{prefix}.shared_experts.down_proj.weight")

    # Routing (match infer's kimi_moe_routing_noaux_tc exactly)
    logits = (W_gate @ x).astype(np.float32)
    l2(logits, "router_logits")
    scores = 1.0 / (1.0 + np.exp(-logits))
    biased = scores + b_corr
    if args.topk_idx:
        topk_idx = np.array([int(s) for s in args.topk_idx.split(",")], dtype=np.int32)
    else:
        topk_idx = np.argsort(-biased)[:K].astype(np.int32)
    topk_w = scores[topk_idx]
    topk_w = topk_w / topk_w.sum()
    topk_w = (topk_w * scaling).astype(np.float32)
    print(f"  topk_idx={topk_idx.tolist()}")
    print(f"  topk_w={topk_w.tolist()}")

    # Shared
    sg = silu(Wsg @ x) * (Wsu @ x)
    shared = Wsd @ sg
    l2(shared, "shared_out")

    # Routed experts
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
        Wg = dequant_sym4(g_pack, g_sc)
        Wu = dequant_sym4(u_pack, u_sc)
        Wd = dequant_sym4(d_pack, d_sc)

        gv = Wg @ x
        uv = Wu @ x
        glu = silu(gv) * uv
        eo  = (Wd @ glu).astype(np.float32)
        if k == 0:
            l2(eo, f"expert{e}_out (k=0)")
        moe_accum += topk_w[k] * eo

    l2(moe_accum, "moe_accum")
    print(f"  residual (input again, for comparison): L2={np.linalg.norm(x):.4g}")
    # combined layer out = residual + shared + moe_accum, where residual is
    # pre-post-norm hidden (we don't have that here — caller computes outside)

if __name__ == "__main__":
    main()
