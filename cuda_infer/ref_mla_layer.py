#!/usr/bin/env python3
"""Reference dump for one MLA attention layer from Kimi K2.6.

Produces bin files consumed by test_mla.cu:

    hidden_in.bin       float32 [H=7168]
    kv_history.bin      bf16    [seq_len-1, 576]  # compressed latent + k_rope
    q_abs_ref.bin       float32 [num_heads=64, kv_lora_rank=512]
    q_rope_ref.bin      float32 [num_heads=64, qk_rope_head_dim=64]
    kv_comp_new.bin     float32 [kv_lora_rank=512]  # appended this step
    k_rope_new.bin      float32 [qk_rope_head_dim=64]
    attn_out_ref.bin    float32 [num_heads=64, kv_lora_rank=512]  # before W_UV
    layer_out_ref.bin   float32 [H=7168]                           # after o_proj (no residual)

Also dumps the absorbed weights:
    W_Q_abs.bin         float32 [num_heads, q_lora_rank=1536, kv_lora_rank=512]
    W_O_abs.bin         float32 [num_heads * kv_lora_rank=32768, H=7168]

We run the math by hand (no transformers library) so we avoid pulling in DeepSeek
modeling code and can control every step. The Kimi K2.6 weight format uses
compressed-tensors symmetric int4 for MoE; MLA attention projections are bf16
(uncompressed), so we can load them directly from safetensors.
"""
import argparse, json, os, struct, sys
import numpy as np
import torch

DEFAULTS = dict(
    hidden_size=7168, num_heads=64,
    q_lora_rank=1536, kv_lora_rank=512,
    qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
    rope_theta=50000.0,
    yarn_factor=64.0, yarn_beta_fast=32.0, yarn_beta_slow=1.0,
    yarn_orig_max_pos=4096, yarn_mscale=1.0, yarn_mscale_all_dim=1.0,
    rms_eps=1e-5,
)

def read_safetensors_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        data_start = 8 + n
    return hdr, data_start

def load_tensor(path, hdr, data_start, name):
    info = hdr[name]
    beg, end = info["data_offsets"]
    nbytes = end - beg
    with open(path, "rb") as f:
        f.seek(data_start + beg)
        raw = f.read(nbytes)
    shape = tuple(info["shape"])
    dt = info["dtype"]
    if dt == "BF16":
        t = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(shape).clone()
    elif dt == "F32":
        t = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(shape).clone()
    elif dt == "I32":
        t = torch.frombuffer(bytearray(raw), dtype=torch.int32).reshape(shape).clone()
    else:
        raise ValueError(f"unsupported dtype {dt}")
    return t

def find_layer_shard(index_json, layer_idx, tensor_name):
    wm = index_json["weight_map"]
    full = f"language_model.model.layers.{layer_idx}.self_attn.{tensor_name}"
    return wm[full], full

def rms_norm(x, weight, eps):
    x32 = x.to(torch.float32)
    rms = torch.sqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return (x32 / rms * weight.to(torch.float32)).to(x.dtype)

def yarn_frequencies(cfg):
    dim = cfg["qk_rope_head_dim"]
    base = cfg["rope_theta"]
    factor = cfg["yarn_factor"]
    beta_fast = cfg["yarn_beta_fast"]
    beta_slow = cfg["yarn_beta_slow"]
    orig = cfg["yarn_orig_max_pos"]
    mscale = cfg["yarn_mscale"]
    mscale_all = cfg["yarn_mscale_all_dim"]

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    # yarn NTK correction: interpolate extrapolation/interpolation bands by wavelength
    def find_correction_dim(num_rot, d, base_, max_pos):
        return d * np.log(max_pos / (num_rot * 2 * np.pi)) / (2 * np.log(base_))
    lo = int(np.floor(find_correction_dim(beta_fast, dim, base, orig)))
    hi = int(np.ceil(find_correction_dim(beta_slow, dim, base, orig)))
    lo = max(lo, 0); hi = min(hi, dim - 1)
    half = dim // 2
    ramp = torch.arange(half, dtype=torch.float32)
    # Linear ramp between lo/2 and hi/2 (pairs)
    lo_h, hi_h = lo / 2, hi / 2
    ramp_mask = torch.clamp((ramp - lo_h) / max(hi_h - lo_h, 0.001), 0.0, 1.0)
    inv_mask = 1.0 - ramp_mask
    # Interpolate extrapolation (1.0) vs interpolation (factor)
    freqs_interp = freqs / factor
    freqs_yarn = freqs_interp * inv_mask + freqs * ramp_mask  # high-freq => extrapolate
    # Actually DeepSeek yarn: low-freq bands (slow rotation) interpolate; high-freq extrapolate
    # The standard yarn formulation: new = orig_interp * (1 - mask) + orig * mask, where mask
    # rises with frequency. We match deepseek's modeling_deepseek.py convention.
    mscale_factor = get_mscale(factor, mscale) * get_mscale(factor, mscale_all)
    return freqs_yarn, mscale_factor

def get_mscale(scale, mscale):
    if scale <= 1: return 1.0
    return 0.1 * mscale * float(np.log(scale)) + 1.0

def apply_rope(x, pos, freqs, mscale):
    """x: [..., rope_dim] interpreted as pairs. Applies rotary at position `pos`.
    DeepSeek convention: pair (x[i], x[i+half])."""
    dim = x.shape[-1]
    half = dim // 2
    t = torch.tensor(float(pos), dtype=torch.float32)
    angles = t * freqs                                  # [half]
    cos = (torch.cos(angles) * mscale).to(x.dtype)
    sin = (torch.sin(angles) * mscale).to(x.dtype)
    x0 = x[..., :half]
    x1 = x[..., half:]
    out = torch.empty_like(x)
    out[..., :half] = x0 * cos - x1 * sin
    out[..., half:] = x1 * cos + x0 * sin
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="Kimi-K2.6 dir")
    ap.add_argument("--layer", type=int, default=1, help="layer index (>=1 for MoE+MLA)")
    ap.add_argument("--seq-len", type=int, default=8, help="total context length incl current token")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = DEFAULTS
    os.makedirs(args.out, exist_ok=True)

    index_path = os.path.join(args.model_dir, "model.safetensors.index.json")
    idx = json.load(open(index_path))

    # Load MLA weights for the requested layer
    names = ["q_a_proj.weight", "q_a_layernorm.weight",
             "q_b_proj.weight",
             "kv_a_proj_with_mqa.weight", "kv_a_layernorm.weight",
             "kv_b_proj.weight", "o_proj.weight"]
    w = {}
    for n in names:
        shard, full = find_layer_shard(idx, args.layer, n)
        shard_path = os.path.join(args.model_dir, shard)
        hdr, ds = read_safetensors_header(shard_path)
        w[n] = load_tensor(shard_path, hdr, ds, full)
        print(f"{n:36s} {tuple(w[n].shape)} {w[n].dtype}")

    H       = cfg["hidden_size"]
    nh      = cfg["num_heads"]
    qlora   = cfg["q_lora_rank"]
    kvlora  = cfg["kv_lora_rank"]
    d_nope  = cfg["qk_nope_head_dim"]
    d_rope  = cfg["qk_rope_head_dim"]
    d_v     = cfg["v_head_dim"]

    # Reshape projection weights
    Wqa = w["q_a_proj.weight"].to(torch.float32)                       # [1536, 7168]
    qa_norm_w = w["q_a_layernorm.weight"].to(torch.float32)             # [1536]
    Wqb = w["q_b_proj.weight"].to(torch.float32)                       # [12288=64*192, 1536]
    Wqb = Wqb.view(nh, d_nope + d_rope, qlora)                         # [64, 192, 1536]
    Wqb_nope = Wqb[:, :d_nope, :]                                      # [64, 128, 1536]
    Wqb_rope = Wqb[:, d_nope:, :]                                      # [64, 64, 1536]

    Wkva = w["kv_a_proj_with_mqa.weight"].to(torch.float32)             # [576, 7168]
    kva_norm_w = w["kv_a_layernorm.weight"].to(torch.float32)           # [512]
    Wkvb = w["kv_b_proj.weight"].to(torch.float32)                     # [16384=64*256, 512]
    Wkvb = Wkvb.view(nh, d_nope + d_v, kvlora)                         # [64, 256, 512]
    Wkvb_k = Wkvb[:, :d_nope, :]                                       # [64, 128, 512]
    Wkvb_v = Wkvb[:, d_nope:, :]                                       # [64, 128, 512]

    Wo = w["o_proj.weight"].to(torch.float32)                          # [7168, 8192=64*128]
    Wo_per_head = Wo.view(H, nh, d_v).permute(1, 2, 0)                 # [64, 128, 7168]

    # --- Precompute absorbed weights
    # W_Q_abs[h] = Wqb_nope[h]^T @ Wkvb_k[h]  -> [1536, 512]
    W_Q_abs = torch.einsum("hdq,hdk->hqk", Wqb_nope, Wkvb_k)            # [64, 1536, 512]
    # W_O_abs_head[h] = Wkvb_v[h]^T @ Wo_per_head[h] -> [512, 7168]
    W_O_abs = torch.einsum("hdk,hdo->hko", Wkvb_v, Wo_per_head)         # [64, 512, 7168]
    W_O_abs_flat = W_O_abs.reshape(nh * kvlora, H)                      # [32768, 7168]

    # Round-trip through bf16 to match the kernel's storage dtype — otherwise the
    # test reports bf16 rounding error as a "mismatch" when there's no bug.
    W_Q_abs      = W_Q_abs.to(torch.bfloat16).to(torch.float32)
    W_O_abs_flat = W_O_abs_flat.to(torch.bfloat16).to(torch.float32)

    # --- Synthetic inputs
    g = torch.Generator().manual_seed(args.seed)
    hidden_history = torch.randn(args.seq_len - 1, H, generator=g, dtype=torch.float32) * 0.1
    hidden_now     = torch.randn(H,              generator=g, dtype=torch.float32) * 0.1

    # Build KV history by running steps 0..seq_len-2 through the kv_a + norm + RoPE
    kv_hist_comp = []
    kr_hist      = []
    freqs, mscale = yarn_frequencies(cfg)
    for t in range(args.seq_len - 1):
        kv_lat = hidden_history[t] @ Wkva.T                             # [576]
        kv_comp = rms_norm(kv_lat[:kvlora], kva_norm_w, cfg["rms_eps"]) # [512]
        kr      = apply_rope(kv_lat[kvlora:], t, freqs, mscale)         # [64]
        kv_hist_comp.append(kv_comp)
        kr_hist.append(kr)
    kv_hist_comp = torch.stack(kv_hist_comp, dim=0) if kv_hist_comp else torch.zeros(0, kvlora)
    kr_hist      = torch.stack(kr_hist,      dim=0) if kr_hist      else torch.zeros(0, d_rope)

    # --- Current step
    pos = args.seq_len - 1
    q_lora    = hidden_now @ Wqa.T                                     # [1536]
    q_lora_n  = rms_norm(q_lora, qa_norm_w, cfg["rms_eps"])             # [1536]

    # Non-absorbed reference: compute q_nope and q_rope directly
    q_nope_ref = torch.einsum("hdq,q->hd", Wqb_nope, q_lora_n)          # [64, 128]
    q_rope_raw = torch.einsum("hdq,q->hd", Wqb_rope, q_lora_n)          # [64, 64]
    q_rope     = apply_rope(q_rope_raw, pos, freqs, mscale)             # [64, 64]

    # Absorbed form: q_abs[h] = q_lora_n @ W_Q_abs[h]  == q_nope @ Wkvb_k[h]
    q_abs = torch.einsum("hqk,q->hk", W_Q_abs, q_lora_n)                # [64, 512]
    # Sanity: q_abs should equal q_nope @ Wkvb_k
    q_abs_check = torch.einsum("hd,hdk->hk", q_nope_ref, Wkvb_k)
    print("q_abs vs nope@Wkvb_k  max_abs_err:", (q_abs - q_abs_check).abs().max().item())

    # Current KV
    kv_lat_now  = hidden_now @ Wkva.T                                   # [576]
    kv_comp_new = rms_norm(kv_lat_now[:kvlora], kva_norm_w, cfg["rms_eps"])  # [512]
    k_rope_new  = apply_rope(kv_lat_now[kvlora:], pos, freqs, mscale)   # [64]

    # Append to cache (absorbed-form cache stores only [512 + 64])
    kv_comp_all = torch.cat([kv_hist_comp, kv_comp_new[None]], dim=0)   # [T, 512]
    kr_all      = torch.cat([kr_hist,      k_rope_new[None]],  dim=0)   # [T, 64]
    T = kv_comp_all.shape[0]

    # --- Attention (absorbed form)
    # scores[h, t] = q_abs[h]·kv_comp_all[t]  +  q_rope[h]·kr_all[t]
    scores = torch.einsum("hk,tk->ht", q_abs, kv_comp_all) + torch.einsum("hr,tr->ht", q_rope, kr_all)
    # DeepSeek V3 yarn convention:
    #   cos/sin are multiplied by  mscale = get_mscale(f, mscale) * get_mscale(f, mscale_all)
    #   softmax_scale is multiplied by get_mscale(f, mscale_all)^2 when mscale_all != 0
    # For Kimi: mscale = mscale_all = 1.0, factor = 64 → each get_mscale = 1.416.
    scale = 1.0 / np.sqrt(d_nope + d_rope)
    if cfg["yarn_mscale_all_dim"]:
        m_all = get_mscale(cfg["yarn_factor"], cfg["yarn_mscale_all_dim"])
        scale = scale * m_all * m_all
    print(f"softmax_scale = {scale:.6f}  (mscale applied to cos/sin = {mscale:.6f})")
    scores = scores * scale
    # causal is implicit here: we already only have t <= pos
    probs = torch.softmax(scores.to(torch.float32), dim=-1)             # [64, T]
    attn_out = torch.einsum("ht,tk->hk", probs.to(torch.float32), kv_comp_all.to(torch.float32))  # [64, 512]

    # --- Output projection via absorbed W_O
    layer_out_absorbed = attn_out.reshape(-1) @ W_O_abs_flat             # [7168]

    # Cross-check: non-absorbed path should give the same result
    # V_full[h, t, 128] = kv_comp[t] @ Wkvb_v[h]    — build for sanity
    Vfull = torch.einsum("tk,hdk->thd", kv_comp_all, Wkvb_v)             # [T, 64, 128]
    attn_v = torch.einsum("ht,thd->hd", probs, Vfull)                    # [64, 128]
    layer_out_full = attn_v.reshape(-1) @ Wo.T                           # [7168]
    diff = (layer_out_absorbed - layer_out_full).abs().max().item()
    print(f"absorbed vs non-absorbed layer_out max_abs_err: {diff:.6g}")

    # --- Dump
    def dump(a, name):
        p = os.path.join(args.out, name)
        a = a.contiguous().to(torch.float32).cpu().numpy()
        a.tofile(p)
        print(f"  {name:24s} shape={a.shape} dtype={a.dtype}")

    dump(hidden_now,        "hidden_in.bin")
    # KV history as bf16 concatenated [kv_comp | k_rope] per token for cache-realism
    kv_hist_packed = torch.cat([kv_hist_comp.to(torch.bfloat16),
                                kr_hist.to(torch.bfloat16)], dim=-1)
    # Write bf16 raw
    out_path = os.path.join(args.out, "kv_history.bin")
    kv_hist_packed.contiguous().view(torch.uint16).cpu().numpy().tofile(out_path)
    print(f"  kv_history.bin           shape={tuple(kv_hist_packed.shape)} dtype=bf16")

    dump(q_abs,             "q_abs_ref.bin")
    dump(q_rope,            "q_rope_ref.bin")
    dump(kv_comp_new,       "kv_comp_new.bin")
    dump(k_rope_new,        "k_rope_new.bin")
    dump(attn_out,          "attn_out_ref.bin")
    dump(layer_out_absorbed,"layer_out_ref.bin")
    dump(W_Q_abs,           "W_Q_abs.bin")
    dump(W_O_abs_flat,      "W_O_abs.bin")

    # Also dump the inputs that our kernel chain needs
    dump(Wqa,               "Wqa.bin")
    dump(qa_norm_w,         "qa_norm_w.bin")
    dump(Wqb_rope.reshape(nh * d_rope, qlora), "Wqb_rope.bin")
    dump(Wkva,              "Wkva.bin")
    dump(kva_norm_w,        "kva_norm_w.bin")

    # Config knobs + pos + T for test_mla to consume
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({
            "pos": pos, "T": T,
            "H": H, "num_heads": nh,
            "q_lora_rank": qlora, "kv_lora_rank": kvlora,
            "qk_nope_head_dim": d_nope, "qk_rope_head_dim": d_rope,
            "v_head_dim": d_v,
            "softmax_scale": scale,
            "mscale": float(mscale),
            "rms_eps": cfg["rms_eps"],
        }, f, indent=2)
    print(f"wrote {args.out}/meta.json")

if __name__ == "__main__":
    main()
