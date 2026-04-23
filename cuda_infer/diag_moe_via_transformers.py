#!/usr/bin/env python3
"""Ultimate cross-check: build Kimi's own DeepseekV3MoE module with real weights
from layer N, run its forward on the hidden input dumped from infer_kimi, and
compare magnitudes to our observed numbers.

If this matches infer_kimi, the model really does produce those magnitudes and
we're missing something in how it's meant to be used.
If it doesn't, we have an architectural bug to fix.
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
        return torch.from_numpy(np.frombuffer(raw, dtype=np.int32).reshape(shape).copy())
    if dt == "BF16":
        return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(shape).clone()
    if dt == "F32":
        return torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(shape).clone()
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

def dequant_sym4_torch(packed_i32: torch.Tensor, scale_bf16: torch.Tensor) -> torch.Tensor:
    """packed_i32 [out, in/8] int32; scale_bf16 [out, in/32] bf16 → f32 [out, in]"""
    packed_np = packed_i32.numpy()
    out_dim, packed_cols = packed_np.shape
    in_dim = packed_cols * 8
    shifts = np.arange(8, dtype=np.uint32) * 4
    nib_u = (packed_np.astype(np.uint32)[:, :, None] >> shifts) & 0xF
    # compressed-tensors sym-int4 biased rep: signed = unsigned - 8 (not two's complement)
    nib_s = nib_u.astype(np.int32) - 8
    W_int = torch.from_numpy(nib_s.reshape(out_dim, in_dim)).to(torch.float32)
    scale_f = scale_bf16.to(torch.float32).repeat_interleave(32, dim=1)
    return W_int * scale_f

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--hidden", required=True)
    args = ap.parse_args()

    # Import Kimi's own modeling code for DeepseekV3MoE
    sys.path.insert(0, args.model_dir)
    from modeling_deepseek import DeepseekV3MoE, DeepseekV3MLP, MoEGate
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    text_cfg = cfg.text_config

    print(f"Config: H={text_cfg.hidden_size} moe_int={text_cfg.moe_intermediate_size} "
          f"ne={text_cfg.n_routed_experts} K={text_cfg.num_experts_per_tok} "
          f"scaling={text_cfg.routed_scaling_factor} norm_topk={text_cfg.norm_topk_prob} "
          f"scoring={text_cfg.scoring_func} topk_method={text_cfg.topk_method}")

    pull = load_all(args.model_dir)
    prefix = f"language_model.model.layers.{args.layer}.mlp"

    # Build an empty MoE module (CPU, f32 for easy inspection)
    moe = DeepseekV3MoE(text_cfg).to(torch.float32).eval()

    # Load gate
    moe.gate.weight.data.copy_(pull(f"{prefix}.gate.weight").to(torch.float32))
    moe.gate.e_score_correction_bias.data.copy_(pull(f"{prefix}.gate.e_score_correction_bias"))

    # Load shared experts
    moe.shared_experts.gate_proj.weight.data.copy_(pull(f"{prefix}.shared_experts.gate_proj.weight").to(torch.float32))
    moe.shared_experts.up_proj.weight.data.copy_(pull(f"{prefix}.shared_experts.up_proj.weight").to(torch.float32))
    moe.shared_experts.down_proj.weight.data.copy_(pull(f"{prefix}.shared_experts.down_proj.weight").to(torch.float32))

    # Load routed experts — dequantize each
    for e in range(text_cfg.n_routed_experts):
        base = f"{prefix}.experts.{e}"
        Wg = dequant_sym4_torch(pull(f"{base}.gate_proj.weight_packed"),
                                pull(f"{base}.gate_proj.weight_scale"))
        Wu = dequant_sym4_torch(pull(f"{base}.up_proj.weight_packed"),
                                pull(f"{base}.up_proj.weight_scale"))
        Wd = dequant_sym4_torch(pull(f"{base}.down_proj.weight_packed"),
                                pull(f"{base}.down_proj.weight_scale"))
        moe.experts[e].gate_proj.weight.data.copy_(Wg)
        moe.experts[e].up_proj.weight.data.copy_(Wu)
        moe.experts[e].down_proj.weight.data.copy_(Wd)
        if e % 64 == 0:
            print(f"  loaded expert {e}/{text_cfg.n_routed_experts}")

    # Prepare input in [B=1, T=1, H] shape like production
    x_np = np.fromfile(args.hidden, dtype=np.float32)
    x = torch.from_numpy(x_np).to(torch.float32).view(1, 1, -1)
    print(f"input: L2={x.view(-1).norm().item():.4g} shape={tuple(x.shape)}")

    with torch.no_grad():
        out = moe(x)  # This returns the MoE output (moe_infer + shared_experts)

    out_flat = out.view(-1)
    print(f"\ntransformers DeepseekV3MoE output:")
    print(f"  L2      = {out_flat.norm().item():.4g}")
    print(f"  mean    = {out_flat.mean().item():.4g}")
    print(f"  RMS     = {out_flat.std().item():.4g}")
    print(f"  first6  = {out_flat[:6].tolist()}")
    print(f"\nExpected from infer_kimi L{args.layer}: moe_accum+shared ≈ 1097 (combine w/ residual → ~1100)")

if __name__ == "__main__":
    main()
