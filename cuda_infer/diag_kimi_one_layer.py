#!/usr/bin/env python3
"""Load ONE Kimi K2.6 DeepseekV3DecoderLayer via transformers + Kimi's own
modeling_deepseek.py, materialise its weights (dequanting experts via the
compressed-tensors library), and run forward on the exact hidden state my
infer_kimi dumped at the same layer boundary.

Lets us compare per-substep against the transformers reference:
  - post-attention residual output
  - post-MLP final output

If the reference matches my GPU output at a layer, my implementation is right
for that layer. If it diverges, we've located the remaining bug.

Usage:
    python3 diag_kimi_one_layer.py --model-dir kimi-k2.6 --layer 1 \
        --hidden /tmp/kimi_L1_input.bin \
        --attnres-ref /tmp/kimi_L1_attnres.bin \
        --output-ref /tmp/kimi_L1_output.bin
"""
import argparse, json, os, struct, sys, gc
import numpy as np
import torch

# Patch for transformers >= 5.0 compat with Kimi's modeling_deepseek.py
import transformers.utils.import_utils as _u
if not hasattr(_u, "is_torch_fx_available"):
    _u.is_torch_fx_available = lambda: False

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

def dequant_expert_proj(model_dir, layer_idx, expert, proj, idx):
    base = f"language_model.model.layers.{layer_idx}.mlp.experts.{expert}.{proj}"
    packed = pull(model_dir, f"{base}.weight_packed", idx)
    scale  = pull(model_dir, f"{base}.weight_scale",  idx)
    shape  = pull(model_dir, f"{base}.weight_shape",  idx)
    orig = torch.Size(shape.tolist())
    unpacked = unpack_from_int32(packed, 4, orig)          # signed int8 in [-8, 7]
    scale_full = scale.to(torch.float32).repeat_interleave(32, dim=1)
    return (unpacked.to(torch.float32) * scale_full)

def compare(a, b, tag):
    a = a.view(-1).float().cpu()
    b = b.view(-1).float().cpu()
    diff = (a - b).abs()
    ref_max = b.abs().max().item()
    print(f"  {tag:20s} got L2={a.norm().item():.4g}  ref L2={b.norm().item():.4g}  "
          f"max_abs={diff.max().item():.4g}  mean_abs={diff.mean().item():.4g}  "
          f"rel={100*diff.max().item()/max(ref_max,1e-9):.2f}%")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--hidden", required=True, help="f32 [H] input hidden from infer_kimi")
    ap.add_argument("--attnres-ref", default=None, help="f32 [H] expected post-attention residual")
    ap.add_argument("--output-ref",  default=None, help="f32 [H] expected post-layer output")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = ap.parse_args()

    # Load Kimi's modeling_deepseek.py through the HF transformers dynamic-module
    # loader so relative imports (from .configuration_deepseek import …) resolve.
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from transformers import AutoConfig
    DeepseekV3DecoderLayer = get_class_from_dynamic_module(
        "modeling_deepseek.DeepseekV3DecoderLayer",
        args.model_dir,
    )

    cfg = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    text_cfg = cfg.text_config
    text_cfg.rope_scaling = None
    text_cfg._attn_implementation = "eager"

    idx = json.load(open(os.path.join(args.model_dir, "model.safetensors.index.json")))["weight_map"]

    dtype = getattr(torch, args.dtype)
    layer = DeepseekV3DecoderLayer(text_cfg, layer_idx=args.layer).to(dtype).eval()

    pre = f"language_model.model.layers.{args.layer}"

    # ---- Load MLA weights
    def get(n): return pull(args.model_dir, f"{pre}.{n}", idx).to(dtype)
    layer.input_layernorm.weight.data.copy_(get("input_layernorm.weight"))
    layer.post_attention_layernorm.weight.data.copy_(get("post_attention_layernorm.weight"))
    layer.self_attn.q_a_proj.weight.data.copy_(get("self_attn.q_a_proj.weight"))
    layer.self_attn.q_a_layernorm.weight.data.copy_(get("self_attn.q_a_layernorm.weight"))
    layer.self_attn.q_b_proj.weight.data.copy_(get("self_attn.q_b_proj.weight"))
    layer.self_attn.kv_a_proj_with_mqa.weight.data.copy_(get("self_attn.kv_a_proj_with_mqa.weight"))
    layer.self_attn.kv_a_layernorm.weight.data.copy_(get("self_attn.kv_a_layernorm.weight"))
    layer.self_attn.kv_b_proj.weight.data.copy_(get("self_attn.kv_b_proj.weight"))
    layer.self_attn.o_proj.weight.data.copy_(get("self_attn.o_proj.weight"))

    # ---- MLP: dense if first_k_dense, MoE otherwise
    is_dense = args.layer < text_cfg.first_k_dense_replace
    if is_dense:
        layer.mlp.gate_proj.weight.data.copy_(get("mlp.gate_proj.weight"))
        layer.mlp.up_proj.weight.data.copy_(get("mlp.up_proj.weight"))
        layer.mlp.down_proj.weight.data.copy_(get("mlp.down_proj.weight"))
    else:
        layer.mlp.gate.weight.data.copy_(get("mlp.gate.weight"))
        layer.mlp.gate.e_score_correction_bias.data.copy_(
            pull(args.model_dir, f"{pre}.mlp.gate.e_score_correction_bias", idx).to(dtype))
        layer.mlp.shared_experts.gate_proj.weight.data.copy_(get("mlp.shared_experts.gate_proj.weight"))
        layer.mlp.shared_experts.up_proj.weight.data.copy_(get("mlp.shared_experts.up_proj.weight"))
        layer.mlp.shared_experts.down_proj.weight.data.copy_(get("mlp.shared_experts.down_proj.weight"))
        ne = text_cfg.n_routed_experts
        for e in range(ne):
            if e % 64 == 0:
                print(f"  dequanting expert {e}/{ne}", flush=True)
            layer.mlp.experts[e].gate_proj.weight.data.copy_(
                dequant_expert_proj(args.model_dir, args.layer, e, "gate_proj", idx).to(dtype))
            layer.mlp.experts[e].up_proj.weight.data.copy_(
                dequant_expert_proj(args.model_dir, args.layer, e, "up_proj", idx).to(dtype))
            layer.mlp.experts[e].down_proj.weight.data.copy_(
                dequant_expert_proj(args.model_dir, args.layer, e, "down_proj", idx).to(dtype))
            gc.collect()

    # ---- Input
    x = torch.from_numpy(np.fromfile(args.hidden, dtype=np.float32)).to(dtype).view(1, 1, -1)
    H = x.shape[-1]
    print(f"\ninput: L2={x.view(-1).float().norm().item():.4g}")

    # causal mask: [1, 1, 1, 1] of zero (no masking for single token)
    attention_mask = torch.zeros(1, 1, 1, 1, dtype=dtype)
    position_ids = torch.tensor([[0]], dtype=torch.long)

    # ---- Forward
    with torch.no_grad():
        hidden_states = x

        # Step manually to match our decomposition
        residual = hidden_states
        hn = layer.input_layernorm(hidden_states)
        attn_out, _, _ = layer.self_attn(
            hidden_states=hn,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
        )
        attnres = residual + attn_out

        residual2 = attnres
        hn2 = layer.post_attention_layernorm(attnres)
        mlp_out = layer.mlp(hn2)
        out = residual2 + mlp_out

    print(f"\n--- substep magnitudes (layer {args.layer}) ---")
    print(f"  input_norm         L2={hn.view(-1).float().norm():.4g}")
    print(f"  attn_out           L2={attn_out.view(-1).float().norm():.4g}")
    print(f"  attnres            L2={attnres.view(-1).float().norm():.4g}")
    print(f"  post_attn_norm     L2={hn2.view(-1).float().norm():.4g}")
    print(f"  mlp_out            L2={mlp_out.view(-1).float().norm():.4g}")
    print(f"  output             L2={out.view(-1).float().norm():.4g}")

    if args.attnres_ref:
        ref = torch.from_numpy(np.fromfile(args.attnres_ref, dtype=np.float32))
        compare(attnres, ref, "attnres vs GPU")
    if args.output_ref:
        ref = torch.from_numpy(np.fromfile(args.output_ref, dtype=np.float32))
        compare(out, ref, "output vs GPU")

if __name__ == "__main__":
    main()
