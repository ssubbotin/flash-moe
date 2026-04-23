#!/usr/bin/env python3
"""Load Moonlight-16B-A3B-Instruct (DeepSeek-V3 arch, same family as Kimi K2.6)
in transformers and dump the residual-stream L2 norm after each decoder layer
for a simple input. Answers the question: does the published modeling code
produce stable residuals in production, or does it explode like we see?

Also dumps (to a pickle) each layer's residual + post-attn-norm + MoE output
so we can study the formula empirically.
"""
import argparse, os, sys, time, pickle
import numpy as np
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", default="/tmp/moonlight_residuals.pkl")
    ap.add_argument("--tokens", default="1008",
        help="CSV token IDs for prompt (default: 'The' for Kimi vocab)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    sys.path.insert(0, args.model_dir)
    from transformers import AutoModelForCausalLM, AutoConfig

    # Kimi's modeling_deepseek.py (symlinked to Moonlight's) may break against
    # transformers 5.x due to removed imports. Patch if needed.
    try:
        from transformers.utils.import_utils import is_torch_fx_available  # noqa
    except ImportError:
        import transformers.utils.import_utils as _u
        _u.is_torch_fx_available = lambda: False

    # Moonlight's config has no rope_scaling; Kimi's modeling_deepseek.py assumes
    # one and KeyErrors on ["type"]. transformers >=4.45 helpfully adds a default
    # {"rope_theta": ..., "rope_type": "default"} which the Kimi code doesn't
    # understand. Force rope_scaling=None to take Kimi's no-scaling branch.
    cfg = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    cfg_text = cfg if not hasattr(cfg, "text_config") else cfg.text_config
    cfg_text.rope_scaling = None
    cfg.rope_scaling = None
    # FlashAttention path asserts attention_mask is not None; eager works on CPU
    cfg._attn_implementation = "eager"
    cfg_text._attn_implementation = "eager"

    dtype = getattr(torch, args.dtype)
    print(f"Loading {args.model_dir} in {dtype} on {args.device}...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, trust_remote_code=True, dtype=dtype,
        low_cpu_mem_usage=True, config=cfg,
    ).to(args.device).eval()
    print(f"Loaded in {time.time() - t0:.1f}s", flush=True)

    tokens = [int(t) for t in args.tokens.split(",") if t.strip()]
    input_ids = torch.tensor([tokens], device=args.device)
    print(f"Input token IDs: {tokens}", flush=True)

    # Hook every decoder layer's output
    layer_outputs = []
    def make_hook(idx):
        def hook(module, input_, output):
            # output for DeepseekV3DecoderLayer is a tuple; [0] is the hidden state
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            h_f = h.float().view(-1)
            layer_outputs.append({
                "layer": idx,
                "shape": tuple(h.shape),
                "L2": float(h_f.norm().item()),
                "RMS": float(h_f.std().item()),
                "mean": float(h_f.mean().item()),
                "first6": h_f[:6].tolist(),
                "nan": int(torch.isnan(h_f).sum().item()),
                "inf": int(torch.isinf(h_f).sum().item()),
            })
        return hook

    # DeepseekV3Model has `layers` ModuleList under `model`
    # For DeepseekV3ForCausalLM: model.model.layers
    layers = model.model.layers
    print(f"Registering hooks on {len(layers)} decoder layers", flush=True)
    handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)]

    # Also hook each layer's mlp (dense or MoE) to see MLP contribution L2
    mlp_outputs = []
    def make_mlp_hook(idx):
        def h(m, input_, output):
            t = output if isinstance(output, torch.Tensor) else output[0]
            f = t.float().view(-1)
            mlp_outputs.append({"layer": idx, "kind": type(m).__name__,
                                "L2": float(f.norm().item()),
                                "RMS": float(f.std().item()),
                                "mean": float(f.mean().item())})
        return h
    for i, layer in enumerate(layers):
        handles.append(layer.mlp.register_forward_hook(make_mlp_hook(i)))

    # And hook self_attn output to see attention contribution L2
    attn_outputs = []
    def make_attn_hook(idx):
        def h(m, input_, output):
            t = output if isinstance(output, torch.Tensor) else output[0]
            f = t.float().view(-1)
            attn_outputs.append({"layer": idx,
                                 "L2": float(f.norm().item()),
                                 "RMS": float(f.std().item())})
        return h
    for i, layer in enumerate(layers):
        handles.append(layer.self_attn.register_forward_hook(make_attn_hook(i)))

    # Hook embed + final norm + lm_head for completeness
    embed_out = {}
    def embed_hook(m, i_, o):
        h = o.float().view(-1)
        embed_out.update({"L2": float(h.norm().item()),
                          "RMS": float(h.std().item()),
                          "mean": float(h.mean().item()),
                          "first6": h[:6].tolist()})
    handles.append(model.model.embed_tokens.register_forward_hook(embed_hook))

    final_norm_out = {}
    def fn_hook(m, i_, o):
        h = o.float().view(-1)
        final_norm_out.update({"L2": float(h.norm().item()),
                               "RMS": float(h.std().item())})
    handles.append(model.model.norm.register_forward_hook(fn_hook))

    print("Running forward...", flush=True)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_hidden_states=False)

    logits = out.logits[0, -1].float()  # [vocab]
    top5 = torch.topk(logits, 5)
    print("\n=== per-layer hidden state (post-layer) ===")
    print(f"  embed          L2={embed_out['L2']:.4g}  RMS={embed_out['RMS']:.4g}  mean={embed_out['mean']:.4g}")
    for info, mlp_info, attn_info in zip(layer_outputs, mlp_outputs, attn_outputs):
        print(f"  L{info['layer']:02d}  residual={info['L2']:<10.4g}  attn_out={attn_info['L2']:<10.4g}  mlp_out={mlp_info['L2']:<10.4g}  ({mlp_info['kind']})")
    print(f"  final_norm     L2={final_norm_out['L2']:.4g}  RMS={final_norm_out['RMS']:.4g}")
    print(f"\n  top5 logits    {[(int(i), float(v)) for i, v in zip(top5.indices, top5.values)]}")

    for h in handles: h.remove()

    with open(args.out, "wb") as f:
        pickle.dump({"embed": embed_out,
                     "layers": layer_outputs,
                     "mlp": mlp_outputs,
                     "attn": attn_outputs,
                     "final_norm": final_norm_out,
                     "top5": [(int(i), float(v)) for i, v in zip(top5.indices, top5.values)]}, f)
    print(f"\nwrote {args.out}")

if __name__ == "__main__":
    main()
