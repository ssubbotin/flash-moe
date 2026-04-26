#!/usr/bin/env python3
"""Dump per-layer hidden states from Qwen/Qwen3.6-35B-A3B-FP8 as a numerical
oracle for porting the model to the flash-moe CUDA engine.

For each prompt, captures at every decoder layer:
    input            -> hidden state arriving at the layer (= input_layernorm input)
    post_attn_resid  -> hidden after attn + residual add (= post_attention_layernorm input)
    mlp_out          -> MoE/MLP block output (before residual add)
    post_layer       -> hidden state leaving the layer  (= post_attn_resid + mlp_out)

Plus once per prompt:
    embed       -> token embeddings going into layer 0
    final_norm  -> final RMS-normed hidden
    logits      -> lm_head output  [seq_len, vocab]
    input_ids   -> tokenizer ids
    prompt_text -> the original prompt
    config      -> a dict snapshot of the model config

Usage:
    python3 dump_qwen36_oracle.py inspect            # print one decoder layer's child modules
    python3 dump_qwen36_oracle.py dump --out /var/lib/qwen36-oracle
"""
import argparse, json, os, sys
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

MODEL_ID = os.environ.get("QWEN36_MODEL", "Qwen/Qwen3.6-35B-A3B-FP8")
HF_CACHE = os.environ.get("HF_HOME", "/var/lib/vllm-gemma4/hf-cache")
# Honor a direct local snapshot path to avoid HF cache write-locks.
LOCAL_PATH = os.environ.get("QWEN36_LOCAL")

PROMPTS = [
    ("hello",        "Hello! How are you today?"),
    ("france",       "The capital of France is"),
    ("math",         "2 + 2 ="),
    ("code",         "def fibonacci(n):"),
]

def load_model_and_tok():
    src = LOCAL_PATH or MODEL_ID
    print(f"loading tokenizer: {src}", flush=True)
    tok = AutoTokenizer.from_pretrained(src, trust_remote_code=False, local_files_only=bool(LOCAL_PATH))
    print(f"loading model:     {src}  (~35GB FP8 -> bf16 compute)", flush=True)
    cfg = AutoConfig.from_pretrained(src, trust_remote_code=False, local_files_only=bool(LOCAL_PATH))
    # Workaround: transformers' FP8 quantizer tries to read `intermediate_size` on
    # the inner text config, but Qwen3_5Moe only sets `moe_intermediate_size` /
    # `shared_expert_intermediate_size`. Mirror it across both so the quantizer
    # can wrap experts without crashing.
    text_cfg = getattr(cfg, "text_config", cfg)
    if not hasattr(text_cfg, "intermediate_size") or getattr(text_cfg, "intermediate_size", None) is None:
        fallback = getattr(text_cfg, "moe_intermediate_size", None) or \
                   getattr(text_cfg, "shared_expert_intermediate_size", None)
        if fallback is not None:
            text_cfg.intermediate_size = int(fallback)
            if hasattr(cfg, "intermediate_size"):
                pass
            else:
                cfg.intermediate_size = int(fallback)
            print(f"  patched intermediate_size -> {fallback}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        src,
        config=cfg,
        trust_remote_code=False,
        dtype="auto",
        device_map="cuda:0",
        local_files_only=bool(LOCAL_PATH),
    )
    model.eval()
    return tok, model

def inspect():
    tok, model = load_model_and_tok()
    # find the decoder layers list — different attribute names across families
    base = model
    for attr in ("model", "language_model"):
        if hasattr(base, attr):
            base = getattr(base, attr)
    layers = base.layers if hasattr(base, "layers") else base.layer
    L = layers[0]
    print("\n--- one decoder layer module tree ---")
    for name, child in L.named_children():
        sub = list(child.named_children())
        print(f"  L0.{name:30s}  {child.__class__.__name__}  ({len(sub)} sub)")
        for sname, sc in sub:
            print(f"      .{sname:24s}  {sc.__class__.__name__}")
    print(f"\nnum layers = {len(layers)}, hidden = {L.input_layernorm.weight.shape[0] if hasattr(L,'input_layernorm') else '?'}")

def find_layers(model):
    base = model
    for attr in ("model", "language_model"):
        if hasattr(base, attr):
            base = getattr(base, attr)
    return base, base.layers

def to_np(t):
    if isinstance(t, tuple):
        t = t[0]
    return t.detach().to("cpu", dtype=torch.float32).numpy()

def dump():
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok, model = load_model_and_tok()
    base, layers = find_layers(model)
    cfg = model.config.to_dict()
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    print(f"saved config to {out_dir/'config.json'}")

    prompts = []
    for tag, text in PROMPTS:
        ids = tok.encode(text, add_special_tokens=False)
        prompts.append((tag, text, ids))

    # capture buffers populated by hooks: capture[layer_idx]['input'] etc.
    capture = [{} for _ in range(len(layers))]
    hooks = []

    # ---- factories: each closure binds the layer index ----
    def mk_pre(name):
        def factory(idx):
            def h(_m, inp):
                capture[idx][name] = to_np(inp[0])
            return h
        return factory
    def mk_post(name):
        def factory(idx):
            def h(_m, _inp, out):
                capture[idx][name] = to_np(out)
            return h
        return factory

    pre_input        = mk_pre('input')
    pre_postattn     = mk_pre('post_attn_resid')
    post_mlp         = mk_post('mlp_out')
    post_h_norm      = mk_post('h_norm')          # input_layernorm output
    post_qkv_pre     = mk_post('qkv_pre_conv')    # in_proj_qkv output (pre conv)
    post_qkv_conv    = mk_post('qkv_post_conv')   # conv1d output (pre SiLU; conv stores pre-act)
    post_z           = mk_post('z')               # in_proj_z output
    post_a_raw       = mk_post('a_raw')           # in_proj_a output (alpha logits)
    post_b_raw       = mk_post('b_raw')           # in_proj_b output (beta logits)
    pre_norm_gated   = mk_pre('core_attn_out')    # input to norm (= delta-rule output)
    post_norm_gated  = mk_post('gated_norm_out')  # norm output (post RMSNormGated)
    post_outproj     = mk_post('attn_out')        # out_proj output (= linear_attn output before residual)
    post_attn_block  = mk_post('attn_block_out')  # whole linear_attn / self_attn module output

    for li, L in enumerate(layers):
        if hasattr(L, "input_layernorm"):
            hooks.append(L.input_layernorm.register_forward_pre_hook(pre_input(li)))
            hooks.append(L.input_layernorm.register_forward_hook(post_h_norm(li)))
        if hasattr(L, "post_attention_layernorm"):
            hooks.append(L.post_attention_layernorm.register_forward_pre_hook(pre_postattn(li)))
        # mlp on Qwen3-Moe is the MoE block; for non-Moe it's the MLP
        mlp = getattr(L, "mlp", None) or getattr(L, "block_sparse_moe", None) or getattr(L, "moe", None)
        if mlp is not None:
            hooks.append(mlp.register_forward_hook(post_mlp(li)))

        # Per-layer attention: either GatedDeltaNet (linear_attn) or full attention (self_attn).
        attn = getattr(L, "linear_attn", None)
        if attn is not None:
            hooks.append(attn.register_forward_hook(post_attn_block(li)))
            for sub_name, key_name in (("in_proj_qkv","qkv_pre_conv"),
                                       ("in_proj_z",  "z"),
                                       ("in_proj_a",  "a_raw"),
                                       ("in_proj_b",  "b_raw"),
                                       ("conv1d",     "qkv_post_conv"),
                                       ("out_proj",   "attn_out")):
                sub = getattr(attn, sub_name, None)
                if sub is not None:
                    hooks.append(sub.register_forward_hook(mk_post(key_name)(li)))
            # norm (RMSNormGated): hook input AND output
            sub = getattr(attn, "norm", None)
            if sub is not None:
                hooks.append(sub.register_forward_pre_hook(pre_norm_gated(li)))
                hooks.append(sub.register_forward_hook(post_norm_gated(li)))
        else:
            sa = getattr(L, "self_attn", None)
            if sa is not None:
                hooks.append(sa.register_forward_hook(post_attn_block(li)))
                for sub_name, key_name in (("q_proj","q_proj_out"),
                                           ("k_proj","k_proj_out"),
                                           ("v_proj","v_proj_out"),
                                           ("o_proj","attn_out")):
                    sub = getattr(sa, sub_name, None)
                    if sub is not None:
                        hooks.append(sub.register_forward_hook(mk_post(key_name)(li)))

    try:
        for tag, text, ids in prompts:
            print(f"\n=== prompt '{tag}' ({len(ids)} tokens): {text[:80]} ===", flush=True)
            # reset capture
            for d in capture: d.clear()

            with torch.inference_mode():
                inp = torch.tensor([ids], dtype=torch.long, device="cuda:0")
                out = model(input_ids=inp,
                            output_hidden_states=True,
                            output_attentions=False,
                            use_cache=False)
            hs = out.hidden_states  # tuple of (L+1) tensors
            logits = out.logits

            # post_layer for layer i = hidden_states[i+1]
            payload = {
                "tag": tag,
                "prompt_text": text,
                "input_ids": np.asarray(ids, dtype=np.int64),
                "embed": to_np(hs[0]),
                "final_norm": to_np(hs[-1]),  # this is actually the last hidden after final norm in HF
                "logits": to_np(logits),
                "num_layers": len(layers),
            }
            for li in range(len(layers)):
                # capture[li] may miss substeps if the model arch named things differently
                d = capture[li]
                for k, v in d.items():
                    payload[f"L{li:02d}_{k}"] = v
                payload[f"L{li:02d}_post_layer"] = to_np(hs[li+1])

            fn = out_dir / f"oracle_{tag}.npz"
            np.savez_compressed(fn, **payload)
            print(f"  saved {fn}  ({fn.stat().st_size/1e6:.1f} MB)", flush=True)

            # tiny diagnostic
            top1 = logits[0, -1].argmax().item()
            print(f"  top-1 next id = {top1}  decoded={tok.decode([top1])!r}", flush=True)
            print(f"  embed L2 = {np.linalg.norm(payload['embed']):.3f}, "
                  f"final L2 = {np.linalg.norm(payload['final_norm']):.3f}", flush=True)
            for li in (0, 1, len(layers)//2, len(layers)-1):
                pl = payload[f"L{li:02d}_post_layer"]
                print(f"  L{li:02d} post_layer L2 = {np.linalg.norm(pl):.3f} "
                      f"min={pl.min():.3f} max={pl.max():.3f}", flush=True)
    finally:
        for h in hooks: h.remove()
    print("\ndone.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("inspect")
    d = sub.add_parser("dump")
    d.add_argument("--out", default="/var/lib/qwen36-oracle")
    args = ap.parse_args()
    if args.cmd == "inspect":
        inspect()
    elif args.cmd == "dump":
        dump()
