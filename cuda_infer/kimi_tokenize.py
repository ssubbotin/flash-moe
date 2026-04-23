#!/usr/bin/env python3
"""kimi_tokenize.py — tokenizer bridge for Kimi K2.6.

Uses the model's own tokenization_kimi.py (via HF AutoTokenizer, trust_remote_code)
to encode/decode text into/from the flash-moe infer_kimi --tokens CSV format.

Usage:
    # Encode a prompt → CSV of token IDs (stdout) usable as --tokens
    python3 kimi_tokenize.py encode "Hello, world!" --model-dir kimi-k2.6

    # Decode a CSV of token IDs → text (stdout)
    python3 kimi_tokenize.py decode "1,2,3,4" --model-dir kimi-k2.6

    # Apply chat template and encode
    python3 kimi_tokenize.py chat user:"What is 2+2?" --model-dir kimi-k2.6

The tokenizer loads tiktoken.model + tokenization_kimi.py from the model dir.
"""
import argparse, sys, os

def load_tok(model_dir):
    # trust_remote_code=True lets transformers pick up tokenization_kimi.py from model_dir
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

def cmd_encode(args):
    tok = load_tok(args.model_dir)
    ids = tok.encode(args.text, add_special_tokens=args.add_special)
    print(",".join(str(i) for i in ids))

def cmd_decode(args):
    tok = load_tok(args.model_dir)
    ids = [int(x) for x in args.ids.replace(",", " ").split() if x.strip()]
    text = tok.decode(ids, skip_special_tokens=args.skip_special)
    sys.stdout.write(text)
    if not text.endswith("\n"):
        sys.stdout.write("\n")

def cmd_chat(args):
    """Build a single-turn chat message via chat_template.jinja, encode it."""
    tok = load_tok(args.model_dir)
    # args.messages is a list of "role:content" strings
    msgs = []
    for m in args.messages:
        if ":" not in m:
            print(f"bad message '{m}' (expect role:content)", file=sys.stderr); sys.exit(1)
        role, content = m.split(":", 1)
        msgs.append({"role": role.strip(), "content": content})
    out = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                  return_tensors=None)
    # Different transformers versions return: list[int] | dict | BatchEncoding
    if hasattr(out, "input_ids"):
        ids = out.input_ids
        if hasattr(ids, "tolist"): ids = ids.tolist()
        if isinstance(ids, list) and ids and isinstance(ids[0], list): ids = ids[0]
    elif isinstance(out, dict):
        ids = out.get("input_ids", out)
        if isinstance(ids, list) and ids and isinstance(ids[0], list): ids = ids[0]
    else:
        ids = out
    print(",".join(str(int(i)) for i in ids))

def cmd_info(args):
    tok = load_tok(args.model_dir)
    print("vocab_size :", tok.vocab_size)
    print("bos_token  :", tok.bos_token, "->", tok.bos_token_id)
    print("eos_token  :", tok.eos_token, "->", tok.eos_token_id)
    print("pad_token  :", tok.pad_token, "->", tok.pad_token_id)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.environ.get("KIMI_MODEL_DIR", "kimi-k2.6"))
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("encode", help="text → CSV of token ids")
    e.add_argument("text")
    e.add_argument("--add-special", action="store_true", help="prepend BOS")
    e.set_defaults(func=cmd_encode)

    d = sub.add_parser("decode", help="CSV/space-separated ids → text")
    d.add_argument("ids")
    d.add_argument("--skip-special", action="store_true", help="omit special tokens")
    d.set_defaults(func=cmd_decode)

    c = sub.add_parser("chat", help="apply chat template to role:content messages")
    c.add_argument("messages", nargs="+", help="e.g. user:\"Hi\" assistant:\"Hello\"")
    c.set_defaults(func=cmd_chat)

    i = sub.add_parser("info", help="print vocab size + special tokens")
    i.set_defaults(func=cmd_info)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
