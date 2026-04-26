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

def cmd_serve(args):
    """Long-lived stdin/stdout protocol for the C driver.

    Each request is one line: <verb>\\t<arg1>\\t<arg2>...
    Each response is: OK\\t<payload-without-newlines>\\n   (or ERR\\t<msg>\\n)

    Verbs:
      info                                     -> OK\\t<vocab>\\t<bos>\\t<eos1,eos2,...>
      encode\\t<base64-utf8-text>               -> OK\\t<csv ids>
      chat\\t<base64-jsonl-messages>            -> OK\\t<csv ids>
      decode_full\\t<csv ids>                   -> OK\\t<base64-utf8-text>
      decode_stream_reset                      -> OK
      decode_stream_push\\t<id>                 -> OK\\t<base64-utf8-fragment>
      ping                                     -> OK\\tpong
      quit                                     -> OK and exit

    The text payloads are base64 to avoid all line/tab/newline escaping issues.
    """
    import base64, json, sys
    tok = load_tok(args.model_dir)

    eos_ids = []
    if tok.eos_token_id is not None:
        ids = tok.eos_token_id if isinstance(tok.eos_token_id, list) else [tok.eos_token_id]
        eos_ids.extend(int(i) for i in ids)
    # Kimi-style models often have several end-of-message ids; include any that look like one.
    for name in ("eot_token_id", "im_end_id"):
        v = getattr(tok, name, None)
        if isinstance(v, int) and v not in eos_ids:
            eos_ids.append(v)
    added = getattr(tok, "added_tokens_encoder", {}) or {}
    for s, i in added.items():
        if any(k in s.lower() for k in ("eos", "end_of", "eot", "<|im_end|>")):
            if int(i) not in eos_ids:
                eos_ids.append(int(i))

    stream_ids = []
    stream_text = ""
    sys.stdout.write("READY\n"); sys.stdout.flush()

    def b64d(s): return base64.b64decode(s).decode("utf-8")
    def b64e(s): return base64.b64encode(s.encode("utf-8")).decode("ascii")
    def reply(*parts):
        sys.stdout.write("\t".join(parts)); sys.stdout.write("\n"); sys.stdout.flush()

    for line in sys.stdin:
        line = line.rstrip("\n")
        if not line: continue
        parts = line.split("\t")
        verb = parts[0]
        try:
            if verb == "info":
                reply("OK", str(tok.vocab_size), str(tok.bos_token_id or -1),
                      ",".join(str(i) for i in eos_ids) if eos_ids else "-1")
            elif verb == "encode":
                text = b64d(parts[1])
                ids = tok.encode(text, add_special_tokens=False)
                reply("OK", ",".join(str(i) for i in ids))
            elif verb == "chat":
                msgs = json.loads(b64d(parts[1]))
                out = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                              tokenize=True, return_tensors=None)
                if hasattr(out, "input_ids"):
                    ids = out.input_ids
                    if hasattr(ids, "tolist"): ids = ids.tolist()
                    if isinstance(ids, list) and ids and isinstance(ids[0], list): ids = ids[0]
                elif isinstance(out, dict):
                    ids = out.get("input_ids", out)
                    if isinstance(ids, list) and ids and isinstance(ids[0], list): ids = ids[0]
                else:
                    ids = out
                reply("OK", ",".join(str(int(i)) for i in ids))
            elif verb == "decode_full":
                ids = [int(x) for x in parts[1].split(",") if x.strip()]
                reply("OK", b64e(tok.decode(ids, skip_special_tokens=True)))
            elif verb == "decode_stream_reset":
                stream_ids = []; stream_text = ""
                reply("OK")
            elif verb == "decode_stream_push":
                stream_ids.append(int(parts[1]))
                full = tok.decode(stream_ids, skip_special_tokens=False)
                # only emit the suffix vs what we've already shown
                if full.startswith(stream_text):
                    frag = full[len(stream_text):]
                else:
                    frag = full  # detokenizer rewrote earlier text; surface the whole thing
                stream_text = full
                reply("OK", b64e(frag))
            elif verb == "ping":
                reply("OK", "pong")
            elif verb == "quit":
                reply("OK"); return
            else:
                reply("ERR", "unknown verb: " + verb)
        except Exception as e:
            reply("ERR", repr(e))

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

    s = sub.add_parser("serve", help="line-based stdin/stdout protocol for the C driver")
    s.set_defaults(func=cmd_serve)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
