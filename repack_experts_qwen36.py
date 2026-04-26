#!/usr/bin/env python3
"""repack_experts_qwen36.py — repack Qwen3.6-35B-A3B-FP8 routed experts into
one contiguous binary per layer for fast pread streaming.

Per-layer file layout (packed_experts/layer_{N}.bin):
    expert_0 | expert_1 | ... | expert_255
where each expert_block has fixed sub-layout:
    gate_proj.weight              FP8 [512, 2048] = 1,048,576 B
    gate_proj.weight_scale_inv    BF16 [4, 16]    =       128 B
    up_proj.weight                FP8 [512, 2048] = 1,048,576 B
    up_proj.weight_scale_inv      BF16 [4, 16]    =       128 B
    down_proj.weight              FP8 [2048, 512] = 1,048,576 B
    down_proj.weight_scale_inv    BF16 [16, 4]    =       128 B
Total per expert: 3,146,112 B (~3 MB), per layer: 256 * 3.146M = ~768 MB,
total all 40 layers: ~30 GB.

Expert e's block lives at  e * EXPERT_BYTES  in layer_N.bin → one pread per
active expert (~3 MB) instead of 6 separate reads through the safetensors
index. Inspired by repack_experts_kimi.py.

Usage:
    python3 repack_experts_qwen36.py --model qwen36-fp8 \
                                     --out   qwen36-fp8/packed_experts \
                                     [--layers 0-39] [--verify]
"""
import argparse, os, struct, json, sys
from pathlib import Path

EXPERT_BYTES_GATE_W = 512 * 2048      # FP8 byte
EXPERT_BYTES_GATE_S = 4 * 16 * 2      # BF16 byte
EXPERT_BYTES_UP_W   = 512 * 2048
EXPERT_BYTES_UP_S   = 4 * 16 * 2
EXPERT_BYTES_DOWN_W = 2048 * 512
EXPERT_BYTES_DOWN_S = 16 * 4 * 2
EXPERT_BYTES = (EXPERT_BYTES_GATE_W + EXPERT_BYTES_GATE_S +
                EXPERT_BYTES_UP_W   + EXPERT_BYTES_UP_S   +
                EXPERT_BYTES_DOWN_W + EXPERT_BYTES_DOWN_S)
NUM_EXPERTS = 256

SUBOFFS = [
    ("mlp.experts.{e}.gate_proj.weight",            EXPERT_BYTES_GATE_W, "F8_E4M3"),
    ("mlp.experts.{e}.gate_proj.weight_scale_inv",  EXPERT_BYTES_GATE_S, "BF16"),
    ("mlp.experts.{e}.up_proj.weight",              EXPERT_BYTES_UP_W,   "F8_E4M3"),
    ("mlp.experts.{e}.up_proj.weight_scale_inv",    EXPERT_BYTES_UP_S,   "BF16"),
    ("mlp.experts.{e}.down_proj.weight",            EXPERT_BYTES_DOWN_W, "F8_E4M3"),
    ("mlp.experts.{e}.down_proj.weight_scale_inv",  EXPERT_BYTES_DOWN_S, "BF16"),
]

def read_shard_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n).decode()), 8 + n

def find_tensor(headers, full_name):
    for path, (h, base) in headers.items():
        if full_name in h:
            t = h[full_name]
            beg, end = t["data_offsets"]
            return path, base + beg, end - beg, t["dtype"]
    return None

def repack_layer(layer_idx, model_dir, out_dir, headers, verify):
    out_path = Path(out_dir) / f"layer_{layer_idx}.bin"
    if out_path.exists() and not verify:
        # idempotent: skip if already exists and is correctly sized
        if out_path.stat().st_size == NUM_EXPERTS * EXPERT_BYTES:
            print(f"  layer {layer_idx}: already packed, skipping")
            return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prefix = f"model.language_model.layers.{layer_idx}"
    print(f"  layer {layer_idx}: writing {out_path}", flush=True)

    with open(out_path, "wb") as out:
        for e in range(NUM_EXPERTS):
            for tmpl, expected_size, expected_dtype in SUBOFFS:
                full = prefix + "." + tmpl.format(e=e)
                info = find_tensor(headers, full)
                if info is None:
                    print(f"  ERROR: tensor {full} not found in any shard", file=sys.stderr)
                    sys.exit(1)
                shard_path, off, n, dtype = info
                if n != expected_size or dtype != expected_dtype:
                    print(f"  ERROR: {full} size={n} dtype={dtype} expected={expected_size} {expected_dtype}", file=sys.stderr)
                    sys.exit(1)
                with open(shard_path, "rb") as f:
                    f.seek(off)
                    data = f.read(n)
                out.write(data)
            if e % 64 == 0:
                print(f"    expert {e}/{NUM_EXPERTS}", flush=True)

    actual = out_path.stat().st_size
    expected = NUM_EXPERTS * EXPERT_BYTES
    if actual != expected:
        print(f"  ERROR: {out_path} size {actual} != expected {expected}", file=sys.stderr)
        sys.exit(1)

    if verify:
        # spot-check expert 0 and expert 255
        with open(out_path, "rb") as f:
            for sample_e in (0, 64, 128, 192, NUM_EXPERTS - 1):
                f.seek(sample_e * EXPERT_BYTES)
                blk = f.read(EXPERT_BYTES)
                # check block sizes line up
                so = 0
                for tmpl, n, _ in SUBOFFS:
                    so += n
                if so != EXPERT_BYTES:
                    print(f"  ERROR: sub-offsets sum {so} != {EXPERT_BYTES}", file=sys.stderr)
                    sys.exit(1)
        print(f"  layer {layer_idx}: verified")

def parse_layers(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-"); out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",  default="qwen36-fp8")
    ap.add_argument("--out",    default="qwen36-fp8/packed_experts")
    ap.add_argument("--layers", default="0-39")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    print(f"reading shard headers from {args.model}...", flush=True)
    headers = {}
    for fn in sorted(os.listdir(args.model)):
        if fn.endswith(".safetensors"):
            p = os.path.join(args.model, fn)
            h, base = read_shard_header(p)
            headers[p] = (h, base)
    print(f"  {len(headers)} shards indexed")

    layers = parse_layers(args.layers)
    for li in layers:
        repack_layer(li, args.model, args.out, headers, args.verify)

    print("done.")

if __name__ == "__main__":
    main()
