#!/usr/bin/env python3
"""
repack_experts_kimi.py — convert Kimi K2.6 compressed-tensors sym-int4 routed
experts from safetensors shards into a streaming-friendly on-disk layout.

Output: one file per MoE layer, layers 1..60 (layer 0 is dense — skipped).
Per-layer file contains 384 expert blocks back-to-back, each block:

    gate_packed   int32[2048, 896]    (7,340,032 B)
    gate_scale    bf16 [2048, 224]    (  917,504 B)
    up_packed     int32[2048, 896]    (7,340,032 B)
    up_scale      bf16 [2048, 224]    (  917,504 B)
    down_packed   int32[7168, 256]    (7,340,032 B)
    down_scale    bf16 [7168,  64]    (  917,504 B)
    -----------------------------------------------
    total per expert                   24,772,608 B   (23.625 MiB)

One pread(layer_fd, expert_idx * 24_772_608, 24_772_608) returns everything
needed for one expert forward.

Usage:
    python3 repack_experts_kimi.py \
        --model-dir kimi-k2.6 \
        --out-dir   kimi-k2.6/packed_experts \
        --layers 1-60 \
        [--verify]

Layers may be a comma+range list like "1-5,30,60".
"""
import argparse, json, os, struct, sys, mmap
from pathlib import Path
from typing import Tuple

# -------- constants for Kimi K2.6 --------
NUM_ROUTED_EXPERTS   = 384
MOE_INTERMEDIATE     = 2048
HIDDEN_SIZE          = 7168
GROUP_SIZE           = 32
BYTES_PER_I32        = 4
BYTES_PER_BF16       = 2

GATE_PACKED_SHAPE = (MOE_INTERMEDIATE, HIDDEN_SIZE // 8)     # (2048, 896)
GATE_SCALE_SHAPE  = (MOE_INTERMEDIATE, HIDDEN_SIZE // GROUP_SIZE)  # (2048, 224)
DOWN_PACKED_SHAPE = (HIDDEN_SIZE, MOE_INTERMEDIATE // 8)     # (7168, 256)
DOWN_SCALE_SHAPE  = (HIDDEN_SIZE, MOE_INTERMEDIATE // GROUP_SIZE)  # (7168, 64)
UP_PACKED_SHAPE   = GATE_PACKED_SHAPE
UP_SCALE_SHAPE    = GATE_SCALE_SHAPE

GATE_PACKED_BYTES = GATE_PACKED_SHAPE[0] * GATE_PACKED_SHAPE[1] * BYTES_PER_I32
GATE_SCALE_BYTES  = GATE_SCALE_SHAPE[0]  * GATE_SCALE_SHAPE[1]  * BYTES_PER_BF16
UP_PACKED_BYTES   = GATE_PACKED_BYTES
UP_SCALE_BYTES    = GATE_SCALE_BYTES
DOWN_PACKED_BYTES = DOWN_PACKED_SHAPE[0] * DOWN_PACKED_SHAPE[1] * BYTES_PER_I32
DOWN_SCALE_BYTES  = DOWN_SCALE_SHAPE[0]  * DOWN_SCALE_SHAPE[1]  * BYTES_PER_BF16

EXPERT_BLOCK_BYTES = (GATE_PACKED_BYTES + GATE_SCALE_BYTES +
                      UP_PACKED_BYTES   + UP_SCALE_BYTES   +
                      DOWN_PACKED_BYTES + DOWN_SCALE_BYTES)
assert EXPERT_BLOCK_BYTES == 24_772_608, EXPERT_BLOCK_BYTES

LAYER_FILE_BYTES = EXPERT_BLOCK_BYTES * NUM_ROUTED_EXPERTS   # 9 072 MiB per layer


# -------- safetensors reading --------

class ShardMap:
    """Shard-map + cached file handles. Reads one tensor on demand via mmap."""
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        idx_path = os.path.join(model_dir, "model.safetensors.index.json")
        with open(idx_path) as f:
            self.weight_map = json.load(f)["weight_map"]
        # tensor name → (shard, offset, nbytes, dtype, shape)
        self._header_cache = {}
        self._mmap_cache = {}

    def _get_header(self, shard: str):
        if shard in self._header_cache:
            return self._header_cache[shard]
        path = os.path.join(self.model_dir, shard)
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        self._header_cache[shard] = (hdr, 8 + n)
        return hdr, 8 + n

    def _get_mmap(self, shard: str):
        if shard in self._mmap_cache:
            return self._mmap_cache[shard]
        f = open(os.path.join(self.model_dir, shard), "rb")
        mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
        self._mmap_cache[shard] = mm
        return mm

    def close(self):
        for mm in self._mmap_cache.values():
            mm.close()

    def read(self, name: str) -> Tuple[memoryview, str, Tuple[int, ...]]:
        shard = self.weight_map[name]
        hdr, base = self._get_header(shard)
        info = hdr[name]
        beg, end = info["data_offsets"]
        mm = self._get_mmap(shard)
        mv = memoryview(mm)[base + beg : base + end]
        return mv, info["dtype"], tuple(info["shape"])


# -------- repack logic --------

def _expect(mv, dtype, shape, want_dtype, want_shape, name):
    if dtype != want_dtype:
        raise RuntimeError(f"{name}: dtype {dtype} != {want_dtype}")
    if tuple(shape) != tuple(want_shape):
        raise RuntimeError(f"{name}: shape {shape} != {want_shape}")
    prod = 1
    for d in want_shape: prod *= d
    expect_n = prod * (4 if want_dtype == "I32" else 2)
    if len(mv) != expect_n:
        raise RuntimeError(f"{name}: {len(mv)} bytes, expected {expect_n}")


def repack_layer(shards: ShardMap, layer_idx: int, out_path: str, verify: bool):
    prefix = f"language_model.model.layers.{layer_idx}.mlp.experts"
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as out:
        for e in range(NUM_ROUTED_EXPERTS):
            base = f"{prefix}.{e}"

            def pull(proj, suffix, want_dtype, want_shape):
                mv, dt, sh = shards.read(f"{base}.{proj}.{suffix}")
                _expect(mv, dt, sh, want_dtype, want_shape, f"L{layer_idx}E{e}.{proj}.{suffix}")
                return mv

            # Sequence matches the on-disk layout doc above
            out.write(pull("gate_proj", "weight_packed", "I32",  GATE_PACKED_SHAPE))
            out.write(pull("gate_proj", "weight_scale",  "BF16", GATE_SCALE_SHAPE))
            out.write(pull("up_proj",   "weight_packed", "I32",  UP_PACKED_SHAPE))
            out.write(pull("up_proj",   "weight_scale",  "BF16", UP_SCALE_SHAPE))
            out.write(pull("down_proj", "weight_packed", "I32",  DOWN_PACKED_SHAPE))
            out.write(pull("down_proj", "weight_scale",  "BF16", DOWN_SCALE_SHAPE))

            if (e + 1) % 64 == 0 or e == NUM_ROUTED_EXPERTS - 1:
                print(f"  L{layer_idx}: {e+1}/{NUM_ROUTED_EXPERTS} experts written", flush=True)

        out.flush()
        os.fsync(out.fileno())

    size = os.path.getsize(tmp)
    if size != LAYER_FILE_BYTES:
        raise RuntimeError(f"layer {layer_idx}: wrote {size} bytes, expected {LAYER_FILE_BYTES}")
    os.replace(tmp, out_path)

    if verify:
        verify_spot_check(shards, layer_idx, out_path)


def verify_spot_check(shards: ShardMap, layer_idx: int, layer_path: str):
    """Re-read a few experts' sub-blocks from the packed file and byte-compare
    with the originals from the safetensors shards."""
    checks = [0, 1, 200, NUM_ROUTED_EXPERTS // 2, NUM_ROUTED_EXPERTS - 1]
    fd = os.open(layer_path, os.O_RDONLY)
    try:
        for e in checks:
            base = f"language_model.model.layers.{layer_idx}.mlp.experts.{e}"
            off = e * EXPERT_BLOCK_BYTES
            for name, nbytes, dtype, shape, proj, suf in [
                ("gate_packed", GATE_PACKED_BYTES, "I32",  GATE_PACKED_SHAPE, "gate_proj", "weight_packed"),
                ("gate_scale",  GATE_SCALE_BYTES,  "BF16", GATE_SCALE_SHAPE,  "gate_proj", "weight_scale"),
                ("up_packed",   UP_PACKED_BYTES,   "I32",  UP_PACKED_SHAPE,   "up_proj",   "weight_packed"),
                ("up_scale",    UP_SCALE_BYTES,    "BF16", UP_SCALE_SHAPE,    "up_proj",   "weight_scale"),
                ("down_packed", DOWN_PACKED_BYTES, "I32",  DOWN_PACKED_SHAPE, "down_proj", "weight_packed"),
                ("down_scale",  DOWN_SCALE_BYTES,  "BF16", DOWN_SCALE_SHAPE,  "down_proj", "weight_scale"),
            ]:
                buf = os.pread(fd, nbytes, off)
                if len(buf) != nbytes:
                    raise RuntimeError(f"L{layer_idx}E{e} short pread {name}")
                mv, _, _ = shards.read(f"{base}.{proj}.{suf}")
                if bytes(buf) != bytes(mv):
                    raise RuntimeError(f"L{layer_idx}E{e}.{name} MISMATCH")
                off += nbytes
        print(f"  L{layer_idx}: verification PASSED (experts {checks})")
    finally:
        os.close(fd)


def parse_layers(spec: str):
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part: continue
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--layers", default="1-60",
        help="comma+range list, e.g. '1-60' or '1,5,10-15'")
    ap.add_argument("--verify", action="store_true",
        help="spot-check 5 experts per layer by re-reading and byte-comparing")
    ap.add_argument("--skip-existing", action="store_true",
        help="skip any layer whose output file already exists at the correct size")
    args = ap.parse_args()

    layers = parse_layers(args.layers)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print(f"Model dir : {args.model_dir}")
    print(f"Out dir   : {args.out_dir}")
    print(f"Layers    : {layers[0]}..{layers[-1]} ({len(layers)} total)")
    print(f"Per layer : {LAYER_FILE_BYTES/1e9:.2f} GB")
    print(f"Total     : {len(layers) * LAYER_FILE_BYTES/1e9:.2f} GB")

    shards = ShardMap(args.model_dir)
    try:
        for li in layers:
            out_path = os.path.join(args.out_dir, f"layer_{li}.bin")
            if args.skip_existing and os.path.exists(out_path) and \
               os.path.getsize(out_path) == LAYER_FILE_BYTES:
                print(f"Layer {li}: already exists, skipping")
                continue
            print(f"Layer {li}: writing {out_path}")
            repack_layer(shards, li, out_path, args.verify)
    finally:
        shards.close()

    print("done")


if __name__ == "__main__":
    main()
