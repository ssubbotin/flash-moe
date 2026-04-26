#!/usr/bin/env python3
"""Extract a Qwen3.6 oracle .npz into a directory of raw f32 .bin files plus a
manifest, so the C++ substep validation test can read tensors without an npz
parser.

Each tensor becomes one file `<key>.bin` written as little-endian float32 in
C-contiguous order.  The manifest is a JSON map  key -> {shape, dtype, nbytes}.

Usage:
    python3 oracle_to_bin.py --in qwen36-oracle/oracle_france.npz \\
                             --out qwen36-oracle/france_bin
"""
import argparse, json, os
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--in",  dest="inp", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
d = np.load(args.inp, allow_pickle=True)

manifest = {}
for k in d.files:
    a = d[k]
    if a.dtype.kind == "U" or a.dtype == object:
        # strings (prompt_text, tag) — keep in manifest, no bin
        manifest[k] = {"object": str(a.item() if a.shape == () else a.tolist())}
        continue
    if a.dtype.kind in ("i", "u"):
        # integers: keep as int64 (input_ids) for fidelity
        out_arr = np.ascontiguousarray(a.astype(np.int64))
        dtype = "i64"
    else:
        out_arr = np.ascontiguousarray(a.astype(np.float32))
        dtype = "f32"
    out_arr.tofile(out / f"{k}.bin")
    manifest[k] = {"shape": list(map(int, a.shape)), "dtype": dtype,
                   "nbytes": int(out_arr.nbytes)}

(out / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(f"wrote {len([k for k,v in manifest.items() if 'shape' in v])} tensors to {out}")
print(f"manifest: {out/'manifest.json'}")
