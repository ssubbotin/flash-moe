#!/usr/bin/env bash
# Usage: ./bench.sh [cold|warm] [tokens] [vram_cache_mib]
# Examples:
#   ./bench.sh cold 30          # cold cache, no VRAM cache, 30 tokens
#   ./bench.sh warm 100 20480   # warm cache, 20 GB VRAM cache, 100 tokens
#
# The harness gates every run on correctness: the first three lines of
# generated text must match bench/expected.txt (" Paris.<|im_end|>" etc).
# Any mismatch exits non-zero BEFORE the timing number is reported.
set -euo pipefail

MODE="${1:-warm}"
TOKENS="${2:-30}"
VRAM_MIB="${3:-0}"

cd "$(dirname "$0")/.."

if [[ "$MODE" == "cold" ]]; then
    sync
    echo 3 > /proc/sys/vm/drop_caches
fi

# The infer binary's only VRAM-cache control knob is DISABLE_VRAM_CACHE. When
# set (any value) the cache is skipped entirely; otherwise the cache claims all
# free VRAM minus 1 GB. The legacy ENABLE_VRAM_CACHE=<mib> interface is not
# honored by the binary — we keep the $VRAM_MIB positional argument as a
# boolean toggle (0 = off, non-zero = on) for compatibility with the apu_infer
# script signature.
unset ENABLE_VRAM_CACHE
if [[ "$VRAM_MIB" == "0" ]]; then
    export DISABLE_VRAM_CACHE=1
else
    unset DISABLE_VRAM_CACHE
fi

LOG=$(mktemp /tmp/bench.XXXXXX.log)
./infer --prompt 'The capital of France is' --tokens "$TOKENS" \
        --experts ../model-safetensors/packed_experts > "$LOG" 2>&1 || {
    echo "FAIL: infer exited non-zero"
    cat "$LOG"
    rm -f "$LOG"
    exit 2
}

# Extract the generated text (between "[generating]" and "[done]" markers).
GEN=$(awk '/^\[generating\]/{flag=1; next} /^\[done\]/{flag=0} flag' "$LOG")

# Correctness gate: first 3 lines of generated text must match expected.txt.
if ! printf '%s\n' "$GEN" | head -3 | diff -q - bench/expected.txt > /dev/null 2>&1; then
    echo "FAIL: output mismatch"
    echo "--- expected ---"
    cat bench/expected.txt
    echo "--- got (first 5 lines) ---"
    printf '%s\n' "$GEN" | head -5
    rm -f "$LOG"
    exit 3
fi

# Extract tok/s and per-token ms from the [done] line.
TOKPS=$(grep '^\[done\]' "$LOG" | grep -oP '[0-9.]+(?= tok/s)')
PER_TOK_MS=$(grep '^\[done\]' "$LOG" | grep -oP '[0-9.]+(?= ms/token)')

printf '%s mode=%s tokens=%s vram=%sMiB tok/s=%s per_tok_ms=%s\n' \
    "$(date +%H:%M:%S)" "$MODE" "$TOKENS" "$VRAM_MIB" "$TOKPS" "$PER_TOK_MS"
rm -f "$LOG"
