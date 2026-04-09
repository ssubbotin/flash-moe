#!/usr/bin/env bash
# Usage: ./profile.sh
# Captures per-phase timing breakdown for rocm_infer on MI300X (gfx942).
#
# Ubuntu 24.04 ROCm 7.2 ships rocprof, but this script uses the in-engine
# --timing flag (per-layer phase timers) instead of rocprofiler/rocprofv2
# trace capture for simpler, reproducible phase attribution. Same goal:
# figure out where time goes per token.
set -euo pipefail
cd "$(dirname "$0")/.."

export ENABLE_VRAM_CACHE=20480

# 30 tokens is long enough that the steady-state [timing] lines dominate
# the output; the first 2-3 are cold-cache outliers and we drop them.
./infer --prompt 'The capital of France is' --tokens 30 --timing \
        --experts ../model-safetensors/packed_experts > /tmp/profile_run.log 2>&1

echo "--- per-layer phase breakdown (last 10 tokens, averaged) ---"
grep '\[timing\] Per-layer avg' /tmp/profile_run.log | tail -10 | \
    awk -F'[ =]+' '{
        for (i=1; i<=NF; i++) {
            if ($i ~ /^(norm|attn|oproj|route|shared|io|expert|combine)$/) {
                phase = $i; val = $(i+1); total[phase]+=val; count[phase]++
            }
        }
    }
    END {
        printf "phase        avg_ms_per_layer   x60_per_token\n"
        for (p in total) printf "%-12s %16.3f %14.2f\n", p, total[p]/count[p], (total[p]/count[p])*60
    }'

echo ""
echo "--- last 10 done lines ---"
grep '^\[done\]' /tmp/profile_run.log | tail -10 || \
    grep 'ms/token' /tmp/profile_run.log | tail -10

rm -f /tmp/profile_run.log
