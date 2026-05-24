#!/usr/bin/env bash
# Test-time compute scaling study (paper Figure 1).
#
# Step 1: sample N raw triplets per CaRB dev sentence with the trained model
#         (cached to <config-dir>/n_study/cache_dev_<N>.jsonl).
# Step 2: sweep n from 1..N and plot the lenient vs frequency F1 curve.
#
# Usage: ./scripts/plot_n_study.sh <config-dir> [N]
#   e.g. ./scripts/plot_n_study.sh configs/lsoie_ex_2500 1024
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${1:-}" ]]; then
    echo "Usage: $0 <config-dir> [N]"
    echo "  e.g. $0 configs/lsoie_ex_2500 1024"
    exit 1
fi

OUT="$1"
N="${2:-1024}"
CONFIG="$(find "$OUT" -maxdepth 1 -name '*.yaml' | head -n1)"
CKPT="$OUT/weights.pt"
CACHE="$OUT/n_study/cache_dev_${N}.jsonl"
PLOT_PREFIX="$OUT/n_study/curve"

mkdir -p "$OUT/n_study"

if [[ ! -f "$CACHE" ]]; then
    uv run python -m diffopenie.evaluation.sample_cache \
        --config "$CONFIG" \
        --checkpoint-path "$CKPT" \
        --input-sentences benchmarks/CaRB/data/dev.txt \
        --n "$N" \
        --out "$CACHE"
else
    echo "[plot_n_study] reusing existing cache: $CACHE"
fi

uv run python -m diffopenie.plotting.lsoie_cache_curve_plot \
    --config "$CONFIG" \
    --cache "$CACHE" \
    --out-prefix "$PLOT_PREFIX"
