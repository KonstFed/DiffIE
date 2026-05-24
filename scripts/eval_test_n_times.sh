#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${1:-}" ]]; then
    echo "Usage: $0 <config-dir>"
    echo "  e.g. $0 configs/lsoie_ex_2500"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONHASHSEED=38
export CUBLAS_WORKSPACE_CONFIG=:16:8

OUT="$1"
CONFIG="$(find "$OUT" -maxdepth 1 -name '*.yaml' | head -n1)"
CKPT="$OUT/weights.pt"

SEEDS="38 39 40 41 42 43 44 45 46 47"

mkdir -p \
  "$OUT/benchie" \
  "$OUT/wire57" \
  "$OUT/carb_test"

uv run python -m diffopenie.evaluation.benchie_eval \
  --config "$CONFIG" \
  --checkpoint-path "$CKPT" \
  --output-dir "$OUT/benchie" \
  --seeds $SEEDS

uv run python -m diffopenie.evaluation.wire57_eval \
  --config "$CONFIG" \
  --checkpoint-path "$CKPT" \
  --output-dir "$OUT/wire57" \
  --seeds $SEEDS

uv run python -m diffopenie.evaluation.carb_eval \
  --config "$CONFIG" \
  --checkpoint-path "$CKPT" \
  --input-sentences benchmarks/CaRB/data/test.txt \
  --gold benchmarks/CaRB/data/gold/test.tsv \
  --output-dir "$OUT/carb_test" \
  --seeds $SEEDS
