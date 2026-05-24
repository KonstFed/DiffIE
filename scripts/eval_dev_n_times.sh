#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

DEV_SENTS="benchmarks/CaRB/data/dev.txt"
GOLD="benchmarks/CaRB/data/gold/dev.tsv"
SEEDS="38 39 40 41 42 43 44 45 46 47"

for dir in configs/*/; do
    name="$(basename "$dir")"
    weights="${dir}weights.pt"
    config="$(find "$dir" -maxdepth 1 -name '*.yaml' | head -n1)"

    if [[ -z "$config" ]]; then
        echo "[$name] skip: no yaml found"
        continue
    fi
    if [[ ! -f "$weights" ]]; then
        echo "[$name] skip: weights.pt not found"
        continue
    fi

    out="${dir}carb_dev"
    echo ""
    echo "=== [$name] $(basename "$config") -> $out"
    uv run python -m diffopenie.evaluation.carb_eval \
        --config "$config" \
        --checkpoint-path "$weights" \
        --input-sentences "$DEV_SENTS" \
        --gold "$GOLD" \
        --output-dir "$out" \
        --seeds $SEEDS
done
