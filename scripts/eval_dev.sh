#!/usr/bin/env bash
# Run CaRB dev predictions for every configs/<exp>/ that has a
# weights.pt + yaml. Predictions land in <exp>/carb_dev/.
# Run from repo root (or anywhere — script cd's to repo root itself).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

DEV_SENTS="benchmarks/CaRB/data/dev.txt"

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
        --output-dir "$out"
done
