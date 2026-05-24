#!/usr/bin/env bash
# Score every configs/<exp>/carb_dev/extractions.tsv with the real CaRB
# scorer (1-to-1 and binary matchers) against CaRB dev gold.
# Run from repo root (or anywhere — script cd's to repo root itself).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

for dir in configs/*/; do
    name="$(basename "$dir")"
    tabbed="${dir}carb_dev/extractions.tsv"

    if [[ ! -f "$tabbed" ]]; then
        echo "[$name] skip: $tabbed not found"
        continue
    fi

    echo ""
    echo "=== $name ==="
    (
        cd benchmarks/CaRB
        echo "-- CaRB --"
        uv run python carb.py \
            --gold data/gold/dev.tsv \
            --tabbed "../../$tabbed" \
            --out "/tmp/carb_dev_${name}_1to1.dat"
        echo "-- CaRB (1-1) --"
        uv run python carb.py \
            --gold data/gold/dev.tsv \
            --tabbed "../../$tabbed" \
            --out "/tmp/carb_dev_${name}_binary.dat" \
            --binary
    )
done
