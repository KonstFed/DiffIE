#!/usr/bin/env bash
# Fetch datasets + benchmarks needed to train/evaluate DiffIE.
# Run from the repo root: ./load_data.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$REPO_ROOT/dataset/cycleoie"
mkdir -p "$REPO_ROOT/benchmarks"

# ── Benchmarks (public GitHub) ──────────────────────────────────────────────

if [ ! -d "$REPO_ROOT/benchmarks/CaRB" ]; then
    echo "[CaRB] cloning into benchmarks/CaRB"
    git clone https://github.com/dair-iitd/CaRB.git "$REPO_ROOT/benchmarks/CaRB"
    # CaRB's oie_readers/extraction.py imports from a sklearn submodule
    # removed in sklearn 0.22+. Patch to use the public location.
    sed -i.bak \
        's|from sklearn.preprocessing.data import binarize|from sklearn.preprocessing import binarize|' \
        "$REPO_ROOT/benchmarks/CaRB/oie_readers/extraction.py"
    rm -f "$REPO_ROOT/benchmarks/CaRB/oie_readers/extraction.py.bak"
else
    echo "[CaRB] already present — skipping"
fi

if [ ! -d "$REPO_ROOT/benchmarks/WiRe57" ]; then
    echo "[WiRe57] cloning"
    git clone https://github.com/rali-udem/WiRe57.git "$REPO_ROOT/benchmarks/WiRe57"
else
    echo "[WiRe57] already present — skipping"
fi

if [ ! -d "$REPO_ROOT/benchmarks/benchie" ]; then
    echo "[BenchIE] cloning"
    git clone https://github.com/gkiril/benchie.git "$REPO_ROOT/benchmarks/benchie"
else
    echo "[BenchIE] already present — skipping"
fi

# ── LSOIE-G (CycleOIE repo) ─────────────────────────────────────────────────
# Configs reference:
#   dataset/cycleoie/lsoie-g-examples.tsv
#   dataset/cycleoie/lsoie-g-principles.csv
# Sourced from https://github.com/Jinsns/CycleOIE (renamed to match configs).

CYCLEOIE_RAW="https://raw.githubusercontent.com/Jinsns/CycleOIE/main/data/lsoie"

if [ ! -f "$REPO_ROOT/dataset/cycleoie/lsoie-g-examples.tsv" ]; then
    echo "[LSOIE-G examples] downloading"
    curl -L -o "$REPO_ROOT/dataset/cycleoie/lsoie-g-examples.tsv" \
        "$CYCLEOIE_RAW/examples/lsoie-g-1.0.tsv"
else
    echo "[LSOIE-G examples] already present — skipping"
fi

if [ ! -f "$REPO_ROOT/dataset/cycleoie/lsoie-g-principles.csv" ]; then
    echo "[LSOIE-G principles] downloading"
    curl -L -o "$REPO_ROOT/dataset/cycleoie/lsoie-g-principles.csv" \
        "$CYCLEOIE_RAW/principles/lsoie-g-1.0.csv"
else
    echo "[LSOIE-G principles] already present — skipping"
fi

echo ""
echo "Done. Verify with:"
echo "  ls $REPO_ROOT/benchmarks/CaRB/data/gold/dev.tsv"
echo "  ls $REPO_ROOT/dataset/cycleoie/lsoie-g-examples.tsv"
