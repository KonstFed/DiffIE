#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

OUT=${1:-configs/lsoie_ex_2500}
SEEDS="38 39 40 41 42 43 44 45 46 47"

uv run python - "$OUT" $SEEDS << 'EOF'
import json, re, subprocess, sys, tempfile, os
from pathlib import Path
import numpy as np

out = Path(sys.argv[1])
seeds = [int(s) for s in sys.argv[2:]]
carb_dir = Path("benchmarks/CaRB")
carb_gold = "data/gold/test.tsv"
keys = ["auc", "precision", "recall", "f1"]
col_w = 10

def run_carb(tsv_abs, binary=False):
    with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as f:
        tmp = f.name
    cmd = [sys.executable, "carb.py", "--gold", carb_gold,
           "--tabbed", tsv_abs, "--out", tmp]
    if binary:
        cmd.append("--binary")
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(carb_dir))
    os.unlink(tmp)
    last = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    m = re.search(r"AUC:\s*([\d.]+).*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\]", last)
    return {"auc": float(m[1]), "precision": float(m[2]),
            "recall": float(m[3]), "f1": float(m[4])} if m else None

def load_json(dir_, seed):
    p = out / dir_ / f"metrics_{seed}.json"
    return json.loads(p.read_text()) if p.exists() else None

def print_table(name, rows, keys):
    print(f"\n{'='*60}")
    print(name)
    print(f"{'='*60}")
    print("seed".ljust(12) + "".join(k.ljust(col_w) for k in keys))
    print("-" * (12 + col_w * len(keys)))
    vals = {k: [] for k in keys}
    for seed, m in rows:
        if m is None:
            print(f"{str(seed):<12}missing")
            continue
        line = str(seed).ljust(12)
        for k in keys:
            v = m.get(k, float("nan"))
            line += f"{v:.4f}".ljust(col_w)
            vals[k].append(v)
        print(line)
    print("-" * (12 + col_w * len(keys)))
    for label, fn in [("mean", np.mean), ("std", np.std)]:
        line = label.ljust(12)
        for k in keys:
            arr = vals[k]
            line += (f"{fn(arr):.4f}" if arr else "n/a").ljust(col_w)
        print(line)

benchie_rows = [(s, load_json("benchie", s)) for s in seeds]
wire57_rows  = [(s, load_json("wire57",  s)) for s in seeds]

std_rows, bin_rows = [], []
for s in seeds:
    tsv = str((out / "carb_test" / f"extractions_{s}.tsv").resolve())
    if not Path(tsv).exists():
        std_rows.append((s, None))
        bin_rows.append((s, None))
        continue
    std_rows.append((s, run_carb(tsv, binary=False)))
    bin_rows.append((s, run_carb(tsv, binary=True)))

print_table("BenchIE",            benchie_rows, ["precision", "recall", "f1"])
print_table("WiRe57",             wire57_rows,  ["precision", "recall", "f1"])
print_table("CaRB",               std_rows,     keys)
print_table("CaRB (binary / 1-1)", bin_rows,    keys)
EOF
