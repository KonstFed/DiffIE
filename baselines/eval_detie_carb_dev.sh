#!/usr/bin/env bash
# Produce DetIE predictions on the CaRB dev sentences and score them with the
# CaRB scorer (standard + 1-1 / binary) against benchmarks/CaRB/data/gold/dev.tsv.
#
# This fills the DetIE F1 cell of the rebuttal quality/cost/VRAM table. It reuses
# the exact DetIE invocation from time_detie_carb_dev.sh (which is known to run),
# then auto-discovers the extraction file DetIE writes, converts it to CaRB
# tabbed format, and scores it.
#
# Prereqs (identical to timing):
#   - DetIE checkout at baselines/DetIE  (bash baselines/setup_detie_local.sh)
#   - version_243 checkpoint bundle present
#   - the DetIE micromamba env ACTIVATED (see baselines/README.md)
#   - the CaRB scorer runnable from benchmarks/CaRB (its own deps)
#
# Usage (from repo root, DetIE env active):
#   bash baselines/eval_detie_carb_dev.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINES_DIR="$ROOT_DIR/baselines"
DETIE_DIR="${DETIE_DIR:-$BASELINES_DIR/DetIE}"
DEV_FILE="${DEV_FILE:-$ROOT_DIR/benchmarks/CaRB/data/dev.txt}"
GOLD="${GOLD:-$ROOT_DIR/benchmarks/CaRB/data/gold/dev.tsv}"
VERSION="${DETIE_VERSION:-243}"
OUT_DIR="${OUT_DIR:-$BASELINES_DIR/detie_carb_dev_pred}"
CARB_DIR="$ROOT_DIR/benchmarks/CaRB"
EVAL_ROOT="$DETIE_DIR/modules/model/evaluation"

# --- sanity checks -----------------------------------------------------------
[[ -d "$DETIE_DIR" ]]  || { echo "ERROR: DetIE checkout not found at $DETIE_DIR (run baselines/setup_detie_local.sh)" >&2; exit 1; }
[[ -f "$DEV_FILE" ]]   || { echo "ERROR: dev sentences not found at $DEV_FILE" >&2; exit 1; }
[[ -f "$GOLD" ]]       || { echo "ERROR: CaRB dev gold not found at $GOLD" >&2; exit 1; }
CKPT="$DETIE_DIR/results/logs/default/version_${VERSION}/checkpoints/best.ckpt"
[[ -f "$CKPT" ]]       || { echo "ERROR: DetIE checkpoint missing: $CKPT" >&2; exit 1; }

mkdir -p "$OUT_DIR"

# --- swap dev sentences into DetIE's benchmark input (restore on exit) --------
TARGET_DIR="$EVAL_ROOT/oie-benchmark-stanovsky/raw_sentences"
TARGET_FILE="$TARGET_DIR/all.txt"
BACKUP_FILE="$TARGET_FILE.detie_eval_dev.bak"
mkdir -p "$TARGET_DIR"
if [[ -f "$TARGET_FILE" && ! -f "$BACKUP_FILE" ]]; then
  cp "$TARGET_FILE" "$BACKUP_FILE"
fi
restore() { [[ -f "$BACKUP_FILE" ]] && mv -f "$BACKUP_FILE" "$TARGET_FILE" || true; }
trap restore EXIT
cp "$DEV_FILE" "$TARGET_FILE"

NUM_SENTENCES="$(grep -c . "$DEV_FILE")"
echo "DetIE eval on $NUM_SENTENCES CaRB-dev sentences (version_$VERSION)"

# --- marker so we can find files DetIE writes during this run ----------------
MARKER="$(mktemp)"
sleep 1  # ensure any file written after this has a strictly newer mtime

# --- run DetIE prediction (same command as the timing script) ----------------
echo "== running DetIE modules/model/test.py =="
(
  cd "$DETIE_DIR"
  PYTHONPATH=. python3 modules/model/test.py "model.best_version=$VERSION"
) 2>&1 | tee "$OUT_DIR/detie_run.log" >&2

# --- discover the extraction file DetIE just wrote ---------------------------
echo "== discovering DetIE prediction output =="
mapfile -t CANDIDATES < <(
  find "$EVAL_ROOT" -type f -newer "$MARKER" \
       \( -name '*.txt' -o -name '*.tsv' -o -name '*.dat' -o -name '*out*' \) \
    2>/dev/null | sort
)
rm -f "$MARKER"

if [[ ${#CANDIDATES[@]} -eq 0 ]]; then
  cat >&2 <<EOF
ERROR: no files were written under $EVAL_ROOT during the run.
DetIE's test.py may only print metrics without dumping predictions, or it writes
elsewhere. Options:
  1. Search the tree yourself:  find "$DETIE_DIR" -newer "$OUT_DIR/detie_run.log" -type f
  2. Point this script at the file:  DETIE_PRED_FILE=/path/to/preds bash $0
EOF
  exit 1
fi

echo "Files written during run:"
printf '  %s\n' "${CANDIDATES[@]}"

# Allow an explicit override; otherwise pick the largest candidate (the dump of
# per-sentence extractions is almost always the biggest new file).
if [[ -n "${DETIE_PRED_FILE:-}" ]]; then
  PRED_RAW="$DETIE_PRED_FILE"
else
  PRED_RAW="$(ls -S "${CANDIDATES[@]}" 2>/dev/null | head -n1)"
fi
echo "Using DetIE prediction file: $PRED_RAW"
echo "---- first 5 lines (verify the format before trusting the score) ----"
head -n 5 "$PRED_RAW" >&2 || true
echo "---------------------------------------------------------------------"

# --- convert DetIE output -> CaRB tabbed --------------------------------------
# CaRB tabbed = sentence \t confidence \t relation \t arg0 \t arg1 [\t arg2 ...]
# DetIE dumps extractions in a tab-separated format; the converter below is
# tolerant of the two shapes we've seen and auto-detects which columns hold
# the sentence / confidence / relation / args. If your dump differs, the head
# printout above tells you exactly what to adjust in _convert().
PRED_TABBED="$OUT_DIR/extractions.tsv"
python3 - "$PRED_RAW" "$PRED_TABBED" <<'PY'
import re, sys
src, dst = sys.argv[1], sys.argv[2]

def looks_num(s):
    try:
        float(s); return True
    except ValueError:
        return False

out = []
with open(src, encoding="utf-8", errors="replace") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line.strip():
            continue
        cols = [c.strip() for c in line.split("\t")]
        if len(cols) < 4:
            # Try the "( arg0 ; rel ; arg1 )" style occasionally emitted.
            m = re.match(r"^(.*?)\t.*\((.*?);(.*?);(.*?)\)", line)
            if not m:
                continue
            sent, a0, rel, a1 = (m.group(1).strip(), m.group(2).strip(),
                                 m.group(3).strip(), m.group(4).strip())
            out.append([sent, "1.0", rel, a0, a1])
            continue
        sent = cols[0]
        # Locate the confidence column (first numeric field after the sentence).
        conf_i = next((i for i in range(1, len(cols)) if looks_num(cols[i])), None)
        if conf_i is None:
            # No explicit confidence: assume [sent, rel, arg0, arg1, ...].
            rel, args = cols[1], cols[2:]
            conf = "1.0"
        else:
            conf = cols[conf_i]
            rest = cols[conf_i + 1:]
            if len(rest) < 3:
                continue
            rel, args = rest[0], rest[1:]
        row = [sent, conf, rel] + args
        out.append(row)

with open(dst, "w", encoding="utf-8") as f:
    for row in out:
        f.write("\t".join(row) + "\n")
print(f"Wrote {len(out)} extractions to {dst}", file=sys.stderr)
PY

[[ -s "$PRED_TABBED" ]] || { echo "ERROR: conversion produced 0 rows — inspect $PRED_RAW format (see head above)." >&2; exit 1; }

# --- score with the CaRB scorer (standard + 1-1) -----------------------------
echo "== scoring with CaRB =="
(
  cd "$CARB_DIR"
  echo "-- CaRB (standard) --"
  python carb.py --gold "$GOLD" --tabbed "$PRED_TABBED" --out /dev/null
  echo "-- CaRB (1-1 / binary) --"
  python carb.py --gold "$GOLD" --tabbed "$PRED_TABBED" --out /dev/null --binary
)

echo ""
echo "Done. Predictions: $PRED_TABBED"
echo "Read F1 from the 'Optimal (precision, recall, F1)' lines above."
