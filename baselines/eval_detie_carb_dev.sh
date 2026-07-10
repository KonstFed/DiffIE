#!/usr/bin/env bash
# Produce DetIE (v243, LSOIE) predictions on the CaRB dev sentences and score
# them with OUR CaRB scorer (standard + 1-1/binary) against
# benchmarks/CaRB/data/gold/dev.tsv. Fills the DetIE F1 cell of the rebuttal
# quality/cost/VRAM table.
#
# How it works (no DetIE code edits): DetIE's detie_predict.py has hardcoded
# input/output paths (data/carb_sentences.txt -> systems_output/detie243_output.txt,
# VERSION=243) and writes "ollie" format. We temporarily swap the dev sentences
# into that input file, run the predictor, copy out the ollie output, then score
# it with our carb.py (which has an --ollie reader). Original files are restored.
#
# Prereqs (same as timing):
#   - DetIE checkout with the version_243 checkpoint bundle present
#   - the DetIE micromamba env ACTIVATED (see baselines/README.md)
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
CARB6="$DETIE_DIR/modules/model/evaluation/carb-openie6"

# --- sanity checks -----------------------------------------------------------
[[ -d "$DETIE_DIR" ]] || { echo "ERROR: DetIE checkout not found at $DETIE_DIR" >&2; exit 1; }
[[ -d "$CARB6" ]]     || { echo "ERROR: carb-openie6 dir not found at $CARB6" >&2; exit 1; }
[[ -f "$DEV_FILE" ]]  || { echo "ERROR: dev sentences not found at $DEV_FILE" >&2; exit 1; }
[[ -f "$GOLD" ]]      || { echo "ERROR: CaRB dev gold not found at $GOLD" >&2; exit 1; }
[[ -f "$CARB6/detie_predict.py" ]] || { echo "ERROR: detie_predict.py missing in $CARB6" >&2; exit 1; }
CKPT="$DETIE_DIR/results/logs/default/version_${VERSION}/checkpoints/best.ckpt"
[[ -f "$CKPT" ]] || { echo "ERROR: DetIE checkpoint missing: $CKPT" >&2; exit 1; }

mkdir -p "$OUT_DIR"

INPUT_FILE="$CARB6/data/carb_sentences.txt"      # detie_predict.py reads this
OUTPUT_FILE="$CARB6/systems_output/detie${VERSION}_output.txt"  # ...and writes this
IN_BAK="$INPUT_FILE.dev_eval.bak"
OUT_BAK="$OUTPUT_FILE.dev_eval.bak"

# Back up the originals (CaRB *test* sentences + the released test output) and
# restore them on exit so the checkout is left untouched.
[[ -f "$INPUT_FILE"  && ! -f "$IN_BAK"  ]] && cp "$INPUT_FILE"  "$IN_BAK"  || true
[[ -f "$OUTPUT_FILE" && ! -f "$OUT_BAK" ]] && cp "$OUTPUT_FILE" "$OUT_BAK" || true
restore() {
  [[ -f "$IN_BAK"  ]] && mv -f "$IN_BAK"  "$INPUT_FILE"  || true
  [[ -f "$OUT_BAK" ]] && mv -f "$OUT_BAK" "$OUTPUT_FILE" || true
}
trap restore EXIT

# Swap in dev sentences.
cp "$DEV_FILE" "$INPUT_FILE"
NUM_SENTENCES="$(grep -c . "$DEV_FILE")"
echo "DetIE (v$VERSION) prediction on $NUM_SENTENCES CaRB-dev sentences"

# --- run the DetIE predictor -------------------------------------------------
# Must run from carb-openie6 (detie_predict.py prepends ../../../../ to the ckpt
# path and uses os.getcwd() for data/ and systems_output/), with the DetIE root
# on PYTHONPATH so `config.*` and `modules.*` import.
echo "== running detie_predict.py =="
(
  cd "$CARB6"
  PYTHONPATH="$DETIE_DIR" python3 detie_predict.py
) 2>&1 | tee "$OUT_DIR/detie_predict.log" >&2

[[ -s "$OUTPUT_FILE" ]] || { echo "ERROR: predictor did not write $OUTPUT_FILE" >&2; exit 1; }

# Keep a copy of the dev predictions in the DiffIE repo.
DEV_PRED="$OUT_DIR/detie${VERSION}_dev_output.txt"
cp "$OUTPUT_FILE" "$DEV_PRED"
echo "Saved dev predictions to $DEV_PRED"
echo "---- head ----"; head -3 "$DEV_PRED" >&2; echo "--------------"

# --- score with OUR CaRB scorer (ollie reader), both matchers ----------------
echo "== scoring with benchmarks/CaRB/carb.py --ollie =="
(
  cd "$CARB_DIR"
  echo "-- CaRB (standard) --"
  python carb.py --gold "$GOLD" --ollie "$DEV_PRED" --out /dev/null
  echo "-- CaRB (1-1 / binary) --"
  python carb.py --gold "$GOLD" --ollie "$DEV_PRED" --out /dev/null --binary
)

echo ""
echo "Done. DetIE dev predictions: $DEV_PRED"
echo "Take F1 from the 'Optimal (precision, recall, F1)' lines above for the table."
