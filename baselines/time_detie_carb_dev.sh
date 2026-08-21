#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINES_DIR="$ROOT_DIR/baselines"
DETIE_DIR="${DETIE_DIR:-$BASELINES_DIR/DetIE}"
DEV_FILE="${DEV_FILE:-$ROOT_DIR/benchmarks/CaRB/data/dev.txt}"
VERSION="${DETIE_VERSION:-243}"
REPEATS="${REPEATS:-5}"
WARMUP="${WARMUP:-1}"
OUT_CSV="${OUT_CSV:-$BASELINES_DIR/detie_carb_dev_timing.csv}"
LOG_DIR="${LOG_DIR:-$BASELINES_DIR/detie_carb_dev_timing_logs}"

if [[ ! -d "$DETIE_DIR" ]]; then
  echo "ERROR: DetIE checkout not found at $DETIE_DIR" >&2
  echo "Run: bash baselines/setup_detie_local.sh" >&2
  exit 1
fi

if [[ ! -f "$DEV_FILE" ]]; then
  echo "ERROR: CaRB dev file not found at $DEV_FILE" >&2
  exit 1
fi

CKPT="$DETIE_DIR/results/logs/default/version_${VERSION}/checkpoints/best.ckpt"
HPARAMS="$DETIE_DIR/results/logs/default/version_${VERSION}/hparams.yaml"
if [[ ! -f "$CKPT" || ! -f "$HPARAMS" ]]; then
  echo "ERROR: DetIE checkpoint bundle is missing." >&2
  echo "Expected:" >&2
  echo "  $CKPT" >&2
  echo "  $HPARAMS" >&2
  exit 1
fi

TARGET_DIR="$DETIE_DIR/modules/model/evaluation/oie-benchmark-stanovsky/raw_sentences"
TARGET_FILE="$TARGET_DIR/all.txt"
BACKUP_FILE="$TARGET_FILE.detie_original.bak"
mkdir -p "$TARGET_DIR" "$LOG_DIR"

if [[ -f "$TARGET_FILE" && ! -f "$BACKUP_FILE" ]]; then
  cp "$TARGET_FILE" "$BACKUP_FILE"
fi
cp "$DEV_FILE" "$TARGET_FILE"

NUM_SENTENCES="$(wc -l < "$DEV_FILE" | tr -d ' ')"

run_once() {
  local label="$1"
  local log_path="$LOG_DIR/${label}.log"
  (
    cd "$DETIE_DIR"
    PYTHONPATH=. python3 modules/model/test.py "model.best_version=$VERSION"
  ) 2>&1 | tee "$log_path" >&2
  python3 - "$log_path" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(errors="replace")
matches = re.findall(r"(?<![A-Za-z])(?:\d+\.\d+|\d+)(?![A-Za-z])", text)
if matches:
    print(matches[-1])
PY
}

for i in $(seq 1 "$WARMUP"); do
  echo "Warmup $i/$WARMUP"
  run_once "warmup_${i}" >/dev/null
done

echo "repeat,num_sentences,total_seconds,sentences_per_second,detie_version" > "$OUT_CSV"
for i in $(seq 1 "$REPEATS"); do
  echo "Measured repeat $i/$REPEATS"
  seconds="$(run_once "repeat_${i}")"
  if [[ -z "$seconds" ]]; then
    echo "ERROR: Could not parse timing seconds from repeat $i log." >&2
    exit 1
  fi
  sent_per_sec="$(awk -v n="$NUM_SENTENCES" -v s="$seconds" 'BEGIN { printf "%.6f", n / s }')"
  printf "%s,%s,%s,%s,%s\n" "$i" "$NUM_SENTENCES" "$seconds" "$sent_per_sec" "$VERSION" >> "$OUT_CSV"
done

echo "Wrote timing CSV to $OUT_CSV"
echo "Logs are in $LOG_DIR"
