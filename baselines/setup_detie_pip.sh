#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINES_DIR="$ROOT_DIR/baselines"
DETIE_DIR="${DETIE_DIR:-$BASELINES_DIR/DetIE}"
VENV_DIR="${DETIE_VENV_DIR:-$BASELINES_DIR/.venvs/detie}"
DETIE_REPO="${DETIE_REPO:-https://github.com/sberbank-ai/DetIE}"
CUDA_TAG="${DETIE_TORCH_CUDA_TAG:-cu111}"

mkdir -p "$BASELINES_DIR"

if [[ ! -d "$DETIE_DIR/.git" ]]; then
  if [[ -e "$DETIE_DIR" ]]; then
    echo "ERROR: $DETIE_DIR exists but is not a git checkout." >&2
    exit 1
  fi
  git clone "$DETIE_REPO" "$DETIE_DIR"
else
  echo "DetIE checkout already exists at $DETIE_DIR"
fi

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade "pip<24" "setuptools<60" wheel

python -m pip install \
  "torch==1.7.1+${CUDA_TAG}" \
  "torchvision==0.8.2+${CUDA_TAG}" \
  -f "https://download.pytorch.org/whl/torch_stable.html"

REQ_FILTERED="$(mktemp)"
grep -Ev '^(torch|lapsolver)==' "$DETIE_DIR/context/requirements.txt" > "$REQ_FILTERED"
python -m pip install -r "$REQ_FILTERED"
rm -f "$REQ_FILTERED"

cat <<'EOF'

NOTE: setup_detie_pip.sh skipped lapsolver because pip often fails to build it
on cluster nodes. For inference this may be okay. If DetIE later errors with
"No module named lapsolver", use the micromamba setup instead:

  bash baselines/setup_detie_local.sh
EOF

python -m spacy download en_core_web_sm
python - <<'PY'
import nltk
nltk.download("stopwords")
PY

cat <<EOF

DetIE pip environment is ready.

Activate it with:
  source "$VENV_DIR/bin/activate"

If torch install fails, try another CUDA wheel tag:
  DETIE_TORCH_CUDA_TAG=cu101 bash baselines/setup_detie_pip.sh
  DETIE_TORCH_CUDA_TAG=cu102 bash baselines/setup_detie_pip.sh

Then run:
  bash "$BASELINES_DIR/time_detie_carb_dev.sh"
EOF
