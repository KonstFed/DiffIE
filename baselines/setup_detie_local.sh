#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINES_DIR="$ROOT_DIR/baselines"
DETIE_DIR="${DETIE_DIR:-$BASELINES_DIR/DetIE}"
ENV_PREFIX="${DETIE_ENV_PREFIX:-$BASELINES_DIR/.envs/detie}"
MAMBA_ROOT="${MAMBA_ROOT_PREFIX:-$BASELINES_DIR/.micromamba}"
DETIE_REPO="${DETIE_REPO:-https://github.com/sberbank-ai/DetIE}"
export MAMBA_ROOT_PREFIX="$MAMBA_ROOT"

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

if command -v micromamba >/dev/null 2>&1; then
  MAMBA_BIN="$(command -v micromamba)"
else
  MAMBA_BIN="$MAMBA_ROOT/bin/micromamba"
  if [[ ! -x "$MAMBA_BIN" ]]; then
    echo "Installing micromamba locally under $MAMBA_ROOT"
    mkdir -p "$MAMBA_ROOT"
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
      | tar -xvj -C "$MAMBA_ROOT" bin/micromamba
  fi
fi

eval "$("$MAMBA_BIN" shell hook -s bash)"

if [[ ! -d "$ENV_PREFIX" ]]; then
  micromamba create -y -p "$ENV_PREFIX" -c pytorch -c defaults -c conda-forge \
    python=3.8 pip cmake cython numpy=1.19.4 pandas=1.1.5 \
    pytorch=1.7.1 cudatoolkit=11.0 "mkl<2024.1" "intel-openmp<2024.1"
else
  echo "Environment already exists at $ENV_PREFIX"
fi

micromamba activate "$ENV_PREFIX"
micromamba install -y -p "$ENV_PREFIX" -c pytorch -c defaults -c conda-forge \
  "mkl<2024.1" "intel-openmp<2024.1"
python -m pip install --upgrade "pip<24" "setuptools<60" wheel

REQ_FILTERED="$(mktemp)"
grep -Ev '^(torch|lapsolver)==' "$DETIE_DIR/context/requirements.txt" > "$REQ_FILTERED"
python -m pip install -r "$REQ_FILTERED"
rm -f "$REQ_FILTERED"

cat <<'EOF'

NOTE: skipped DetIE's lapsolver pin. It fails to build on many modern cluster
nodes and is not expected to be needed for the inference timing path. If the
timing script later errors with "No module named lapsolver", we will patch that
specific import path instead of compiling lapsolver.
EOF

python -m spacy download en_core_web_sm
python - <<'PY'
import nltk
nltk.download("stopwords")
PY

cat <<EOF

DetIE local environment is ready.

Activate it with:
  eval "\$($MAMBA_BIN shell hook -s bash)"
  micromamba activate "$ENV_PREFIX"

Next, download the DetIE bundle from:
  https://drive.google.com/drive/folders/1SGeQWcFwmL4BaMbCTxVw5-oU69vPW_d-?usp=sharing

Copy the LSOIE checkpoint folder so these files exist:
  $DETIE_DIR/results/logs/default/version_243/checkpoints/best.ckpt
  $DETIE_DIR/results/logs/default/version_243/hparams.yaml

Then run:
  bash "$BASELINES_DIR/time_detie_carb_dev.sh"
EOF
