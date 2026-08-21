#!/usr/bin/env bash
# Download all DiffIE artifacts (weights + predictions + n-study caches) from
# Hugging Face into configs/. After this every reproduction script works
# without retraining.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

REPO="${DIFFIE_HF_REPO:-KonstFed/diffIE}"

echo "[hf] $REPO -> configs/"
uv run --with "huggingface_hub[cli]" hf download "$REPO" --local-dir configs
