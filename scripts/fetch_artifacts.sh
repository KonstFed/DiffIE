#!/usr/bin/env bash
# Download all DiffIE artifacts (weights + predictions + n-study caches) from
# Hugging Face into configs/. After this every reproduction script works
# without retraining.
#
# The repo is private during review — log in once with a read token:
#     uvx --from "huggingface_hub[cli]" hf auth login
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

REPO="TBD"

echo "[hf] $REPO -> configs/"
uv run --with "huggingface_hub[cli]" hf download "$REPO" --local-dir configs
