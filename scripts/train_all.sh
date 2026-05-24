#!/usr/bin/env bash
# Train every paper experiment in sequence (data ablation + MDLM ablation +
# primary model). Wraps diffopenie.evaluation.run_final_experiments which:
#   - calls diffopenie.training.train_example as a clean subprocess per experiment,
#   - moves checkpoint_best.pt to weights.pt inside each configs/<exp>/.
#
# Pass-through args:
#   --only 50,2500    only run the named experiments
#   --dry-run         print commands without executing
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

uv run python -m diffopenie.evaluation.run_final_experiments "$@"
