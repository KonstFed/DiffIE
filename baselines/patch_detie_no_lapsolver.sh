#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DETIE_DIR="${DETIE_DIR:-$ROOT_DIR/baselines/DetIE}"
MODELS_PY="$DETIE_DIR/modules/model/models.py"

if [[ ! -f "$MODELS_PY" ]]; then
  echo "ERROR: DetIE models.py not found at $MODELS_PY" >&2
  exit 1
fi

python3 - "$MODELS_PY" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
old = "from lapsolver import solve_dense\n"
new = """try:\n    from lapsolver import solve_dense\nexcept ModuleNotFoundError:\n    from scipy.optimize import linear_sum_assignment\n\n    def solve_dense(cost):\n        return linear_sum_assignment(cost)\n"""

if old not in text:
    if "from scipy.optimize import linear_sum_assignment" in text:
        print(f"DetIE lapsolver fallback already present in {path}")
        raise SystemExit(0)
    raise SystemExit(f"Could not find expected lapsolver import in {path}")

path.write_text(text.replace(old, new, 1))
print(f"Patched DetIE lapsolver fallback in {path}")
PY
