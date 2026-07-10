#!/bin/bash
# Re-time DiffIE and the MC-dropout tagger on ONE A100 (same GPU as the Qwen
# baseline) so all three efficiency numbers are comparable.
#   sbatch scripts/timing_a100.slurm
# Needs valid weights.pt for both configs (verify: python -c "import zipfile;
# print(zipfile.ZipFile('<w>.pt').testzip())" -> None).

#SBATCH --job-name=timing-ab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-gpu=80G
#SBATCH --time=02:00:00
#SBATCH --output=timing-a100-%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

NS="1 2 4 8 16 32 64 128 256 512"

echo "=== DiffIE ==="
uv run python -m diffopenie.evaluation.timing_benchmark \
    --config configs/lsoie_ex_2500/lsoie_ex_2500_config.yaml \
    --checkpoint-path configs/lsoie_ex_2500/weights.pt \
    --input-sentences benchmarks/CaRB/data/dev.txt \
    --ns $NS --limit 200 --repeats 3 --warmup 10 --device cuda \
    --out timing_diffie_a100.csv

echo "=== MC-dropout tagger ==="
uv run python -m diffopenie.evaluation.timing_benchmark \
    --config configs/lsoie_ex_tagger_2500/lsoie_ex_tagger_2500_config.yaml \
    --checkpoint-path configs/lsoie_ex_tagger_2500/weights.pt \
    --input-sentences benchmarks/CaRB/data/dev.txt \
    --ns $NS --limit 200 --repeats 3 --warmup 10 --device cuda \
    --out timing_tagger_a100.csv

echo "done -> timing_diffie_a100.csv, timing_tagger_a100.csv"
