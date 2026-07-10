#!/bin/bash
# Serve an OpenAI-compatible vLLM endpoint for Qwen on one A100 via Slurm
# (GPU-heavy -> must go through Slurm; default queue workq).
#
#   sbatch scripts/serve_qwen_vllm.slurm         # start the server
#   squeue --me                                  # wait until state R (RUNNING)
#   grep -m1 'connect with' vllm-<jobid>.log     # get the node + URL
#
# Then, from a shell that can reach that node (another Slurm job, or the login
# node if the compute-node port is reachable), run the caller:
#   OPENAI_BASE_URL=http://<node>:8000/v1 API_KEY=EMPTY \
#   uv run --with openai python -m diffopenie.evaluation.llm_extract \
#     --input-sentences benchmarks/CaRB/data/dev.txt \
#     --gold benchmarks/CaRB/data/gold/dev.tsv \
#     --gold-examples benchmarks/CaRB/data/gold/test.tsv \
#     --model "$MODEL" --limit 200 --workers 1   # then again --workers 8
#
# Stop the server when done:  scancel <jobid>

#SBATCH --job-name=vllm-qwen
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-gpu=80G
#SBATCH --time=12:00:00
#SBATCH --output=vllm-%j.log

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"    # 80GB A100 also fits 14B / 32B-AWQ
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-8192}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

echo "=================================================================="
echo " vLLM server  (node=$(hostname))"
echo "   model:    $MODEL"
echo "   connect with: OPENAI_BASE_URL=http://$(hostname):${PORT}/v1 API_KEY=EMPTY"
echo "=================================================================="

# --host 0.0.0.0 so another node can reach it; xgrammar enables the JSON
# response_format the extractor uses. If `uv` is not on the node PATH, activate
# a venv that has vllm instead (source ~/venvs/<env>/bin/activate; vllm serve ...).
exec uv run --with vllm vllm serve "$MODEL" \
    --host 0.0.0.0 --port "$PORT" \
    --dtype bfloat16 --max-model-len "$MAX_LEN" \
    --guided-decoding-backend xgrammar
