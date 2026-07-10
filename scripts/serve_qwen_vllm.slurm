#!/bin/bash
# Deploy an OpenAI-compatible vLLM server (Qwen) on one A100 for the LLM OpenIE
# baseline. Submit with:  sbatch scripts/serve_qwen_vllm.slurm
#
# Once RUNNING, read vllm_<jobid>.log for the line telling you which node it is
# on, then point the extractor at it:
#   OPENAI_BASE_URL=http://<node>:8000/v1 API_KEY=EMPTY \
#   uv run --with openai python -m diffopenie.evaluation.llm_extract \
#     --input-sentences benchmarks/CaRB/data/dev.txt \
#     --gold benchmarks/CaRB/data/gold/dev.tsv \
#     --gold-examples benchmarks/CaRB/data/gold/test.tsv \
#     --model "$MODEL" --workers 8 --limit 200
#
# The server blocks until the job's time limit; cancel with scancel <jobid>.

#SBATCH --job-name=vllm-qwen
#SBATCH --gres=gpu:a100:1            # TODO: match your cluster's GPU gres syntax
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=vllm_%j.log
##SBATCH --partition=gpu             # TODO: uncomment + set your partition
##SBATCH --account=your_account      # TODO: uncomment + set if your site requires it

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"   # A100-80GB can run 14B / 32B-AWQ
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-8192}"

echo "=================================================================="
echo " vLLM server"
echo "   node:       $(hostname)"
echo "   model:      $MODEL"
echo "   endpoint:   http://$(hostname):$PORT/v1"
echo "   connect w/: OPENAI_BASE_URL=http://$(hostname):$PORT/v1 API_KEY=EMPTY"
echo "=================================================================="

# Serve. --dtype bfloat16 for A100; guided decoding backend enables the
# response_format={"type":"json_object"} the extractor uses.
exec uv run --with vllm vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --dtype bfloat16 \
    --max-model-len "$MAX_LEN" \
    --guided-decoding-backend xgrammar
