#!/bin/bash
# Evaluate the hybrid agent on one GPU: starts the model server (vLLM), waits until it answers, runs run.py.
#
#   smoke test (30 min partition, 2 games, 1 seed):   sbatch cluster/eval.sbatch
#   full dev run:   sbatch -p gpu_a100_il -t 12:00:00 --export=ALL,GAMES=dev,SEEDS=3,LABEL=hybrid-dev cluster/eval.sbatch
#   core alone on the same node (control):            add  ,NO_LLM=1  to --export
#
# Results land in runs/<date>-<label>.json; copy them to the laptop with:  scp 'uc3:<repo>/runs/*.json' runs/
# NOT YET TESTED on the cluster (written 2026-11-12).
#SBATCH --partition=gpu_a100_short
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --job-name=arc-hybrid
#SBATCH --output=cluster/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p cluster/logs
export PATH=$HOME/.local/bin:$PATH
WS=$(ws_find arc)
MODEL=${MODEL:-cyankiwi/Qwen3.6-27B-AWQ-INT4}
PORT=${PORT:-8000}

if [ -z "${NO_LLM:-}" ]; then
  export HF_HOME=$WS/hf HF_HUB_OFFLINE=1
  "$WS/vllm/bin/vllm" serve "$MODEL" --port "$PORT" --max-model-len 16384 --gpu-memory-utilization 0.92 \
      --reasoning-parser qwen3 > "cluster/logs/vllm-$SLURM_JOB_ID.log" 2>&1 &
  VLLM=$!
  trap 'kill $VLLM 2>/dev/null' EXIT
  for _ in $(seq 90); do curl -sf "localhost:$PORT/v1/models" >/dev/null && break; sleep 10; done
  curl -sf "localhost:$PORT/v1/models" >/dev/null || { echo "model server did not start"; tail -40 "cluster/logs/vllm-$SLURM_JOB_ID.log"; exit 1; }
  export LLM_BASE_URL=http://localhost:$PORT/v1 LLM_MODEL=$MODEL LLM_EXTRA='{"top_k": 20}'
fi

# export OPERATION_MODE=offline     # uncomment if the compute nodes have no internet access
uv run run.py ${GAMES:-vc33 ls20} --seeds "${SEEDS:-1}" --label "${LABEL:-hybrid-smoke}"
