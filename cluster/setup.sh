#!/bin/bash
# One-time setup on a bwUniCluster 3.0 LOGIN node (needs internet). From the repository folder run:
#     bash cluster/setup.sh
# It creates a workspace (model weights are ~20 GB, too big for $HOME), installs uv, the agent's Python
# environment, vLLM, downloads the model and the game files. Safe to run again.
# NOT YET TESTED on the cluster (written 2026-11-12); expect small fixes on first use.
set -euo pipefail

MODEL=${MODEL:-cyankiwi/Qwen3.6-27B-AWQ-INT4}   # 4-bit build with vision, Apache-2.0, needs vllm >= 0.19

WS=$(ws_find arc 2>/dev/null || true)
[ -n "$WS" ] || WS=$(ws_allocate arc 60)        # 60 days; extend with: ws_extend arc 60
echo "workspace: $WS"
export UV_CACHE_DIR=$WS/uv-cache HF_HOME=$WS/hf

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH

uv sync                                          # the agent's environment (.venv in the repository)
[ -x "$WS/vllm/bin/vllm" ] || { uv venv "$WS/vllm" --python 3.12 && uv pip install --python "$WS/vllm/bin/python" "vllm>=0.19"; }
"$WS/vllm/bin/python" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL'))"

uv run run.py vc33 --budget 5                    # downloads the game files into environment_files/
echo "setup done. Smoke test:  sbatch cluster/eval.sbatch"
