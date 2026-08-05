#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/root/autodl-tmp/StableAvatar"
PYTHON="/root/autodl-tmp/envs/stableavatar/bin/python"
PORT="${PORT:-8400}"
GPU_MEMORY_MODE="${GPU_MEMORY_MODE:-model_cpu_offload}"
WAIT_FOR_GPU="${WAIT_FOR_GPU:-0}"

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PORT
export GPU_MEMORY_MODE
export GPU_IDLE_TIMEOUT="${GPU_IDLE_TIMEOUT:-300}"
export GRADIO_SERVER_NAME="0.0.0.0"

mkdir -p "$ROOT/runtime/uploads" "$ROOT/runtime/outputs"

if [[ ! -e /tmp/stableavatar ]]; then
    ln -s "$ROOT/runtime" /tmp/stableavatar
fi

while true; do
    gpu_list="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
    if [[ -n "${gpu_list//[[:space:]]/}" && "$gpu_list" != *"No devices were found"* ]]; then
        break
    fi
    if [[ "$WAIT_FOR_GPU" != "1" ]]; then
        printf '%s\n' 'No NVIDIA GPU is visible. Recreate or reattach a GPU container, then rerun this script.' >&2
        exit 2
    fi
    printf '%s\n' 'Waiting for an NVIDIA GPU to become visible...' >&2
    sleep 15
done

printf 'GPU: '
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
printf 'PyTorch: '
"$PYTHON" -c 'import torch; print(torch.__version__, "CUDA build", torch.version.cuda, "available", torch.cuda.is_available())'

exec "$PYTHON" -m gunicorn \
    --bind "0.0.0.0:${PORT}" \
    --workers 1 \
    --threads 2 \
    --timeout 3600 \
    --graceful-timeout 30 \
    server:app
