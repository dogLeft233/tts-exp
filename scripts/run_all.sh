#!/usr/bin/env bash
# run_all.sh - Sequential pipeline runner (issue #1)
# Usage:
#   ./run_all.sh             # full run using scripts/config.yaml
#   ./run_all.sh --smoke     # smoke test, 1 configured sample
#   ./run_all.sh --config scripts/configs/aishell1_100_zh.yaml
# Per Q11 of plan: fail-fast-free (failures recorded, run continues).
# Per Q21: each step is independently rerunnable via --run_id reuse.

set -uo pipefail

SMOKE_FLAG=""
CONFIG_FLAG=""
if [ "${1:-}" = "--smoke" ]; then
  SMOKE_FLAG="--smoke"
  shift
fi
if [ "${1:-}" = "--config" ]; then
  if [ -z "${2:-}" ]; then
    echo "[run_all] --config requires a path" >&2
    exit 2
  fi
  CONFIG_FLAG="--config $2"
  shift 2
fi
if [ "$#" -gt 0 ]; then
  echo "[run_all] unknown arguments: $*" >&2
  exit 2
fi

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
export RUN_ID

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

# Ensure python from ditto env is available (steps 00-03,05 run here;
# 04 switches to syncnet env via subprocess internally).
DITTO_PYTHON="${AUTODL_TMP:-/root/autodl-tmp}/envs/ditto/bin/python"
if [ ! -x "$DITTO_PYTHON" ]; then
  DITTO_PYTHON="python3"
fi
# Also need LD_LIBRARY_PATH for tensorrt (cuDNN 8) — export if present
if [ -d "/root/autodl-tmp/envs/ditto/opt/cudnn8/lib" ]; then
  export LD_LIBRARY_PATH="/root/autodl-tmp/envs/ditto/opt/cudnn8/lib:${LD_LIBRARY_PATH:-}"
fi
# HF mirror for Qwen3-TTS model download (huggingface.co is blocked in China)
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

RUN_DIR="$REPO_DIR/runs/$RUN_ID"
mkdir -p "$RUN_DIR"
echo "[run_all] RUN_ID=$RUN_ID python=$DITTO_PYTHON"
echo "[run_all] RUN_DIR=$RUN_DIR"
echo "[run_all] smoke=$SMOKE_FLAG"

# Snapshot base + override config and git commit for reproducibility.
cp scripts/config.yaml "$RUN_DIR/config.yaml"
if [ -n "$CONFIG_FLAG" ]; then
  CONFIG_PATH="${CONFIG_FLAG#--config }"
  cp "$CONFIG_PATH" "$RUN_DIR/config_override.yaml"
fi
git rev-parse HEAD > "$RUN_DIR/git_commit.txt" 2>/dev/null || echo "unknown" > "$RUN_DIR/git_commit.txt"

STEPS=("00_datacheck.py" "01_asr.py" "02_tts.py" "03_ditto.py" "04_eval.py" "05_report.py")
FAILS=()

for step in "${STEPS[@]}"; do
  echo ""
  echo "===== $step$SMOKE_FLAG ====="
    if $DITTO_PYTHON "scripts/$step" --run_id "$RUN_ID" $SMOKE_FLAG $CONFIG_FLAG; then
    echo "[ok] $step"
  else
    rc=$?
    echo "[fail] $step (exit $rc)"
    FAILS+=("$step($rc)")
  fi
done

echo ""
echo "============================================================"
if [ "${#FAILS[@]}" -eq 0 ]; then
  echo "[run_all] ALL OK: $RUN_DIR"
  exit 0
else
  echo "[run_all] FAILURES: ${FAILS[*]}"
  echo "[run_all] artifacts at: $RUN_DIR"
  exit 1
fi