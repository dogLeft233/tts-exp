#!/bin/bash
# R1 TTS-TFG matrix: 5 providers × 10 samples each, loudnorm off
# Each provider runs as a separate pipeline (02→03→04→05).
#
# Prerequisites:
#   1. scripts/configs/r1_*.yaml exist (provider-specific overrides)
#   2. server env vars: DASHSCOPE_API_KEY (DashScope), FISH_AUDIO_API_KEY (Fish)
#   3. transcripts cached from previous run (01_transcript/)
#   4. ditto env: /root/autodl-tmp/envs/ditto (or whichever has requests, soundfile)
#   5. syncnet env: /root/autodl-tmp/envs/syncnet
#
# Usage: bash scripts/run_r1_matrix.sh
set -euo pipefail

export HF_ENDPOINT=https://hf-mirror.com
DITTO_ENV="/root/autodl-tmp/envs/ditto"
SYNCNET_ENV="/root/autodl-tmp/envs/syncnet"
CUDNN_LIB="$DITTO_ENV/opt/cudnn8/lib"
REPO="/root/autodl-tmp/repos/tts-exp"
TR_CP="/root/autodl-tmp/checkpoints/ditto_trt_Ampere_Plus"
TR_CFG="/root/autodl-tmp/checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl"
BASE_ID="r1"  # prefix for run_ids

export LD_LIBRARY_PATH="$CUDNN_LIB:$LD_LIBRARY_PATH"

MODELS=(
  "faster_qwen3:scripts/configs/r1_faster_qwen3.yaml"
  "dashscope_vc:scripts/configs/r1_dashscope_vc.yaml"
  "dashscope_flash:scripts/configs/r1_dashscope_flash.yaml"
  "dashscope_cv3:scripts/configs/r1_dashscope_cv3.yaml"
  "fish_audio:scripts/configs/r1_fish_audio.yaml"
)

for ENTRY in "${MODELS[@]}"; do
  SLUG="${ENTRY%%:*}"
  CFG="${ENTRY##*:}"
  TS=$(date -u +%Y%m%dT%H%M%SZ)
  RUN_ID="${BASE_ID}_${SLUG}_${TS}"
  echo "======== $(date) RUN $RUN_ID ========"

  # 00_datacheck (skip if already passed)
  # 01_transcript — copy from strict run cache
  mkdir -p "$REPO/runs/$RUN_ID/01_transcript"
  cp "$REPO/runs/aishell1_strict_20260707T081223Z/01_transcript/"*.txt "$REPO/runs/$RUN_ID/01_transcript/" 2>/dev/null || true
  cp "$REPO/runs/aishell1_strict_20260707T081223Z/01_transcript/transcript.json" "$REPO/runs/$RUN_ID/01_transcript/" 2>/dev/null || true

  # 02_tts
  echo "--- 02_tts $SLUG ---"
  source "$DITTO_ENV/bin/activate"
  CUDA_VISIBLE_DEVICES=0 python "$REPO/scripts/02_tts.py" \
    --run_id "$RUN_ID" \
    --config "$CFG"
  deactivate

  # 03_ditto
  echo "--- 03_ditto $SLUG ---"
  source "$DITTO_ENV/bin/activate"
  export LD_LIBRARY_PATH="$CUDNN_LIB:$LD_LIBRARY_PATH"
  CUDA_VISIBLE_DEVICES=0 python "$REPO/scripts/03_ditto.py" --run_id "$RUN_ID"
  # Clean stale .tmp.mp4 files
  find "$REPO/runs/$RUN_ID/03_ditto" -name '*.tmp.mp4' -delete 2>/dev/null || true
  deactivate

  # 04_eval
  echo "--- 04_eval $SLUG ---"
  source "$SYNCNET_ENV/bin/activate"
  python "$REPO/scripts/04_eval.py" --run_id "$RUN_ID"
  deactivate

  # 05_report
  echo "--- 05_report $SLUG ---"
  source "$SYNCNET_ENV/bin/activate"
  python "$REPO/scripts/05_report.py" --run_id "$RUN_ID"
  deactivate

  echo "======== $(date) DONE $RUN_ID ========"
done

echo "======== ALL PROVIDERS DONE ========"
