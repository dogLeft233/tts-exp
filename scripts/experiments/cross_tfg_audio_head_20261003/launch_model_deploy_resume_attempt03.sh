#!/usr/bin/env bash
set -euo pipefail
WORKER="/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/model_deploy_resume_attempt03.py"
PYTHON="/root/autodl-tmp/wav2lip-syncnet/envs/wav2lip/bin/python"
OUT="/dev/shm/cross_tfg_audio_head_20261003/model_deploy_attempt03"
mkdir -p "$OUT/control"
for JOB in musetalk ditto; do
  CONTROL="$OUT/control/$JOB"
  mkdir "$CONTROL"
  STARTED="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  CMD="$PYTHON $WORKER --job $JOB"
  nohup "$PYTHON" "$WORKER" --job "$JOB" >"$CONTROL/worker.stdout.log" 2>"$CONTROL/worker.stderr.log" < /dev/null &
  PID=$!
  printf '%s\n' "$PID" > "$CONTROL/pid.txt"
  printf '%s\n' "$CMD" > "$CONTROL/command.txt"
  printf '%s\n' "$STARTED" > "$CONTROL/started_utc.txt"
  printf '%s\n' "RUNNING" > "$CONTROL/launch_status.txt"
  printf '%s\n' "$PID"
done
