#!/usr/bin/env bash
set -euo pipefail
PY=/root/autodl-tmp/wav2lip-syncnet/envs/wav2lip/bin/python
WORKER=/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/model_deploy_worker_attempt02.py
CODE_ROOT=/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/model_sources_attempt02
OUT=/dev/shm/cross_tfg_audio_head_20261003/model_deploy_attempt02
mkdir -p "$OUT"
for job in musetalk ditto; do
  job_root="$OUT/$job"
  mkdir -p "$job_root"
  if [ -e "$job_root/launcher.pid" ] || [ -e "$job_root/deployment_status.json" ]; then
    echo "refusing to reuse prior job namespace: $job_root" >&2
    exit 3
  fi
  nohup "$PY" "$WORKER" --job "$job" --output-root "$OUT" --code-root "$CODE_ROOT" \
    >"$job_root/launcher.stdout.log" 2>"$job_root/launcher.stderr.log" </dev/null &
  pid=$!
  printf '%s\n' "$pid" >"$job_root/launcher.pid"
  printf '{"job":"%s","pid":%s,"worker":"%s","started_unix":%s}\n' "$job" "$pid" "$WORKER" "$(date +%s)" >"$job_root/launch_manifest.json"
done
printf '{"status":"STARTED_BACKGROUND","root":"%s","pids":{"musetalk":%s,"ditto":%s}}\n' "$OUT" "$(cat "$OUT/musetalk/launcher.pid")" "$(cat "$OUT/ditto/launcher.pid")"