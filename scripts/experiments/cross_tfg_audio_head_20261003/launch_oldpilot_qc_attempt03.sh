#!/usr/bin/env bash
set -euo pipefail
PY=/root/autodl-tmp/wav2lip-syncnet/envs/wav2lip/bin/python
SCRIPT=/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/engineering_qc_oldpilot_attempt03.py
BASE=/dev/shm/cross_tfg_audio_head_20261003
OUT="$BASE/oldpilot_qc_attempt03"
RUN="$OUT/run"
STATUS="$OUT/control"
if [ -e "$OUT" ]; then
  echo "refusing to reuse existing attempt03 namespace: $OUT" >&2
  exit 3
fi
mkdir -p "$STATUS"
nohup bash -c 'set +e; "$1" "$2" --output-root "$3"; rc=$?; printf "%s\n" "$rc" > "$4/runner_exit_code"; exit "$rc"' _ "$PY" "$SCRIPT" "$RUN" "$STATUS" \
  >"$STATUS/runner.stdout.log" 2>"$STATUS/runner.stderr.log" </dev/null &
pid=$!
printf '%s\n' "$pid" >"$STATUS/runner.pid"
printf '{"status":"STARTED_BACKGROUND","pid":%s,"python":"%s","script":"%s","output_root":"%s","started_unix":%s}\n' "$pid" "$PY" "$SCRIPT" "$RUN" "$(date +%s)" >"$STATUS/launch_manifest.json"
printf '{"status":"STARTED_BACKGROUND","pid":%s,"output_root":"%s","status_root":"%s"}\n' "$pid" "$RUN" "$STATUS"