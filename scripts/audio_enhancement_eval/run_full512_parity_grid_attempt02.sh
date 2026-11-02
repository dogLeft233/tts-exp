#!/usr/bin/env bash
set -euo pipefail

CODE=/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/proxy_native_full512_parity_attempt03/visual_grid_attempt01/code
JOB=/root/autodl-tmp/tts-research-storage-20261001/job_control/cross_tfg_audio_head_20261003/proxy_native_full512_parity_attempt03_visual_grid_attempt01
OUT=/dev/shm/cross_tfg_audio_head_20261003/proxy_native_full512_parity_attempt03/visual_grid_attempt01/output
PY=/root/autodl-tmp/wav2lip-syncnet/envs/wav2lip/bin/python
SCRIPT=$CODE/render_full512_parity_grid_attempt01.py
INPUT=$CODE/visual_grid_inputs_attempt01.json

mkdir -p "$JOB"
if [ -e "$OUT" ]; then
  echo 'refusing existing visual grid output' >&2
  exit 40
fi
printf '%s  %s\n' '33289897a7ae4fd88c7317bfea318612aebbed5bd15501bc32e9b232c33bbfd6' "$SCRIPT" | sha256sum -c -
printf '%s  %s\n' '43665d4c3678ea4199743c16d2fe4ea1346863e10db83b582a28beac1184648d' "$INPUT" | sha256sum -c -
printf '{"status":"STARTED","pid":%s,"source_sha256":"33289897a7ae4fd88c7317bfea318612aebbed5bd15501bc32e9b232c33bbfd6","input_sha256":"43665d4c3678ea4199743c16d2fe4ea1346863e10db83b582a28beac1184648d"}\n' "$$" > "$JOB/start.json"
set +e
PYTHONDONTWRITEBYTECODE=1 "$PY" -B "$SCRIPT" --input-manifest "$INPUT" --output-root "$OUT" > "$JOB/stdout.log" 2> "$JOB/stderr.log"
rc=$?
set -e
printf '{"status":"%s","exit_code":%s,"finished_utc":"%s"}\n' "$([ "$rc" -eq 0 ] && echo COMPLETE || echo FAILED)" "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$JOB/exit.json"
if [ "$rc" -ne 0 ]; then
  exit "$rc"
fi
test -s "$OUT/proxy_native_full512_frame_grid.png"
test -s "$OUT/visual_grid_receipt.json"
printf '%s\n' 'VISUAL_GRID_PASS'
