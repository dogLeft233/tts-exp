#!/usr/bin/env bash
set +e
NS=/root/tts-research-system-20261001/cross_tfg_audio_head_20261003
TASK=$NS/stagea_dev_attempt01
CODE=$TASK/code
CTRL=$NS/job_control/stagea_dev_attempt01_attempt03
OUT=/dev/shm/cross_tfg_audio_head_20261003/stagea_dev_attempt01/run
PLAN=$TASK/plan/plan.json
ROWS=$TASK/plan/planned_rows.jsonl
RUNPY=/root/autodl-tmp/wav2lip-syncnet/envs/syncnet/bin/python
W2L=/root/autodl-tmp/wav2lip-syncnet/Wav2Lip
SYNC=/root/autodl-tmp/wav2lip-syncnet/syncnet_python
FACE=/root/autodl-tmp/tts-research-20260930/portable_source_axis_20260930/inputs/primary_face.png
SOURCE_VIDEO=/dev/shm/cross_tfg_audio_head_20261003/stage_a_plan_attempt01/source/static_portrait_reference_25fps.mp4
FFMPEG_SHA=36d94a605d612e4090d1b8aec889d0c0801c6eafb1593c90f5c0dfd2e2966a45
FFPROBE_SHA=d4f3ef9c12be756793cad83dd2004d89f49c1c4094053bfbbe7e28925c8fa4fd
PLAN_SHA=bf68725df298d68b4205ef205236e4046b4ba180c8e6da622fcaa9b039d38585
mkdir -p "$CTRL"
printf '%s\n' "$$" > "$CTRL/worker.pid"
date -u +%Y-%m-%dT%H:%M:%SZ > "$CTRL/started_utc.txt"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CROSS_TFG_STAGEA_RUNTIME_SETTINGS="$CTRL/runtime_settings.json"
if [ -e "$OUT" ]; then
  printf '%s\n' 'OUTPUT_DIRECTORY_ALREADY_EXISTS' > "$CTRL/launch_error.txt"
  printf '%s\n' 98 > "$CTRL/exitcode"
  exit 98
fi
if [ "$(sha256sum "$PLAN" | awk '{print $1}')" != "$PLAN_SHA" ]; then
  printf '%s\n' 'PLAN_SHA_MISMATCH' > "$CTRL/launch_error.txt"
  printf '%s\n' 97 > "$CTRL/exitcode"
  exit 97
fi
"$RUNPY" "$CODE/stagea_entry_attempt01.py" \
  --plan "$PLAN" --planned-rows "$ROWS" --expected-plan-raw-sha256 "$PLAN_SHA" \
  --output-root "$OUT" --project-root "$CODE" \
  --wav2lip-python /root/autodl-tmp/wav2lip-syncnet/envs/wav2lip/bin/python \
  --wav2lip-root "$W2L" \
  --generator-child "$CODE/scripts/experiments/cross_tfg_audio_head_20261003/engineering_qc_oldpilot_attempt04.py" \
  --syncnet-root "$SYNC" --syncnet-python /root/autodl-tmp/wav2lip-syncnet/envs/syncnet/bin/python \
  --syncnet-checkpoint "$SYNC/data/syncnet_v2.model" \
  --syncnet-evaluator "$CODE/scripts/experiments/cross_tfg_audio_head_20261003/stagea_syncnet_evaluator.py" \
  --ffmpeg /usr/bin/ffmpeg --ffprobe /usr/bin/ffprobe \
  --ffmpeg-sha256 "$FFMPEG_SHA" \
  --pcm16-lineage-sha256 f634bad9c287a2f7f8cd8ffa62674bdd9e97dbf4b0f00dbf44725e4ca4ca36a0 \
  --generator-timeout 900 --syncnet-timeout 1800 --ffmpeg-timeout 300 \
  > "$CTRL/stagea.stdout" 2> "$CTRL/stagea.stderr"
RUN_RC=$?
printf '%s\n' "$RUN_RC" > "$CTRL/stagea_exitcode"
date -u +%Y-%m-%dT%H:%M:%SZ > "$CTRL/finished_utc.txt"
if [ -d "$OUT" ]; then
  "$RUNPY" - "$OUT" "$CTRL/jpeg_cleanup.json" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
expected = Path('/dev/shm/cross_tfg_audio_head_20261003/stagea_dev_attempt01/run')
if root.resolve() != expected:
    raise SystemExit('cleanup refused unexpected output root')
deleted = 0
deleted_bytes = 0
for path in root.rglob('*.jpg'):
    size = path.stat().st_size
    path.unlink()
    deleted += 1
    deleted_bytes += size
Path(sys.argv[2]).write_text(json.dumps({'scope':str(root),'deleted_jpeg_count':deleted,'deleted_jpeg_bytes':deleted_bytes,'kept_primary_audio_and_track_artifacts':True},sort_keys=True,indent=2)+'\n',encoding='utf-8')
PY
  CLEAN_RC=$?
  printf '%s\n' "$CLEAN_RC" > "$CTRL/jpeg_cleanup_exitcode"
  if [ "$CLEAN_RC" -eq 0 ]; then
    mkdir -p "$TASK/persist_attempt01"
    AVAIL=$(df -B1 "$TASK" | awk 'NR==2 {print $4}')
    SIZE=$(du -sb "$OUT" | awk '{print $1}')
    if [ -n "$AVAIL" ] && [ -n "$SIZE" ] && [ "$AVAIL" -gt "$((SIZE + 50000000))" ]; then
      TMP="$TASK/persist_attempt01/.run.tar.gz.partial"
      tar -czf "$TMP" -C "$(dirname "$OUT")" "$(basename "$OUT")"
      TAR_RC=$?
      if [ "$TAR_RC" -eq 0 ]; then
        mv "$TMP" "$TASK/persist_attempt01/run.tar.gz"
        sha256sum "$TASK/persist_attempt01/run.tar.gz" > "$CTRL/persist_archive_sha256.txt"
        wc -c < "$TASK/persist_attempt01/run.tar.gz" > "$CTRL/persist_archive_bytes.txt"
        printf '%s\n' 0 > "$CTRL/archive_exitcode"
      else
        printf '%s\n' "$TAR_RC" > "$CTRL/archive_exitcode"
        rm -f "$TMP"
      fi
    else
      printf '%s\n' 'PERSIST_ARCHIVE_SKIPPED_INSUFFICIENT_SPACE_OUTPUT_REMAINS_TMPFS' > "$CTRL/archive_exitcode"
    fi
  fi
fi
printf '%s\n' "$RUN_RC" > "$CTRL/exitcode"
exit "$RUN_RC"
