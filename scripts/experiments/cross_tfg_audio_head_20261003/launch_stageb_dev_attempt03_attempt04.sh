#!/usr/bin/env bash
set -euo pipefail
NS=/root/tts-research-system-20261001/cross_tfg_audio_head_20261003
TASK=$NS/stageb_dev_attempt03
CODE=$TASK/code
CTRL=$NS/job_control/stageb_dev_attempt03_attempt04
OUT=/dev/shm/cross_tfg_audio_head_20261003/stageb_dev_attempt03/run
PLAN=$TASK/plan/plan.json
ROWS=$TASK/plan/planned_rows.jsonl
FIT_TAR=$NS/fit_subset_attempt07/persist_attempt07/run02.tar.gz
ASSET_DIR=$TASK/assets
CHECKPOINT=$ASSET_DIR/shared_audio_head.pt
RUNPY=/root/autodl-tmp/wav2lip-syncnet/envs/syncnet/bin/python
W2L=/root/autodl-tmp/wav2lip-syncnet/Wav2Lip
SYNC=/root/autodl-tmp/wav2lip-syncnet/syncnet_python
FACE=/root/autodl-tmp/tts-research-20260930/portable_source_axis_20260930/inputs/primary_face.png
VIDEO=/dev/shm/cross_tfg_audio_head_20261003/stage_a_plan_attempt01/source/static_portrait_reference_25fps.mp4
PLAN_SHA=43f10f334ea904d071836195733a1d1bdabe479a38061156ac38d5083d7d8632
FIT_TAR_SHA=55ae75e8159d90d17ecbe261fde25801e8af30b6a4a505bed095af5c2342dde3
HEAD_SHA=bc5aeec15f9197ca0c1d0d606eebe41ff2dbf6620767eee5bc37ca4b9395aaed
FFMPEG_SHA=36d94a605d612e4090d1b8aec889d0c0801c6eafb1593c90f5c0dfd2e2966a45
FFPROBE_SHA=d4f3ef9c12be756793cad83dd2004d89f49c1c4094053bfbbe7e28925c8fa4fd
mkdir -p "$CTRL"
trap 'rc=$?; printf "%s\n" "$rc" > "$CTRL/exitcode"; date -u +%Y-%m-%dT%H:%M:%SZ > "$CTRL/finished_utc.txt"' EXIT
printf '%s\n' "$$" > "$CTRL/worker.pid"
date -u +%Y-%m-%dT%H:%M:%SZ > "$CTRL/started_utc.txt"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export CROSS_TFG_STAGEA_RUNTIME_SETTINGS="$CTRL/runtime_settings.json"
if [ -e "$OUT" ]; then printf '%s\n' 'OUTPUT_DIRECTORY_ALREADY_EXISTS' > "$CTRL/launch_error.txt"; printf '%s\n' 98 > "$CTRL/exitcode"; exit 98; fi
mkdir -p "$TASK" "$ASSET_DIR"
test "$(sha256sum "$FIT_TAR" | awk '{print $1}')" = "$FIT_TAR_SHA" || { echo FIT_ARCHIVE_SHA_MISMATCH > "$CTRL/launch_error.txt"; exit 96; }
if [ -e "$CHECKPOINT" ]; then
  test "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" = "$HEAD_SHA" || { echo EXISTING_FIT_CHECKPOINT_SHA_MISMATCH > "$CTRL/launch_error.txt"; exit 93; }
else
  TMP="$CHECKPOINT.partial"
  tar -xOzf "$FIT_TAR" run02/shared_audio_head.pt > "$TMP" || { echo FIT_CHECKPOINT_EXTRACT_FAILED > "$CTRL/launch_error.txt"; exit 95; }
  test "$(sha256sum "$TMP" | awk '{print $1}')" = "$HEAD_SHA" || { echo FIT_CHECKPOINT_SHA_MISMATCH > "$CTRL/launch_error.txt"; exit 94; }
  mv "$TMP" "$CHECKPOINT"
fi
test "$(sha256sum "$PLAN" | awk '{print $1}')" = "$PLAN_SHA" || { echo PLAN_SHA_MISMATCH > "$CTRL/launch_error.txt"; exit 97; }
test "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" = "$HEAD_SHA" || { echo HEAD_SHA_MISMATCH > "$CTRL/launch_error.txt"; exit 93; }
test "$(git -C "$W2L" rev-parse HEAD)" = bac9a81e63ecc153202353372e5724b83d9e6322
test "$(git -C "$SYNC" rev-parse HEAD)" = 907c0b579c2e2d83f0eae1b2ac9e720cde4e5623
test "$(sha256sum "$W2L/checkpoints/wav2lip_gan.pth" | awk '{print $1}')" = ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8
test "$(sha256sum "$SYNC/data/syncnet_v2.model" | awk '{print $1}')" = 961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442
test "$(sha256sum "$FACE" | awk '{print $1}')" = e3a695edf22a89cac40184453131019874a65e15592593d5be30e7f71373d9c6
test "$(sha256sum "$VIDEO" | awk '{print $1}')" = c76617141188acfea5f4c122513b982997ec45ebaa35a2b6ed9f67101e9c40de
test "$(sha256sum /usr/bin/ffmpeg | awk '{print $1}')" = "$FFMPEG_SHA"
test "$(sha256sum /usr/bin/ffprobe | awk '{print $1}')" = "$FFPROBE_SHA"
if ! "$RUNPY" -c 'import torch,numpy,cv2,scripts.audio_enhancement; from scripts.audio_enhancement.heads import SpectralResidualHead; assert str(torch.__version__) == "2.5.1+cu124" and torch.cuda.is_available(); print("stageb_preflight",torch.__version__,numpy.__version__,torch.cuda.is_available())' > "$CTRL/preflight.log" 2>&1; then
  printf '%s\n' PREFLIGHT_FAILED > "$CTRL/launch_error.txt"
  exit 90
fi
test ! -e "$OUT" || { echo OUTPUT_COLLISION > "$CTRL/launch_error.txt"; exit 98; }
set +e
"$RUNPY" "$CODE/stageb_entry_attempt03.py" --plan "$PLAN" --planned-rows "$ROWS" --expected-plan-raw-sha256 "$PLAN_SHA" \
  --output-root "$OUT" --project-root "$CODE" --wav2lip-python /root/autodl-tmp/wav2lip-syncnet/envs/wav2lip/bin/python \
  --wav2lip-root "$W2L" --generator-child "$CODE/scripts/experiments/cross_tfg_audio_head_20261003/engineering_qc_oldpilot_attempt04.py" \
  --syncnet-root "$SYNC" --syncnet-python "$RUNPY" --syncnet-checkpoint "$SYNC/data/syncnet_v2.model" \
  --syncnet-evaluator "$CODE/scripts/experiments/cross_tfg_audio_head_20261003/stagea_syncnet_evaluator.py" \
  --ffmpeg /usr/bin/ffmpeg --ffprobe /usr/bin/ffprobe --ffmpeg-sha256 "$FFMPEG_SHA" \
  --pcm16-lineage-sha256 f634bad9c287a2f7f8cd8ffa62674bdd9e97dbf4b0f00dbf44725e4ca4ca36a0 \
  --generator-timeout 900 --syncnet-timeout 1800 --ffmpeg-timeout 300 \
  > "$CTRL/stageb.stdout" 2> "$CTRL/stageb.stderr"
RUN_RC=$?
set -e
printf '%s\n' "$RUN_RC" > "$CTRL/stageb_exitcode"
date -u +%Y-%m-%dT%H:%M:%SZ > "$CTRL/finished_utc.txt"
if [ -d "$OUT" ]; then
  if "$RUNPY" - "$OUT" "$CTRL/jpeg_cleanup.json" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); expected=Path('/dev/shm/cross_tfg_audio_head_20261003/stageb_dev_attempt03/run')
if root.resolve()!=expected: raise SystemExit('cleanup refused unexpected output root')
deleted=deleted_bytes=0
for p in root.rglob('*.jpg'):
 n=p.stat().st_size; p.unlink(); deleted+=1; deleted_bytes+=n
Path(sys.argv[2]).write_text(json.dumps({'scope':str(root),'deleted_jpeg_count':deleted,'deleted_jpeg_bytes':deleted_bytes,'kept_primary_audio_and_track_artifacts':True},sort_keys=True,indent=2)+'\n')
PY
  then CLEAN_RC=0; else CLEAN_RC=$?; fi
  printf '%s\n' "$CLEAN_RC" > "$CTRL/jpeg_cleanup_exitcode"
  if [ "$CLEAN_RC" -eq 0 ]; then
    mkdir -p "$TASK/persist_attempt01"
    AVAIL=$(df -B1 "$TASK" | awk 'NR==2 {print $4}'); SIZE=$(du -sb "$OUT" | awk '{print $1}')
    if [ -n "$AVAIL" ] && [ -n "$SIZE" ] && [ "$AVAIL" -gt "$((SIZE+50000000))" ]; then
      TMPARCH="$TASK/persist_attempt01/.run.tar.gz.partial"
      tar -czf "$TMPARCH" -C "$(dirname "$OUT")" "$(basename "$OUT")" && mv "$TMPARCH" "$TASK/persist_attempt01/run.tar.gz"
      ARC_RC=$?; printf '%s\n' "$ARC_RC" > "$CTRL/archive_exitcode"
      if [ "$ARC_RC" -eq 0 ]; then sha256sum "$TASK/persist_attempt01/run.tar.gz" > "$CTRL/persist_archive_sha256.txt"; wc -c < "$TASK/persist_attempt01/run.tar.gz" > "$CTRL/persist_archive_bytes.txt"; fi
    else echo PERSIST_ARCHIVE_SKIPPED_OUTPUT_REMAINS_TMPFS > "$CTRL/archive_exitcode"; fi
  fi
fi
printf '%s\n' "$RUN_RC" > "$CTRL/exitcode"
exit "$RUN_RC"
