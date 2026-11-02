#!/usr/bin/env bash
set -euo pipefail

NS=/root/tts-research-system-20261001/cross_tfg_audio_head_20261003
TASK=$NS/stageb_musetalk_attempt03
CODE=$TASK/code
PLAN_DIR=$TASK/plan
CTRL=$NS/job_control/stageb_musetalk_attempt03_attempt01
OUT=/dev/shm/cross_tfg_audio_head_20261003/stageb_musetalk_attempt03/run
BUNDLE=$TASK/musetalk_stageb_attempt03_bundle.zip
UNPACK=$TASK/.unpack_attempt01
ORCHPY=/root/autodl-tmp/wav2lip-syncnet/envs/syncnet/bin/python
GENPY=/root/autodl-tmp/wav2lip-syncnet/envs/wav2lip/bin/python
SCOREPY=/root/autodl-tmp/wav2lip-syncnet/envs/syncnet/bin/python
W2L=/root/autodl-tmp/wav2lip-syncnet/Wav2Lip
SYNC=/root/autodl-tmp/wav2lip-syncnet/syncnet_python
SCOREDIR=$SYNC
FACE=/root/autodl-tmp/tts-research-20260930/portable_source_axis_20260930/inputs/primary_face.png
SOURCE_VIDEO=/dev/shm/cross_tfg_audio_head_20261003/stage_a_plan_attempt01/source/static_portrait_reference_25fps.mp4
MUSE_SOURCE=$NS/model_sources_attempt02/MuseTalk
MUSE_MODEL=/dev/shm/cross_tfg_audio_head_20261003/model_deploy_attempt16/output/musetalk_workspace
MUSE_CODE=$NS/model_deploy_attempt16/code
BASE_OVERLAY=$NS/model_deploy_attempt09/code/overlay_python310_attempt02
FUTURE_OVERLAY=$NS/model_deploy_attempt10/code/future_overlay_attempt02
PLAN_SHA=926fa4ffd908e42e9e75b1562443f44964adafbb9db4af8e4850ab5e7e04c5c3
PLAN_PAYLOAD_SHA=4c6b6c63d13f6e6b58895106aa7e22ca6dccdc821ef6a3f88797976794286c2d
ROWS_SHA=9c720da9a353227385355b6e713592c1af01d3b3524f84083aeba3cba7835b33
PACKET_SHA=5f6bbff95f163b04aea57b4de31c80d34aa79e8779f48fc0914263bc873fb0a1
BUNDLE_SHA=62f936087557f34595036c115e4edf728254400f0b1058a003110d3fb8de6176
BUNDLE_MANIFEST_SHA=295a911c4d3e33b071acd459c62598ce17fbc87d00cb811d5d5cccd9f696a839
SHIM_SHA=f5be56d59d44665d69ab0a5b18bdddb3d58a1efbffeafcea14ff2f4b56d585ba
FRONTEND_SHA=7f78a14485eebaa20e7afd893a46b2ead26db5544194d988d7d9890d6a43ebbc
CORE_SHA=30c894ea31541cdde76a5a915bf635a36838a1cdb11f761c334c8986e71a01bc
PCM16_LINEAGE_SHA=f634bad9c287a2f7f8cd8ffa62674bdd9e97dbf4b0f00dbf44725e4ca4ca36a0
SYNCNET_EVALUATOR_SHA=b15762e1837fd28eaba8fe9e261ffb35b164246da555cb3a7e3bbc7c645f818b
HEAD_SHA=286d085674fe1eb10fb4fff51559bc9923a947327d971a9ab67738a2d1b9cecc
W2L_HEAD=bac9a81e63ecc153202353372e5724b83d9e6322
SYNC_HEAD=907c0b579c2e2d83f0eae1b2ac9e720cde4e5623
W2L_CHECKPOINT_SHA=ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8
SYNC_CHECKPOINT_SHA=961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442
FFMPEG_SHA=36d94a605d612e4090d1b8aec889d0c0801c6eafb1593c90f5c0dfd2e2966a45
FFPROBE_SHA=d4f3ef9c12be756793cad83dd2004d89f49c1c4094053bfbbe7e28925c8fa4fd
FACE_SHA=e3a695edf22a89cac40184453131019874a65e15592593d5be30e7f71373d9c6
SOURCE_VIDEO_SHA=c76617141188acfea5f4c122513b982997ec45ebaa35a2b6ed9f67101e9c40de
MUSE_COMMIT=0a89dec45a0192b824e3cf4daf96c239440c5ed8
MUSE_MODEL_REV=2bcb936e2fddb4d86db4c62fd45b387d0c061571
MUSE_REPORT=/dev/shm/cross_tfg_audio_head_20261003/model_deploy_attempt16/output/musetalk_portrait_silent.native_report.json
MUSE_REPORT_SHA=3004566cff8ce3736fcb393f14e804e5aec2d7ec529f0df8bb5cfb594ca948d0

mkdir -p "$CTRL"
if [ -e "$CTRL/worker.pid" ] || [ -e "$CTRL/exitcode" ] || [ -e "$CTRL/started_utc.txt" ]; then
  printf '%s\n' CONTROL_ATTEMPT_ALREADY_USED > "$CTRL/launch_error.txt"
  exit 97
fi
printf '%s\n' "$$" > "$CTRL/worker.pid"
date -u +%Y-%m-%dT%H:%M:%SZ > "$CTRL/started_utc.txt"
RUN_RC=99
finish() {
  rc=$?
  if [ ! -e "$CTRL/exitcode" ]; then printf '%s\n' "$rc" > "$CTRL/exitcode"; fi
  if [ ! -e "$CTRL/finished_utc.txt" ]; then date -u +%Y-%m-%dT%H:%M:%SZ > "$CTRL/finished_utc.txt"; fi
}
trap finish EXIT

fail() {
  code=$1
  shift
  printf '%s\n' "$*" > "$CTRL/launch_error.txt"
  printf '%s\n' "$code" > "$CTRL/exitcode"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$CTRL/finished_utc.txt"
  exit "$code"
}

test -f "$BUNDLE" || fail 91 BUNDLE_MISSING
test "$(sha256sum "$BUNDLE" | awk '{print $1}')" = "$BUNDLE_SHA" || fail 92 BUNDLE_SHA_MISMATCH
unzip -tq "$BUNDLE" > "$CTRL/bundle_test.log" 2>&1 || fail 93 BUNDLE_ZIP_INTEGRITY_FAILED
test ! -e "$CODE" || fail 94 CODE_DIRECTORY_ALREADY_EXISTS
test ! -e "$PLAN_DIR" || fail 95 PLAN_DIRECTORY_ALREADY_EXISTS
test ! -e "$UNPACK" || fail 96 UNPACK_DIRECTORY_ALREADY_EXISTS
mkdir "$UNPACK"
unzip -q "$BUNDLE" -d "$UNPACK" || fail 93 BUNDLE_EXTRACTION_FAILED
test "$(sha256sum "$UNPACK/bundle_manifest.json" | awk '{print $1}')" = "$BUNDLE_MANIFEST_SHA" || fail 92 BUNDLE_MANIFEST_SHA_MISMATCH
mkdir -p "$(dirname "$CODE")" "$(dirname "$PLAN_DIR")"
mv "$UNPACK/code" "$CODE"
mv "$UNPACK/plan" "$PLAN_DIR"
rmdir "$UNPACK"

test "$(sha256sum "$PLAN_DIR/plan.json" | awk '{print $1}')" = "$PLAN_SHA" || fail 92 PLAN_RAW_SHA_MISMATCH
test "$(sha256sum "$PLAN_DIR/planned_rows.jsonl" | awk '{print $1}')" = "$ROWS_SHA" || fail 92 PLANNED_ROWS_SHA_MISMATCH
test "$(sha256sum "$PLAN_DIR/packet_manifest.json" | awk '{print $1}')" = "$PACKET_SHA" || fail 92 PACKET_MANIFEST_SHA_MISMATCH
test "$(sha256sum "$CODE/musetalk_five_arg_shim_attempt03.py" | awk '{print $1}')" = "$SHIM_SHA" || fail 92 SHIM_SHA_MISMATCH
test "$(sha256sum "$CODE/run_cross_frontend_dev_attempt03.py" | awk '{print $1}')" = "$FRONTEND_SHA" || fail 92 FRONTEND_RUNNER_SHA_MISMATCH
test "$(sha256sum "$CODE/scripts/experiments/cross_tfg_audio_head_20261003/run_stagea_dev_attempt02.py" | awk '{print $1}')" = "$CORE_SHA" || fail 92 STAGEA_CORE_SHA_MISMATCH
test "$(sha256sum "$CODE/scripts/experiments/cross_tfg_audio_head_20261003/stagea_pcm16_lineage.py" | awk '{print $1}')" = "$PCM16_LINEAGE_SHA" || fail 92 PCM16_LINEAGE_SHA_MISMATCH
test "$(sha256sum "$CODE/scripts/experiments/cross_tfg_audio_head_20261003/stagea_syncnet_evaluator.py" | awk '{print $1}')" = "$SYNCNET_EVALUATOR_SHA" || fail 92 SYNCNET_EVALUATOR_SHA_MISMATCH
test ! -e "$OUT" || fail 98 OUTPUT_ROOT_ALREADY_EXISTS
test "$(git -C "$W2L" rev-parse HEAD)" = "$W2L_HEAD" || fail 96 WAV2LIP_SOURCE_HEAD_MISMATCH
test "$(git -C "$SYNC" rev-parse HEAD)" = "$SYNC_HEAD" || fail 96 SYNCNET_SOURCE_HEAD_MISMATCH
test "$(sha256sum "$W2L/checkpoints/wav2lip_gan.pth" | awk '{print $1}')" = "$W2L_CHECKPOINT_SHA" || fail 96 WAV2LIP_CHECKPOINT_MISMATCH
test "$(sha256sum "$SYNC/data/syncnet_v2.model" | awk '{print $1}')" = "$SYNC_CHECKPOINT_SHA" || fail 96 SYNCNET_CHECKPOINT_MISMATCH
test "$(sha256sum "$FACE" | awk '{print $1}')" = "$FACE_SHA" || fail 96 PRIMARY_FACE_SHA_MISMATCH
test "$(sha256sum "$SOURCE_VIDEO" | awk '{print $1}')" = "$SOURCE_VIDEO_SHA" || fail 96 SOURCE_PORTRAIT_VIDEO_SHA_MISMATCH
test "$(sha256sum /usr/bin/ffmpeg | awk '{print $1}')" = "$FFMPEG_SHA" || fail 96 FFMPEG_SHA_MISMATCH
test "$(sha256sum /usr/bin/ffprobe | awk '{print $1}')" = "$FFPROBE_SHA" || fail 96 FFPROBE_SHA_MISMATCH
test "$(git -C "$MUSE_SOURCE" rev-parse HEAD)" = "$MUSE_COMMIT" || fail 96 MUSETALK_SOURCE_COMMIT_MISMATCH
test "$(sha256sum "$MUSE_REPORT" | awk '{print $1}')" = "$MUSE_REPORT_SHA" || fail 96 MUSETALK_QUALIFICATION_REPORT_SHA_MISMATCH
test -d "$MUSE_MODEL" && test -d "$MUSE_CODE" && test -d "$BASE_OVERLAY" && test -d "$FUTURE_OVERLAY" || fail 96 MUSETALK_RUNTIME_ROOT_MISSING
test "$("$ORCHPY" -B -c 'import json,sys; p=json.load(open(sys.argv[1])); print(p["plan_sha256"])' "$PLAN_DIR/plan.json")" = "$PLAN_PAYLOAD_SHA" || fail 92 PLAN_PAYLOAD_SHA_MISMATCH

GPU_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
test -n "$GPU_FREE" || fail 90 GPU_MEMORY_QUERY_FAILED
test "$GPU_FREE" -ge 12000 || fail 90 GPU_FREE_MEMORY_BELOW_12G
printf 'gpu_free_mib=%s\n' "$GPU_FREE" > "$CTRL/gpu_preflight.txt"

export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
if ! "$ORCHPY" -B -c 'import torch,numpy,cv2,scripts.audio_enhancement; from scripts.audio_enhancement.heads import SpectralResidualHead; torch.set_num_threads(1); assert str(torch.__version__) == "2.5.1+cu124" and torch.cuda.is_available(); print("orchestrator",torch.__version__,numpy.__version__,torch.cuda.is_available())' > "$CTRL/orchestrator_preflight.log" 2>&1; then
  fail 90 ORCHESTRATOR_RUNTIME_PREFLIGHT_FAILED
fi
if ! env PYTHONPATH="$BASE_OVERLAY:$FUTURE_OVERLAY:$MUSE_CODE" "$GENPY" -B -c 'import torch,numpy,cv2,soundfile; assert str(torch.__version__) == "2.5.1+cu124" and torch.cuda.is_available(); print("musetalk_runtime",torch.__version__,numpy.__version__,torch.cuda.is_available())' > "$CTRL/musetalk_preflight.log" 2>&1; then
  fail 90 MUSETALK_RUNTIME_PREFLIGHT_FAILED
fi
if ! "$SCOREPY" -B -c 'import torch,numpy,scipy,cv2,python_speech_features; torch.set_num_threads(1); assert str(torch.__version__) == "2.5.1+cu124"; print("syncnet_runtime",torch.__version__,numpy.__version__,torch.get_num_threads())' > "$CTRL/syncnet_preflight.log" 2>&1; then
  fail 90 SYNCNET_RUNTIME_PREFLIGHT_FAILED
fi

set +e
"$ORCHPY" -B "$CODE/run_cross_frontend_dev_attempt03.py" \
  --stagea-core "$CODE/scripts/experiments/cross_tfg_audio_head_20261003/run_stagea_dev_attempt02.py" \
  --expected-stagea-core-sha256 "$CORE_SHA" \
  --plan "$PLAN_DIR/plan.json" --planned-rows "$PLAN_DIR/planned_rows.jsonl" \
  --expected-plan-raw-sha256 "$PLAN_SHA" --output-root "$OUT" --project-root "$CODE" \
  --wav2lip-python "$GENPY" --wav2lip-root "$W2L" \
  --generator-child "$CODE/musetalk_five_arg_shim_attempt03.py" \
  --syncnet-root "$SYNC" --syncnet-python "$SCOREPY" --syncnet-checkpoint "$SYNC/data/syncnet_v2.model" \
  --syncnet-evaluator "$CODE/scripts/experiments/cross_tfg_audio_head_20261003/stagea_syncnet_evaluator.py" \
  --ffmpeg /usr/bin/ffmpeg --ffprobe /usr/bin/ffprobe --ffmpeg-sha256 "$FFMPEG_SHA" \
  --pcm16-lineage-sha256 "$PCM16_LINEAGE_SHA" \
  --generator-timeout 1800 --syncnet-timeout 1800 --ffmpeg-timeout 300 \
  > "$CTRL/stageb.stdout" 2> "$CTRL/stageb.stderr"
RUN_RC=$?
set -e
printf '%s\n' "$RUN_RC" > "$CTRL/stageb_exitcode"
date -u +%Y-%m-%dT%H:%M:%SZ > "$CTRL/runner_finished_utc.txt"

if [ -d "$OUT" ]; then
  if "$ORCHPY" -B - "$OUT" "$CTRL/jpeg_cleanup.json" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); expected=Path('/dev/shm/cross_tfg_audio_head_20261003/stageb_musetalk_attempt03/run')
if root.resolve()!=expected: raise SystemExit('cleanup refused unexpected output root')
deleted=deleted_bytes=0
for p in root.rglob('*.jpg'):
    n=p.stat().st_size; p.unlink(); deleted+=1; deleted_bytes+=n
Path(sys.argv[2]).write_text(json.dumps({'scope':str(root),'deleted_jpeg_count':deleted,'deleted_jpeg_bytes':deleted_bytes,'kept_primary_audio_and_track_artifacts':True},sort_keys=True,indent=2)+'\n',encoding='utf-8')
PY
  then
    CLEAN_RC=0
  else
    CLEAN_RC=$?
  fi
  printf '%s\n' "$CLEAN_RC" > "$CTRL/jpeg_cleanup_exitcode"
  if [ "$CLEAN_RC" -eq 0 ]; then
    mkdir -p "$TASK/persist_attempt01"
    AVAIL=$(df -B1 "$TASK" | awk 'NR==2 {print $4}')
    SIZE=$(du -sb "$OUT" | awk '{print $1}')
    if [ -n "$AVAIL" ] && [ -n "$SIZE" ] && [ "$AVAIL" -gt "$((SIZE + 50000000))" ]; then
      TMPARCH=$TASK/persist_attempt01/.run.tar.gz.partial
      if tar -czf "$TMPARCH" -C "$(dirname "$OUT")" "$(basename "$OUT")"; then
        mv "$TMPARCH" "$TASK/persist_attempt01/run.tar.gz"
        sha256sum "$TASK/persist_attempt01/run.tar.gz" > "$CTRL/persist_archive_sha256.txt"
        wc -c < "$TASK/persist_attempt01/run.tar.gz" > "$CTRL/persist_archive_bytes.txt"
        printf '%s\n' 0 > "$CTRL/archive_exitcode"
      else
        TAR_RC=$?
        printf '%s\n' "$TAR_RC" > "$CTRL/archive_exitcode"
        rm -f "$TMPARCH"
      fi
    else
      printf '%s\n' ARCHIVE_SKIPPED_INSUFFICIENT_PERSIST_SPACE_OUTPUT_REMAINS_IN_TMPFS > "$CTRL/archive_exitcode"
    fi
  fi
fi

printf '%s\n' "$RUN_RC" > "$CTRL/exitcode"
date -u +%Y-%m-%dT%H:%M:%SZ > "$CTRL/finished_utc.txt"
trap - EXIT
exit "$RUN_RC"
