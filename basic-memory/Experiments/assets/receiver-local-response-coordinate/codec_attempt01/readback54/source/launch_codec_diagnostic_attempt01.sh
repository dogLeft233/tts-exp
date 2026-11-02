#!/usr/bin/env bash
set -u
ROOT=/home/wjj/tts-audio/tts-exp
TASK="$ROOT/.codex/scratch/native_attribute_response_probe_20261015_codec_diagnostic_attempt01"
CONTROL="$TASK/control_attempt01"
OUTPUT="$TASK/output_attempt01"
OLD="$ROOT/.codex/scratch/native_attribute_response_probe_20261015_output_owned_musetalk_attempt01"
INPUT="$OLD/provenance/output_owned_musetalk_input_manifest_attempt01.json"
BASE="$OLD/readout_attempt02/readout_manifest.json"
PYTHON=/home/wjj/.venvs/w2l_syncnet_v100_attempt01/bin/python
RUNNER="$TASK/source/codec_diagnostic_attempt01.py"
PROTOCOL="$TASK/provenance/codec_diagnostic_protocol_attempt01.json"
HELPER="$ROOT/runs/cross_tfg_audio_head_20261009_v100_native_deployment_attempt01/code/scripts/audio_enhancement/overdub.py"
EVALUATOR="$ROOT/scripts/experiments/cross_tfg_audio_head_20261003/stagea_syncnet_evaluator.py"
SUMMARY="$OLD/source/output_owned_musetalk_readout_executor_attempt03.py"
SYNC_ROOT=/dev/shm/cross_tfg_native_axis_injection_pilot_20261013_attempt01/clock_contract_syncnet_root_attempt05
CHECKPOINT="$SYNC_ROOT/data/syncnet_v2.model"
FFMPEG=/usr/bin/ffmpeg
FFPROBE=/usr/bin/ffprobe
EXPECTED_RUNNER=4befcb7705790fd732aab7dead62cadd2ebc597d95845e0a851e2cb290d9cf29
EXPECTED_PROTOCOL=062d01aae34493ea82bb781672b7a6d6a18f7cf0fa3a2b549e8df82b3ab932ee
EXPECTED_INPUT=673ee09cdba130ee70698553f6bd52ca4b4633cf85c94e2092de66bbb36b4889
EXPECTED_BASE=1b4f454463a4387b470c744b0f9269b991e2c4e4fd0164132c40a1c87a655c4f
EXPECTED_HELPER=0febe6f2b43ed4fb089bdf36e2e208cc1cd61681d8e0ebfd7f59cd56dd00b4b6
EXPECTED_EVALUATOR=b15762e1837fd28eaba8fe9e261ffb35b164246da555cb3a7e3bbc7c645f818b
EXPECTED_SUMMARY=79bb2df7c3f0ccb8642ae778dce84856e887fce86035df6d0057766980e68f44
EXPECTED_SYNCNET=bd722bf844700d6a923dec2f10d7cacd65c0c6b22d09aa5d44d2509660e900d6
EXPECTED_CHECKPOINT=961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442
if [[ "${1-}" == "__worker" ]]; then
  cd "$ROOT" || exit 80
  started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  "$PYTHON" -B "$RUNNER" \
    --input-manifest "$INPUT" --baseline-readout-manifest "$BASE" --output-root "$OUTPUT" \
    --overdub-source "$HELPER" --stagea-evaluator "$EVALUATOR" --summary-source "$SUMMARY" \
    --syncnet-python "$PYTHON" --syncnet-root "$SYNC_ROOT" --checkpoint "$CHECKPOINT" \
    --ffmpeg "$FFMPEG" --ffprobe "$FFPROBE" \
    > "$CONTROL/worker.stdout.log" 2> "$CONTROL/worker.stderr.log" &
  worker_pid=$!
  worker_ticks=$(awk '{print $22}' "/proc/$worker_pid/stat" 2>/dev/null || printf 'unavailable')
  {
    printf 'schema=native_attribute_response_codec_diagnostic_worker_identity_v1\n'
    printf 'started_utc=%s\n' "$started_utc"
    printf 'worker_pid=%s\n' "$worker_pid"
    printf 'worker_start_ticks=%s\n' "$worker_ticks"
    printf 'supervisor_pid=%s\n' "$$"
    printf 'cwd=%s\n' "$ROOT"
    printf 'python=%s\n' "$PYTHON"
    printf 'argv=%s -B %s --input-manifest %s --baseline-readout-manifest %s --output-root %s --overdub-source %s --stagea-evaluator %s --summary-source %s --syncnet-python %s --syncnet-root %s --checkpoint %s --ffmpeg %s --ffprobe %s\n' "$PYTHON" "$RUNNER" "$INPUT" "$BASE" "$OUTPUT" "$HELPER" "$EVALUATOR" "$SUMMARY" "$PYTHON" "$SYNC_ROOT" "$CHECKPOINT" "$FFMPEG" "$FFPROBE"
    printf 'pythonpath_effective=%s\n' "${PYTHONPATH-}"
    printf 'execution_scope=six serial SyncNet scores only; FFV1 codec diagnostic; no TFG forward\n'
  } > "$CONTROL/worker_identity.txt.tmp"
  mv "$CONTROL/worker_identity.txt.tmp" "$CONTROL/worker_identity.txt"
  wait "$worker_pid"
  worker_rc=$?
  finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  {
    printf 'schema=native_attribute_response_codec_diagnostic_terminal_v1\n'
    printf 'status=TERMINAL\n'
    printf 'worker_pid=%s\n' "$worker_pid"
    printf 'worker_start_ticks=%s\n' "$worker_ticks"
    printf 'worker_returncode=%s\n' "$worker_rc"
    printf 'started_utc=%s\n' "$started_utc"
    printf 'finished_utc=%s\n' "$finished_utc"
    printf 'stdout=%s\n' "$CONTROL/worker.stdout.log"
    printf 'stderr=%s\n' "$CONTROL/worker.stderr.log"
  } > "$CONTROL/terminal_receipt.txt.tmp"
  mv "$CONTROL/terminal_receipt.txt.tmp" "$CONTROL/terminal_receipt.txt"
  exit "$worker_rc"
fi
if [[ "${1-}" != "--launch" ]]; then
  printf 'BLOCKED=explicit --launch required\n' >&2
  exit 88
fi
if [[ -e "$CONTROL" || -e "$OUTPUT" ]]; then
  printf 'ERROR=output_or_control_exists\n' >&2
  exit 91
fi
for item in "$RUNNER:$EXPECTED_RUNNER" "$PROTOCOL:$EXPECTED_PROTOCOL" "$INPUT:$EXPECTED_INPUT" "$BASE:$EXPECTED_BASE" "$HELPER:$EXPECTED_HELPER" "$EVALUATOR:$EXPECTED_EVALUATOR" "$SUMMARY:$EXPECTED_SUMMARY" "$SYNC_ROOT/SyncNetInstance.py:$EXPECTED_SYNCNET" "$CHECKPOINT:$EXPECTED_CHECKPOINT"; do
  file=${item%%:*}; expected=${item##*:}; actual=$(sha256sum "$file" | awk '{print $1}') || exit 92
  if [[ "$actual" != "$expected" ]]; then printf 'ERROR=pin_mismatch file=%s actual=%s expected=%s\n' "$file" "$actual" "$expected" >&2; exit 93; fi
done
apps=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader) || exit 94
if [[ -n "$apps" ]]; then printf 'ERROR=GPU_compute_owner_present %s\n' "$apps" >&2; exit 95; fi
ffmpeg -hide_banner -encoders 2>&1 | grep -E '[[:space:]]ffv1[[:space:]]' >/dev/null || exit 96
ffmpeg -hide_banner -decoders 2>&1 | grep -E '[[:space:]]ffv1[[:space:]]' >/dev/null || exit 97
mkdir "$CONTROL" || exit 98
cd "$ROOT" || exit 99
original_pythonpath="${PYTHONPATH-}"
if [[ -n "$original_pythonpath" ]]; then export PYTHONPATH="$ROOT:$original_pythonpath"; else export PYTHONPATH="$ROOT"; fi
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
launcher_pid=$$
launcher_ticks=$(awk '{print $22}' "/proc/$launcher_pid/stat" 2>/dev/null || printf 'unavailable')
launcher_sha=$(sha256sum "$0" | awk '{print $1}')
nohup bash "$0" __worker </dev/null > "$CONTROL/supervisor.log" 2>&1 &
supervisor_pid=$!
supervisor_ticks=$(awk '{print $22}' "/proc/$supervisor_pid/stat" 2>/dev/null || printf 'unavailable')
{
  printf 'schema=native_attribute_response_codec_diagnostic_launch_v1\n'
  printf 'started_utc=%s\n' "$started_utc"
  printf 'launcher_script_sha256=%s\n' "$launcher_sha"
  printf 'launcher_pid=%s\n' "$launcher_pid"
  printf 'launcher_start_ticks=%s\n' "$launcher_ticks"
  printf 'supervisor_pid=%s\n' "$supervisor_pid"
  printf 'supervisor_start_ticks=%s\n' "$supervisor_ticks"
  printf 'cwd=%s\n' "$ROOT"
  printf 'python=%s\n' "$PYTHON"
  printf 'runner_sha256=%s\n' "$EXPECTED_RUNNER"
  printf 'protocol_sha256=%s\n' "$EXPECTED_PROTOCOL"
  printf 'input_manifest_sha256=%s\n' "$EXPECTED_INPUT"
  printf 'formal_h264_manifest_sha256=%s\n' "$EXPECTED_BASE"
  printf 'pythonpath_original=%s\n' "$original_pythonpath"
  printf 'pythonpath_effective=%s\n' "$PYTHONPATH"
  printf 'pythonpath_policy=prepend repo root; preserve inherited value\n'
  printf 'output_root=%s\n' "$OUTPUT"
  printf 'control_root=%s\n' "$CONTROL"
  printf 'score_invocations_max=6; sequential=true\n'
} > "$CONTROL/launch_receipt.txt.tmp"
mv "$CONTROL/launch_receipt.txt.tmp" "$CONTROL/launch_receipt.txt"
for i in $(seq 1 30); do [[ -f "$CONTROL/worker_identity.txt" ]] && break; sleep 0.1; done
printf 'LAUNCH_ACCEPTED\n'
cat "$CONTROL/launch_receipt.txt"
if [[ -f "$CONTROL/worker_identity.txt" ]]; then cat "$CONTROL/worker_identity.txt"; else printf 'WORKER_IDENTITY_NOT_YET_WRITTEN\n'; fi