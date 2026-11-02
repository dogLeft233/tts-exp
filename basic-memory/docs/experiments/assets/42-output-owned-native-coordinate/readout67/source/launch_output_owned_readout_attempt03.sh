#!/usr/bin/env bash
set -u
ROOT=/home/wjj/tts-audio/tts-exp
TASK="$ROOT/.codex/scratch/native_attribute_response_probe_20261015_output_owned_musetalk_attempt01"
CONTROL="$TASK/control_attempt02"
OUTPUT="$TASK/readout_attempt02"
INPUT="$TASK/provenance/output_owned_musetalk_input_manifest_attempt01.json"
ADMISSION="$TASK/provenance/root_output_owned_readout_admission_attempt02.json"
PYTHON=/home/wjj/.venvs/w2l_syncnet_v100_attempt01/bin/python
EXECUTOR="$TASK/source/output_owned_musetalk_readout_executor_attempt03.py"
EXPECTED_EXEC=79bb2df7c3f0ccb8642ae778dce84856e887fce86035df6d0057766980e68f44
EXPECTED_INPUT=673ee09cdba130ee70698553f6bd52ca4b4633cf85c94e2092de66bbb36b4889
EXPECTED_ADMISSION=53608ef4d471995a1404e8f1af870096b619fcad83535c7177eb265686f1f2b7

if [[ "${1-}" == "__worker" ]]; then
  cd "$ROOT" || exit 80
  started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  original_pythonpath="${READOUT_ORIGINAL_PYTHONPATH-}"
  effective_pythonpath="${PYTHONPATH-}"
  "$PYTHON" -B "$EXECUTOR" --input-manifest "$INPUT" --output-root "$OUTPUT" --execute \
    > "$CONTROL/worker.stdout.log" 2> "$CONTROL/worker.stderr.log" &
  worker_pid=$!
  worker_ticks=$(awk '{print $22}' "/proc/$worker_pid/stat" 2>/dev/null || printf 'unavailable')
  {
    printf 'schema=output_owned_musetalk_readout_worker_identity_v1\n'
    printf 'started_utc=%s\n' "$started_utc"
    printf 'worker_pid=%s\n' "$worker_pid"
    printf 'worker_start_ticks=%s\n' "$worker_ticks"
    printf 'supervisor_pid=%s\n' "$$"
    printf 'cwd=%s\n' "$ROOT"
    printf 'python=%s\n' "$PYTHON"
    printf 'argv=%s -B %s --input-manifest %s --output-root %s --execute\n' "$PYTHON" "$EXECUTOR" "$INPUT" "$OUTPUT"
    printf 'pythonpath_original=%s\n' "$original_pythonpath"
    printf 'pythonpath_effective=%s\n' "$effective_pythonpath"
    printf 'syncnet_subprocess_env=inherited from worker\n'
  } > "$CONTROL/worker_identity.txt.tmp"
  mv "$CONTROL/worker_identity.txt.tmp" "$CONTROL/worker_identity.txt"
  wait "$worker_pid"
  worker_rc=$?
  finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  {
    printf 'schema=output_owned_musetalk_readout_terminal_v1\n'
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
  printf 'BLOCKED=explicit --launch required; this template is not a launch request\n' >&2
  exit 88
fi
if [[ "$EXPECTED_INPUT" == WAIT_* || "$EXPECTED_ADMISSION" == WAIT_* ]]; then
  printf 'BLOCKED=waiting_for_renderer_manifest_and_root_admission\n' >&2
  exit 90
fi
if [[ -e "$CONTROL" || -e "$OUTPUT" ]]; then
  printf 'ERROR=output_or_control_exists\n' >&2
  exit 91
fi
for item in "$EXECUTOR:$EXPECTED_EXEC" "$INPUT:$EXPECTED_INPUT" "$ADMISSION:$EXPECTED_ADMISSION"; do
  file=${item%%:*}
  expected=${item##*:}
  actual=$(sha256sum "$file" | awk '{print $1}') || exit 92
  if [[ "$actual" != "$expected" ]]; then
    printf 'ERROR=pin_mismatch file=%s actual=%s expected=%s\n' "$file" "$actual" "$expected" >&2
    exit 93
  fi
done
mkdir "$CONTROL" || exit 94
cd "$ROOT" || exit 95
original_pythonpath="${PYTHONPATH-}"
export READOUT_ORIGINAL_PYTHONPATH="$original_pythonpath"
if [[ -n "$original_pythonpath" ]]; then
  export PYTHONPATH="$ROOT:$original_pythonpath"
else
  export PYTHONPATH="$ROOT"
fi
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
launcher_pid=$$
launcher_ticks=$(awk '{print $22}' "/proc/$launcher_pid/stat" 2>/dev/null || printf 'unavailable')
launcher_sha=$(sha256sum "$0" | awk '{print $1}')
nohup bash "$0" __worker </dev/null > "$CONTROL/supervisor.log" 2>&1 &
supervisor_pid=$!
supervisor_ticks=$(awk '{print $22}' "/proc/$supervisor_pid/stat" 2>/dev/null || printf 'unavailable')
{
  printf 'schema=output_owned_musetalk_readout_launch_v1\n'
  printf 'started_utc=%s\n' "$started_utc"
  printf 'launcher_script_sha256=%s\n' "$launcher_sha"
  printf 'launcher_pid=%s\n' "$launcher_pid"
  printf 'launcher_start_ticks=%s\n' "$launcher_ticks"
  printf 'supervisor_pid=%s\n' "$supervisor_pid"
  printf 'supervisor_start_ticks=%s\n' "$supervisor_ticks"
  printf 'cwd=%s\n' "$ROOT"
  printf 'python=%s\n' "$PYTHON"
  printf 'argv=%s -B %s --input-manifest %s --output-root %s --execute\n' "$PYTHON" "$EXECUTOR" "$INPUT" "$OUTPUT"
  printf 'pythonpath_original=%s\n' "$original_pythonpath"
  printf 'pythonpath_effective=%s\n' "$PYTHONPATH"
  printf 'pythonpath_policy=prepend repo root; preserve inherited value\n'
  printf 'syncnet_subprocess_env=inherited from worker\n'
  printf 'admission_sha256=%s\n' "$EXPECTED_ADMISSION"
  printf 'executor_sha256=%s\n' "$EXPECTED_EXEC"
  printf 'input_manifest_sha256=%s\n' "$EXPECTED_INPUT"
  printf 'input_manifest=%s\n' "$INPUT"
  printf 'output_root=%s\n' "$OUTPUT"
} > "$CONTROL/launch_receipt.txt.tmp"
mv "$CONTROL/launch_receipt.txt.tmp" "$CONTROL/launch_receipt.txt"
for i in $(seq 1 30); do
  [[ -f "$CONTROL/worker_identity.txt" ]] && break
  sleep 0.1
done
printf 'LAUNCH_ACCEPTED\n'
cat "$CONTROL/launch_receipt.txt"
if [[ -f "$CONTROL/worker_identity.txt" ]]; then
  cat "$CONTROL/worker_identity.txt"
else
  printf 'WORKER_IDENTITY_NOT_YET_WRITTEN\n'
fi