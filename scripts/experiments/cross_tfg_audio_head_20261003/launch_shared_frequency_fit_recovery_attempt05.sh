#!/usr/bin/env bash
set -euo pipefail
NS=/dev/shm/cross_tfg_audio_head_20261003/shared_frequency_fit_recovery_attempt04
CODE="$NS/code"
CTRL="$NS/control"
RUN="$NS/run"
DITTO_CTRL="$NS/ditto_control"
PY=/root/autodl-tmp/wav2lip-syncnet/envs/wav2lip/bin/python
PROJECT=/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/shared_frequency_profiles_fit_attempt03
WRAPPER="$CODE/recover_shared_frequency_fit_attempt04.py"
CORE="$PROJECT/scripts/experiments/cross_tfg_audio_head_20261003/run_stagea_dev_attempt03.py"
ADAPTER="$PROJECT/scripts/experiments/cross_tfg_audio_head_20261003/run_cross_frontend_dev_attempt03.py"
PROFILE="$PROJECT/scripts/audio_enhancement/shared_frequency_profile_v1.py"
PLAN="$PROJECT/plan/plan.json"
ROWS="$PROJECT/plan/planned_rows.jsonl"
GPU_WRAPPER="$PROJECT/gpu_admission_wrapper_attempt05.py"
GPU_POLICY="$PROJECT/gpu_admission_policy_attempt06.json"
GPU_ENTRY="$PROJECT/scripts/experiments/cross_tfg_audio_head_20261003/run_cross_frontend_fit_profiles_attempt02.py"
OLD_RUN=/dev/shm/cross_tfg_audio_head_20261003/shared_frequency_profiles_fit_attempt03/run
EXPECTED_PLAN=1f0a290b612850b29b21233e4ffbfe5d0b48538341fdeeb853ce6cec945a99b7
EXPECTED_CORE=d12c863ffcd5eacde7b0d09b0abc097429d0318ccea3b059d022d5b2e52e2a94
EXPECTED_ADAPTER=7f78a14485eebaa20e7afd893a46b2ead26db5544194d988d7d9890d6a43ebbc
EXPECTED_PROFILE=fd7303f513fdbada80da50e5dd395e9441142f1658544765d7523f1618f55253
EXPECTED_WRAPPER=ac2c381d9238cbb4cd399478d561b54116757c2db2ff0b4b6186a1f9513f5e79
EXPECTED_GPU_WRAPPER=411be16642d1e3c31576ec837b887d0e09303cdf3f95a2f5bdd48e5c683e9ef9
EXPECTED_GPU_POLICY=88797076267eb5b7f08abc00c2543f7b6484f9597bf606992dff100728df62e1
EXPECTED_GPU_ENTRY=dacda9729db79ec96c73c19402866b78fb3d695b2560ca6f285ce4ce0c056660
EXPECTED_FFMPEG=36d94a605d612e4090d1b8aec889d0c0801c6eafb1593c90f5c0dfd2e2966a45
EXPECTED_PCM16=f634bad9c287a2f7f8cd8ffa62674bdd9e97dbf4b0f00dbf44725e4ca4ca36a0
sha() { sha256sum "$1" | awk '{print $1}'; }
fail() { printf '%s\n' "$1" >&2; exit 2; }
[[ -x "$PY" ]] || fail "missing pinned Wav2Lip Python"
[[ -f "$WRAPPER" ]] || fail "missing recovery wrapper"
[[ ! -e "$RUN" ]] || fail "recovery run path already exists"
[[ ! -e "$DITTO_CTRL" ]] || fail "Ditto control path already exists"
[[ "$(sha "$WRAPPER")" == "$EXPECTED_WRAPPER" ]] || fail "recovery wrapper SHA mismatch"
[[ "$(sha "$CORE")" == "$EXPECTED_CORE" ]] || fail "Stage-A core SHA mismatch"
[[ "$(sha "$ADAPTER")" == "$EXPECTED_ADAPTER" ]] || fail "adapter runner SHA mismatch"
[[ "$(sha "$PROFILE")" == "$EXPECTED_PROFILE" ]] || fail "shared profile source SHA mismatch"
[[ "$(sha "$PLAN")" == "$EXPECTED_PLAN" ]] || fail "frozen fit plan SHA mismatch"
[[ "$(sha "$GPU_WRAPPER")" == "$EXPECTED_GPU_WRAPPER" ]] || fail "frozen GPU admission wrapper SHA mismatch"
[[ "$(sha "$GPU_POLICY")" == "$EXPECTED_GPU_POLICY" ]] || fail "frozen GPU admission policy SHA mismatch"
[[ "$(sha "$GPU_ENTRY")" == "$EXPECTED_GPU_ENTRY" ]] || fail "frozen fit entrypoint SHA mismatch"
[[ "$(sha /usr/bin/ffmpeg)" == "$EXPECTED_FFMPEG" ]] || fail "FFmpeg SHA mismatch"
[[ -f "$OLD_RUN/results.jsonl" && -f "$OLD_RUN/result_manifest.json" ]] || fail "failed source ledger or manifest missing"
mkdir -p "$CTRL/tmp" "$CTRL/cache"
shm_free=$(df --output=avail -B1 /dev/shm | tail -n 1 | tr -d ' ')
root_free=$(df --output=avail -B1 / | tail -n 1 | tr -d ' ')
printf '{"phase":"PREFLIGHT","shm_free_bytes":%s,"root_free_bytes":%s,"plan_raw_sha256":"%s","wrapper_sha256":"%s"}\n' "$shm_free" "$root_free" "$(sha "$PLAN")" "$(sha "$WRAPPER")" > "$CTRL/preflight.json"
if (( shm_free < 1073741824 )); then fail "less than 1 GiB free in SHM for failed-row recovery"; fi
export TMPDIR="$CTRL/tmp" TMP="$CTRL/tmp" TEMP="$CTRL/tmp"
export XDG_CACHE_HOME="$CTRL/cache" MPLCONFIGDIR="$CTRL/cache/matplotlib"
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
cat > "$CTRL/worker.sh" <<'WORKER'
#!/usr/bin/env bash
set -uo pipefail
NS=/dev/shm/cross_tfg_audio_head_20261003/shared_frequency_fit_recovery_attempt04
CODE="$NS/code"
CTRL="$NS/control"
RUN="$NS/run"
DITTO_CTRL="$NS/ditto_control"
PY=/root/autodl-tmp/wav2lip-syncnet/envs/wav2lip/bin/python
PROJECT=/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/shared_frequency_profiles_fit_attempt03
WRAPPER="$CODE/recover_shared_frequency_fit_attempt04.py"
CORE="$PROJECT/scripts/experiments/cross_tfg_audio_head_20261003/run_stagea_dev_attempt03.py"
ADAPTER="$PROJECT/scripts/experiments/cross_tfg_audio_head_20261003/run_cross_frontend_dev_attempt03.py"
PROFILE="$PROJECT/scripts/audio_enhancement/shared_frequency_profile_v1.py"
PLAN="$PROJECT/plan/plan.json"
ROWS="$PROJECT/plan/planned_rows.jsonl"
GPU_WRAPPER="$PROJECT/gpu_admission_wrapper_attempt05.py"
GPU_POLICY="$PROJECT/gpu_admission_policy_attempt06.json"
GPU_ENTRY="$PROJECT/scripts/experiments/cross_tfg_audio_head_20261003/run_cross_frontend_fit_profiles_attempt02.py"
OLD_RUN=/dev/shm/cross_tfg_audio_head_20261003/shared_frequency_profiles_fit_attempt03/run
EXPECTED_PLAN=1f0a290b612850b29b21233e4ffbfe5d0b48538341fdeeb853ce6cec945a99b7
EXPECTED_CORE=d12c863ffcd5eacde7b0d09b0abc097429d0318ccea3b059d022d5b2e52e2a94
EXPECTED_WRAPPER=ac2c381d9238cbb4cd399478d561b54116757c2db2ff0b4b6186a1f9513f5e79
EXPECTED_GPU_WRAPPER=411be16642d1e3c31576ec837b887d0e09303cdf3f95a2f5bdd48e5c683e9ef9
EXPECTED_GPU_POLICY=88797076267eb5b7f08abc00c2543f7b6484f9597bf606992dff100728df62e1
EXPECTED_GPU_ENTRY=dacda9729db79ec96c73c19402866b78fb3d695b2560ca6f285ce4ce0c056660
EXPECTED_FFMPEG=36d94a605d612e4090d1b8aec889d0c0801c6eafb1593c90f5c0dfd2e2966a45
EXPECTED_PCM16=f634bad9c287a2f7f8cd8ffa62674bdd9e97dbf4b0f00dbf44725e4ca4ca36a0
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$PROJECT"
mkdir -p "$RUN"
printf '%s\n' "$$" > "$CTRL/worker.pid"
statline=$(cat "/proc/$$/stat")
rest=${statline##*) }
set -- $rest
printf '%s\n' "${20}" > "$CTRL/worker_start_ticks"
printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$CTRL/worker_started_utc"
"$PY" -B "$GPU_WRAPPER" \
  --gate-wrapper-sha256 "$EXPECTED_GPU_WRAPPER" \
  --gate-entry "$WRAPPER" --gate-entry-sha256 "$EXPECTED_WRAPPER" \
  --gate-plan "$PLAN" --gate-plan-sha256 "$EXPECTED_PLAN" \
  --gate-gpu-policy "$GPU_POLICY" --gate-gpu-policy-sha256 "$EXPECTED_GPU_POLICY" \
  --gate-gpu-event-log "$CTRL/gpu_admission_events.jsonl" \
  --gate-core-path "$CORE" --gate-core-sha256 "$EXPECTED_CORE" \
  --gate-generator-child-path "$PROJECT/scripts/experiments/cross_tfg_audio_head_20261003/engineering_qc_oldpilot_attempt04.py" \
  --gate-generator-child-sha256 1c04011b79b38a4cff7a0235bc960acc12ee3f9aafc333bbf1633ab25eb3a29d \
  --gate-syncnet-root /root/autodl-tmp/wav2lip-syncnet/syncnet_python \
  --gate-syncnet-python /root/autodl-tmp/wav2lip-syncnet/envs/syncnet/bin/python \
  --gate-syncnet-evaluator "$PROJECT/scripts/experiments/cross_tfg_audio_head_20261003/stagea_syncnet_evaluator.py" \
  --gate-syncnet-evaluator-sha256 b15762e1837fd28eaba8fe9e261ffb35b164246da555cb3a7e3bbc7c645f818b \
  --stagea-core "$CORE" --adapter-runner "$ADAPTER" --profile-source "$PROFILE" \
  --reuse-run "$OLD_RUN" --ditto-control-dir "$DITTO_CTRL" \
  --plan "$PLAN" --planned-rows "$ROWS" --expected-plan-raw-sha256 "$EXPECTED_PLAN" \
  --output-root "$RUN" --project-root "$PROJECT" \
  --wav2lip-python "$PY" --wav2lip-root /root/autodl-tmp/wav2lip-syncnet/Wav2Lip \
  --generator-child "$PROJECT/scripts/experiments/cross_tfg_audio_head_20261003/engineering_qc_oldpilot_attempt04.py" \
  --syncnet-root /root/autodl-tmp/wav2lip-syncnet/syncnet_python \
  --syncnet-python /root/autodl-tmp/wav2lip-syncnet/envs/syncnet/bin/python \
  --syncnet-checkpoint /root/autodl-tmp/wav2lip-syncnet/syncnet_python/data/syncnet_v2.model \
  --syncnet-evaluator "$PROJECT/scripts/experiments/cross_tfg_audio_head_20261003/stagea_syncnet_evaluator.py" \
  --ffmpeg /usr/bin/ffmpeg --ffprobe /usr/bin/ffprobe \
  --ffmpeg-sha256 "$EXPECTED_FFMPEG" --pcm16-lineage-sha256 "$EXPECTED_PCM16" \
  --generator-timeout 900 --syncnet-timeout 1800 --ffmpeg-timeout 300 \
  > "$CTRL/worker.stdout.log" 2> "$CTRL/worker.stderr.log"
rc=$?
printf '%s\n' "$rc" > "$CTRL/worker.exitcode.tmp"
mv "$CTRL/worker.exitcode.tmp" "$CTRL/worker.exitcode"
printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$CTRL/worker_finished_utc"
exit "$rc"
WORKER
chmod 700 "$CTRL/worker.sh"
nohup /bin/bash "$CTRL/worker.sh" </dev/null >/dev/null 2>&1 &
launcher_pid=$!
printf '%s\n' "$launcher_pid" > "$CTRL/launcher.pid"
printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$CTRL/launcher_started_utc"
sleep 2
if [[ ! -s "$CTRL/worker.pid" || ! -s "$CTRL/worker_start_ticks" ]]; then
  fail "worker failed to publish its PID/start ticks"
fi
printf '{"status":"STARTED","launcher_pid":%s,"worker_pid":%s,"worker_start_ticks":"%s","run":"%s","control":"%s"}\n' \
 "$launcher_pid" "$(cat "$CTRL/worker.pid")" "$(cat "$CTRL/worker_start_ticks")" "$RUN" "$CTRL"
