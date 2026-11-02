#!/usr/bin/env bash
set -euo pipefail
B=/home/data2/wangjiajun/av-evaluation
D=/home/data2/wangjiajun/DyStreamDeploy
S=$B/experiments/dystream_lips_full104_20261031
R=$B/jobs/dystream_lips_full104_rms24_nine_20261031_attempt01
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-7272933b-1219-b5f8-a9dd-ceade9bbde37
export GPU_UUID=$CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TMPDIR=$D/tmp XDG_CACHE_HOME=$D/cache NUMBA_CACHE_DIR=$D/cache/numba MPLCONFIGDIR=$D/cache/matplotlib
export PATH=$B/tools/bin:$PATH
cd "$S"
trap 'echo "FAILED at line $LINENO"; touch "$R/orchestrator_failed"' ERR
test -f "$R/preparation.json"
test -f "$R/dependency_acceptance.json"
test -f "$R/recovery02_authorization.json"
test ! -e "$R/orchestrator_complete"
test ! -e "$R/resume02_started"
touch "$R/resume02_started"
mkdir -p "$R/logs" "$R/stages" "$R/source_snapshot_resume02"
cp ./*.py ./*.sh "$R/source_snapshot_resume02/"
taskset -c 0-3 "$B/envs/eval-base/bin/python" verify_runtime_resume02.py > "$R/logs/runtime_before.log" 2>&1
cp "$R/scorer_runtime_comparison.json" "$R/scorer_runtime_before.json"
"$B/envs/eval-base/bin/python" -c "from prepare import RUN,write; write(RUN/'status.json',dict(stage='generation',status='RUNNING'))"
taskset -c 0-3 "$D/env/bin/python" -u generate_native_export_resume02.py > "$R/logs/generate.log" 2>&1 &
echo $! > "$R/generate.pid"
wait "$(cat "$R/generate.pid")"
taskset -c 0-3 "$B/envs/eval-base/bin/python" -u syncnet.py > "$R/logs/syncnet_controller.log" 2>&1
taskset -c 0-3 "$B/envs/eval-base/bin/python" checkpoint.py > "$R/logs/checkpoint.log" 2>&1
taskset -c 0-3 "$B/envs/eval-base/bin/python" audit_syncnet_geometry.py > "$R/logs/syncnet_geometry.log" 2>&1
taskset -c 0-3 "$B/envs/eval-base/bin/python" -u nine.py > "$R/logs/nine_controller.log" 2>&1
taskset -c 0-3 "$B/envs/eval-base/bin/python" audit_native_export_resume02.py --run "$R" > "$R/logs/audit.log" 2>&1
taskset -c 0-3 "$B/envs/eval-base/bin/python" verify_statistics.py --run "$R" > "$R/logs/statistics_audit.log" 2>&1
taskset -c 0-3 "$B/envs/eval-base/bin/python" verify_runtime_resume02.py > "$R/logs/runtime_after.log" 2>&1
cp "$R/scorer_runtime_comparison.json" "$R/scorer_runtime_after.json"
taskset -c 0-3 "$B/envs/eval-base/bin/python" finish.py > "$R/logs/package.log" 2>&1
touch "$R/orchestrator_complete"
echo ALL_COMPLETE
