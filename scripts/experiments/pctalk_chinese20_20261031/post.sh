set -euo pipefail
B=/home/data2/wangjiajun/av-evaluation
R=$B/jobs/pctalk_chinese20_rms24_20261031_attempt01
S=$B/experiments/pctalk_chinese20_20261031
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
cd "$S"
while ! test -f "$R/stages/syncnet_complete.json"; do
    if test -f "$R/failure.json" || test -f "$R/generation_failure.json"; then exit 1; fi
    sleep 20
done
cp "$S/analyze.py" "$S/package.py" "$R/source_snapshot/"
taskset -c 0-3 "$B/envs/eval-base/bin/python" analyze.py > "$R/logs/analysis.log" 2>&1
taskset -c 0-3 "$B/envs/eval-base/bin/python" package.py > "$R/logs/package.log" 2>&1
