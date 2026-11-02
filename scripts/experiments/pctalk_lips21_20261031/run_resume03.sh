#!/usr/bin/env bash
set -euo pipefail
B=/home/data2/wangjiajun/av-evaluation
D=/home/data2/wangjiajun/PCTalkDeploy
S=$B/experiments/pctalk_lips21_20261031
R=$B/jobs/pctalk_lips21_rms24_nine_20261031_attempt01
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-7272933b-1219-b5f8-a9dd-ceade9bbde37
export GPU_UUID=$CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TMPDIR=$D/tmp XDG_CACHE_HOME=$D/cache NUMBA_CACHE_DIR=$D/cache/numba MPLCONFIGDIR=$D/cache/matplotlib
export PATH=$B/tools/bin:$PATH
cd "$S"
trap 'echo "FAILED at line $LINENO"; touch "$R/orchestrator_failed"' ERR
test -f "$R/smoke_complete.json"
test ! -e "$R/orchestrator_resume03_started"
touch "$R/orchestrator_resume03_started"
cp ./*.py ./run.sh ./run_resume03.sh "$R/source_snapshot/"
cp "$R/scorer_runtime_comparison.json" "$R/scorer_runtime_before.json"
taskset -c 0-3 "$D/env/bin/python" -u generate_resource_wait03.py > "$R/logs/generate.log" 2>&1 &
echo $! > "$R/generate.pid"
wait "$(cat "$R/generate.pid")"
taskset -c 0-3 "$B/envs/eval-base/bin/python" -u syncnet.py > "$R/logs/syncnet_controller.log" 2>&1
taskset -c 0-3 "$B/envs/eval-base/bin/python" - <<'PY'
from prepare import RUN,write
import json,numpy as np
rows=json.loads((RUN/'scoring_inputs.json').read_text())['rows']
scores={r['id']:r for r in json.loads((RUN/'syncnet/scores.json').read_text())}
out={}
for name,ids in [('strict_quality17',sorted({r['source'] for r in rows if r['strict_quality']})),('all21',sorted({r['source'] for r in rows}))]:
 n=[scores[s+'_N']['C'] for s in ids];t=[scores[s+'_T']['C'] for s in ids]
 out[name]=dict(n=len(ids),N_mean=float(np.mean(n)),T_mean=float(np.mean(t)),delta=float(np.mean(np.array(t)-n)),T_higher=int(np.sum(np.array(t)>n)))
write(RUN/'syncnet_checkpoint.json',dict(status='PASS',subsets=out,note='Nine-endpoint adjusted inference pending; no author-cohort replication claim'))
eligible=[r['id'] for r in rows if min(r['samples']//640,r['actual_video_frames'])>=125]
write(RUN/'predeclared_support.json',dict(status='FROZEN_BEFORE_EIGHT_SCORERS',classifier_eligible_ids=eligible,
 face_support_threshold=.95,policy='One N and one T per complete speaker; no replacement/imputation or threshold relaxation'))
PY
taskset -c 0-3 "$B/envs/eval-base/bin/python" -u nine.py > "$R/logs/nine_controller.log" 2>&1
taskset -c 0-3 "$B/envs/eval-base/bin/python" audit.py --run "$R" > "$R/logs/audit.log" 2>&1
taskset -c 0-3 "$B/envs/eval-base/bin/python" verify_runtime.py > "$R/logs/runtime_after.log" 2>&1
cp "$R/scorer_runtime_comparison.json" "$R/scorer_runtime_after.json"
taskset -c 0-3 "$B/envs/eval-base/bin/python" finish.py > "$R/logs/package.log" 2>&1
touch "$R/orchestrator_complete"
echo ALL_COMPLETE
