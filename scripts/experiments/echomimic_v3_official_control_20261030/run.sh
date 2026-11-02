set -euo pipefail
D=/home/data2/wangjiajun/EchoMimicV3-Flash
B=/home/data2/wangjiajun/av-evaluation
S="$B/experiments/echomimic_v3_official_control_20261030"
R="$B/jobs/echomimic_v3_official_control_20261030_attempt01"
export CUDA_VISIBLE_DEVICES=GPU-de7ff5ff-3510-739f-917c-cb98892de759 CUDA_DEVICE_ORDER=PCI_BUS_ID GPU_UUID=GPU-de7ff5ff-3510-739f-917c-cb98892de759
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HOME="$D/cache/huggingface" TORCH_HOME="$D/cache/torch" XDG_CACHE_HOME="$D/cache" TMPDIR="$D/tmp" MPLCONFIGDIR="$D/cache/matplotlib"
export PATH="$B/tools/bin:$PATH" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
test ! -e "$R"
"$D/env/bin/python" "$S/control.py" prepare
mkdir "$R/logs"
cp "$S/control.py" "$S/run.sh" "$R/"
sha256sum "$S/control.py" "$S/run.sh" > "$R/source_sha256.txt"
function resource_gate() {
 while true; do
  read -r free util < <(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits | tr ',' ' ')
  if [ "$free" -ge 24576 ] && [ "$util" -le 85 ]; then break; fi
  echo "WAITING_GPU free=$free util=$util"
  sleep 30
 done
}
resource_gate
"$D/env/bin/python" "$S/control.py" generation > "$R/logs/generation.log" 2>&1 &
gen_pid=$!
echo "$gen_pid" > "$R/generation.pid"
(
 while kill -0 "$gen_pid" 2>/dev/null; do
  date -u +%FT%TZ
  nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader,nounits | awk -F, -v pid="$gen_pid" '$1+0==pid'
  sleep 5
 done
) > "$R/logs/own_generator_nvidia.csv" &
monitor_pid=$!
wait "$gen_pid"
wait "$monitor_pid"
resource_gate
export HF_HOME="$B/cache/huggingface" TORCH_HOME="$B/cache/torch" XDG_CACHE_HOME="$B/cache" TMPDIR="$B/tmp" LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7
"$B/envs/eval-base/bin/python" "$S/control.py" score > "$R/logs/scoring.log" 2>&1
"$B/envs/eval-base/bin/python" "$S/control.py" analyze > "$R/logs/analysis.log" 2>&1
echo 'OFFICIAL_CONTROL_COMPLETE'
