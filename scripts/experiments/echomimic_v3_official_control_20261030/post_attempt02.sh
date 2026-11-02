set -euo pipefail
B=/home/data2/wangjiajun/av-evaluation
S="$B/experiments/echomimic_v3_official_control_20261030"
R="$B/jobs/echomimic_v3_official_control_20261030_attempt01"
export CUDA_VISIBLE_DEVICES=GPU-de7ff5ff-3510-739f-917c-cb98892de759 CUDA_DEVICE_ORDER=PCI_BUS_ID GPU_UUID=GPU-de7ff5ff-3510-739f-917c-cb98892de759
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HOME="$B/cache/huggingface" TORCH_HOME="$B/cache/torch" XDG_CACHE_HOME="$B/cache" TMPDIR="$B/tmp" MPLCONFIGDIR="$B/cache/matplotlib"
export PATH="$B/tools/bin:$PATH" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7
exec 9>"$R/score_repair.lock"
flock -n 9
read -r free util < <(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits | tr ',' ' ')
test "$free" -ge 24576
"$B/envs/eval-base/bin/python" "$S/score_repair.py" > "$R/logs/scoring_attempt02.log" 2>&1
"$B/envs/eval-base/bin/python" "$S/shift_control.py" > "$R/logs/offset_control_attempt02.log" 2>&1
"$B/envs/eval-base/bin/python" "$S/finalize.py" > "$S/package_attempt02.log" 2>&1
echo CONTROL_AND_ARCHIVE_COMPLETE
