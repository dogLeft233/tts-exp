#!/usr/bin/env bash
set -euo pipefail
R=/home/data2/wangjiajun/DyStreamDeploy
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-GPU-3464c5ee-e2ed-81e4-e275-b612a929bfe1}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MAX_JOBS=4
export TMPDIR=$R/tmp XDG_CACHE_HOME=$R/cache
export HF_HOME=$R/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_HOME=$R/cache/torch TORCH_EXTENSIONS_DIR=$R/cache/torch-extensions
export NUMBA_CACHE_DIR=$R/cache/numba MPLCONFIGDIR=$R/cache/matplotlib
export PYTHONPYCACHEPREFIX=$R/cache/pycache PYTHONDONTWRITEBYTECODE=1
if [ -f /usr/share/glvnd/egl_vendor.d/50_mesa.json ]; then
 export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
 export LIBGL_ALWAYS_SOFTWARE=1
fi
export PYTHONUNBUFFERED=1 GRADIO_ANALYTICS_ENABLED=False WANDB_MODE=disabled
cd /home/data2/wangjiajun/DyStream
exec "$R/env/bin/python" "/home/data2/wangjiajun/av-evaluation/experiments/dystream_lips_full104_20261031/worker.py" "$@"
