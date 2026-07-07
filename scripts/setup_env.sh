# ============================================================
# setup_env.sh - Idempotent environment setup for tts-exp
# Strictly per official docs:
#   - ditto.md (user notes)
#   - https://github.com/antgroup/ditto-talkinghead (README, environment.yaml)
#   - https://github.com/andimarafioti/faster-qwen3-tts (README)
#   - https://github.com/joonson/syncnet_python (README)
# Acceptance: issue #1
# ============================================================

set -uo pipefail

# ------------------------------------------------------------
# Paths (deterministic; /root/autodl-tmp persists across instance restarts)
# ------------------------------------------------------------
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
AUTODL_TMP="/root/autodl-tmp"
ENVS_DIR="$AUTODL_TMP/envs"
THIRD_PARTY="$REPO_DIR/third_party"
CHECKPOINTS_DIR="$AUTODL_TMP/checkpoints"
DOWNLOADS_DIR="$AUTODL_TMP/downloads"

mkdir -p "$ENVS_DIR" "$THIRD_PARTY" "$CHECKPOINTS_DIR" "$DOWNLOADS_DIR"

# ------------------------------------------------------------
# Activate conda base shell (for conda CLI)
# ------------------------------------------------------------
source /root/miniconda3/etc/profile.d/conda.sh

# ------------------------------------------------------------
# HF mirror (China-friendly; see Q18 of plan)
# ------------------------------------------------------------
export HF_ENDPOINT="https://hf-mirror.com"

# ------------------------------------------------------------
# Clone third-party repos (idempotent)
# Per Q19 of plan: clone original repos as-is, scripts wrap them.
# ------------------------------------------------------------
clone_if_missing() {
  local url="$1" path="$2"
  if [ -d "$path/.git" ]; then
    echo "[skip] already cloned: $path"
  else
    echo "[clone] $url -> $path"
    git clone --depth 1 "$url" "$path"
  fi
}

clone_if_missing https://github.com/antgroup/ditto-talkinghead "$THIRD_PARTY/ditto-talkinghead"
clone_if_missing https://github.com/joonson/syncnet_python "$THIRD_PARTY/syncnet_python"

# ------------------------------------------------------------
# Create ditto env from ditto's official environment.yaml (idempotent)
# Strictly per https://github.com/antgroup/ditto-talkinghead README:
#   conda env create -f environment.yaml
# This installs python 3.10, torch 2.5.1 cu121, tensorrt 8.6.1,
# librosa, scikit-image, opencv-python-headless, polygraphy, colored, numpy 2.0.1 etc.
# ------------------------------------------------------------
DITTO_ENV="$ENVS_DIR/ditto"
if conda env list | awk '{print $1}' | grep -qx "$DITTO_ENV"; then
  echo "[skip] ditto env exists at $DITTO_ENV"
else
  echo "[env] creating ditto env from ditto-talkinghead/environment.yaml"
  conda env create -f "$THIRD_PARTY/ditto-talkinghead/environment.yaml" -p "$DITTO_ENV"
fi

conda activate "$DITTO_ENV"

# Free system disk: remove conda cache after env create (idempotent env skips this on reruns)
conda clean -a -y > /dev/null 2>&1 || true

# ------------------------------------------------------------
# Extras per ditto.md (NOT in ditto's environment.yaml)
# ditto.md prescribes (NVIDIA index needed for tensorrt-libs):
#   python -m pip install tensorrt==8.6.1 tensorrt-bindings==8.6.1
#   python -m pip install tensorrt-libs==8.6.1 --extra-index-url https://pypi.nvidia.com
# The aliyun pip mirror doesn't carry tensorrt, so explicit install here.
# ------------------------------------------------------------
echo "[pip] tensorrt==8.6.1 + bindings + libs (per ditto.md)"
pip install tensorrt==8.6.1 tensorrt-bindings==8.6.1 --extra-index-url https://pypi.nvidia.com
pip install tensorrt-libs==8.6.1 --extra-index-url https://pypi.nvidia.com

echo "[pip] onnxruntime-gpu mediapipe einops (per ditto.md)"
pip install onnxruntime-gpu mediapipe einops

# Ditto inference deps (from antgroup/ditto-talkinghead README pip section,
# not fully captured by environment.yaml via aliyun mirror)
pip install --quiet filetype imageio opencv-python-headless scikit-image cython cuda-python==12.6.2.post1 imageio-ffmpeg colored polygraphy

# ------------------------------------------------------------
# faster-qwen3-tts (PyTorch 2.5.1 already in ditto env satisfies requirement)
# Per https://github.com/andimarafioti/faster-qwen3-tts README: pip install faster-qwen3-tts
# ------------------------------------------------------------
echo "[pip] installing faster-qwen3-tts"
pip install faster-qwen3-tts

# Pin huggingface_hub to the range faster-qwen3-tts/transformers expect.
# A prior `pip install -U "huggingface_hub[cli]"` may have dragged it to v1.x
# (breaking faster-qwen3-tts). Force the compatible range.
pip install --quiet "huggingface_hub[cli]>=0.36,<1.0"

# Convenience deps for orchestrating scripts
pip install --quiet python-dotenv pyyaml requests

# ------------------------------------------------------------
# cuDNN 8 for TensorRT mode (per ditto.md)
# TRT engine in ditto_trt_Ampere_Plus expects cuDNN 8 (libcudnn.so.8)
# ditto.md prescribes: extract cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz
#   into $CONDA_PREFIX/opt/cudnn8, set LD_LIBRARY_PATH
# ------------------------------------------------------------
CUDNN8_DIR="$DITTO_ENV/opt/cudnn8"
TARBALL_NAME="cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz"
TARBALL_PATH="$DOWNLOADS_DIR/$TARBALL_NAME"

if [ -f "$CUDNN8_DIR/lib/libcudnn.so.8" ]; then
  echo "[skip] cuDNN 8 already extracted to $CUDNN8_DIR"
elif [ -f "$TARBALL_PATH" ]; then
  echo "[cudnn] extracting $TARBALL_NAME"
  tmp_extract="$(mktemp -d)"
  tar -xJf "$TARBALL_PATH" -C "$tmp_extract"
  mkdir -p "$CUDNN8_DIR"
  cp -a "$tmp_extract"/cudnn-linux-*/lib "$CUDNN8_DIR/"
  cp -a "$tmp_extract"/cudnn-linux-*/include "$CUDNN8_DIR/"
  rm -rf "$tmp_extract"
else
  echo "[manual] cuDNN 8 tarball missing:"
  echo "         1. Download 'Local Installer for Linux x86_64 (Tar)' of cuDNN 8 for CUDA 12.x"
  echo "            from https://developer.nvidia.com/cudnn-archive"
  echo "         2. Place at: $TARBALL_PATH"
  echo "         3. Re-run this script."
  echo "         TRT online mode will fail; PyTorch fallback unaffected."
fi

export LD_LIBRARY_PATH="$CUDNN8_DIR/lib:${LD_LIBRARY_PATH:-}"

# ------------------------------------------------------------
# Download Ditto checkpoints from HuggingFace (idempotent)
# Per antgroup/ditto-talkinghead README:
#   git lfs install
#   git clone https://huggingface.co/digital-avatar/ditto-talkinghead checkpoints
# `hf download` lays out files directly under the --local-dir root, so
# $CHECKPOINTS_DIR/ditto-talkinghead/{ditto_cfg,ditto_onnx,ditto_pytorch,ditto_trt_Ampere_Plus}
# To match README's flat `./checkpoints/<sub>` convention (and config.yaml paths),
# we symlink each subdir to $CHECKPOINTS_DIR/<sub>.
# ------------------------------------------------------------
DITTO_REPO_DIR="$CHECKPOINTS_DIR/ditto-talkinghead"
DITTO_CKPT_SENTINEL="$DITTO_REPO_DIR/ditto_trt_Ampere_Plus/decoder_fp16.engine"
if [ -f "$DITTO_CKPT_SENTINEL" ]; then
  echo "[skip] ditto checkpoints present"
else
  echo "[hf] downloading digital-avatar/ditto-talkinghead via $HF_ENDPOINT"
  # faster-qwen3-tts and transformers pin huggingface_hub<1.0; install cli extras
  # in that compatible range to avoid breaking them.
  if ! command -v hf >/dev/null 2>&1; then
    pip install --quiet "huggingface_hub[cli]>=0.36,<1.0"
  fi
  # hf CLI supersedes the deprecated `huggingface-cli` (>=0.30).
  if command -v hf >/dev/null 2>&1; then
    hf download digital-avatar/ditto-talkinghead \
      --repo-type model \
      --exclude ".DS_Store" \
      --local-dir "$DITTO_REPO_DIR" || \
      echo "[warn] hf download failed - retry with: source /etc/network_turbo && rerun"
  else
    huggingface-cli download digital-avatar/ditto-talkinghead \
      --repo-type model \
      --include "ditto_cfg/*" --include "ditto_onnx/*" \
      --include "ditto_pytorch/*" --include "ditto_trt_Ampere_Plus/*" \
      --local-dir "$DITTO_REPO_DIR" || \
      echo "[warn] huggingface-cli download failed - retry with: source /etc/network_turbo && rerun"
  fi
fi

# Flatten layout via symlinks: $CHECKPOINTS_DIR/<sub> -> ditto-talkinghead/<sub>
# Space-efficient; matches README convention & config.yaml paths.
for sub_dir in "$DITTO_REPO_DIR"/*/; do
  name="$(basename "$sub_dir")"
  [ "$name" = ".cache" ] && continue
  if [ ! -e "$CHECKPOINTS_DIR/$name" ]; then
    ln -s "ditto-talkinghead/$name" "$CHECKPOINTS_DIR/$name"
    echo "[symlink] $CHECKPOINTS_DIR/$name -> ditto-talkinghead/$name"
  fi
done

conda deactivate

# ------------------------------------------------------------
# Create syncnet env (idempotent)
# Per https://github.com/joonson/syncnet_python README:
#   conda env create -f environment.yml          (GPU)
#   conda env create -f environment-cpu.yml      (CPU fallback per Q24 of plan)
# ------------------------------------------------------------
SYNCNET_ENV="$ENVS_DIR/syncnet"
if conda env list | awk '{print $1}' | grep -qx "$SYNCNET_ENV"; then
  echo "[skip] syncnet env exists at $SYNCNET_ENV"
else
  echo "[env] creating syncnet env from syncnet_python/environment.yml"
  if ! conda env create -f "$THIRD_PARTY/syncnet_python/environment.yml" -p "$SYNCNET_ENV"; then
    echo "[env] GPU env failed; falling back to environment-cpu.yml"
    conda env create -f "$THIRD_PARTY/syncnet_python/environment-cpu.yml" -p "$SYNCNET_ENV"
  fi
fi

# Download SyncNet pretrained model (idempotent; per README `sh download_model.sh`)
SYNCNET_MODEL_SENTINEL="$THIRD_PARTY/syncnet_python/data/syncnet_v2.model"
if [ -f "$SYNCNET_MODEL_SENTINEL" ]; then
  echo "[skip] syncnet_v2.model present"
else
  echo "[syncnet] running download_model.sh"
  # Disable `set -u` around conda activate: the MKL activation hook in the
  # syncnet env references unbound MKL_INTERFACE_LAYER, which is harmless
  # but kills the parent shell under `set -u`. Restore afterwards.
  set +u
  conda activate "$SYNCNET_ENV"
  ( cd "$THIRD_PARTY/syncnet_python" && bash download_model.sh ) \
    || echo "[warn] download_model.sh failed - check network / proxy"
  conda deactivate
  set -u
fi

# ------------------------------------------------------------
# Env activation helper printed for the user
# ------------------------------------------------------------
cat <<EOF

============================================================
 setup_env.sh complete
============================================================
  ditto env:        $DITTO_ENV
  syncnet env:      $SYNCNET_ENV
  ditto checkpoints: $CHECKPOINTS_DIR
  third_party:      $THIRD_PARTY
  HF_ENDPOINT:      $HF_ENDPOINT
  cuDNN 8 (TRT):    $CUDNN8_DIR  (LD_LIBRARY_PATH prepended)
------------------------------------------------------------
To activate ditto env + path exports in a new shell:

  source $REPO_DIR/scripts/setup_env.sh

(this is safe to repeat — all steps are idempotent)
============================================================
EOF