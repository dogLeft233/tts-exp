# Hallo3 Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Install Hallo3 on the Seetacloud RTX 4080 host using an existing compatible PyTorch environment when safe, otherwise an isolated environment, and complete one verified inference.

**Architecture:** Persistent source, model files, logs, and outputs live under `/root/autodl-tmp/hallo3`. The environment is selected read-only from existing Conda environments and only a clearly dedicated compatible environment is reused; otherwise a new `hallo3` environment is created. Hallo3's pinned PyTorch/CUDA packages and mutually conflicting OpenCV packages are excluded from the bulk requirements install and installed deliberately.

**Tech Stack:** Ubuntu 20.04/22.04, Conda, Python 3.10, PyTorch 2.4.0, torchvision 0.19.0, CUDA 12.1, Hugging Face Hub, ffmpeg, Hallo3/CogVideoX.

---

### Task 1: Inspect the host without changing it

**Files:**
- Create: `/root/autodl-tmp/hallo3/logs/host-inspection.txt`

- [ ] **Step 1: Verify the persistent data mount and GPU**

Run through SSH:

```bash
if [ -e /root/autodl-tmp/hallo3 ] && [ ! -d /root/autodl-tmp/hallo3/.git ]; then
  printf '%s\n' '/root/autodl-tmp/hallo3 exists but is not a Hallo3 checkout' >&2
  exit 2
fi
mkdir -p /root/autodl-tmp/hallo3/logs
{
  printf 'host=%s\n' "$(hostname)"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  df -h /root/autodl-tmp
  command -v conda || true
  command -v ffmpeg || true
} | tee /root/autodl-tmp/hallo3/logs/host-inspection.txt
```

Expected: an NVIDIA GPU with approximately 32GB VRAM, writable persistent storage, and either an existing Conda installation or a clear path to one.

- [ ] **Step 2: Enumerate existing environments and PyTorch versions**

Run:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list
for env in $(conda env list | awk '/^\// {print $1}'); do
  conda run -p "$env" python -c 'import sys; print(sys.executable); import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())' 2>&1 || true
done | tee -a /root/autodl-tmp/hallo3/logs/host-inspection.txt
```

Use an existing environment only when it is clearly dedicated to Hallo3 and reports Python 3.10, `torch 2.4.0`, `torchvision 0.19.0`, and CUDA availability. Do not modify JoyVASA, IMTalker, or another model's environment merely because it has PyTorch installed.

### Task 2: Prepare the source tree and selected environment

**Files:**
- Create or reuse: `/root/autodl-tmp/hallo3/`
- Create: `/root/autodl-tmp/hallo3/logs/install.log`

- [ ] **Step 1: Clone Hallo3 without overwriting an existing checkout**

Run:

```bash
if [ ! -d /root/autodl-tmp/hallo3/.git ]; then
  git clone https://github.com/fudan-generative-vision/hallo3.git /root/autodl-tmp/hallo3
fi
git -C /root/autodl-tmp/hallo3 status --short
git -C /root/autodl-tmp/hallo3 rev-parse HEAD
```

If the directory exists but is not a Hallo3 checkout, stop rather than deleting it.

- [ ] **Step 2: Create an isolated environment when no compatible dedicated environment exists**

Run after Task 1 selects the environment:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
export HALLO_ENV=hallo3
conda create -y -n hallo3 python=3.10
conda run -n "$HALLO_ENV" python -m pip install --upgrade pip setuptools wheel
conda run -n "$HALLO_ENV" python -m pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
```

If Task 1 found a clearly dedicated compatible environment, set `HALLO_ENV` to that environment name instead and skip `conda create` and the two `pip install` commands above. Keep the exported `HALLO_ENV` variable in the shell used for later tasks.

- [ ] **Step 3: Install sanitized Hallo3 dependencies**

Run from the repository root using the `HALLO_ENV` exported in Task 2:

```bash
cd /root/autodl-tmp/hallo3
awk -F== '!($1 ~ /^(torch|torchvision|triton|nvidia-|opencv-contrib-python|opencv-python|opencv-python-headless)$/)' requirements.txt > /tmp/hallo3-requirements.txt
conda run -n "$HALLO_ENV" python -m pip install -r /tmp/hallo3-requirements.txt
conda run -n "$HALLO_ENV" python -m pip install opencv-python-headless==4.9.0.80
conda run -n "$HALLO_ENV" python -m pip check | tee /root/autodl-tmp/hallo3/logs/install.log
```

The single headless OpenCV package replaces all three OpenCV entries from the upstream requirements file. If `deepspeed==0.14.4` fails to build, record the complete error and first test inference without removing unrelated packages; do not silently install an unpinned replacement.

- [ ] **Step 4: Install and verify ffmpeg**

Run:

```bash
if ! command -v ffmpeg >/dev/null 2>&1; then
  apt-get update
  apt-get install -y ffmpeg
fi
ffmpeg -version | sed -n '1p'
```

### Task 3: Download Hallo3 checkpoints and sample assets

**Files:**
- Create: `/root/autodl-tmp/hallo3/pretrained_models/`
- Create: `/root/autodl-tmp/hallo3/logs/checkpoints.log`

- [ ] **Step 1: Download the official model repository**

Run from `/root/autodl-tmp/hallo3`:

```bash
conda run -n "$HALLO_ENV" python -m pip install 'huggingface_hub[cli]==0.25.2'
HF_HUB_DISABLE_XET=1 conda run -n "$HALLO_ENV" huggingface-cli download fudan-generative-ai/hallo3 --local-dir /root/autodl-tmp/hallo3/pretrained_models --local-dir-use-symlinks False 2>&1 | tee /root/autodl-tmp/hallo3/logs/checkpoints.log
```

- [ ] **Step 2: Verify the required checkpoint layout**

Run:

```bash
test -f /root/autodl-tmp/hallo3/pretrained_models/hallo3/latest
test -f /root/autodl-tmp/hallo3/pretrained_models/cogvideox-5b-i2v-sat/vae/3d-vae.pt
test -f /root/autodl-tmp/hallo3/pretrained_models/t5-v1_1-xxl/config.json
test -f /root/autodl-tmp/hallo3/pretrained_models/wav2vec/wav2vec2-base-960h/config.json
test -f /root/autodl-tmp/hallo3/pretrained_models/audio_separator/Kim_Vocal_2.onnx
find /root/autodl-tmp/hallo3/pretrained_models/face_analysis/models -type f -printf '%f\n' | sort
```

If the Hugging Face download does not include the face-analysis or face-landmarker files, run:

```bash
FACE_DIR=/root/autodl-tmp/hallo3/pretrained_models/face_analysis/models
mkdir -p "$FACE_DIR"
wget -qO /tmp/buffalo_l.zip https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
unzip -j -o /tmp/buffalo_l.zip '*.onnx' -d "$FACE_DIR"
wget -qO "$FACE_DIR/face_landmarker_v2_with_blendshapes.task" https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
rm -f /tmp/buffalo_l.zip
```

- [ ] **Step 3: Verify official sample inputs**

Run:

```bash
test -f /root/autodl-tmp/hallo3/examples/inference/images/0001.jpg
test -f /root/autodl-tmp/hallo3/examples/inference/audios/0001.wav
awk 'NR == 1 {print; exit}' /root/autodl-tmp/hallo3/examples/inference/input.txt > /root/autodl-tmp/hallo3/one-sample-input.txt
```

### Task 4: Run import, runtime, and minimal inference checks

**Files:**
- Create: `/root/autodl-tmp/hallo3/logs/validation.log`
- Create: `/root/autodl-tmp/hallo3/outputs/`

- [ ] **Step 1: Verify the selected runtime**

Run:

```bash
cd /root/autodl-tmp/hallo3
conda run -n "$HALLO_ENV" python -c 'import sys, torch, torchvision; print(sys.executable); print("torch", torch.__version__); print("torchvision", torchvision.__version__); print("torch_cuda", torch.version.cuda); print("cuda_available", torch.cuda.is_available()); print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")'
```

- [ ] **Step 2: Verify Hallo3 imports from its repository root**

Run:

```bash
cd /root/autodl-tmp/hallo3
conda run -n "$HALLO_ENV" python -c 'import torch; from hallo3.sample_video import sampling_main; print("hallo3 import ok", torch.__version__)'
```

- [ ] **Step 3: Run one official short sample**

Run:

```bash
cd /root/autodl-tmp/hallo3
nvidia-smi --query-gpu=timestamp,memory.used,memory.total --format=csv,noheader -lms 500 > /root/autodl-tmp/hallo3/logs/gpu-memory.csv &
MONITOR_PID=$!
CUDA_VISIBLE_DEVICES=0 conda run -n "$HALLO_ENV" bash scripts/inference_long_batch.sh /root/autodl-tmp/hallo3/one-sample-input.txt /root/autodl-tmp/hallo3/outputs 2>&1 | tee /root/autodl-tmp/hallo3/logs/validation.log
STATUS=${PIPESTATUS[0]}
kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true
exit "$STATUS"
```

The official configuration uses 13 frames per sampling window and 50 diffusion steps. This is the smallest upstream inference path and must be attempted before any custom batch.

- [ ] **Step 4: Validate the generated MP4 streams and memory**

Run:

```bash
find /root/autodl-tmp/hallo3/outputs -type f -name '*_with_audio.mp4' -size +1k -print -exec ffprobe -v error -show_entries stream=codec_type -of csv=p=0 {} \;
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
```

Expected: at least one non-empty MP4 with both `video` and `audio` stream lines. Record the exact output path, runtime versions, and peak observed memory in `validation.log`.

### Task 5: Report the deployment result

**Files:**
- Modify: `docs/superpowers/specs/2026-07-25-hallo3-deployment-design.md` only if the verified environment differs from the planned branch.
- Create: `/root/autodl-tmp/hallo3/DEPLOYMENT.md`

- [ ] **Step 1: Record the remote environment and commands**

Write `/root/autodl-tmp/hallo3/DEPLOYMENT.md` with the selected environment name, Python/PyTorch/torchvision/CUDA versions, repository commit, checkpoint verification result, sample command, output path, and peak GPU memory.

- [ ] **Step 2: Stop cleanly on missing artifacts or OOM**

If checkpoint download, import, or inference fails, preserve the logs and report the first actionable error. Do not claim Hallo3 is installed and working until the MP4 stream check passes.
