# New Server TFG Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy and validate LatentSync, EchoMimic, and AniPortrait at 512x512 on the RTX 4080 server.

**Architecture:** Store all persistent artifacts under `/autodl-pub/tfg_exp` and source trees under `/autodl-pub/tfg_models`. Create one Python 3.10 environment per model, keeping incompatible packages isolated. A single-sample validation writes logs and MP4s to the experiment directory before a model can start the 26-item comparison batch.

**Tech Stack:** Ubuntu 22.04, NVIDIA RTX 4080/CUDA 13 driver, Python 3.10, PyTorch CUDA wheels, Conda/Mamba, ffmpeg, Hugging Face Hub.

---

### Task 1: Bootstrap Persistent Workspace and Python Environment Manager

**Files:**
- Create: `/autodl-pub/tfg_exp/{audio,faces,static_videos,generation,logs}`
- Create: `/autodl-pub/tfg_models`
- Create: `/autodl-pub/tfg_exp/DEPLOYMENT.md`

- [ ] **Step 1: Install Miniforge and confirm the CUDA runtime is visible**

```bash
wget -qO /tmp/Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash /tmp/Miniforge3.sh -b -p /opt/conda
/opt/conda/bin/conda --version
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
```

Expected: a Conda version plus `NVIDIA GeForce RTX 4080` and about `32760 MiB`.

- [ ] **Step 2: Create the persistent data directories**

```bash
mkdir -p /autodl-pub/tfg_exp/audio /autodl-pub/tfg_exp/faces /autodl-pub/tfg_exp/static_videos
mkdir -p /autodl-pub/tfg_exp/generation /autodl-pub/tfg_exp/logs /autodl-pub/tfg_models
```

- [ ] **Step 3: Record host facts and planned output layout**

```markdown
# RTX 4080 TFG Deployment

- Host: `connect.westd.seetacloud.com:14453`
- GPU: NVIDIA RTX 4080, 32GB VRAM
- Inputs: `/autodl-pub/tfg_exp/{audio,faces,static_videos}`
- Outputs: `/autodl-pub/tfg_exp/generation/<model>/{natural_raw,tts_raw}`
- Logs: `/autodl-pub/tfg_exp/logs/<model>-<run>.log`
```

- [ ] **Step 4: Verify persistent space**

Run: `df -h /autodl-pub`

Expected: at least `5 TB` free.

### Task 2: Transfer Experiment Inputs and Existing Model Sources

**Files:**
- Create: `/autodl-pub/tfg_exp/manifest.txt`
- Create: `/autodl-pub/tfg_models/{LatentSync,echomimic,AniPortrait}`

- [ ] **Step 1: Create an explicit experiment manifest**

```text
natural_raw/1.wav
...
natural_raw/13.wav
tts_raw/1.wav
...
tts_raw/13.wav
faces/1.png
...
faces/13.png
static_videos/1.mp4
...
static_videos/13.mp4
```

- [ ] **Step 2: Copy assets from the prior server using rsync over SSH**

```bash
rsync -aP -e 'ssh -p 1986' wjj@10.1.114.114:/data/wjj/tfg_exp/ /autodl-pub/tfg_exp/
rsync -aP -e 'ssh -p 1986' wjj@10.1.114.114:/data/wjj/tfg_models/LatentSync/ /autodl-pub/tfg_models/LatentSync/
rsync -aP -e 'ssh -p 1986' wjj@10.1.114.114:/data/wjj/tfg_models/echomimic/ /autodl-pub/tfg_models/echomimic/
rsync -aP -e 'ssh -p 1986' wjj@10.1.114.114:/data/wjj/tfg_models/AniPortrait/ /autodl-pub/tfg_models/AniPortrait/
```

- [ ] **Step 3: Verify all expected experiment inputs exist**

Run: `test $(find /autodl-pub/tfg_exp/audio/natural_raw -name '*.wav' | wc -l) -eq 13 && test $(find /autodl-pub/tfg_exp/audio/tts_raw -name '*.wav' | wc -l) -eq 13 && test $(find /autodl-pub/tfg_exp/faces -name '*.png' | wc -l) -eq 13`

Expected: exit code `0`.

### Task 3: Configure and Validate LatentSync 1.6

**Files:**
- Modify: `/autodl-pub/tfg_models/LatentSync/scripts/inference.py`
- Create: `/autodl-pub/tfg_exp/generation/latentsync/test-command.sh`
- Create: `/autodl-pub/tfg_exp/logs/latentsync-test.log`

- [ ] **Step 1: Create the isolated environment and install CUDA-enabled dependencies**

```bash
/opt/conda/bin/conda create -y -n latentsync python=3.10
/opt/conda/bin/conda run -n latentsync pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
/opt/conda/bin/conda run -n latentsync pip install -r /autodl-pub/tfg_models/LatentSync/requirements.txt
```

- [ ] **Step 2: Add a device-independent fp16 selection**

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weight_dtype = torch.float16 if device.type == "cuda" else torch.float32
```

Replace hard-coded `cuda:4` and compute-capability-only fp16 conditions in `scripts/inference.py` with this block.

- [ ] **Step 3: Run a one-sample 512x512 test using the official stage-2 512 configuration**

```bash
/opt/conda/bin/conda run -n latentsync python -m scripts.inference --unet_config_path configs/unet/stage2_512.yaml --inference_ckpt_path checkpoints/latentsync_unet.pt --video_path /autodl-pub/tfg_exp/static_videos/1.mp4 --audio_path /autodl-pub/tfg_exp/audio/natural_raw/1.wav --video_out_path /autodl-pub/tfg_exp/generation/latentsync/test-1.mp4 --inference_steps 20 --guidance_scale 1.5
```

- [ ] **Step 4: Verify the output has video and audio streams**

Run: `ffprobe -v error -show_entries stream=codec_type -of csv=p=0 /autodl-pub/tfg_exp/generation/latentsync/test-1.mp4`

Expected: one `video` line and one `audio` line.

### Task 4: Configure and Validate EchoMimic

**Files:**
- Modify: `/autodl-pub/tfg_models/echomimic/infer_audio2vid.py`
- Create: `/autodl-pub/tfg_exp/generation/echomimic/test.yaml`
- Create: `/autodl-pub/tfg_exp/logs/echomimic-test.log`

- [ ] **Step 1: Create the isolated Python 3.10 environment**

```bash
/opt/conda/bin/conda create -y -n echomimic python=3.10
/opt/conda/bin/conda run -n echomimic pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
/opt/conda/bin/conda run -n echomimic pip install -r /autodl-pub/tfg_models/echomimic/requirements.txt
/opt/conda/bin/conda run -n echomimic pip install xformers
```

- [ ] **Step 2: Enable xformers on the created diffusion pipeline**

```python
if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
    pipe.enable_xformers_memory_efficient_attention()
```

Add immediately after `pipe = pipe.to(device, dtype=weight_dtype)` in `infer_audio2vid.py`.

- [ ] **Step 3: Write a one-case input configuration**

```yaml
test_cases:
  "/autodl-pub/tfg_exp/faces/1.png":
    - "/autodl-pub/tfg_exp/audio/natural_raw/1.wav"
```

- [ ] **Step 4: Run and verify the test output**

Run: `/opt/conda/bin/conda run -n echomimic python -u infer_audio2vid.py --config /autodl-pub/tfg_exp/generation/echomimic/test.yaml`

Expected: a non-empty `withaudio.mp4` under EchoMimic `output/`, with 512x512 video and an audio stream.

### Task 5: Configure and Validate AniPortrait

**Files:**
- Modify: `/autodl-pub/tfg_models/AniPortrait/scripts/audio2vid.py`
- Create: `/autodl-pub/tfg_exp/generation/aniportrait/test.yaml`
- Create: `/autodl-pub/tfg_exp/logs/aniportrait-test.log`

- [ ] **Step 1: Create the isolated Python 3.10 environment**

```bash
/opt/conda/bin/conda create -y -n aniportrait python=3.10
/opt/conda/bin/conda run -n aniportrait pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
/opt/conda/bin/conda run -n aniportrait pip install -r /autodl-pub/tfg_models/AniPortrait/requirements.txt
/opt/conda/bin/conda run -n aniportrait pip install xformers
```

- [ ] **Step 2: Use explicit CUDA fp16 inference and xformers**

```python
weight_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu", dtype=weight_dtype)
pipe.enable_xformers_memory_efficient_attention()
```

Replace hard-coded `.cuda()` calls and any `torch.set_default_dtype()` mutation with these local values in `scripts/audio2vid.py`.

- [ ] **Step 3: Run a short 512x512 test with the official audio-driven pipeline**

```bash
/opt/conda/bin/conda run -n aniportrait python -m scripts.audio2vid --config /autodl-pub/tfg_exp/generation/aniportrait/test.yaml -W 512 -H 512 -L 60
```

- [ ] **Step 4: Verify the generated MP4**

Run: `find /autodl-pub/tfg_models/AniPortrait/output -name '*.mp4' -size +1k -print`

Expected: at least one result, containing video and audio streams when inspected with `ffprobe`.

### Task 6: Record Verified Configurations and Schedule Batches

**Files:**
- Modify: `docs/tfg_models/TFG_DEPLOY_SUMMARY.md`
- Create: `/autodl-pub/tfg_exp/generation/{latentsync,echomimic,aniportrait}/batch-{natural_raw,tts_raw}.yaml`

- [ ] **Step 1: Record package versions and peak memory for each successful test**

```bash
/opt/conda/bin/conda run -n <model> python -c 'import torch; print(torch.__version__, torch.version.cuda)'
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
```

- [ ] **Step 2: Generate separate natural and TTS manifests for every model**

Each manifest must reference IDs `1` through `13`, with exactly one face image/static face video and its matching condition audio file per item.

- [ ] **Step 3: Launch batches only after recording a successful single-sample verification**

Run each model in a separate `nohup` process and redirect logs to `/autodl-pub/tfg_exp/logs/`.

- [ ] **Step 4: Update the local deployment summary**

Add the new server's environment versions, the exact commands used, test output paths, and whether all 26 outputs were started or completed.
