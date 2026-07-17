# Remote GPU Model Deployment

## When to use

Deploying ML models to remote GPU servers in network-constrained environments (AutoDL/SeetaCloud with limited HF access).

## Server inventory

### 旧服务器 (GTX 1080 Ti × 8) — lip-sync inference
```
sshpass -p 'zTnKbLMb1MXhtJz2' ssh -p 1986 wjj@10.1.114.114
```
- 8× GTX 1080 Ti (11GB, CUDA 11.4 max)
- PyTorch max: 2.0.1+cu118
- Works: Wav2Lip, MuseTalk 1.5
- Disk: `/data/wjj/tfg_models/`, `/data/wjj/tfg_exp/`

### 评估服务器 (RTX 4080) — SyncNet + historical runs
```
sshpass -p 'qUz0z/KoV/Ik' ssh -p 20398 root@connect.westd.seetacloud.com
```
- RTX 4080 SUPER (32GB)
- SyncNet env: `/root/autodl-tmp/envs/syncnet`
- Repo: `/root/autodl-tmp/repos/tts-exp`
- Runs: `runs/tfg_*` directories

## Weight download strategy (ordered by reliability)

1. **hf-mirror.com** (no proxy needed, fastest):
   ```bash
   wget --no-check-certificate "https://hf-mirror.com/{org}/{repo}/resolve/main/{file}?download=true" -O output
   ```

2. **Proxy** (unstable, slow for large files):
   ```bash
   source /etc/network_turbo
   ```

3. **Local download + SCP** (most reliable for >1GB):
   ```bash
   # Download locally, then:
   sshpass -p 'PASS' scp -P PORT file root@server:/path/
   ```

## Conda environment management

### Greenfield env creation
```bash
conda create -y -n NAME python=3.10 pytorch=X.X torchaudio=X.X pytorch-cuda=12.4 -c pytorch
conda install -y -n NAME -c conda-forge [packages]
```

### Critical traps
- **GraalPy**: `conda create` may resolve to GraalPy instead of CPython → verify with `python -c "import sys; print(sys.implementation.name)"`
- **pip upgrade cascade**: `pip install` can silently upgrade torch → pin all versions
- **ONNX Runtime CUDA**: `onnxruntime-gpu` needed for CUDA ONNX providers (not just `onnxruntime`)
- **PyTorch 2.6+**: `torch.load` defaults to `weights_only=True` → patch to `weights_only=False` for legacy checkpoints

### Environment reuse
When a new model's deps clash with a clean env, try reusing an existing env first:
```bash
# Check what's already installed
conda run -n EXISTING_ENV pip list | grep -iE "torch|diffusers|transformers|insightface"

# Install only missing deps
conda run -n EXISTING_ENV pip install [missing-only]
```

## Common code patches for lip-sync models

### 1. torch.load weights_only (PyTorch ≥2.6)
```bash
sed -i 's/torch\.load(/torch.load(weights_only=False, /g' file.py
# Then fix positional args to keyword:
sed -i 's/weights_only=False, \([^)]*\))/\1, weights_only=False)/g' file.py
```

### 2. Path replace() bug
```python
# BAD: str.replace corrupts extensions
temp = output_path.replace(name, name + '-temp')  # "4.mp4" → "4-temp.mp4-temp"

# GOOD: use pathlib
temp = str(Path(output_path).parent / (name + '-temp' + Path(output_path).suffix))
```

### 3. Batch inference script template
```bash
#!/bin/bash
LOG=/path/to/batch.log
ENV=env_name
SRC=/path/to/source
for id in $(seq 1 13); do
    OUTFILE="$OUT/${id}.mp4"
    [ -f "$OUTFILE" ] && [ -s "$OUTFILE" ] && continue  # skip existing
    conda run -n "$ENV" python inference.py \
        --input ... --output "$OUTFILE" >> "$LOG" 2>&1
    echo "EXIT=$?" | tee -a "$LOG"
done
```

## Model compatibility matrix

| Model | Min VRAM | CUDA | PyTorch | Notes |
|-------|----------|------|---------|-------|
| Wav2Lip | 4GB | 11.x | 2.0.1 | TorchScript→state_dict conversion |
| MuseTalk 1.5 | 8GB | 11.x | 2.0.1 | vae_type="v15" |
| LatentSync | 4GB | 12.x | 2.6.0 | Old GPU OOM at 512×512 |
| JoyVASA | 3GB | 12.x | 2.4.1 | Fastest (23s/video) |
| V-Express | 9GB | 12.x | 2.6.0 | Slow (6min/video); kps pre-extraction needed |
| EchoMimic | 11GB | 12.x | — | Abandoned (slow + VRAM) |
| AniPortrait | >11GB | 12.x | — | OOM on 1080 Ti |
| Hallo2 | 24GB | **11.8 only** | 2.2.2 | English only |

## Adding a model to SyncNet eval

```bash
# 1. Create run dir
mkdir -p runs/tfg_MODEL/03_ditto/{natural_raw,tts_raw}

# 2. Upload videos (from local or another server)
# SCP to 20398 server

# 3. Run eval
cd /root/autodl-tmp/repos/tts-exp/scripts
/root/autodl-tmp/envs/syncnet/bin/python 04_eval.py --run_id tfg_MODEL

# 4. Extract scores
find runs/tfg_MODEL/04_eval -name syncnet.json | ...
```
