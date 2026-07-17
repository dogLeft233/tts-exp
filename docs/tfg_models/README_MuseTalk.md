# MuseTalk
**MuseTalk: Real-Time High-Fidelity Video Dubbing via Spatio-Temporal Sampling**
Yue Zhang\*, Zhizhou Zhong\*, Minhao Liu\*, Zhaokang Chen, Bin Wu†, Yubin Zeng,
Chao Zhan, Junxin Huang, Yingjie He, Wenjiang Jiang
Lyra Lab, Tencent Music Entertainment

We introduce `MuseTalk`, a **real-time high quality** lip-syncing model (30fps+ on an NVIDIA Tesla V100).

## Audio Backend
- **Audio encoder**: frozen `whisper-tiny` (384-dim embeddings)
- **Architecture**: UNet from `stable-diffusion-v1-4`, audio→image via cross-attention
- **NOT** a diffusion model — single-step latent inpainting

## Requirements
- Python 3.10, CUDA 11.7
- PyTorch 2.0.1
- MMEngine, MMCV==2.0.1, MMDet==3.1.0, MMPose==1.1.0
- ffmpeg-static

## Installation
```bash
conda create -n MuseTalk python==3.10
conda activate MuseTalk
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install --no-cache-dir -U openmim
mim install mmengine
mim install "mmcv==2.0.1"
mim install "mmdet==3.1.0"
mim install "mmpose==1.1.0"
```

## Download Weights
```bash
sh ./download_weights.sh
```
Weights organized as:
```
./models/
├── musetalk/ musetalk.json + pytorch_model.bin
├── musetalkV15/ musetalk.json + unet.pth
├── syncnet/ latentsync_syncnet.pt
├── dwpose/ dw-ll_ucoco_384.pth
├── face-parse-bisent/ 79999_iter.pth + resnet18-5c106cde.pth
├── sd-vae/ config.json + diffusion_pytorch_model.bin
└── whisper/ config.json + pytorch_model.bin + preprocessor_config.json
```

## Inference
```bash
# v1.5 (recommended)
sh inference.sh v1.5 normal
# or manually:
python -m scripts.inference --inference_config configs/inference/test.yaml \
  --result_dir results/test \
  --unet_model_path models/musetalkV15/unet.pth \
  --unet_config models/musetalkV15/musetalk.json --version v15

# real-time:
sh inference.sh v1.5 realtime
```

Config file `configs/inference/test.yaml`:
- `video_path`: input video/image
- `audio_path`: input audio

## Training
1. Data preparation: `python -m scripts.preprocess --config ./configs/training/preprocess.yaml`
2. Stage 1: `sh train.sh stage1`
3. Stage 2: `sh train.sh stage2`
- Stage 1: batch 32 → ~74GB/GPU (H20)
- Stage 2: batch 2 + grad accum 8 → ~85GB/GPU

## Key Parameters
- `bbox_shift`: adjust mask region (positive = more mouth openness)
- Audio: supports Chinese, English, Japanese
- Output: face region 256×256

## Notes
- Resolution limit 256×256 face; use GFPGAN for upscaling
- Known issues: mustache/lip shape/color not well preserved, some jitter
- License: MIT (code), model: any purpose
