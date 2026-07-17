# LatentSync: Taming Stable Diffusion for Lip Sync!
ByteDance, 2024-2025

## Audio Backend
- **Audio encoder**: OpenAI Whisper (mel-spectrogram → audio embeddings)
- **Architecture**: Stable Diffusion UNet + cross-attention for audio conditioning
- **Losses**: TREPA + LPIPS + SyncNet (pixel space)
- **NOT** pixel-space diffusion or two-stage; directly models audio-visual correlations

## Versions
| Version | Resolution | VRAM (inf) | VRAM (train) | Date |
|:-------:|:----------:|:----------:|:------------:|:----:|
| 1.5 | 256×256 | 8 GB | 20 GB | 2025-03 |
| 1.6 | 512×512 | 18 GB | 55 GB | 2025-06 |

## Updates
- **1.6** (2025-06): 512×512 training, mitigates blurriness
- **1.5** (2025-03): temporal layer added, better Chinese video, VRAM reduced to 20 GB

## Setup
```bash
source setup_env.sh
```
Checkpoints:
```
./checkpoints/
├── latentsync_unet.pt
└── whisper/
    └── tiny.pt
```
Or download from HuggingFace: `ByteDance/LatentSync-1.6`

## Inference
### Gradio
```bash
python gradio_app.py
```
### CLI
```bash
./inference.sh
```
### Parameters
- `inference_steps` [20-50]: higher = better quality, slower
- `guidance_scale` [1.0-3.0]: higher = better lip-sync, may cause distortion/jitter

## Data Processing Pipeline
```bash
./data_processing_pipeline.sh
```
Steps:
1. Remove broken videos
2. Resample to 25fps, audio 16kHz
3. Scene detection (PySceneDetect)
4. Split into 5-10s segments
5. Affine transform via InsightFace landmarks, resize 256×256
6. Filter by SyncNet confidence > 3, adjust AV offset to 0
7. Filter by HyperIQA score > 40

## Training
### SyncNet (pretrained, 94% on VoxCeleb2 + HDTF)
```bash
huggingface-cli download ByteDance/LatentSync-1.6 stable_syncnet.pt --local-dir checkpoints
```

### U-Net
```bash
./train_unet.sh
```
Configs:
- `stage1.yaml`: 23 GB VRAM
- `stage2.yaml`: 30 GB VRAM
- `stage2_efficient.yaml`: 20 GB VRAM (slight quality drop)
- `stage1_512.yaml`: 30 GB VRAM
- `stage2_512.yaml`: 55 GB VRAM

## Evaluation
```bash
# Sync confidence score
./eval/eval_sync_conf.sh
# SyncNet accuracy
./eval/eval_syncnet_acc.sh
```

## Notes
- Built on AnimateDiff, inspired by MuseTalk
- Whisper audio encoder (same as MuseTalk and EchoMimic)
- Uses its own SyncNet (trained from 94% accuracy)
