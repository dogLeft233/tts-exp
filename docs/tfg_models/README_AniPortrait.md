# AniPortrait: Audio-Driven Synthesis of Photorealistic Portrait Animations
Huawei Wei, Zejun Yang, Zhisheng Wang
Tencent Games Zhiji, Tencent | arXiv 2403.17694

## Audio Backend
- **Audio encoder**: wav2vec2-base-960h → 2D facial landmarks (intermediate 3D representation)
- **Architecture**: Stable Diffusion UNet + Reference UNet + Pose Guider + Motion Module
- **Pipeline**: audio → 3D mesh → 2D landmarks → SD with motion module → video

## Requirements
- Python >= 3.10, CUDA 11.7

## Installation
```bash
pip install -r requirements.txt
```

## Download Weights
All weights under `./pretrained_weights/`:
```
./pretrained_weights/
├── image_encoder/                  # lambdalabs/sd-image-variations-diffusers
├── sd-vae-ft-mse/                  # stabilityai
├── stable-diffusion-v1-5/          # runwayml
├── wav2vec2-base-960h/             # facebook
├── audio2mesh.pt                   # trained
├── audio2pose.pt                   # trained
├── denoising_unet.pth              # trained
├── film_net_fp16.pt                # frame interpolation (optional, for -acc)
├── motion_module.pth               # trained
├── pose_guider.pth                 # trained
└── reference_unet.pth              # trained
```
Download from HuggingFace: `ZJYang/AniPortrait`

## Inference Modes

### Audio Driven (audio → portrait)
```bash
python -m scripts.audio2vid \
  --config ./configs/prompts/animation_audio.yaml \
  -W 512 -H 512 -acc
```
Add audio/image paths in `animation_audio.yaml`.

### Self Driven (pose → portrait)
```bash
python -m scripts.pose2vid \
  --config ./configs/prompts/animation.yaml \
  -W 512 -H 512 -acc
```

### Face Reenactment (video → portrait)
```bash
python -m scripts.vid2vid \
  --config ./configs/prompts/animation_facereenac.yaml \
  -W 512 -H 512 -acc
```

### Head Pose Control
```bash
python -m scripts.generate_ref_pose \
  --ref_video ./configs/inference/head_pose_temp/pose_ref_video.mp4 \
  --save_path ./configs/inference/head_pose_temp/pose.npy
```
Delete `pose_temp` in `animation_audio.yaml` to enable audio2pose model.

### Frame Interpolation (acceleration)
Add `-acc` flag + download `film_net_fp16.pt`.

### Gradio Web UI
```bash
python -m scripts.app
```

## Training
### Data
- Datasets: VFHQ + CelebV-HQ
```bash
python -m scripts.preprocess_dataset \
  --input_dir VFHQ_PATH \
  --output_dir SAVE_PATH \
  --training_json JSON_PATH
```

### Stage 1
```bash
accelerate launch train_stage_1.py --config ./configs/train/stage1.yaml
```

### Stage 2
Download motion module from AnimateDiff: `mm_sd_v15_v2.ckpt`
```bash
accelerate launch train_stage_2.py --config ./configs/train/stage2.yaml
```

## Notes
- Output: 512×512
- Uses wav2vec2 (unique among these models — different from whisper/HuBERT)
- Audio2Pose and Audio2Mesh models for head pose control
- Supports face reenactment (video-driven) in addition to audio-driven
