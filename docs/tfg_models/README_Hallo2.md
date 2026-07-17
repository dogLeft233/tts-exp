# Hallo2: Long-Duration and High-Resolution Audio-driven Portrait Image Animation
Fudan University, Baidu Inc, Nanjing University
**ICLR 2025**

## Audio Backend
- **Audio encoder**: wav2vec2-base-960h (Facebook)
- **Architecture**: Latent diffusion + denoising UNet + face locator + motion module (AnimateDiff)
- **Key features**: 4K resolution, hour-long video, text-prompt enhanced

## Requirements
- Ubuntu 20.04/22.04, CUDA 11.8
- Tested GPUs: A100
- Python 3.10

## Installation
```bash
git clone https://github.com/fudan-generative-vision/hallo2
cd hallo2
conda create -n hallo python=3.10
conda activate hallo
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
apt-get install ffmpeg
```

## Download Weights
```bash
huggingface-cli download fudan-generative-ai/hallo2 --local-dir ./pretrained_models
```
Structure (10+ components):
```
./pretrained_models/
├── audio_separator/    Kim_Vocal_2.onnx
├── CodeFormer/         codeformer.pth, vqgan_code1024.pth
├── face_analysis/      landmarker, 1k3d68, 2d106det, genderage, glintr100, scrfd
├── facelib/            detection, parsing, yolov5 models
├── hallo2/             net_g.pth, net.pth  (main checkpoints)
├── motion_module/      mm_sd_v15_v2.ckpt
├── realesrgan/         RealESRGAN_x2plus.pth
├── sd-vae-ft-mse/      config.json, diffusion_pytorch_model.safetensors
├── stable-diffusion-v1-5/ unet/ config.json, diffusion_pytorch_model.safetensors
└── wav2vec/            wav2vec2-base-960h/ config.json, model.safetensors, etc.
```

## Input Requirements
- **Source image**: cropped square, face 50-70%, forward facing (<30° rotation)
- **Driving audio**: WAV format, English only (training data is English)

## Inference
### Long-duration animation
```bash
python scripts/inference_long.py --config ./configs/inference/long.yaml
```
Edit config to set `source_image`, `driving_audio`, `save_path`.

Additional options:
- `--pose_weight`: weight of pose
- `--face_weight`: weight of face
- `--lip_weight`: weight of lip
- `--face_expand_ratio`: face region size

### High-resolution (4K) animation
```bash
python scripts/video_sr.py \
  --input_path [input_video] \
  --output_path [output_dir] \
  --bg_upsampler realesrgan \
  --face_upsample -w 1 -s 4
```
Uses CodeFormer for face enhancement + RealESRGAN for background.
> Note: SR component is modified from CodeFormer (S-Lab License 1.0)

## Training
### Data preparation
1. Organize videos in `dataset_name/videos/*.mp4`
2. Preprocess:
```bash
python -m scripts.data_preprocess --input_dir dataset_name/videos --step 1
python -m scripts.data_preprocess --input_dir dataset_name/videos --step 2
```
3. Generate metadata:
```bash
python scripts/extract_meta_info_stage1.py -r path/to/dataset -n dataset_name
python scripts/extract_meta_info_stage2.py -r path/to/dataset -n dataset_name
```

### Training
```bash
accelerate launch -m --config_file accelerate_config.yaml \
  --machine_rank 0 --main_process_ip 0.0.0.0 --main_process_port 20055 \
  --num_machines 1 --num_processes 8 \
  scripts.train_stage1 --config ./configs/train/stage1.yaml
```

## Notes
- **Heaviest model**: needs A100, 10+ pretrained components
- First to achieve 4K + hour-long audio-driven portrait animation
- Uses wav2vec2 audio encoder (same as AniPortrait, different from whisper models)
- Includes vocal separation (audio_separator/Kim_Vocal_2)
- Auto-downloads InsightFace, MediaPipe face landmarks, etc.
