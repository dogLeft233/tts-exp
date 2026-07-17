# EchoMimic: Lifelike Audio-Driven Portrait Animations through Editable Landmark Conditioning
Zhiyuan Chen*, Jiajiong Cao*, Zhiquan Chen, Yuming Li†, Chenguang Ma†
Terminal Technology Department, Alipay, Ant Group
**AAAI 2025**

## Audio Backend
- **Audio encoder**: frozen Whisper-tiny
- **Architecture**: Stable Diffusion UNet + Reference UNet + motion module + face locator
- **Generates**: 512×512 video at 24fps

## Requirements
- Tested: Centos 7.2 / Ubuntu 22.04, CUDA >= 11.7
- Tested GPUs: A100(80G) / RTX 4090D(24G) / V100(16G)
- Python 3.8 / 3.10 / 3.11

## Installation
```bash
git clone https://github.com/BadToBest/EchoMimic
cd EchoMimic
conda create -n echomimic python=3.8
conda activate echomimic
pip install -r requirements.txt
```

## ffmpeg-static
```bash
# Download ffmpeg-4.4-amd64-static.tar.xz
export FFMPEG_PATH=/path/to/ffmpeg-4.4-amd64-static
```

## Download Weights
```bash
git lfs install
git clone https://huggingface.co/BadToBest/EchoMimic pretrained_weights
```
Structure:
```
./pretrained_weights/
├── denoising_unet.pth
├── reference_unet.pth
├── motion_module.pth
├── face_locator.pth
├── sd-vae-ft-mse/   config.json + diffusion_pytorch_model.safetensors
├── sd-image-variations-diffusers/   config.json + pytorch_model.bin
└── audio_processor/ whisper_tiny.pt
```

## Inference
### Audio-Driven
```bash
python -u infer_audio2vid.py
```
Edit `./configs/prompts/animation.yaml`:
```yaml
test_cases:
  "path/to/your/image":
    - "path/to/your/audio"
```

### Audio + Pose Driven
```bash
python -u infer_audio2vid_pose.py
```
Edit `./configs/prompts/animation_pose.yaml`

### Accelerated Inference (10x speedup)
Accelerated models available for Audio Driven and Audio + Landmark modes.
- V100: ~50s/240frames (from ~7min)

### Gradio UI
```bash
python -u webgui.py --server_port=3000
```

## Key Parameters
- Can generate by: audio only, facial landmarks only, or both combined
- `draw_mouse=True`: enable pose-driven mode (no audio)
- Output: 512×512, ~24fps

## Notes
- Supports English and Mandarin Chinese
- Accelerated inference: ~10x speedup
- V2 and V3 also available (semi-body, unified multi-modal)
- License: academic research
