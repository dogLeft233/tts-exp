# New Server TFG Deployment Design

## Goal

Deploy the TFG models that did not complete inference on the previous GTX 1080 Ti server: LatentSync, EchoMimic, and AniPortrait. Validate each model with one 512x512 sample before scheduling the 13 natural-audio and 13 TTS-audio experiment inputs.

## Target Host

- Host: `connect.westd.seetacloud.com:14453`
- GPU: NVIDIA RTX 4080, 32GB VRAM
- RAM: 503GB
- Persistent data: `/autodl-pub` with 5.3TB available
- OS: Ubuntu 22.04.5
- CUDA driver support: CUDA 13.0

## Layout

- `/autodl-pub/tfg_exp`: input audio, face images, static face videos, generated videos, and batch manifests.
- `/autodl-pub/tfg_models/<model>`: isolated source checkout and model weights.
- Each model has an independent Python environment to pin its incompatible dependencies.

## Model Plan

### LatentSync

- Install the official project dependencies in its own environment.
- Use the official stage-2 512x512 configuration and fp16 inference on the RTX 4080.
- Run one static-face-video and audio pair before batch inference.

### EchoMimic

- Install the project in a separate Python 3.10 environment.
- Use fp16 and xformers memory-efficient attention.
- Use the existing 512x512, 30-step, CFG 2.0 configuration; only replace its input manifest.

### AniPortrait

- Install the official project in its own environment.
- Use fp16 and explicitly enable xformers memory-efficient attention in the pipeline.
- Restore the official 512x512 resolution and use a short clip for initial validation; then run full inputs.

## Verification

For every model, verification requires:

1. A successful import of the model's entry point.
2. A completed single-sample 512x512 MP4 containing both video and audio.
3. A recorded command, package versions, output path, duration, and peak GPU memory.

No batch inference starts until its corresponding single-sample check passes.
