# Hallo3 Deployment Design

## Goal

Install and minimally validate Hallo3 on the Seetacloud RTX 4080 host without modifying unrelated environments or the existing local experiment files.

## Target

- Host: `connect.weste.seetacloud.com:12862`
- Persistent data root: `/root/autodl-tmp`
- Hallo3 source and assets: `/root/autodl-tmp/hallo3`
- Official target runtime: Python 3.10, PyTorch 2.4.0, torchvision 0.19.0, CUDA 12.1

## Environment Strategy

1. Inspect the host's Conda environments, GPU driver, CUDA visibility, and installed PyTorch versions without changing anything.
2. Reuse an existing environment only if it satisfies Hallo3's Python, PyTorch, torchvision, and CUDA requirements.
3. Otherwise create an isolated `hallo3` Conda environment using the host's existing Conda installation.
4. Install Hallo3 dependencies while excluding duplicate PyTorch/CUDA packages and replacing the requirements file's mutually conflicting OpenCV packages with one compatible OpenCV package.

## Installation Layout

- Clone Hallo3 into `/root/autodl-tmp/hallo3` unless that directory already contains the expected repository.
- Keep model checkpoints below `/root/autodl-tmp/hallo3/pretrained_models` or the repository's documented checkpoint path.
- Keep validation outputs and logs below `/root/autodl-tmp/hallo3/outputs` and `/root/autodl-tmp/hallo3/logs`.
- Do not alter the prior JoyVASA, IMTalker, or other Conda environments.

## Verification

The installation is considered successful only if:

1. `torch`, `torchvision`, CUDA availability, and GPU name are printed from the selected environment.
2. Hallo3's inference entry point imports successfully.
3. Required checkpoint files are present and readable.
4. One official short inference completes without an out-of-memory or dependency error.
5. The output MP4 is non-empty and contains both video and audio streams when checked with `ffprobe`.

If checkpoints or official sample inputs are unavailable, stop after environment validation and report the exact missing artifact instead of fabricating a result.
