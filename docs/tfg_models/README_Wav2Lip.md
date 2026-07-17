# Wav2Lip: Accurately Lip-syncing Videos In The Wild
ACM Multimedia 2020 | IIIT Hyderabad
Paper: "A Lip Sync Expert Is All You Need for Speech to Lip Generation In the Wild"

## Audio Backend
- **Audio encoder**: Mel-spectrogram (80-dim)
- **Architecture**: Encoder-decoder + pre-trained SyncNet as discriminator
- **Key**: Lip-sync expert (SyncNet) as audio-visual synchronization discriminator

## Requirements
- Python 3.6
- ffmpeg: `sudo apt-get install ffmpeg`
- Face detection model: s3fd.pth → `face_detection/detection/sfd/s3fd.pth`
  - Link: https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth

## Installation
```bash
pip install -r requirements.txt
```

## Download Weights
| Model | Description | Link |
|:-----|:-----------|:-----|
| Wav2Lip | Highly accurate lip-sync | Google Drive (folder) |
| Wav2Lip + GAN | Slightly inferior sync, better visual quality | Google Drive (single) |

## Inference
```bash
python inference.py \
  --checkpoint_path <ckpt> \
  --face <video_file> \
  --audio <audio_file>
```
Result saved to `results/result_voice.mp4`.

### Tips
- `--pads 0 20 0 0`: adjust face bounding box (add bottom padding for chin)
- `--nosmooth`: fix dislocated mouth / double-mouth artifacts
- `--resize_factor`: lower resolution can improve quality
- Wav2Lip (no GAN) needs more tuning but can give better results

## Training
### Data: LRS2 dataset
### Step 1: Preprocess
```bash
python preprocess.py --data_root data_root/main --preprocessed_root lrs2_preprocessed/
```

### Step 2: Train SyncNet expert
```bash
python color_syncnet_train.py --data_root lrs2_preprocessed/ --checkpoint_dir <dir>
```

### Step 3: Train Wav2Lip
```bash
python wav2lip_train.py --data_root lrs2_preprocessed/ \
  --checkpoint_dir <dir> \
  --syncnet_checkpoint_path <ckpt>
```
For visual quality discriminator: use `hq_wav2lip_train.py` (~2 days)

## Evaluation
See `evaluation/` folder for metrics (LSE-D, LSE-C, etc.)

## Notes
- **Classic baseline** — the most widely cited and compared model
- Output: face region at lower resolution (96×96 in original, 192×288 in HD)
- Works for any identity, voice, language; CGI faces and synthetic voices
- Open source for research/non-commercial only; commercial via sync.so API
- Audio: simplest backend (mel-spectrogram, 80-dim) — very different from modern backends
