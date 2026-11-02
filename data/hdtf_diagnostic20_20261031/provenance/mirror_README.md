# HDTF Dataset for Talking Face Generation

This dataset is a curated and preprocessed version of **HDTF (High-Definition Talking Face)**, prepared for tasks such as talking-head generation, video captioning, and multimodal avatar synthesis.

## 📁 File Overview

| File Name           | Description                                                                |
|---------------------|----------------------------------------------------------------------------|
| `videos.zip`        | Original 400 full-length HDTF videos                                       |
| `clips.zip`         | Videos split into 81-frame clips, each representing a short temporal unit  |
| `audios.zip`        | Audio embeddings extracted from original videos using OpenAI Whisper       |
| `poses.zip`         | Stickman-style pose videos (MediaPipe output), aligned with `clips.zip`    |
| `tensors.zip` (split) | Multimodal latent tensors, including:                                     |
|                     | - Latent features of `clips` videos                                        |
|                     | - Latent features of `poses` videos                                        |
|                     | - Text embeddings from captions                                            |
| `metadata.csv`      | Textual descriptions (captions) for each clip, generated via Llava-Video   |
| `train.txt`         | List of training clip file names (matched to `clips.zip`)                  |
| `val.txt`           | List of validation clip file names                                         |
| `test.txt`          | List of test clip file names                                               |

> All modalities (clip, audio, pose, caption, and tensors) are aligned via consistent clip filenames.

---

## 📦 Handling Split Archives

Some large files (e.g., `tensors.zip`) are split due to Hugging Face's 50GB file limit.

To restore the full archive:

```bash
cat tensors.zip.part_* > tensors.zip
unzip tensors.zip
