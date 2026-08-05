# AvatarForcing 部署记录

## 状态

- **状态**: ✅ AISHELL-1 26 条批量推理完成；服务器已关闭
- **容器**: `connect.westc.seetacloud.com:26795`
- **目标模型**: AvatarForcing — One-Step Streaming Talking Avatars (1.3B)
- **官方代码**: [KlingAIResearch/AvatarForcing](https://github.com/KlingAIResearch/AvatarForcing)
- **官方权重**: [lycui/AvatarForcing (HF)](https://huggingface.co/lycui/AvatarForcing)；[Wan2.1-T2V-1.3B (HF)](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)；[wav2vec2-base-960h (HF)](https://huggingface.co/facebook/wav2vec2-base-960h)

## 环境准备

- Anaconda 虚拟环境 (`/root/autodl-tmp/conda/envs/avatarforcing`)，非 conda env name
- Python 3.10、PyTorch 2.5.1+cu121、FlashAttention 2.7.1.post4（GitHub Releases 预编译 wheel）
- 依赖：transformers 4.49.0、diffusers 0.35.2、peft 0.17.1、accelerate 1.11.0 等
- 输入音频补零到 `teacher_len=55` 对应约 55.25s，满足数据管线要求
- 网络限制：HF 直连不可达，所有模型通过 `hf-mirror.com` 镜像下载

## 模型权重

| 权重 | 位置 | 大小 | SHA-256 |
|------|------|:----:|--------|
| model.pt | `checkpoints/` | 19 GB | `63dd0b...` |
| Wan2.1-VAE | `wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth` | 485 MB | `38071a...` |
| T5 encoder | `wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth` | 11 GB | `7cace0...` |
| Wav2Vec2 | `wan_models/wav2vec2-base-960h/model.safetensors` | 361 MB | `8aa76a...` |

## 输入封装

一行三个字段（shlex 分割）：参考图路径、WAV 音频路径、文本 prompt。注意：
- `teacher_len` 对应 `data.teacher_len` 控制音频嵌入截取长度
- 从推理脚本看 `num_output_frames` 指的是潜空间帧数，与输出帧数并非 1:1

## 帧数约束

`num_output_frames - 1` 必须能被 `num_frame_per_block`（默认 4）整除。合法的帧数值为：21, 53, 93... 对应约 0.84s, 2.12s, 3.72s 的物理时长。

## 输出后处理

脚本内置 `ffmpeg` mux 音轨。但音频必须对齐 `teacher_len` 以确保足够送入 Wav2Vec2；推理后需用原音频 re-mux 并 `-shortest` 裁剪。本批次统一用 `--num_output_frames 53`（53-1=52, 52/4=13）生成约 8.36s 视频后后处理完成。

## AISHELL-1 Batch

- **输入**: 13 个 `natural_raw` + 13 个 `tts_raw`；样本 ID `1..13`
- **人像**: 原始 AISHELL-1 13 张参考图（`data/data/image/{1..13}.png`）
- **Natural 音频**: `data/data/audio/{1..13}.wav`（16kHz, mono）
- **TTS 音频**: `runs/r2_faster_qwen3_20260707T145233Z/02_tts/{1..13}.wav`（Qwen3 0.6B ICL），input
- **结果**: 26/26 成功，无失败；SHA-256 本地与远端全部一致

## 输出规格

- **分辨率**: 512×512
- **帧率**: 25 FPS
- **视频编码**: H.264
- **音频编码**: AAC，16kHz, mono（original audio 采样率）
- **总计**: 20MB（26 个文件）

## 性能

- 单样本推理：约 42s（RTX 4080 SUPER）
- 显存峰值：约 18–20GB（batch_size=1, 无 OOM）
- 性能瓶颈：四轮滑动窗口去噪 + VAE 解码

## SyncNet 评分

| 条件 | 样本数 | Sync-C ↑ | Sync-D ↓ |
|------|:---:|:---:|:---:|
| `natural_raw` | 13 | 2.687 | 11.275 |
| `tts_raw` | 13 | 2.577 | 11.773 |
| Δ (TTS − Natural) | — | **−0.110** | **+0.498** |

- 26/26 个视频评分成功，无失败。
- 逐样本结果：`results/avatarforcing/04_eval/`。

## 注意事项

- 输入 WAV 需为 16kHz mono
- Decord 0.6.0 的 Python wheel 标注 cp36 导致 `pip check` 警告，不影响实际运行
- FlashAttention 从 GitHub Releases 直接下载 cu12+torch2.5 预编译 wheel 避免无卡编译
- 真实音频越短时 `teacher_len` 越浪费；更高效的方案是把短音频视为第 1 段并在跳过数据管线中触发提前返回前补零到够用
