# AVTR-1 部署记录

## 状态

- **状态**: ✅ 官方示例和 AISHELL-1 26 条批量推理完成；服务器已关闭
- **容器**: `connect.westd.seetacloud.com:48111`
- **目标模型**: AVTR-1（开放权重）
- **官方代码**: [avtr-1](https://github.com/avaturn-live/avtr-1)
- **官方权重**: [HF](https://huggingface.co/avaturn-live/avtr-1)
- **官方环境**: Python 3.12、PyTorch 2.7.1 + CUDA 12.8、TensorRT 10.11.0.33
- **当前运行环境**: Pixi 管理的独立环境，Torch 2.7.1+cu128

## 环境准备

- 代码在已有 `/root/autodl-tmp/avtr-1` 目录（前一环境迁移），未做覆盖重装
- Pixi cache 已存在，首次运行下载 artifacts 约 2.82GB（44 个文件）
- TRT engines 7/7 已预构建（build-avtr1）
- `pip install git+https://github.com/avaturn-live/avtr-1` 安装完毕

## TRT Engines

| 引擎 | 状态 |
|------|:---:|
| AVTR1 encode | ✅ |
| AVTR1 decode | ✅ |
| Hubert | ✅ |
| Decoder | ✅ |
| Warp (含 grid-sample plugin) | ✅ |
| Stitch | ✅ |
| MODNet matting | ✅ |

## Smoke Test

- **时间**: 2026-07-29
- **GPU**: NVIDIA GeForce RTX 4080 SUPER，32,760 MiB
- **输入**: 官方默认（avatar maria，未指定 `--speech` 时使用静音 + `--duration 60`）
- **命令**: `pixi run generate_offline --duration 60 --bg minimal_office`
- **输出**: `/root/autodl-tmp/avtr-1/avtr1_test.mp4`
- **结果**: 1280×720、25 FPS、60.2s；H.264 视频 + AAC 音频；`ffmpeg` 解码检查通过

## AISHELL-1 Batch

- **输入**: 13 个 `natural_raw` + 13 个 `tts_raw`；样本 ID `1..13`
- **人像**: 自定义 `portraits_dir` 传入原始 AISHELL-1 13 张参考图（`data/data/image/{1..13}.png`）
- **Natural 音频**: `data/data/audio/{1..13}.wav`（16kHz, mono）
- **TTS 音频**: `runs/r2_faster_qwen3_20260707T145233Z/02_tts/{1..13}.wav`（24kHz, mono），faster_qwen3 0.6B ICL
- **批处理脚本**: `scripts/avtr1_batch.py`，一次性加载 13 个 avatar + 遍历条件
- **背景**: `--bg minimal_office`
- **结果**: 26/26 成功，无失败；全部 MP4 SHA-256 与远端一致

## 输出规格

所有 26 个输出视频：
- **分辨率**: 1280×720（720P）
- **帧率**: 25 FPS
- **视频编码**: H.264 (libx264)
- **音频编码**: AAC，单声道
  - `natural_raw`: 16kHz（原音频采样率）
  - `tts_raw`: 24kHz（TTS 输出采样率）
- **总计**: 74MB

## 性能

- 单样本推理：约 3–5s（取决于音频长度）
- 显存峰值：未记录但无 OOM（32GB 卡）
- 与 SoulX-FlashHead 不同：AVTR-1 在 RTX 4080 上接近实时，无需预热 chunk

## SyncNet 评分

| 条件 | 样本数 | Sync-C ↑ | Sync-D ↓ |
|------|:---:|:---:|:---:|
| `natural_raw` | 13 | 5.460 | 8.830 |
| `tts_raw` | 13 | 5.788 | 8.632 |
| Δ (TTS − Natural) | — | **+0.327** | **−0.198** |

- 26/26 个视频评分成功，无失败。
- 逐样本结果：`results/avtr1/04_eval/`。

## 注意事项

- 关机遇阻：容器 PID 1 为 bash，`/sbin/shutdown` 不存在，需使用 `/usr/bin/shutdown`
- License：Renderer/Streamer 为 PolyForm Noncommercial；InsightFace 预训练权重非商用
- AVTR-1 对普通话的自然 raw 音频和 TTS 音频均无 WER/TER 过滤条件
