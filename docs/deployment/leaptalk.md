# LeapTalk 部署记录

## 状态

- **状态**: ✅ AISHELL-1 26 条批量推理完成；服务器已关闭
- **容器**: `connect.westc.seetacloud.com:41601`
- **目标模型**: LeapTalk (ViBT) — 基于 SoulX-FlashHead 1.3B 的 1-step 流式 talking head
- **官方代码**: [LeapTalk](https://github.com/zhangrongxiang/LeapTalk)
- **官方权重**: [SoulX-FlashHead-1_3B (HF)](https://huggingface.co/Soul-AILab/SoulX-FlashHead-1_3B)；[LeapTalk LoRA (HF)](https://huggingface.co/z-rx/leaptalk)；[wav2vec2-base-960h (HF)](https://huggingface.co/facebook/wav2vec2-base-960h)

## 环境准备

- 系统 Python 3.12.13 (deadsnakes PPA)，venv 在数据盘 `/root/autodl-tmp/leaptalk_venv`
- PyTorch 2.7.1+cu126、CUDA 12.6、RTX 4080 SUPER 32GB
- 权重总约 6.5GB：FlashHead Pro 5.7GB、VAE 485MB、wav2vec2 361MB、LeapTalk LoRA+proj+TAE 385MB
- 全部通过 `hf-mirror.com` 代理下载，使用 Seetacloud 内网 `network_turbo` 代理加速
- `mediapipe` 需降至 0.10.21（新版本取消了 `mp.solutions` 顶层 API）
- 未安装 `flash_attn`，使用 PyTorch attention 实现

## 输入

与统一评估协议一致：13 张 AISHELL-1 参考图 × 13 条 natural raw 音频 + 13 条 TTS raw 音频。
- natural raw：16kHz，原始 AISHELL-1 音频
- TTS raw：24kHz，faster_qwen3 0.6B ICL 生成（从 SoulX-FlashHead TTS 输出视频提取）

## 输出规格

- **分辨率**: 512×512
- **帧率**: 25 FPS
- **视频编码**: H.264
- **音频编码**: AAC（原始 MP3 已通过 ffmpeg 替换为原始 WAV 重封装）
  - natural: 16kHz, mono
  - tts: 24kHz, mono
- **总计**: 11MB（26 个文件）
- **结果目录**: `results/leaptalk/{natural_raw,tts_raw}/{1..13}.mp4`

## 性能

- 单样本推理：约 38–44s（RTX 4080 SUPER）
- 生成速度：约 40 FPS（扣除前 2 个预热 chunk 后平均）
- 每 chunk（28 帧）约 0.69s 推理时间 + 0.10s 解码
- 显存峰值：约 8–9GB（使用 `--lite` TAE 轻量模式）
- 固定 seed=42

## SyncNet 评分

| 条件 | 样本数 | Sync-C ↑ | Sync-D ↓ |
|------|:---:|:---:|:---:|
| `natural_raw` | 13 | 6.222 | 8.979 |
| `tts_raw` | 13 | **7.221** | **8.111** |
| Δ (TTS − Natural) | — | **+0.999** | **−0.868** |

- 26/26 个视频评分成功，无失败。
- 逐样本结果：`results/leaptalk/04_eval/`。

## 注意事项

- LeapTalk 的 `CPUFaceHandler` 使用 `mediapipe.solutions.face_detection`，需 `mediapipe==0.10.21`
- 模型加载时需要 `Model_Pro/config.json`，只下载大权重不够；需从 HF 仓库补齐小文件
- LeapTalk 默认封装 MP3 音轨，与项目评估脚本期望的 AAC 不一致，需 post-process remux
- 远程服务器连接已关闭（端口 41601），不再可 SSH
