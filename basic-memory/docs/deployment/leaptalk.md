---
title: leaptalk
type: note
permalink: tts-exp/docs/deployment/leaptalk
---

# LeapTalk 部署记录

## 当前无卡部署（2026-09-17）

- 当前服务器：`connect.weste.seetacloud.com:40091`；部署根目录：`/root/autodl-tmp/tts-exp-leaptalk`。
- 状态：🔄 软件、模型、输入、预检和 GPU 推理脚本已完成；服务器当前没有可用 NVIDIA GPU，因此尚未生成 GPU 推理 MP4。
- 官方代码固定在 commit `5d8fef8dac3d1c9d3b3d5d9cceb6c76acc81e2c1`；模型下载使用 `/etc/network_turbo`、`HF_ENDPOINT=https://hf-mirror.com` 和 `HF_HUB_DISABLE_XET=1`，只保留推理路径需要的文件。完整 revision、文件大小和 SHA256 在 `models/manifest.json`。
- 独立环境：`envs/leaptalk`，Python 3.12.14，PyTorch `2.7.1+cu126`，torchvision/torchaudio `0.22.1+cu126`/`2.7.1+cu126`。为 LeapTalk 的 MediaPipe 路径固定了 `mediapipe==0.10.21`、`opencv-contrib-python==4.10.0.84`、`numpy==1.26.4`、`protobuf==4.25.8`；`pip check` 通过，既有服务器环境未修改。
- 推理配置：官方 Lite、`Model_Pro`、TAE、1 个 denoising step、512×512、25 FPS、16 kHz、BF16、`--compile off`、`--usp off`、单进程单卡 `torchrun`。
- 固定输入：官方 `assets/photo.jpg`；`assets/news.wav` 的前 4 秒，转为单声道 16 kHz PCM16，不做增益或降噪；来源和产物哈希在 `inputs/manifest.json`。
- 系统没有 ffmpeg 命令；`bin/ffmpeg` 已指向环境内的 `imageio-ffmpeg` 二进制，官方音频复用路径和验证脚本均通过该入口运行。

### 已完成验证

- `bin/preflight.sh`：模型、输入、依赖均无缺失；官方 `inference` 导入成功；实际调用 `CPUFaceHandler` 在固定图片上检测到 1 张人脸。
- 预检结果：`imports_ok=true`、`missing_files=[]`、`torch.cuda.is_available()=false`、设备数为 0、`nvidia-smi` 为 `present_but_unavailable`，状态为 `GPU_REQUIRED_PENDING`，退出码 75。
- `bin/run_gpu_smoke.sh` 的无卡门禁已验证：退出码 75，只留下 `preflight.json` 和日志，没有启动推理进程。
- 过期 MediaPipe/OpenCV/NumPy/Protobuf/Tifffile/decord wheel 和 pip 下载缓存已清理；数据盘剩余约 35 GB。当前 wheel cache 仅保留可重建环境所需版本。
- MediaPipe 兼容性注意：`0.10.35` 虽能导入，但不提供代码依赖的 `mp.solutions`；当前版本组合已通过运行级人脸检测。

### 开卡后的运行命令

```bash
cd /root/autodl-tmp/tts-exp-leaptalk
RUN_ID=leaptalk_smoke_01 ./bin/run_gpu_smoke.sh
```

成功后查看 `runs/gpu_smoke_<RUN_ID>/verification.json`、`gpu_samples.csv`、`inference.log`、`review/{head,middle,tail}.png` 和 `leaptalk.mp4`。直接推理可运行：

```bash
./bin/run_inference.sh \
  --audio inputs/smoke_4s.wav \
  --image inputs/photo.jpg \
  --out runs/manual.mp4
```

当前无卡状态下没有科学实验结果；该部署为后续开卡推理提供可复现的运行入口。旧服务器记录和历史实验结果保留在本条目的后续内容中。
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