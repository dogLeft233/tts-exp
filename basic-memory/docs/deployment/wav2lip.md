---
title: wav2lip
type: note
permalink: tts-exp/docs/deployment/wav2lip
tags:
- wav2lip
- 部署
- deployment
- TFG
- 唇形同步
---

# Wav2Lip 本地部署记录

## 状态

- **状态**: ✅ 本地 V100 推理验证通过 (2026-08-12)
- **来源**: third_party/Wav2Lip (官方 Rudrabha/Wav2Lip 仓库, depth-1 克隆)
- **推理权重**: `checkpoints/wav2lip_gan.pth` (436MB, 来自 hf-mirror 镜像 `Nekochu/Wav2Lip`)
- **人脸检测权重**: `face_detection/detection/sfd/s3fd.pth` (89.8MB, adrianbulat.com)

## 显存确认 (2026-08-12 上网核对 + 实测)

- 官方文档/第三方部署指南: Basic 4GB / HQ 6GB / HD 8GB VRAM 即可
- **本机实测峰值: ~4.2GB** (Wav2Lip GAN 推理 + S3FD 检测, batch 默认)
- V100 16GB 完全够用 (比训练/大模型低一个量级)

## 环境 (本地 venv)

| 组件 | 版本 |
|------|------|
| venv | `~/.venvs/wav2lip` (python 3.12) |
| torch / torchvision | 2.6.0+cu124 / 0.21.0 (V100 必须 cu124, CUDA 13 不支持 sm_70) |
| numpy | 2.5.2 (torch 依赖拉入; 无 np.int 问题) |
| librosa | 1.0.0 (需两处 API 修复, 见下) |
| opencv-contrib-python / scipy / tqdm / soundfile | 最新 |

## 官方 requirements 太老, 需手动修复的兼容点

1. `audio.py:10` — `librosa.core.load` → `librosa.load` (librosa 0.10+ 移除 core 别名)
2. `audio.py:18` — `librosa.output.write_wav` → `wavfile.write` (0.8+ 移除 output 模块)
3. `audio.py:100` — `librosa.filters.mel(sr=, n_fft=, ...)` 改为 keyword-only 传参 (librosa 1.0)
4. `face_detection/utils.py:115` — `dtype=np.int` → `np.int32` (numpy 2.x 移除 np.int)

## 使用

```bash
~/.venvs/wav2lip/bin/python inference.py \
  --checkpoint_path checkpoints/wav2lip_gan.pth \
  --face <video.mp4> --audio <speech.wav> \
  --outfile result.mp4
```

- 视频可无声; 音频任意 (内部重采样到 16k mel)
- 人脸检测失败会抛 `ValueError('Face not detected!')` — 先单测帧检测, 见下

## 验证记录 (2026-08-12)

- 输入: syncnet example.avi (317 帧全检测通过) + 16kHz 单声道 wav (6.4s)
- 输出: 224x224 h264+aac mp4, 时长 6.4s, 帧率与原视频一致
- **SyncNet V2 评分: Sync-C=6.460, AV offset=-2** (run_pipeline + run_syncnet 完整流程)
- 推理速度: 6.4s 音频约 11s (V100), 实时率 ~0.6

## 注意事项

- 人脸检测 (S3FD) 对部分样本中段会漏检 (实测 stableavatar natural_raw/6.mp4 157 帧中 78 帧漏检), 报 Face not detected — 换视频或加 `--box`/`--crop` 兜底
- 系统需 ffmpeg (输出 mux 用)
- 权重下载: 官方 sharepoint 链接国内不通, 用 hf-mirror 镜像 (`https://hf-mirror.com/Nekochu/Wav2Lip/resolve/main/wav2lip_gan.pth`)
- 远程参考: 服务器上有 Wav2Lip 结果 (`results/tfg_wav2lip`), Sync-C 旧数据 +1.515
