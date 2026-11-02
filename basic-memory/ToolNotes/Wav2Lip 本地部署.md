---
title: Wav2Lip 本地部署
type: tool_note
permalink: tts-exp/tool-notes/wav2-lip-本地部署
status: active
tags:
- tool-note
- wav2lip
- deployment
---

# Wav2Lip 本地部署

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| Initial Document | August 12, 2026 | user |

## Observations

- [tool] Wav2Lip

## 状态

- **status**: active
- **状态**: ✅ 本地 V100 推理验证通过 (2026-08-12)
- **完整部署记录**: [[wav2lip]] (docs/deployment/wav2lip.md) — 环境/修复/验证细节

## 要点

- 位置: `third_party/Wav2Lip` (官方 Rudrabha/Wav2Lip, depth-1 克隆)
- 权重: `wav2lip_gan.pth` (436MB, hf-mirror 镜像 `Nekochu/Wav2Lip`) + `s3fd.pth` (89.8MB, adrianbulat.com)
- 环境: `~/.venvs/wav2lip` (python 3.12, torch 2.6.0+cu124, librosa 1.0, opencv 5.0)
- 显存: 峰值 ~4.2GB, V100 16GB 完全够用 (官方要求 4-8GB)
- 性能: 6.4s 音频约 11s 推理 (实时率 ~0.6)
- 验证: Sync-C=6.460, AV offset=-2 (SyncNet V2 完整流程)

## 踩坑

- 官方 requirements (torch 1.1) 与新版环境不兼容,需修 4 处 API:
  `librosa.core.load`→`librosa.load`、`librosa.output.write_wav`→`wavfile.write`、`librosa.filters.mel` 改 keyword-only 传参、`np.int`→`np.int32`
- S3FD 人脸检测对部分样本中段会漏检 (stableavatar natural_raw/6.mp4 157 帧漏 78 帧),报 `Face not detected!` — 换视频或用 `--box`/`--crop` 兜底
- 官方 sharepoint 权重链接国内不通,走 hf-mirror

## Relations

- relates_to [[tts-exp]]
- relates_to [[wav2lip]]
