---
title: syncnet
type: note
permalink: tts-exp/docs/deployment/syncnet
tags:
- syncnet
- 部署
- deployment
- 评估
- 评分
- v100
---

# SyncNet V2 本地部署记录

SyncNet V2 是项目唇形同步评估核心 (04_eval.py),在本地 V100 上完成完整部署:人脸检测 (S3FD) + 裁剪 (run_pipeline.py) + 评分 (run_syncnet.py)。代码、权重、环境与服务器 (RTX 4080, syncnet conda env) 版本对齐,同一视频评分结果与服务器一致 (Confidence 2.192 vs 2.193)。

## 状态

- **状态**: ✅ 本地 V100 完整流程验证通过 (2026-08-11), 分数与服务器一致
- **来源**: third_party/syncnet_python (从服务器同步, 含权重)
- **远程参考**: 服务器 `/root/autodl-tmp/envs/syncnet` (conda, torch 2.5.1)

## 环境 (本地 venv, 与服务器版本对齐)

| 组件 | 版本 |
|------|------|
| venv | `~/.venvs/syncnet` (python 3.12) |
| torch / torchaudio / torchvision | 2.5.1+cu124 / 2.5.1 / 0.20.1 |
| numpy | 2.2.6 |
| opencv-contrib-python | 4.13.0.92 |
| scipy | 1.15.2 |
| librosa | 0.11.0 |
| python_speech_features | 0.6 |
| scenedetect | 0.6.7.1 (run_pipeline.py 需要) |
| scikit-learn | 1.7.2 |
| pillow | 12.3.0 |

## 权重位置

- 仓库: `/home/wjj/tts-audio/tts-exp/third_party/syncnet_python/`
- `data/syncnet_v2.model` (54.6MB) — SyncNet 权重
- `detectors/s3fd/weights/sfd_face.pth` (89.8MB) — 人脸检测
- `data/example.avi` — 自带示例

## 完整流程 (04_eval.py 同款两步)

```bash
cd third_party/syncnet_python
# Step 1: 人脸检测 + 裁剪 (需要 scenedetect + s3fd)
~/.venvs/syncnet/bin/python run_pipeline.py --videofile <video.mp4> \
  --reference <tag> --data_dir <work> --overwrite
# Step 2: 评分
~/.venvs/syncnet/bin/python run_syncnet.py --videofile <video.mp4> \
  --reference <tag> --data_dir <work> --initial_model data/syncnet_v2.model
```

- 输出解析: `AV offset`, `Min dist` (→Sync-D), `Confidence` (→Sync-C), 见 scripts/pipeline/04_eval.py
- 服务器运行需把 syncnet env 的 bin (含 ffmpeg) 注入 PATH; 本地 ffmpeg 在系统 PATH

## 验证结果 (同一视频, multiset_pipeline natural_raw/4.mp4)

| 指标 | 本地 V100 | 服务器 RTX 4080 |
|------|:---:|:---:|
| AV offset | 0 | 0 |
| Confidence (Sync-C) | 2.192 | 2.193 |

## 注意事项

- `demo_syncnet.py` 直接喂原帧给网络, 只适合 224x224 输入; 真实视频必须走 run_pipeline.py 裁剪
- 人脸检测用 CUDA (S3FD), V100 上 ~100 it/s
- 系统需 ffmpeg

## Observations

- [location] venv at ~/.venvs/syncnet (python 3.12), torch 2.5.1+cu124 #deployment
- [location] repo at /home/wjj/tts-audio/tts-exp/third_party/syncnet_python/ #deployment
- [location] weights data/syncnet_v2.model (54.6MB) + s3fd/weights/sfd_face.pth (89.8MB) #deployment
- [workflow] 评分必须两步: run_pipeline.py (S3FD 人脸检测裁剪) → run_syncnet.py #evaluation
- [warning] demo_syncnet.py 只适合 224x224 输入, 真实视频必须走 run_pipeline.py 裁剪 #evaluation
- [interface] 输出解析字段: AV offset, Min dist (→Sync-D), Confidence (→Sync-C) #evaluation
- [verification] 同一视频本地 vs 服务器: Confidence 2.192 vs 2.193, AV offset 均 0 #reproducibility
- [dependency] run_pipeline.py 需要 scenedetect + scikit-learn + ffmpeg #dependency
- [performance] S3FD 人脸检测 V100 上 ~100 it/s (CUDA) #benchmark

## Relations

- part_of [[tts-exp]]
- evaluates [[qwen3-tts]]
- used_by [[tedlium3-tts-augmentation]]
- relates_to [[local-network-proxy-notes]]
