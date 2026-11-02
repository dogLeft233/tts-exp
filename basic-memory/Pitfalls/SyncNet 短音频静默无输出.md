---
title: SyncNet 短音频静默无输出
type: pitfall
permalink: tts-exp/pitfalls/sync-net-短音频静默无输出
tags:
- pitfall
- syncnet
- audio
- short-audio
---

# SyncNet 短音频静默无输出

## 现象

短音频(<4s)视频转 25fps 后帧数 <100,`run_pipeline.py` 默认 `min_track=100` 导致 track 为空 → `run_syncnet.py` 静默无输出,评分缺失且无任何报错。

## 原因

SyncNet V2 的 S3FD 人脸检测依赖 track 长度阈值,短音频(常见于 AISHELL 短句)帧数不足时 track 为空,后续直接跳过。

## 修复

评测时传 `--min-track 50`。增强头下游评估(50 valid pairs)即用此参数完成。

## 位置

`third_party/syncnet_python/`,run_pipeline.py / run_syncnet.py。

## Observations
- [area] syncnet
- [symptom] 短音频评分静默缺失,无报错
- [cause] min_track=100 阈值对 <100 帧视频过滤掉全部 track
- [fix] --min-track 50

## Relations
- relates_to [[tts-exp]]
