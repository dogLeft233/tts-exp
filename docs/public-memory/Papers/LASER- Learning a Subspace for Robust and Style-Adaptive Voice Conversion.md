---
title: 'LASER: Learning a Subspace for Robust and Style-Adaptive Voice Conversion'
type: paper
permalink: tts-exp/papers/laser-learning-a-subspace-for-robust-and-style-adaptive-voice-conversion
---

# LASER: Learning a Subspace for Robust and Style-Adaptive Voice Conversion

arXiv 2406.09153。最直接的 HuBERT/WavLM SSL sequence↔sequence soft-DTW 近邻。

用于 voice conversion 的 SSL soft-DTW 对齐方案,与项目 B 主线(constrained soft-DTW)最接近的已发表工作;复现成本较高、长序列 O(TU) 显存。

## Observations
- [status] cited
- [url] https://arxiv.org/abs/2406.09153
- [key_insights] HuBERT/WavLM SSL 序列级 soft-DTW 的先例,证明 SSL↔SSL 对齐可行
- [key_insights] 复现成本高,项目内先用 offline hard DTW/linear 对照

## Relations
- part_of [[TTS Feature Alignment and Duration Model Research]]
