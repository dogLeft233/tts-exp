---
title: 'Wav2Sem: Plug-and-Play Audio Semantic Decoupling for 3D Speech-Driven Facial
  Animation'
type: paper
permalink: tts-exp/papers/wav2-sem-plug-and-play-audio-semantic-decoupling-for-3-d-speech-driven-facial-animation
---

# Wav2Sem: Plug-and-Play Audio Semantic Decoupling for 3D Speech-Driven Facial Animation

CVPR 2025。表征→TFG 的已发表因果先例,项目 34 号 near-homophone 分析直接按此范式实现。

对近同音词注入句级 BERT 语义向量后,SSL 空间 L2 可分离度提升(Wav2Vec2 0.0397→0.0701 ≈ +77%;HuBERT 0.2689→0.2909),并传递到六个 TFG 基线唇形几何改善(VOCASET FaceFormer LVE 4.1090→3.9891;BIWI FaceDiffuse MVE 6.8088→6.5725)。

## Observations
- [status] cited
- [authors] Li et al.
- [url] https://arxiv.org/abs/2505.23290
- [key_insights] 干预"音频语义可分离度"→ 唇形同步改善的因果先例,为项目把可分性指标关联 SyncNet-C 提供已发表范式
- [key_insights] 项目 scripts/34_near_homophone_analysis.py 已按此范式实现中文送气/发音部位/声调对照

## Relations
- part_of [[TTS Feature Alignment and Duration Model Research]]
- relates_to [[tts-exp]]
