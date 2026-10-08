---
title: 'HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction
  of Hidden Units'
type: paper
permalink: tts-exp/papers/hu-bert-self-supervised-speech-representation-learning-by-masked-prediction-of-hidden-units
---

# HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units

arXiv 2106.07447。项目所有 feature alignment / 增强头工作的骨干特征模型。

Masked prediction 学习 hidden units;frame rate 约 50Hz(320 sample stride);layer 0/6/11/12 是项目常用离散层。音素/词信息集中在高层(与 Pasad 2023 一致)。

## Observations
- [status] cited
- [authors] Hsu, Tsai, Bolte, Salakhutdinov, Mohamed
- [url] https://arxiv.org/abs/2106.07447
- [key_insights] 项目 frozen backbone: facebook/hubert-base-ls960, 离线缓存可用
- [key_insights] HuBERT frame index 不是 phone boundary,需要 MFA 提供音素边界

## Relations
- part_of [[TTS Feature Alignment and Duration Model Research]]
