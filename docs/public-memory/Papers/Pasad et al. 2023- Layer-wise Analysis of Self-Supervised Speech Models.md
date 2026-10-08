---
title: 'Pasad et al. 2023: Layer-wise Analysis of Self-Supervised Speech Models'
type: paper
permalink: tts-exp/papers/pasad-et-al.-2023-layer-wise-analysis-of-self-supervised-speech-models
---

# Pasad et al. 2023: Layer-wise Analysis of Self-Supervised Speech Models

ICASSP 2023, arXiv 2211.03929。层选择是系统性决定因素。

HuBERT 族(WavLM/AV-HuBERT)音素/词信息集中在高层;wav2vec2/XLS-R-53 族在中间层达峰、高层下降。项目当前 0/6/11/12 离散层对 HuBERT 合理,对 XLS-R 可能错过最优中间层,应单独标定。

## Observations
- [status] cited
- [authors] Pasad et al.
- [url] https://arxiv.org/abs/2211.03929
- [key_insights] B2 实验(逐层可分离性曲线)的理论依据 — XLS-R 需标定中间层

## Relations
- part_of [[TTS Feature Alignment and Duration Model Research]]
