---
title: 'Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search'
type: paper
permalink: tts-exp/papers/glow-tts-a-generative-flow-for-text-to-speech-via-monotonic-alignment-search
---

# Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search

arXiv 2005.11129。可微 duration/soft alignment 的 TTS 先例。

训练时 MAS/maximum path 得到 duration;推理不运行 MAS,使用 duration predictor + ceil/length regulator。与 VITS/DISSC 同属"可微 duration/warping alignment"主线。

## Observations
- [status] cited
- [authors] Kim, Kim, Kong, Sung
- [url] https://arxiv.org/abs/2005.11129
- [key_insights] MAS 训练/推理不对称范式(duration predictor 推理)是项目 C 主线的参考

## Relations
- part_of [[TTS Feature Alignment and Duration Model Research]]
