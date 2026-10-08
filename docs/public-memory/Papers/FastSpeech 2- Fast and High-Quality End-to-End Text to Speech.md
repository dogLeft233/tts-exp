---
title: 'FastSpeech 2: Fast and High-Quality End-to-End Text to Speech'
type: paper
permalink: tts-exp/papers/fast-speech-2-fast-and-high-quality-end-to-end-text-to-speech
---

# FastSpeech 2: Fast and High-Quality End-to-End Text to Speech

arXiv 2006.04558。C 主线(duration predictor + length regulator)的标杆模型。

FastSpeech2 用 MFA duration 作训练 target,推理由 predictor 预测 duration;使用 log(duration+1) + masked MSE。TTS duration/length regulator 通常只允许一个 phone → 多个 acoustic frames,不支持 generic DTW 任意双向占用。

## Observations
- [status] cited
- [authors] Ren et al.
- [url] https://arxiv.org/abs/2006.04558
- [key_insights] duration expansion/length regulator 是"TTS rhythm 是目标"时的成熟路线
- [key_insights] 推理需要 transcript/phone sequence;natural waveform-only 无 transcript 时不适用

## Relations
- part_of [[TTS Feature Alignment and Duration Model Research]]
