---
title: 'Soft-DTW: A Differentiable Loss Function for Time-Series'
type: paper
permalink: tts-exp/papers/soft-dtw-a-differentiable-loss-function-for-time-series
---

# Soft-DTW: A Differentiable Loss Function for Time-Series

Cuturi 2017 (ICML)。项目 constrained soft-DTW 主线的理论基座。

soft-DTW 将 hard min 替换为 soft-min,对所有合法 paths 可微聚合;支持变长/time stretch,但本身没有 phone boundary、duration 或 output length constraint。单独使用会 collapse,需要 phone anchor/sil mask/band/regularizer。

## Observations
- [status] cited
- [authors] Marco Cuturi
- [url] https://proceedings.mlr.press/v70/cuturi17a.html
- [key_insights] soft-DTW 的 expected alignment matrix 是软占用,行列和不一定为 1,不能直接当 OT coupling
- [key_insights] 仅作 loss 时无逐帧 target 标签;当前项目不优先引入,先做 explicit duration/pause target

## Relations
- part_of [[TTS Feature Alignment and Duration Model Research]]
