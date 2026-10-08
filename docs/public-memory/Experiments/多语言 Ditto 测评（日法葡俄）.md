---
title: 多语言 Ditto 测评（日法葡俄）
type: experiment
permalink: tts-exp/experiments/多语言-ditto-测评日法葡俄
tags:
- multilingual
- legacy
- ditto
- syncnet
- cross-language
---

# 多语言 Ditto 测评（日法葡俄）

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| Initial Document | August 14, 2026 | user |

2026-07-30 用 legacy 管线（13 样本协议）对 MDC 多语言采集的日语/法语/葡语/俄语音频做 Ditto + SyncNet 测评，检验中文朗读域 TTS 优势是否跨语言成立。数据获取背景见 [[multilingual-datasets]]。

## 方法

- 每语言 ≈13 条 utterance + faster_qwen3 0.6B ICL 配对 TTS，Ditto TRT 生成 natural_raw / tts_raw 双臂，SyncNet V2 评分
- 产物：`runs/legacy_{ja,fr,pt,ru}_20260730T*/`（完整 00–05 步）

## 结果

| 语言 | n | Natural C | TTS C | ΔC | p | 方向 |
|---|---:|---:|---:|---:|---:|---|
| 日语 ja | 13 | 5.590 | 5.541 | −0.049 | 0.803 | ❌ |
| 法语 fr | 12 | 6.474 | 6.468 | −0.067 | 0.889 | ❌ |
| 葡语 pt | 11 | 6.903 | 4.942 | −1.899 | 0.076 | ❌ (边际) |
| 俄语 ru | 13 | 4.828 | 5.900 | **+1.072** | **0.0027** | ✅ 显著 |

- 俄语 paired t p=0.0027、Cohen's d=1.04；Sync-D 方向 Δ+0.329 (p=0.12, ns)
- 葡语 tts 出现低分样本（min 0.459，疑似 TTS 克隆失败拉低均值）

## 结论

1. 4 语言中仅俄语显著复现 TTS 优势（+1.07）；日/法无差异，葡语边际负——TTS 唇同步红利不是跨语言普适现象
2. 与英文 LibriSpeech(+0.16 ns)/HDTF(−0.23) 一致：只有中文（AISHELL-1）与俄语成立，效应高度语言/语料特异
3. 俄语显著是「非中文语言成立」的首个正例，n=13 小样本，需复核

## Observations

- [status] concluded
- [hypothesis] 中文朗读域的 TTS 优势能跨语言推广
- [result] ja −0.049 / fr −0.067 / pt −1.899 均不成立；ru +1.072 (p=0.0027) 显著成立
- [conclusion] TTS 唇同步红利语言特异：中文、俄语成立；英/日/法/葡不成立
- [boundary] n≈11–13 小样本、legacy 管线协议；非严格 heldout 声明

## Relations

- relates_to [[multilingual-datasets]]
- relates_to [[跨数据集 TFG 测评（5×50 multiset）]]
- relates_to [[tts-exp]]
