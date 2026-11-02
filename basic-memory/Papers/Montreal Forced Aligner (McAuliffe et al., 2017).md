---
title: Montreal Forced Aligner (McAuliffe et al., 2017)
type: paper
permalink: tts-exp/papers/montreal-forced-aligner-mc-auliffe-et-al.-2017
---

# Montreal Forced Aligner (McAuliffe et al., 2017)

Interspeech 2017。项目所有 alignment 工作(MFA-linear target、phone pooling、hard DTW)的边界来源。

Kaldi acoustic-model forced aligner:audio + transcript + pronunciation dictionary + acoustic model → word/phone intervals。英语数据 phone boundary mean error 约 17–25ms,median 约 11ms,90% <50ms,仍有 2–5% token error ≥100ms。log-likelihood 是相对 best-path 分数,不是绝对 confidence。

## Observations
- [status] cited
- [authors] McAuliffe, Socolof, Mihuc, Wagner, Sonderegger
- [url] https://www.isca-archive.org/interspeech_2017/mcauliffe17_interspeech.html
- [key_insights] MFA 是有前提的 boundary estimator,不是无条件精确切分器
- [key_insights] 使用前先 mfa validate,审计 OOV/unaligned/短 interval/duration deviation
- [key_insights] 当前 AISHELL/普通话注意发音变体、连读、sil/spn、短 phone、transcript mismatch

## Relations
- part_of [[TTS Feature Alignment and Duration Model Research]]
