---
title: TTS 音素可分性机制 LRS3 严格 Atlas 与自然时钟构造 2026-09-21
type: experiment
permalink: tts-exp/experiments/tts-音素可分性机制-lrs3-严格-atlas-与自然时钟构造-2026-09-21
status: concluded
protocol: phone_separability_mechanism_v1
scientific_status: ATLAS_SUPPORTED_RULE_GATE_NEGATIVE
tags:
- tts
- phoneme-separability
- lrs3
- natural-clock
- exploratory
---

# TTS 音素可分性机制 LRS3 严格 Atlas 与自然时钟构造 2026-09-21

## Observations
- [status] concluded
- [progress] 本次 `phone_separability_mechanism_v1` 的 LRS3 机制与规则分支已完成，科学状态为 `ATLAS_SUPPORTED_RULE_GATE_NEGATIVE`；这不等于整个自然音频增强问题已被证明不可行。
- [protocol] 240 paired samples / 480 assets；0 missing；FIT/DEV 按 source_group 隔离，E_SEEN 40 groups 只作已见评估，不参与选择。主模型 HuBERT layer 6，交叉模型 XLSR layer 10。
- [evidence] HuBERT E_SEEN T−N accuracy：full +0.09260（95% CI [+0.06777,+0.11913]）、core +0.10517（[+0.07762,+0.13438]）、matched_1frame +0.10576（[+0.07415,+0.14027]）；margin 分别 +0.01303、+0.01471、+0.01332。XLSR 对应 +0.08999、+0.09459、+0.07670，CI 均不跨 0。
- [boundary] boundary_start 的优势较小（HuBERT +0.03939、XLSR +0.03092），boundary_end 仅4个 E_SEEN groups，DEV 无 eligible groups；不能把边界统计当主证据。中段和 matched_1frame 仍有稳定优势。
- [mechanism] 来源域 AUC N-vs-T：HuBERT DEV/E_SEEN=0.704/0.715，XLSR=0.664/0.683；ABX error 也偏向 TTS。域/声学实现差异是候选解释，但这些指标不是 phone 因果证据。
- [construction] FIT-only phone spectral-shape templates 有71个可用标签；生成1050条 natural-clock arm WAV，10 arms，全部 waveform contract 通过。输出保持自然样本数、mask外 PCM 和自然时间轴；E_SEEN 未参与选择。
- [candidate_gate] 候选 HuBERT/XLSR 重抽取已完成。HuBERT DEV 的6个 selectable rules 均未通过预注册 gate；最佳 `PHONE_SHAPE_BETA1_SPEECH` core accuracy +0.003893（95% CI [-0.001155,+0.009166]），margin +0.000718（CI [+0.000425,+0.001010]），远低于 TTS core baseline +0.076801。XLSR 为 held-out 交叉核验，未改变选择。
- [training] `SKIPPED_NO_RULE_PASSES_DEV_GATE`；没有进入条件神经训练，避免在规则信号为负/不确定时把训练结果误当机制证据。
- [validation] 独立 checker：480 bindings、1050 construction rows、atlas statistics 均 PASS；测试 14 passed；candidate、training decision、report 均已写入 run。
- [artifact] full run：`runs/phone_separability_mechanism_full_v2_20260921/`；报告 `08_report/report.md`；候选选择 `07_candidates/selection.json`；候选 runner：`scripts/experiments/phone_separability_mechanism/candidate.py`。
- [decision] LRS3 原始 TTS 的操作性 phone probe 优势得到严格复现，但本轮自然时钟规则未复现它。这个结果更具体地支持“优势不是简单的全局谱形/响度/局部规则可直接搬运”，下一步应转向更明确的时序-动态、声码器/录音域或可微目标设计，并重新注册门槛；不能把它写成 TFG/SyncNet 因果结论。
## Relations

- implements [[TTS 音素可分性机制与自然时钟增强探索 Implementation Spec]]
- follows [[LRS3 音素优势与自然音频规则增强实现审计 2026-09-20]]
- relates_to [[TTS 增强 TFG 的原因：证据、排除项与因果分叉]]
