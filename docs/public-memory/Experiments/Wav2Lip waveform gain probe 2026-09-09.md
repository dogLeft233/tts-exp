---
title: Wav2Lip waveform gain probe 2026-09-09
type: experiment
permalink: tts-exp/experiments/wav2-lip-waveform-gain-probe-2026-09-09
status: concluded
date: '2026-09-09'
report: openspec/changes/probe-wav2lip-waveform-gain/
tags:
- wav2lip
- replacement
- parallel
- concluded
---

# Wav2Lip waveform gain probe 2026-09-09

历史响度机制未确立且曾受Ditto重复噪声混杂。用确定性Wav2Lip、固定自然音轨和可精确实现的波形标量，测试驱动前端的增益响应。

本次已按spec完成实验、独立复核和BM收尾；数值结论见Observations。总交接 `openspec/parallel-replacement-next-20260909.md`，历史证据综述 `openspec/replacement-history-review-20260909.md`。本轮共五路：新增四路和既有视觉教师missingness审计；CPU并行准备，GPU单锁串行，不能把并行当作多个GPU模型同时驻留。

## Observations
- [status] concluded
- [question] 在固定Wav2Lip和固定自然音轨下，确定性±3 dB波形增益是否产生natural-anchor replacement收益。
- [protocol] `openspec/changes/probe-wav2lip-waveform-gain/`；16条/8组，候选为GAIN_PLUS/GAIN_MINUS，评估音频始终是original natural PCM，不训练、不下载模型。
- [result] 工程结论为 `GO`，科学结论为 `NO_WAVEFORM_GAIN_ESTABLISHED`。实际生成32个视频、36个score cells（32候选+4 fresh controls），低于34/36上限。
- [result] GAIN_PLUS：ΔC mean=+0.017156，99%CI=[+0.000336,+0.036350]；ΔD mean=-0.004982，99%CI=[-0.051531,+0.040553]；ΔA mean=-0.007733，99%CI=[-0.060844,+0.040418]；三项联合正组3/8。
- [result] GAIN_MINUS：ΔC mean=+0.007527，99%CI=[-0.021623,+0.042866]；ΔD mean=+0.001007，99%CI=[-0.015816,+0.021161]；ΔA mean=+0.000311，99%CI=[-0.017906,+0.020990]；三项联合正组1/8。
- [conclusion] 波形响度增益没有通过完整replacement门槛；不能据此设计waveform head，也不能把轻微ΔC正值当作稳定机制。
- [validation] `validation.json` 为 PASS，`independent=true`；validator独立重建PCM、验证原始自然音轨绑定、从visual/audio embedding重建矩阵并独立计算统计。fresh replay/parity的matrix、endpoint、offset均复核通过。
- [review] 修复score result self-hash、media/model/plan绑定和stale error；最终分析、验证、review互相绑定。
- [artifact] `runs/wav2lip_waveform_gain_20260910_v1/`；实际预算记录在`final.json`，为32 videos / 36 score cells。
- [boundary] `replacement_confirmed=false`、`waveform_head_authorized=false`、`generalization_established=false`、`historical_shift_gate_repaired=false`。
- [next] 该方向可作为阴性排除项关闭；后续优先研究语义/音素结构与视觉teacher证据，不继续调增益幅度。
## Relations

- follows [[Wav2Lip natural content residual continuation 2026-09-09]]
- relates_to [[Wav2Lip natural temporal contrast probe 2026-09-09]]
- relates_to [[Wav2Lip reference conditioning interaction 2026-09-09]]
- relates_to [[Wav2Lip residual local response audit 2026-09-09]]
- pairs_with [[LRS3 visual teacher missingness audit 2026-09-09]]
- relates_to [[Wav2Lip parallel evidence contract audit 2026-09-09]]

## Changelog

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 按用户要求复核历史并设计五路并行交接；本change严格校验通过，尚未执行 | September 9, 2026 | user |
| 完成E2固定波形增益探针；32视频/36评分，独立validator PASS，未建立waveform gain replacement收益 | September 9, 2026 | agent |