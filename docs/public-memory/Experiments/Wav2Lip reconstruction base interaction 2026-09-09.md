---
title: Wav2Lip reconstruction base interaction 2026-09-09
type: experiment
permalink: tts-exp/experiments/wav2-lip-reconstruction-base-interaction-2026-09-09
status: concluded
date: '2026-09-09'
report: openspec/changes/probe-wav2lip-reconstruction-base-interaction/
tags:
- wav2lip
- replacement
- parallel
- concluded
---

# Wav2Lip reconstruction base interaction 2026-09-09

masked条件中正确内容有价值，但同一增量加到完整natural无收益。固定增量不调参，只检验该响应是否依赖重建基底及其非线性交互。

本次已按spec完成实验、独立复核和BM收尾；数值结论见Observations。总交接 `openspec/parallel-replacement-next-20260909.md`，历史证据综述 `openspec/replacement-history-review-20260909.md`。本轮共五路：新增四路和既有视觉教师missingness审计；CPU并行准备，GPU单锁串行，不能把并行当作多个GPU模型同时驻留。

## Observations
- [status] concluded
- [question] masked条件中有效的content increment，是否依赖reconstruction base，或在完整natural base上产生interaction。
- [protocol] `openspec/changes/probe-wav2lip-reconstruction-base-interaction/`；固定16条/8组，比较N、N_CONTENT、BASE、BASE_CONTENT、BASE_WRONG，候选只在控制通过后生成。
- [result] 工程结论为 `GO`，科学结论为 `NO_BASE_CONTENT_GAIN_ESTABLISHED`。实际生成48个视频、52个score cells（48候选+4 fresh controls），低于50/52上限。
- [result] BASE_CONTENT/N的ΔC mean=+0.026522，99%CI=[-0.003972,+0.056085]，联合正组5/8，未过主门槛。interaction mean=+0.016891，99%CI=[+0.000822,+0.033042]，联合正组6/8，`signal=false`。
- [result] BASE/N ΔC=-0.005850，99%CI=[-0.047730,+0.061998]；BASE_WRONG/N ΔC=+0.005601，99%CI=[-0.024090,+0.036958]；均不能支持replacement或base-dependent content gain。
- [conclusion] 正确content increment在masked/重建条件中的局部响应，不能迁移为完整natural replacement收益；interaction虽有小的正均值，也未达到预设机制门槛。
- [validation] `validation.json` 为 PASS，`independent=true`；独立复原D三seed factorial、重建SyncNet矩阵和统计，并复核D reconstruction hash及fresh controls。
- [review] 修复了候选score的media hash/self-hash绑定、实际预算记录和stale error；最终artifact已重评分并重新绑定。
- [artifact] `runs/wav2lip_reconstruction_base_interaction_20260910_v1/`；实际预算记录在`final.json`，为48 videos / 52 score cells。
- [boundary] `replacement_confirmed=false`、`waveform_head_authorized=false`、`generalization_established=false`、`historical_shift_gate_repaired=false`。
- [next] 当前固定content-residual/base-interaction构造不值得继续调参；后续转向语义/音素结构或视觉teacher可观测性研究。
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
| 完成E4重建基底交互探针；48视频/52评分，独立validator PASS，未建立base-dependent replacement收益 | September 9, 2026 | agent |