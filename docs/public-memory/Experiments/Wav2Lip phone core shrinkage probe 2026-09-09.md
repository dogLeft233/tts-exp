---
title: Wav2Lip phone core shrinkage probe 2026-09-09
type: experiment
permalink: tts-exp/experiments/wav2-lip-phone-core-shrinkage-probe-2026-09-09
status: concluded
date: '2026-09-09'
report: openspec/changes/probe-wav2lip-phone-core-shrinkage/
tags:
- wav2lip
- replacement
- parallel
- concluded
---

# Wav2Lip phone core shrinkage probe 2026-09-09

全时域平滑没有已确认收益，但它与保留音素边界、仅约束音素内部变化的问题不同。用已存phone-core边界和同支持等范数平滑对照做一次固定检验。

本次已按spec完成实验、独立复核和BM收尾；数值结论见Observations。总交接 `openspec/parallel-replacement-next-20260909.md`，历史证据综述 `openspec/replacement-history-review-20260909.md`。本轮共五路：新增四路和既有视觉教师missingness审计；CPU并行准备，GPU单锁串行，不能把并行当作多个GPU模型同时驻留。

## Observations
- [status] concluded
- [question] 保留音素核心边界、并与generic smoothing匹配扰动范数，是否比同样natural-anchor下的通用平滑更有利。
- [protocol] `openspec/changes/probe-wav2lip-phone-core-shrinkage/`；固定16条/8组，先独立重建音素核心与同范数generic候选，任何退化不调参、不强行生成视频。
- [result] 工程结论为 `GO`，科学结论为 `INPUT_DEGENERATE`。16条全部保留，发现2条候选PCM/输入退化ID：`lrs3_7VRzn8hc5mc_00016`、`lrs3_7c5t6FkvUG0_00001`；因此实际0视频/0 score cells。
- [conclusion] 这不是“phone-core无效”的科学阴性，而是当前构造无法提供满足协议的非退化候选；没有进入TFG评分，也没有replacement结论。
- [validation] `validation.json` 为 PASS，`independent=true`；独立重建mask、weights和退化cohort，确认所有16条仍在分母。
- [review] E3退化路径已改为必须经过独立validator，不能由runner自填PASS；实际预算如实写入`final.json`。
- [artifact] `runs/wav2lip_phone_core_shrinkage_20260910_v1/`；最终预算为0 videos / 0 score cells。
- [boundary] `replacement_confirmed=false`、`waveform_head_authorized=false`、`generalization_established=false`、`historical_shift_gate_repaired=false`。
- [next] 若继续音素方向，应先单独设计非退化构造/可达性小实验；不要把本轮退化结果写成phone-core机制被否定。
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
| 完成E3音素核心收缩探针；输入退化，0视频/0评分，独立validator PASS；不作phone-core科学阴性 | September 9, 2026 | agent |