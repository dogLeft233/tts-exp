---
title: Wav2Lip parallel evidence contract audit 2026-09-09
type: experiment
permalink: tts-exp/experiments/wav2-lip-parallel-evidence-contract-audit-2026-09-09
status: concluded
date: '2026-09-09'
report: openspec/changes/audit-wav2lip-parallel-evidence-contract/
tags:
- wav2lip
- replacement
- parallel
- concluded
---

# Wav2Lip parallel evidence contract audit 2026-09-09

最新A/B/C的validator没有独立重算主要统计；B还漏实现了spec已要求的已知延迟配对搜索域。先以缓存审计恢复可信证据，不重跑模型。

本次已按spec完成实验、独立复核和BM收尾；数值结论见Observations。总交接 `openspec/parallel-replacement-next-20260909.md`，历史证据综述 `openspec/replacement-history-review-20260909.md`。本轮共五路：新增四路和既有视觉教师missingness审计；CPU并行准备，GPU单锁串行，不能把并行当作多个GPU模型同时驻留。

## Observations
- [status] concluded
- [question] 独立复算最新A/B/C证据，并检查已知延迟配对域是否修复历史F46门禁。
- [protocol] `openspec/changes/audit-wav2lip-parallel-evidence-contract/`；固定16条/8组，只读P/Q/D缓存，0新视频、0评分、0模型调用。
- [result] E0工程结论为 `GO`，科学结论为 `CONTRACT_VIOLATION_REPRODUCED`；没有生成候选，也不授权replacement。
- [result] B旧legacy域通过13/16；按spec的matched lag域通过16/16，matched-domain control=`PASS`。旧域anchor damage mean=+1.162568，95%CI=[+0.844647,+1.482466]，8/8组为正；`f46_boundary_restored=false`。
- [conclusion] 历史问题确实包含控制/评分契约错误：正确配对域本身通过，但旧13/16边界不能被宣称已恢复。该审计只修复证据解释边界，不产生replacement收益。
- [validation] `validation.json` 为 PASS，`independent=true`；A/B/C 从原始embedding、lag域、support和bootstrap独立重算，并检查zero-exposure行。严格OpenSpec、五路聚焦测试均通过。
- [review] 实现审阅发现并修复validator的lag符号/曲线轴错误、陈旧score/media绑定、fresh-control parity校验、self-hash和stale-error问题；当前最终artifact已重新绑定。
- [artifact] `runs/wav2lip_parallel_evidence_contract_20260910_v1/`；`final.json`、`validation.json`、`recomputed_analysis.json`已完成。
- [boundary] `replacement_confirmed=false`、`waveform_head_authorized=false`、`generalization_established=false`、`historical_shift_gate_repaired=false`。
- [next] 后续reference实验必须复用matched-domain控制契约；不要把旧13/16结果或本次控制PASS解释为replacement阳性，也不再复活旧构造。
## Relations

- follows [[Wav2Lip natural content residual continuation 2026-09-09]]
- relates_to [[Wav2Lip natural temporal contrast probe 2026-09-09]]
- relates_to [[Wav2Lip reference conditioning interaction 2026-09-09]]
- relates_to [[Wav2Lip residual local response audit 2026-09-09]]
- pairs_with [[LRS3 visual teacher missingness audit 2026-09-09]]

## Changelog

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 按用户要求复核历史并设计五路并行交接；本change严格校验通过，尚未执行 | September 9, 2026 | user |
| 完成E0独立契约审计；matched-domain 16/16通过，确认旧13/16契约缺陷；独立validator PASS | September 9, 2026 | agent |