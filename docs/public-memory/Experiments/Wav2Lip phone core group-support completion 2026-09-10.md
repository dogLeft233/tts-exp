---
title: Wav2Lip phone core group-support completion 2026-09-10
type: experiment
permalink: tts-exp/experiments/wav2-lip-phone-core-group-support-completion-2026-09-10
status: concluded
date: '2026-09-10'
tags:
- openspec
- concluded
- parallel
---

# Wav2Lip phone core group-support completion 2026-09-10

本次按 OpenSpec 完成了代码实现、GPU 全量 run、独立 validator 和最终审计。A run 使用 a_20260910_r3，工程终态 GO，科学结论为未建立 PHONE_CORE 增益。

## Observations
- [status] concluded
- [question] 原 phone-core 组级支持规则下，保留全部记录的 PHONE_CORE 能否优于完整 natural 及 generic control？
- [openspec] openspec/changes/complete-wav2lip-phone-core-group-support/
- [report] openspec/parallel-next-experiments-20260910.md
- [inputs] openspec/next-experiments-20260910-inputs.json
- [budget] 设计上限34视频/36评分；实际 A run 为34视频/36评分；不训练、不生成TTS、不下载模型。
- [boundary] 已见数据探索/纠错续验；replacement_confirmed=false、waveform_head_authorized=false、generalization_established=false；不修改父run与旧阴性。
- [implementation] 按 source_group 聚合曝光，16条记录全部保留；候选门禁要求控制 PASS 与独立验证 PASS；固定输入、replay/parity 和 PCM/matrix/endpoint 约束均落地。
- [execution] A run a_20260910_r3 完成：8/8组支持，旧 offset 13/16、matched offset 16/16，fresh replay/parity 均 PASS；两条零曝光记录仍在分母：lrs3_7VRzn8hc5mc_00016、lrs3_7c5t6FkvUG0_00001。
- [result] PHONE_CORE ΔC mean=-0.012565，mechanism joint-positive groups=3；最终 scientific_decision=NO_PHONE_CORE_GAIN_ESTABLISHED。
- [validation] control_validation 与 validation 均 PASS；所有产物 self-hash 通过；聚焦 pytest 16 passed，compileall、Ruff 和 OpenSpec strict 通过；独立 validator 复核旧成功 driver 14/14。
- [review] root 最终审计通过：fixed bindings、组级曝光、候选数量、原N音轨、replay/parity、预算和授权 flags 均一致。
- [next] 若继续，需另行设计独立 source-group confirmation；本 run 不授权 waveform head、replacement 或 generalization。
## Relations

- follows [[Wav2Lip phone core shrinkage probe 2026-09-09]]

## Changelog
| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 根据最新结果及源码偏差创建独立OpenSpec计划；尚未执行 | September 10, 2026 | user |
| 开始按 OpenSpec 实现，阶段测试和审阅待完成 | September 10, 2026 | agent |
| A最终实现与root审阅完成：9 tests、compile/OpenSpec strict通过；GPU待执行 | September 10, 2026 | agent/root |
| A GPU run、独立验证与最终审计完成：a_20260910_r3，34视频/36评分，结论未建立 PHONE_CORE 增益 | September 10, 2026 | agent/root |
| 终审补强：独立本地统计、空 core 队列保留、16/8 每组双记录、旧成功 driver replay=14；聚焦 pytest 16、Ruff、compileall、OpenSpec strict 和 validator 均 PASS | September 10, 2026 | agent/root |