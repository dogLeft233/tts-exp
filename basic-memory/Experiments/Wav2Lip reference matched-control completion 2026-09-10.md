---
title: Wav2Lip reference matched-control completion 2026-09-10
type: experiment
permalink: tts-exp/experiments/wav2-lip-reference-matched-control-completion-2026-09-10
status: concluded
date: '2026-09-10'
tags:
- openspec
- concluded
- parallel
---

# Wav2Lip reference matched-control completion 2026-09-10

本次按 OpenSpec 完成了代码实现、历史与 fresh 控制、F46_C 候选、独立 validator 和最终审计。B run 使用 b_20260910_r2，工程终态 GO，科学结论为未建立参考交互。

## Observations
- [status] concluded
- [question] F46自身正确配对域控制通过后，同一固定 CORRECT 残差的响应是否随参考条件变化？
- [openspec] openspec/changes/complete-wav2lip-reference-matched-control/
- [report] openspec/parallel-next-experiments-20260910.md
- [inputs] openspec/next-experiments-20260910-inputs.json
- [budget] 设计上限36视频/54评分；实际 B run 为36视频/54评分；不训练、不生成TTS、不下载模型。
- [boundary] 已见数据探索/纠错续验；replacement_confirmed=false、waveform_head_authorized=false、generalization_established=false；不修改父run与旧阴性。
- [implementation] 固定 signed-lag：natural lag_start=-15、delay lag_start=-10，offset 分别15-j/10-j；F46 matched control 在 F46_C 前独立验收；共享 F0_N anchor 计算 g0/g1/I。
- [execution] B run b_20260910_r2 完成：历史 F46 offset 16/16，fresh F0 replay/F46 repeat/parity/delay 均 PASS；fresh 控制20视频/38评分，之后生成16个 F46_C 候选。
- [result] interaction I mean=0.003677065883，99% CI=[-0.0333733260578,0.0403142108742]，4/8组正；最终 scientific_decision=NO_REFERENCE_INTERACTION_ESTABLISHED。
- [validation] control_validation 与 validation 均 PASS；历史、fresh 和候选 manifest self-hash 通过；聚焦 pytest 12 passed，独立 validator 从 raw embeddings 重建四 cell、共同 anchor、组 bootstrap 和 interaction；compileall、Ruff 和 OpenSpec strict 通过。
- [review] root 最终审计通过：F46 source/frame/box/reference 绑定、两套 lag 域、replay/parity、原N音轨、四 cell、预算和授权 flags 均一致。
- [next] 若继续，需另行设计独立参考/source-group confirmation；本 run 不授权 waveform head、replacement 或 generalization。
## Relations

- follows [[Wav2Lip reference conditioning interaction 2026-09-09]]

## Changelog
| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 根据最新结果及源码偏差创建独立OpenSpec计划；尚未执行 | September 10, 2026 | user |
| B实现、历史CPU门禁与root审阅完成：6 tests、compile/OpenSpec strict通过；fresh GPU待执行 | September 10, 2026 | agent/root |
| B fresh GPU、F46_C、独立验证与最终审计完成：b_20260910_r2，36视频/54评分，结论未建立参考交互 | September 10, 2026 | agent/root |
| 终审补强：独立 raw embeddings/matrix 四 cell 与 bootstrap 重算、F46_N/C 身份冻结；聚焦 pytest 12、Ruff、compileall、OpenSpec strict 和 validator 均 PASS | September 10, 2026 | agent/root |