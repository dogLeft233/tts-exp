---
title: Wav2Lip reference conditioning interaction 2026-09-09
type: experiment
permalink: tts-exp/experiments/wav2-lip-reference-conditioning-interaction-2026-09-09
status: concluded
date: '2026-09-09'
hypothesis: 同一固定CORRECT音频增量相对N的响应是否随静态参考条件变化？
report: openspec/changes/probe-wav2lip-reference-conditioning-interaction/
tags:
- wav2lip
- replacement
- parallel
- concluded
validation_status: independent_reaudit_pending
---

# Wav2Lip reference conditioning interaction 2026-09-09

用户要求在现有视觉教师missingness spec之外发散设计几个可独立并行的小实验。本实体记录该分支已执行的历史产物。2026-09-09后续源码审阅发现验收覆盖不足，当前须由E0独立复核；旧报告不等于已按spec确认。

诊断分支：N/CORRECT × F0/F46，两参考取同一源视频零基第0和46帧与对应历史box，不按嘴张合或评分选图，每cell始终静态93帧。F0两cell只读复用；新增F46 N/C和N repeat及delay控制。四cell统一用F0_N在全U的k0，主量I=(A_N-A_C)_F46-(A_N-A_C)_F0。99%组CI排除0、abs(mean I)>0.05、至少7/8同号才REFERENCE_DEPENDENT_RESPONSE；否则NO_REFERENCE_INTERACTION_ESTABLISHED。

I正可与g1仍负并存，不代表replacement；参考变化包含姿态/光照/crop，不能证明嘴部先验因果。阳性只建议多参考稳健性确认。

## Observations

- [status] concluded
- [hypothesis] 同一固定CORRECT音频增量相对N的响应是否随静态参考条件变化？
- [history] 最新CORRECT/N全U ΔC=+0.008，95%CI跨0，NO_INCREMENT_ESTABLISHED；控制已过，不是replacement已过。shift自由分数、旧TTS谱迁移及旧natural-policy阴性不被新设计覆盖。
- [budget] 36个fresh视频/54个fresh评分上限；新参考控制失败则不生成F46_C。 全部训练、新TTS、vocoder、新模型下载为0。
- [protocol] 公共入口 openspec/parallel-replacement-probes-20260909.md；本条 openspec/changes/probe-wav2lip-reference-conditioning-interaction/{proposal.md,design.md,specs/,tasks.md}。输入P/Q/D清单字节SHA已写公共契约；下游复核间接身份和独立validator。
- [boundary] 已见16记录/8组，非独立确认；replacement_confirmed、waveform_head_authorized、generalization_established、historical_shift_gate_repaired均false。任何诊断阳性不改父全U阴性。
- [parallel] 本条只拥有 scripts/experiments/wav2lip_reference_conditioning_interaction/、同名tests/run、本change/tasks和本BM实体。与另两条及视觉教师审计无科学依赖；CPU可并行，单16G卡使用公共flock排队生成。
- [validation] 历史OpenSpec strict与旧validator返回PASS；源码复核发现独立验收不完整，validation_status=independent_reaudit_pending，不能继续引用为已完成科学独立验收。
- [next] 执行新E0 audit-wav2lip-parallel-evidence-contract，保留旧run只读，旁路重算数字/原spec合规性后更新本实体。

- [result] F0 replay、F46 repeat、parity均通过；F46 delay sensitivity 仅13/16条满足offset差[-6,-4]，冻结要求为至少14/16，因此 delay gate 未通过。
- [result] delay anchor damage mean=+1.267882，95%CI=[+0.975656,+1.585121]，8/8组正；但offset pass数量不足使总体控制失败。
- [legacy_conclusion] 唯一终态为 CONTROL_FAILED；未生成F46_C，不计算主差中差I，不报告reference-dependent response，也不授权replacement或训练。
- [budget] 实际 fresh 视频20、fresh评分38；上限36/54未被用满；训练、新TTS、vocoder、新模型下载均为0。
- [artifact] 运行目录 runs/wav2lip_reference_conditioning_interaction_20260909_v1；F46固定参考、控制、门禁、review、final和原validation已输出。
- [lesson] 参考交互问题当前被delay控制门禁阻塞，不能把F46响应或失败解释为参考依赖结论；应由新E0按原spec重算配对域，不能把legacy 13/16当正确域结论或直接放行F46_C。
- [review] B的design已要求delay lag[-10,20]及offset=10-j，score.metrics却固定15-index，runner直接以legacy U offset判13/16；这是实现遗漏，不是尚待修改的科学门槛。旧validator未重算控制，F0 replay只验U列表。尚未独立计算正确域F46通过数，不能宣布恢复16/16；F46_C仍未生成。
- [correction] 撤回“本轮已按spec完成独立数值验收”的描述；这是源码/产物结构复核，尚未运行新E0或全量重算，不断言全部历史结果错误。
- [report] 历史复核与五路交接为openspec/replacement-history-review-20260909.md、openspec/parallel-replacement-next-20260909.md。

## Relations

- follows [[Wav2Lip natural content residual continuation 2026-09-09]]
- relates_to [[Wav2Lip spectral structure replacement 2026-09-08]]
- relates_to [[LRS3 visual teacher missingness audit 2026-09-09]]
- pairs_with [[Wav2Lip natural temporal contrast probe 2026-09-09]]
- pairs_with [[Wav2Lip residual local response audit 2026-09-09]]

- requires [[Wav2Lip parallel evidence contract audit 2026-09-09]]

## Changelog

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 按用户要求设计独立并行探索OpenSpec，冻结预算、停止边界和独立验收；尚未执行实验 | September 9, 2026 | user |
| 完成B实现与控制；F46 delay仅13/16通过，按门禁终止并形成 CONTROL_FAILED；实际20视频/38评分，未生成F46_C | September 9, 2026 | agent |
| 后续源码审阅发现spec/validator覆盖缺口；纠正已独立验收说法，合并当前状态并移回Observations，原结果和Changelog保留，待E0重算 | September 9, 2026 | user |
