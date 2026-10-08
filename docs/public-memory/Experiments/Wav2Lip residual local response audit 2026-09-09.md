---
title: Wav2Lip residual local response audit 2026-09-09
type: experiment
permalink: tts-exp/experiments/wav2-lip-residual-local-response-audit-2026-09-09
status: concluded
date: '2026-09-09'
hypothesis: 已有CORRECT残差的实际输入支持曝光与固定natural anchor的局部响应是否有关，而不是重选窗口使旧整体结果变阳性？
report: openspec/changes/audit-wav2lip-residual-local-response/
tags:
- wav2lip
- replacement
- parallel
- concluded
validation_status: independent_reaudit_pending
---

# Wav2Lip residual local response audit 2026-09-09

用户要求在现有视觉教师missingness spec之外发散设计几个可独立并行的小实验。本实体记录该分支已执行的历史产物。2026-09-09后续源码审阅发现验收覆盖不足，当前须由E0独立复核；旧报告不等于已按spec确认。

CPU-only回顾性分支。由P construct.K与D used_masks复核干预位置；官方16列mel chunk经5帧SyncNet视觉支持求唯一列并集B，曝光e=|B∩K|/|B|，固定U/k0。每记录内中心化e及行级d=D_N-D_C，组内beta=sum cov_i/sum var_i，local_gain=sum(e*d)/sum(e)，8组等权99%bootstrap。两项CI下界均正且7/8组同正才LOCAL_RESPONSE_ASSOCIATION_TO_CONFIRM，否则NO_LOCAL_POSITIVE_ASSOCIATION_ESTABLISHED；任一整组无方差/曝光则LOCAL_SUPPORT_NOT_IDENTIFIABLE。

设计期仅看K/chunk支持发现2条U无曝光：lrs3_7VRzn8hc5mc_00016、lrs3_7c5t6FkvUG0_00001；所有8组仍有方差。保留两条作全U/零曝光检查，不填局部收益0；组内定义是曝光条件估计量，不能代替完整视频平均。未计算新beta/local_gain。

## Observations

- [status] concluded
- [hypothesis] 已有CORRECT残差的实际输入支持曝光与固定natural anchor的局部响应是否有关，而不是重选窗口使旧整体结果变阳性？
- [history] 最新CORRECT/N全U ΔC=+0.008，95%CI跨0，NO_INCREMENT_ESTABLISHED；控制已过，不是replacement已过。shift自由分数、旧TTS谱迁移及旧natural-policy阴性不被新设计覆盖。
- [budget] 0视频/0fresh评分/0模型调用，不能解码媒体或启动GPU。 全部训练、新TTS、vocoder、新模型下载为0。
- [protocol] 公共入口 openspec/parallel-replacement-probes-20260909.md；本条 openspec/changes/audit-wav2lip-residual-local-response/{proposal.md,design.md,specs/,tasks.md}。输入P/Q/D清单字节SHA已写公共契约；下游复核间接身份和独立validator。
- [boundary] 已见16记录/8组，非独立确认；replacement_confirmed、waveform_head_authorized、generalization_established、historical_shift_gate_repaired均false。任何诊断阳性不改父全U阴性。
- [parallel] 本条只拥有 scripts/experiments/wav2lip_residual_local_response/、同名tests/run、本change/tasks和本BM实体。与另两条及视觉教师审计无科学依赖；CPU可并行，单16G卡使用公共flock排队生成。
- [validation] 历史OpenSpec strict与旧validator返回PASS；源码复核发现独立验收不完整，validation_status=independent_reaudit_pending，不能继续引用为已完成科学独立验收。
- [next] 执行新E0 audit-wav2lip-parallel-evidence-contract，保留旧run只读，旁路重算数字/原spec合规性后更新本实体。

- [result] K/chunk支持重建成功；两个完全零曝光记录为 lrs3_7VRzn8hc5mc_00016、lrs3_7c5t6FkvUG0_00001；所有8组均可辨识。
- [result] 父全U ΔA复现为+0.019221371838，绑定父值+0.019221360662，绝对误差1.12e-8；矩阵独立重建最大误差1.91e-6。
- [result] CORRECT beta mean=+0.063541，99%CI=[-0.129408,+0.317586]；local_gain mean=+0.030062，99%CI=[-0.062287,+0.122876]，均未通过正向门禁。WRONG beta mean=+0.004710；SHUFFLE beta mean=-0.208568，均不改变主结论。
- [legacy_conclusion] 唯一终态为 NO_LOCAL_POSITIVE_ASSOCIATION_ESTABLISHED；局部曝光没有证明可用正向响应，父全局阴性保持不变；不生成局部gate候选、不授权replacement或训练。
- [budget] 实际 fresh 视频0、fresh评分0、模型调用0、GPU占用0。
- [artifact] 运行目录 runs/wav2lip_residual_local_response_20260909_v1；support_indices、exposure、analysis、review、final和原validation已输出。
- [lesson] “整体平均稀释了正收益”的补救解释在当前固定U/K下没有统计支持；这不等于否定所有新的语义局部路线。
- [review] C的validate复用producer row_supports且不重算beta/local_gain/CI；producer用两个边际各7组正替代同一7组联合正、只检查整条零曝光，且support_indices末chunk起点有待核对。当前阴性数字不因此自动失效，但需E0独立重算量化影响。
- [correction] 撤回“本轮已按spec完成独立数值验收”的描述；这是源码/产物结构复核，尚未运行新E0或全量重算，不断言全部历史结果错误。
- [report] 历史复核与五路交接为openspec/replacement-history-review-20260909.md、openspec/parallel-replacement-next-20260909.md。

## Relations

- follows [[Wav2Lip natural content residual continuation 2026-09-09]]
- relates_to [[Wav2Lip spectral structure replacement 2026-09-08]]
- relates_to [[LRS3 visual teacher missingness audit 2026-09-09]]
- pairs_with [[Wav2Lip natural temporal contrast probe 2026-09-09]]
- pairs_with [[Wav2Lip reference conditioning interaction 2026-09-09]]

- requires [[Wav2Lip parallel evidence contract audit 2026-09-09]]

## Changelog

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 按用户要求设计独立并行探索OpenSpec，冻结预算、停止边界和独立验收；尚未执行实验 | September 9, 2026 | user |
| 完成C CPU审计；矩阵/父ΔA复现通过，局部beta与local_gain未过正向门禁；终态 NO_LOCAL_POSITIVE_ASSOCIATION_ESTABLISHED，0/0预算 | September 9, 2026 | agent |
| 后续源码审阅发现spec/validator覆盖缺口；纠正已独立验收说法，合并当前状态并移回Observations，原结果和Changelog保留，待E0重算 | September 9, 2026 | user |
