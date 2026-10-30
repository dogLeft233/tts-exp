---
title: Wav2Lip natural temporal contrast probe 2026-09-09
type: experiment
permalink: tts-exp/experiments/wav2-lip-natural-temporal-contrast-probe-2026-09-09
status: concluded
date: '2026-09-09'
hypothesis: 固定对称时间平滑或增强natural mel的局部变化，在不使用TTS、不移动时间坐标时能否产生natural-anchor兼容的探索收益？
report: openspec/changes/probe-wav2lip-natural-temporal-contrast/
tags:
- wav2lip
- replacement
- parallel
- concluded
validation_status: independent_reaudit_pending
---

# Wav2Lip natural temporal contrast probe 2026-09-09

用户要求在现有视觉教师missingness spec之外发散设计几个可独立并行的小实验。本实体记录该分支已执行的历史产物。2026-09-09后续源码审阅发现验收覆盖不足，当前须由E0独立复核；旧报告不等于已按spec确认。

唯一新候选方向。N/SMOOTH/SHARP，时间核[1,4,6,4,1]/16，R=M-S，前后2列残差0，a=min(0.25,0.5/max|R|)，两方向±aR后clip[-4,4]。不是旧natural-only policy训练续跑，不调核宽/幅度。固定16条/8组、P/F0参考、完整N音轨、前3.84s mel与U=30..57。两个contrast分别要求C/D/A的99%组CI下界全正、mean ΔC>0.05及7/8组三项正；否则NO_TEMPORAL_CONTRAST_GAIN_ESTABLISHED。两方向同时通过仍不构成单调机制。

阳性只建议冻结构造独立source-group确认，再验证波形可达性；本轮仅mel seam，不是可部署泛用波形头。

## Observations

- [status] concluded
- [hypothesis] 固定对称时间平滑或增强natural mel的局部变化，在不使用TTS、不移动时间坐标时能否产生natural-anchor兼容的探索收益？
- [history] 最新CORRECT/N全U ΔC=+0.008，95%CI跨0，NO_INCREMENT_ESTABLISHED；控制已过，不是replacement已过。shift自由分数、旧TTS谱迁移及旧natural-policy阴性不被新设计覆盖。
- [budget] 34个fresh视频/36个fresh评分上限（32候选+2N replay，另2parity评分）。 全部训练、新TTS、vocoder、新模型下载为0。
- [protocol] 公共入口 openspec/parallel-replacement-probes-20260909.md；本条 openspec/changes/probe-wav2lip-natural-temporal-contrast/{proposal.md,design.md,specs/,tasks.md}。输入P/Q/D清单字节SHA已写公共契约；下游复核间接身份和独立validator。
- [boundary] 已见16记录/8组，非独立确认；replacement_confirmed、waveform_head_authorized、generalization_established、historical_shift_gate_repaired均false。任何诊断阳性不改父全U阴性。
- [parallel] 本条只拥有 scripts/experiments/wav2lip_natural_temporal_contrast/、同名tests/run、本change/tasks和本BM实体。与另两条及视觉教师审计无科学依赖；CPU可并行，单16G卡使用公共flock排队生成。
- [validation] 历史OpenSpec strict与旧validator返回PASS；源码复核发现独立验收不完整，validation_status=independent_reaudit_pending，不能继续引用为已完成科学独立验收。
- [next] 执行新E0 audit-wav2lip-parallel-evidence-contract，保留旧run只读，旁路重算数字/原spec合规性后更新本实体。

- [result] SMOOTH/N：mean ΔC=+0.012265，99%CI=[-0.013520,+0.042660]；mean ΔD=+0.010606，99%CI=[-0.030407,+0.055650]；mean ΔA=+0.010606，99%CI=[-0.030407,+0.055650]，三项联合正组数7/8。
- [result] SHARP/N：mean ΔC=-0.014675，99%CI=[-0.032744,+0.001665]；mean ΔD=-0.023216，99%CI=[-0.057321,+0.012667]；mean ΔA=-0.023541，99%CI=[-0.057425,+0.012408]，三项联合正组数1/8。
- [legacy_conclusion] 唯一终态为 NO_TEMPORAL_CONTRAST_GAIN_ESTABLISHED；两方向均未通过，replacement_confirmed、waveform_head_authorized、generalization_established、historical_shift_gate_repaired 均保持 false。
- [budget] 实际 fresh 视频34、fresh评分36；包含2个N replay、2个parity评分、32个候选；训练、新TTS、vocoder、新模型下载均为0。
- [artifact] 运行目录 runs/wav2lip_natural_temporal_contrast_20260909_v1；prepare、controls、candidate、analysis、原validator和final已输出。
- [lesson] natural-only固定时间调制在本16条/8组探索中未产生natural-anchor增益；不进入波形头训练或泛化确认。
- [review] A的validate复用producer temporal_contrast，未从embeddings独立重算C/D/A、组bootstrap或最终标签；旧PASS不足以证明原spec独立验收完成。现有正负数字保留为报告值，E0将检验数值是否变化。
- [correction] 撤回“本轮已按spec完成独立数值验收”的描述；这是源码/产物结构复核，尚未运行新E0或全量重算，不断言全部历史结果错误。
- [report] 历史复核与五路交接为openspec/replacement-history-review-20260909.md、openspec/parallel-replacement-next-20260909.md。

## Relations

- follows [[Wav2Lip natural content residual continuation 2026-09-09]]
- relates_to [[Wav2Lip spectral structure replacement 2026-09-08]]
- relates_to [[LRS3 visual teacher missingness audit 2026-09-09]]
- pairs_with [[Wav2Lip reference conditioning interaction 2026-09-09]]
- pairs_with [[Wav2Lip residual local response audit 2026-09-09]]

- requires [[Wav2Lip parallel evidence contract audit 2026-09-09]]

## Changelog

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 按用户要求设计独立并行探索OpenSpec，冻结预算、停止边界和独立验收；尚未执行实验 | September 9, 2026 | user |
| 完成A实现与实验；controls PASS，32候选无方向通过，终态 NO_TEMPORAL_CONTRAST_GAIN_ESTABLISHED；实际34视频/36评分；更新结论与产物路径 | September 9, 2026 | agent |
| 后续源码审阅发现spec/validator覆盖缺口；纠正已独立验收说法，合并当前状态并移回Observations，原结果和Changelog保留，待E0重算 | September 9, 2026 | user |
