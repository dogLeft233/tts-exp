---
title: Wav2Lip natural content residual continuation 2026-09-09
type: experiment
permalink: tts-exp/experiments/wav2-lip-natural-content-residual-continuation-2026-09-09
status: concluded
date: '2026-09-09'
report: openspec/changes/resume-wav2lip-natural-content-residual/
hypothesis: 修订已知延迟控制搜索坐标后，同模型正确内容条件增量加到完整natural基底上能否产生固定anchor兼容的replacement探索收益？
tags:
- wav2lip
- replacement
- content-residual
- continuation
- concluded
---

# Wav2Lip natural content residual continuation 2026-09-09

结合历史继续一次已被门禁阻断的内容残差检验。直接phone-aligned TTS mel和MAG/ENV未建立收益；历史shift自由分数改善未转化为共享natural anchor改善。masked辅助收益提供另一条有限线索，但必须检验相对完整natural的绝对收益。

父内容残差实验Stage A因offset仅13/16而CONTROL_FAILED，候选尚未执行。后续CPU诊断以配对lag域恢复16/16、三条越界峰全部恢复，未补偿anchor损伤mean=1.162567632539，95%CI=[0.844646827451,1.482466255980]，8/8组正。这个结果支持显式amendment，不是replacement确认。

## Observations

- [status] concluded
- [protocol] 新change为resume-wav2lip-natural-content-residual；冻结原16条/8组、三个seed先融合、N/CORRECT/WRONG/SHUFFLE、幅度和K、U=range(30,58)、候选lag=-15..15与父natural anchor。仅A_DELAY改用lag=-10..20及offset=10-j，固定anchor仍使用未补偿legacy曲线。
- [integrity] P=runs/wav2lip_natural_content_residual_20260908_v2；D=runs/wav2lip_delay_search_support_20260909_v1。新run只读绑定两父run，保留原CONTROL_FAILED及诊断stage_b_authorized=false；本run独立验收通过才授权候选。
- [implementation] 旧runner的all路径未在候选前显式调用独立validator；新入口须使all/candidates/resume都强制核对控制验收和当前绑定。这是执行路径约束，不否认父run已有的独立验收结果。
- [controls] 重算旧13/16和配对16/16；沿用14/16灵敏度门槛和原anchor规则。追加两个固定scorer parity与排序前两条N_REPLAY生成/评分，像素逐值一致、矩阵/端点容差1e-4；失败BLOCKED。
- [budget] 新增最多50视频/52评分：N_REPLAY 2/2、parity 0/2、候选48/48；复用父18视频/36评分，累计68/88。无训练、新TTS或vocoder。跨会话检查相对旧预算增加2视频/4评分。
- [decision] CORRECT/N必须C/D/A三项95%CI下界全>0、至少7/8组三项同正且平均ΔC>0.05；失败NO_INCREMENT_ESTABLISHED并停止该固定构造。再胜WRONG与SHUFFLE且范数有效才CONTENT_RESIDUAL_SIGNAL，否则NATURAL_GAIN_MECHANISM_UNRESOLVED。
- [boundary] 所有结果replacement_confirmed、waveform_head_authorized、generalization_established、historical_shift_gate_repaired均false；seen-record、前3.84s特征支持、完整N音轨的探索，非句义机制。阳性仅建议独立源组确认；本轮不自动启动下一实验或关机。
- [validation] 设计已通过openspec validate resume-wav2lip-natural-content-residual --strict --no-interactive；已只读核对design列出的父资产字节SHA-256，未执行新forward或候选。
- [report] openspec/changes/resume-wav2lip-natural-content-residual/{proposal.md,design.md,tasks.md,specs/wav2lip-natural-content-residual-continuation/spec.md}。
- [next] 固定content-residual构造停止；若继续研究，应另设计独立source-group或新机制验证，不把本轮当作replacement确认。

## Relations

- follows [[Wav2Lip delay search support audit 2026-09-09]]
- extends [[Wav2Lip natural content residual probe 2026-09-08]]
- relates_to [[Wav2Lip historical shift rescore 2026-09-08]]
- contrasts_with [[Wav2Lip spectral structure replacement 2026-09-08]]
- relates_to [[LRS3 masked TTS trajectory-specificity diagnosis]]

## Changelog

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 根据用户请求结合历史设计版本化续跑OpenSpec；完成严格校验与父资产哈希核对，实验待下游实现 | September 9, 2026 | user |
- [result] 新run `runs/wav2lip_natural_content_residual_continuation_20260909_v1` controls通过：旧域13/16、修订配对域16/16；anchor damage mean=1.162567632539，95%CI=[0.844646827451,1.482466255980]，8/8组正。
- [result] 48个候选完成评分；CORRECT/N：ΔC=0.007685 CI[-0.029184,0.039568]，ΔD=0.019940 CI[-0.019390,0.058372]，ΔA=0.019221 CI[-0.020412,0.057966]，三项均未过门。
- [result] CORRECT/WRONG三项CI也跨0；CORRECT/SHUFFLE三项为正，但不能替代natural主假设。
- [conclusion] 终态为NO_INCREMENT_ESTABLISHED：本固定内容残差增量相对完整natural没有建立收益；不是所有语义路线失效，也不是replacement确认。
- [validation] 独立validator PASS，OpenSpec strict PASS，单测10 passed；replacement_confirmed、waveform_head_authorized、generalization_established、historical_shift_gate_repaired均false。
- [budget] 实际新增50视频/52评分（N_REPLAY 2视频/2评分、parity 2评分、候选48视频/48评分）；复用父18视频/36评分；无训练、新TTS或vocoder。
| 完成controls、48候选、独立all验收；得到NO_INCREMENT_ESTABLISHED并保存三组效应与预算 | September 9, 2026 | user |