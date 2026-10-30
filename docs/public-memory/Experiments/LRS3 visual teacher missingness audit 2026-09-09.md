---
title: LRS3 visual teacher missingness audit 2026-09-09
type: experiment
permalink: tts-exp/experiments/lrs3-visual-teacher-missingness-audit-2026-09-09
status: concluded
date: '2026-09-09'
report: openspec/changes/audit-lrs3-visual-teacher-missingness/
tags:
- lrs3
- visual-teacher
- missingness
- retrospective
- concluded
---

# LRS3 visual teacher missingness audit 2026-09-09

用户要求解释上一轮结果、回顾历史并设计下一步给下游。content-residual续跑控制已通过，但相对完整natural的ΔSync-C=+0.008，95%CI=[−0.029,+0.040]，固定anchor也未建立改善。shift自由分数正向、masked条件优于NAT_ONLY、目标残差优于shuffle，均不能代替优于完整natural的replacement证据。建议暂停当前固定残差构造，先低成本审计潜在视觉教师的依据。

旧duration-compatible视觉教师审计有133条/23组缓存、399个特征文件，原eligibility仅115条/21组，父Stage01为BLOCKED。本轮是独立的回顾性CPU缓存分析，不修复父门禁。已完成共同支持与missingness bounds复算；数值结论见Observations。

## Observations
- [status] concluded
- [question] 在缓存样本中，MFA-linear/TTS驱动视频相对natural是否更接近共同支持上的真实口部几何，以及缺失记录是否会改变方向。
- [protocol] `openspec/changes/audit-lrs3-visual-teacher-missingness/`；固定133条/23组，保留全部分母，三臂common-support，CPU-only回顾性审计，0视频/0评分/0模型调用。
- [result] 工程结论为 `GO`，诊断结论为 `INCONCLUSIVE_UNDER_MISSINGNESS`。observed=115/133；missingness bounds L=-0.221363044906、U=+0.213868149021，跨过0。
- [conclusion] 缓存证据不足以证明visual teacher相对natural的正向优势；也不能据此否定visual teacher。父Stage01的BLOCKED和115/133缺失边界保持，不修复门禁、不授权Stage02或replacement。
- [validation] `validation.json` 为 PASS，`independent=true`；独立复核exact pairing、common support、bounds、support index hash和parent hash。
- [review] 保留原始分母与缺失记录，没有用observed-only均值救阳性；最终artifact绑定analysis/validation/review。
- [artifact] `runs/lrs3_visual_teacher_missingness_20260909_v1/`；`final.json`已完成，未创建新模型或视频。
- [boundary] `replacement_confirmed=false`、`waveform_head_authorized=false`、`generalization_established=false`；parent_gate_repaired/stage02_authorized保持false。
- [next] 若继续visual-teacher路线，应先补齐可审计的真实视频/对齐样本，再做独立确认；不要把本轮不确定性当作replacement阳性。
## Relations

- follows [[Wav2Lip natural content residual continuation 2026-09-09]]
- relates_to [[TTS 视觉教师到 replacement-safe 音频控制链路]]
- relates_to [[LRS3 TTS Visual Advantage Audit]]
- contrasts_with [[Wav2Lip historical shift rescore 2026-09-08]]
- relates_to [[Wav2Lip spectral structure replacement 2026-09-08]]

## Changelog

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 回顾历史后设计CPU共同支持与缺失界限OpenSpec，锁定输入hash并完成严格校验；新诊断待下游实现 | September 9, 2026 | user |
| 完成E1缓存缺失审计；115/133 observed且界限跨0，独立validator PASS，结论为INCONCLUSIVE | September 9, 2026 | agent |