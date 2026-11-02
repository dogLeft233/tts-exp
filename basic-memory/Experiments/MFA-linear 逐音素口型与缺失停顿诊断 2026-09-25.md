---
title: MFA-linear 逐音素口型与缺失停顿诊断 2026-09-25
type: experiment
permalink: tts-exp/experiments/mfa-linear-逐音素口型与缺失停顿诊断-2026-09-25
status: concluded
date: '2026-09-25'
result: 464/466音素匹配，中心绝对偏差中位数5 ms；7段缺失停顿使M输出音频中位RMS=0.0286，对照已匹配停顿0.0025；原视频M−N均值Sync-C=-0.698。
conclusion: 全局音素节奏大体保留，未匹配自然停顿回退为TTS语音是明确的局部失败机制；其它视觉/声学差距仍存在。
report: runs/mfa_linear_phoneme_video_audit_20260925/report.md
inputs: runs/aishell1_qwen_mfa_linear_n100_20260816; runs/mfa_linear_nas_short_expanded_20260924
outputs: runs/mfa_linear_phoneme_video_audit_20260925
code_paths:
- scripts/experiments/mfa_linear_phoneme_video_audit.py
tags:
- mfa-linear
- phoneme
- wav2lip
- syncnet
- experiment
---

# MFA-linear 逐音素口型与缺失停顿诊断

复用 NAS 短窗口扩大样本试验中的 10 条 AISHELL-1 Qwen 云端 TTS / MFA-linear Wav2Lip N/M 视频。只研究原始视频、统一自然音轨，不使用 NAS 重定时版本。以原项目的 mandarin_china_mfa、mandarin_mfa 和逐字 .lab 对 MFA-linear 输出音频重新强制对齐；MediaPipe 测 25 fps 嘴部开合；官方 SyncNet V2 裁剪提取固定零偏移逐 200 ms 窗口距离。全段 Sync-C 引用原 10 条官方结果。

## Observations

- [status] concluded，10 位说话人、10 条视频、464/466 个非静音音素重新配对成功。
- [result] 自然音频与 MFA-linear 输出音频的配对音素中心偏差：绝对值中位数 5 ms，90 分位 15 ms，450/464 落在 ±40 ms。原始视频自然音轨 Sync-C：N 均值 5.940，M 均值 5.243，M−N=-0.698。
- [result] N/M 口型开合曲线同时间相关为 0.631–0.844；±5 帧搜索中 10/10 均在 0 帧最相关。节奏总体对应，但局部开合形状不同。
- [mechanism] 7 段 ≥100 ms 自然停顿在 TTS 中缺失，mfa_linear_target 对未匹配自然音素使用全局 TTS 特征回退。缺失停顿处 M 输出音频 RMS 中位数 0.0286，21 段已匹配停顿处 0.0025；7/7 有明显填充能量，6/7 固定零偏移局部 SyncNet 距离对 M 更大。
- [example] a1_049 在 3.11–3.43 s 自然停顿：自然 RMS 0.00229，M RMS 0.02860；后续 /i,tɕ,i/ 重对齐中心提前约 350–370 ms；M 视频局部 SyncNet 距离差 +4.566，并排帧可见停顿期口部动作。
- [limit] 缺失停顿是已核实的局部失败机制，但 5 条没有这类长停顿的样本仍有 Sync-C 差距。MFA 强制对齐不能证明音质；25 fps 不能视觉验证 5 ms；开合代理和跨音素的 SyncNet 窗口不能当作单音素评分。
- [outputs] 完整报告 `runs/mfa_linear_phoneme_video_audit_20260925/report.md`；逐音素/逐停顿数据 `runs/mfa_linear_phoneme_video_audit_20260925/analysis.json`；10 条曲线和 6 张口部并排图；可重跑脚本 `scripts/experiments/mfa_linear_phoneme_video_audit.py`。

## Relations

- relates_to [[NAS 短窗口 10 说话人扩大样本试验 2026-09-24]]

## Changelog

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 用户请求逐音素检查，完成输出音频重新 MFA、口型曲线、局部 SyncNet 和缺失停顿诊断 | September 25, 2026 | user（实验请求）；agent（执行与验证） |
