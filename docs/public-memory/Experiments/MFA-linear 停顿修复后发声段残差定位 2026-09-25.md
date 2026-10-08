---
title: MFA-linear 停顿修复后发声段残差定位 2026-09-25
type: experiment
permalink: tts-exp/experiments/mfa-linear-停顿修复后发声段残差定位-2026-09-25
status: concluded
date: '2026-09-25'
cohort_size: 10
result: F/N 自然音轨 Sync-C 5.477/5.920；纯发声窗口共同−2帧偏移 F−N 距离 10/10 为正，均值 +0.904。
conclusion: 剩余差距进入发声段，下一步拆分局部时间事件与同时间声学/口型形状。
report: basic-memory/docs/experiments/35-mfa-linear-speech-residual-alignment.md
artifact: runs/mfa_linear_silence_pause_fix_20260925/residual_window_analysis.json
tags:
- mfa-linear
- speech
- wav2lip
- syncnet
---

# MFA-linear 停顿修复后发声段残差定位

复用同批 10 条、同一自然音轨和官方 SyncNet V2，对 N/F 视频逐窗口复算。10/10 官方 Sync-C 与冻结分数逐条一致。N/F 均值 5.920/5.477；5 条无缺失长停顿样本仍差 0.531。固定 N 的最佳 −2 帧偏移，纯发声 200 ms 窗口的 F−N 距离在 10/10 条为正，跨样本平均 +0.904。局部距离不是 Sync-C，不能由此唯一归因时序或谱形。

优先查看 a1_019、a1_044（无缺失长停顿且 F=M）及 a1_031 的发声段。先并排比较自然/重建/修复版的 Wav2Lip mel、发声事件和口型，再独立测试有界局部时间重映射与自然声学特征 oracle；用新的 speaker-disjoint 样本和固定自然音轨验证。a1_057 的 `spn` 与旧 M 重算异常不作第一条机制样本。

## Observations

- [status] concluded
- [result] F/N 自然音轨 Sync-C 5.477/5.920；无缺失长停顿的 5 条 N−F=0.531。#syncnet
- [result] 纯发声窗口共同 −2 帧偏移 F−N 距离 10/10 为正，跨样本平均 +0.904。#speech
- [conclusion] 停顿修复后仍有发声段局部音视频特征差异；需用单因素干预分辨时间和声学机制。
- [limit] 同批发现样本；局部距离不是全段 Sync-C，a1_057 旧 M 重算异常。
- [report] [[35-mfa-linear-speech-residual-alignment]]

## Relations

- extends [[MFA-linear 静音与停顿修复 2026-09-25]]
- relates_to [[MFA-linear 声码器与 Wav2Lip 响应拆分 2026-09-25]]

## Changelog

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 复算修复后 N/F 发声段残差并记录后续诊断优先级 | September 25, 2026 | user（进一步分析请求）；agent（执行） |
