---
title: MFA-linear 新说话人发声段残差复核 2026-09-25
type: experiment
permalink: tts-exp/experiments/mfa-linear-新说话人发声段残差复核-2026-09-25
status: concluded
date: '2026-09-25'
cohort_size: 17
result: 自然音轨 Sync-C N/M/F/R=5.576/4.764/4.902/5.447；纯发声固定偏移 F−N 距离17/17为正，均值+0.994。
conclusion: 静音修复改善停顿，但发声段残差仍在新说话人中复现。
report: basic-memory/docs/experiments/36-mfa-linear-heldout-speech-residual.md
inputs: AISHELL-1 n100 中按新说话人和自然时长≥4.8秒冻结的17条
outputs: runs/mfa_linear_heldout_speech_20260925/analysis.json
model: WavLM-L6, HiFi-GAN, Wav2Lip GAN, SyncNet V2
tags:
- mfa-linear
- speech
- wav2lip
- syncnet
- heldout
---

# MFA-linear 新说话人发声段残差复核 2026-09-25

接续 [[MFA-linear 停顿修复后发声段残差定位 2026-09-25]]，先按未进入 n=10 发现集的说话人及自然时长≥4.8 秒固定 AISHELL-1 n100 中 17 条、5 位说话人。N=自然驱动视频，M=旧 MFA-linear，F=静音回退及缺失停顿修复，R=自然语音经同一 WavLM-L6/HiFi-GAN 重建。四路同脸、同 Wav2Lip GAN、同自然音轨 PCM、同官方 SyncNet V2 裁剪；68 个视频与裁剪完整。

平均自然音轨 Sync-C：N/M/F/R=5.576/4.764/4.902/5.447。F−M 全体+0.139；含8段缺失长停顿的6条+0.337且6/6提高；N−F+0.674（14/17）；R−F+0.545（15/17）。全部17条在纯发声窗口的固定自然最佳偏移下 F−N 距离平均为正，跨条均值+0.994。N/F/R整句最佳偏移均为−2帧。

纯发声 Wav2Lip mel MAE：F/R=0.677/0.385，R在17/17更接近N；归一口部开度误差F/R=0.241/0.157，R在16/17更小。每条最高差异发声段中14/17经±160ms恒定mel平移仍以0帧最相似。a1_076的F/N/R Sync-C=2.458/4.548/4.447，停顿已修好但3.52–3.88秒发声段仍明显偏离；a1_082虽F整句高于N，局部发声F−N距离仍为正，提醒不能混用两项指标。

解释：静音修复改善缺失停顿样本，但发声残差在新说话人中复现；直接自然特征重建更接近N，说明声码器单独不足以解释差距。下一步需在预先固定的窗口和对照位置单独比较局部时间重映射与自然声学特征oracle，并用新的验证样本检查收益。本文的最高差异窗口为事后定位，不是独立验证或音素级因果归因。完整协议、逐条表、限制和复现路径见 [[36-mfa-linear-heldout-speech-residual]]。

## Observations

- [status] concluded
- [result] 新说话人17条自然音轨Sync-C N/M/F/R=5.576/4.764/4.902/5.447；F−M在6条停顿受影响样本平均+0.337，6/6上升。#syncnet
- [result] 纯发声固定偏移F−N距离17/17为正，跨条平均+0.994。#speech
- [result] 发声区Wav2Lip mel MAE F/R=0.677/0.385，R在17/17更近；口开误差F/R=0.241/0.157。#wav2lip
- [conclusion] 静音修复不能消除纯发声差距，应独立检验局部时间和同时间声学/口型形状。
- [limit] 候选段事后选取、单脸单模型、5位说话人；局部距离不是Sync-C。
- [report] [[36-mfa-linear-heldout-speech-residual]]

## Relations

- extends [[MFA-linear 停顿修复后发声段残差定位 2026-09-25]]
- relates_to [[MFA-linear 静音与停顿修复 2026-09-25]]

## Changelog

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 17条新说话人全量复核与发声段机制诊断 | September 25, 2026 | user（扩大发声段检查请求）；agent（执行） |