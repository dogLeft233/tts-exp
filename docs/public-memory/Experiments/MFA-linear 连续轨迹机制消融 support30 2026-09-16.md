---
title: MFA-linear 连续轨迹机制消融 support30 2026-09-16
type: experiment
permalink: tts-exp/experiments/mfa-linear-连续轨迹机制消融-support30-2026-09-16
status: concluded
date: September 16, 2026
protocol: mfa_linear_trajectory_ablation_v2_support30
run_path: runs/mfa_linear_trajectory_ablation_v2_support30_20260916
result_status: complete
parent_run: runs/mfa_linear_trajectory_ablation_20260916
tags:
- mfa-linear
- trajectory-ablation
- syncnet
- mechanism
- support30
---

依据 [[MFA-linear 连续轨迹机制消融 support30 疏通 Spec]]，在不改动 v1 原运行的前提下，复用已核验的 S0765 固定 15 条样本、120 条音频、120 个视频、feature 和 score box，仅在新目录按最小共同支持 30 重跑 SyncNet 评分与分析。协议仍固定 8 个音频臂、31 个 lag（-15…+15）、共同窗口 `range(15,F-15)`，T_RAW 使用独立支持集。

## Observations
- [status] concluded
- [progress] support30 疏通运行完成；结果为探索性诊断
- [protocol] `mfa_linear_trajectory_ablation_v2_support30`；父运行 `runs/mfa_linear_trajectory_ablation_20260916/` 保留原 incomplete 状态，新运行 `runs/mfa_linear_trajectory_ablation_v2_support30_20260916/`。
- [reuse] 新目录仅保存输入/复用元数据与复制的 JSON manifest；所有音频、视频、feature、固定 score box 通过绝对路径引用父运行，父级 inputs/audio/videos manifest SHA 已核验，未使用可写链接覆盖原文件。
- [result] SyncNet 评分 225/225 cell、曲线 225/225，失败 0；每个 cell 有 31 个有限 lag 值，14 个非 T_RAW mechanism cell 与 T_RAW 的支持分别记录并均达到 30。
- [continuity] 父运行已有的 120 个合格 cell 与本轮同键结果逐字段一致（C、D、curve_median、k_star、d_zero、support 和 fixed natural lag 均无差异）。
- [main_result] G_own = q_own(T_100)-q_own(N_RAW) 的 Sync-C 均值 +0.109，普通 95% CI [-0.200, 0.398]，因此只能说均值为正、证据不确定；固定自然音轨 q_N 的 R/E/I Sync-C 均值分别为 -0.150/-0.325/-0.427。
- [sensitivity] 仅按当前分数的非 T_RAW 共同支持做描述性筛选：阈值 30/35/40/45/50 的 n 分别为 15/13/12/10/8，G_own Sync-C 均值分别为 +0.109/+0.186/+0.197/+0.133/+0.114；没有重评分、重切片或新增显著性检验。
- [quality] 运行审计确认 225 个唯一 cell、全部有限、原 v1 未改变、无 symlink；py_compile、ruff、8 个协议测试和独立严格产物审计通过。
- [boundary] 单说话人、历史已见样本、没有真实嘴型运动真值，音质/声码器适配仍可能混入；听检未完成，不能据此宣称 TFG 真实改善。
- [report] 详细报告、分数、曲线和 dose 图：`runs/mfa_linear_trajectory_ablation_v2_support30_20260916/report.md`、`analysis.json`、`scores.csv`、`curves.json`、`keep_dose.png`。

## Relations

- implements [[MFA-linear 连续轨迹机制消融 support30 疏通 Spec]]
- extends [[MFA-linear 连续轨迹机制消融 2026-09-16]]
- relates_to [[TTS 原生增益来源的生成端与评估端交叉诊断]]

## Changelog

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 执行 support30 疏通；复用父运行产物完成 225 cell 与敏感性分析，保留不确定性与单说话人边界 | September 16, 2026 | user request / agent execution |