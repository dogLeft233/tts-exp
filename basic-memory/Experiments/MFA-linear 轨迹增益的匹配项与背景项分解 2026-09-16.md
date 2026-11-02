---
title: MFA-linear 轨迹增益的匹配项与背景项分解 2026-09-16
type: experiment
permalink: tts-exp/experiments/mfa-linear-轨迹增益的匹配项与背景项分解-2026-09-16
status: concluded
date: '2026-09-16'
protocol: mfa_linear_trajectory_decomposition_v1
source_run: runs/mfa_linear_trajectory_ablation_v2_support30_20260916
output_run: runs/mfa_linear_trajectory_decomposition_20260916_v10
tags:
- mfa-linear
- trajectory
- syncnet
- decomposition
- support30
---

# MFA-linear 轨迹增益的匹配项与背景项分解 2026-09-16

这是对冻结 support30 历史运行的 CPU-only 事后再分析：S0765 单说话人、15 条已见 utterance、225 个 cell、每条 31 个 lag；不生成音视频、不调用 GPU/模型/API。报告先做逐 cell 的 C/D/B 重算，再把 `delta_C` 分为 `match_gain=D_baseline-D_candidate` 与 `background_gain=B_candidate-B_baseline`，并固定自然基线 lag 做 `C_anchor/search_bonus/O/relative_margin` 诊断。

## Observations

- [status] concluded
- [result] 主分析均使用同一组 `numpy.random.default_rng(20260916)` 的 20,000 次 utterance bootstrap；12 个主统计量使用 Bonferroni 校正区间，alpha_each=0.05/12，分位数方法为 linear。
- [result] N_dose（own N_100−N_000）：`delta_C=+2.206`，Bonferroni CI `[+1.634,+2.838]`；`match_gain=+0.200`（`[-0.174,+0.569]`）；`background_gain=+2.006`（`[+1.410,+2.648]`）；`dominance=+1.807`（`[+1.027,+2.678]`）。
- [result] T_dose（own T_100−T_000）：`delta_C=+2.072`，Bonferroni CI `[+1.585,+2.579]`；`match_gain=+0.119`（`[-0.274,+0.424]`）；`background_gain=+1.953`（`[+1.398,+2.518]`）；`dominance=+1.835`（`[+1.084,+2.662]`）。
- [result] source_interaction=T_dose−N_dose：`delta_C=-0.134`（`[-0.861,+0.517]`），`match_gain=-0.081`（`[-0.349,+0.214]`），`background_gain=-0.053`（`[-0.690,+0.567]`），`dominance=+0.028`（`[-0.623,+0.740]`）。因此本轮没有 TTS 特异性证据；N/T 两来源均改善更符合一般轨迹线索。
- [result] 两个剂量总增益的校正区间均排除 0，且 background_gain 与 dominance 均排除 0；数值上背景中位数上移贡献大于最佳距离项，但这只是 SyncNet 曲线代数分解。
- [result] `G_own=T_100−N_RAW` 均值约 `+0.109`，普通 95% CI 跨 0；阳性参照在本队列不确定。固定自然音轨视角的 T_dose 也为正（普通 95% CI `[+1.584,+2.135]`），但仍是历史评分响应。
- [validation] v10 通过：225 cell、210 mechanism cell、15 reference-only、45 paired rows；重算曲线和所有固定 lag 恒等式最大误差为 0；5 个冻结源文件分析前后 SHA-256 不变；JSON 有限，PNG/report 存在。
- [validation] 独立审计从源曲线重新计算 225 行 C/D/B/k_star/d_zero、k_N/O/C_anchor/search_bonus，并用独立 bootstrap index matrix 逐项复算 12 个校正区间，全部与 v10 输出相符。缺失源的 incomplete 路径和非空输出拒绝也通过验证；专项 pytest 26 个（含旧 ablation 联测）、ruff、py_compile 通过。
- [report] 报告与产物：`runs/mfa_linear_trajectory_decomposition_20260916_v10/report.md`、`analysis.json`、`validation.json`、`inputs.json`、`cell_metrics.csv`、`paired_effects.csv`、`decomposition.png`。
- [boundary] 该结果是单说话人历史样本上的探索性数值响应；T_100 是历史 MFA-linear 条件特征来源，不是原始 TTS 波形。没有真实嘴型真值、人工同步标注或音质因果控制，不能据此宣称真实口型更准、SyncNet 被欺骗或解释所有 TTS 优势；`background_gain` 也不等于错配可分性。

## Relations

- implements [[MFA-linear 轨迹增益的匹配项与背景项分解 Spec]]
- extends [[MFA-linear 连续轨迹机制消融 support30 2026-09-16]]
- relates_to [[LRS3 TTS 原生优势的分数分解与曲线诊断]]

## Implementation

- [implementation] 补齐 `relative_margin` 空值传播与 `valid_n/missing_n`，负向组件区间明确报告抵消方向；逐 cell 恒等式现在记录真实误差，生产向量分解复用标量分解，非法 view 明确失败。

## Changelog

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 按 spec 完成 CPU 再分析、独立重算审计、错误路径验证和报告交付 | September 16, 2026 | user request |
| 复审后补齐 optional/null、负向判读、真实恒等式记录和敏感性测试；v10 用当前脚本 hash 完成最终运行 | September 16, 2026 | user request |
