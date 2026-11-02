---
title: 实现 MFA-linear 连续轨迹机制消融实验
type: task
permalink: tts-exp/tasks/实现-mfa-linear-连续轨迹机制消融实验
status: done
date: '2026-09-16'
tags:
- mfa-linear
- trajectory-ablation
- experiment
close_date: September 16, 2026
---

# 实现 MFA-linear 连续轨迹机制消融实验

依据 [[MFA-linear 连续轨迹机制消融实验 Spec]]，实现并执行固定 S0765 n=15 的音素内轨迹消融实验。任务范围包括 prepare、audio、render、score、analyze 五个阶段、最小协议测试、smoke 连通性检查和全量结果核验。

## Requirements

### Requirement: 实验实现必须严格执行冻结协议
实现 SHALL 只使用 spec 固定的 15 条样本、8 个音频臂、冻结 WavLM/HiFi-GAN/Wav2Lip/SyncNet，并产出可复算的输入、音频、视频、评分和分析文件。

#### Scenario: 协议产物完整
- **WHEN** 全量阶段成功结束
- **THEN** inputs.json、audio_qc.csv、features、audio、videos、scores.csv（225 行）、curves、analysis.json、report.md 和剂量图均存在且身份校验通过。

### Requirement: 失败必须可见
实现 SHALL 在资产、模型、音频、视频或评分失败时保留具体失败列表并将状态标记为 incomplete，不删除失败 cell 伪装成完整实验。

#### Scenario: smoke 与全量验证
- **WHEN** smoke 或全量运行遇到错误
- **THEN** 命令返回非零或明确报告 incomplete，错误包含 sample_id、arm/cell 和阶段，修复后可只补失败项。

## Observations
- [status] done
- [boundary] support30 疏通协议已完成；原 v1 的 incomplete 历史记录保留，不把失败样本伪装成 v1 完整。
- [scope] 原任务脚本与最小测试已实现；新增 `reuse` stage、可配置 `--min-common-windows` 和描述性支持阈值敏感性分析。
- [amendment] 按 [[MFA-linear 连续轨迹机制消融 support30 疏通 Spec]]，固定 15 条样本、8 个臂、模型、crop、lag 和窗口规则不变，只把新 run 的共同支持门槛设为 30，并复用父 run 的已核验音频/视频/feature/score box。
- [result] support30 新运行完成 SyncNet 225/225 cells、曲线 225/225、失败 0；G_own Sync-C 均值 +0.109，普通 95% CI [-0.200, 0.398]，证据不确定；固定自然音轨 R/E/I 为 -0.150/-0.325/-0.427。
- [verification] 父 run 原有 120 个合格 cell 与新 run 同键结果逐字段一致；225 个 cell 唯一且有限、31 点曲线可复算、T_RAW 独立支持记录；py_compile、ruff、8 个协议测试和严格产物审计通过。
- [boundary] 单说话人、历史已见样本、无真实嘴型真值且听检未完成；结果仅是探索性机制诊断，不能宣称真实 TFG 改善。
- [report] support30 运行报告与结果笔记：`runs/mfa_linear_trajectory_ablation_v2_support30_20260916/report.md`；[[MFA-linear 连续轨迹机制消融 support30 2026-09-16]]。
- [close_date] September 16, 2026
## Relations

- implements [[MFA-linear 连续轨迹机制消融实验 Spec]]
- project [[tts-exp]]