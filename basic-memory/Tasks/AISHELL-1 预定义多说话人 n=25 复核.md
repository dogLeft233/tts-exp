---
title: AISHELL-1 预定义多说话人 n=25 复核
type: task
permalink: tts-exp/tasks/aishell-1-预定义多说话人-n-25-复核
status: done
priority: high
current_step: 5
steps:
- 核对现有 AISHELL-1 manifest、speaker 分布、自然/TTS/MFA 对齐产物和 Wav2Lip/SyncNet 入口。
- 设计并冻结 n=25 多说话人 cohort，明确 speaker 数量、每 speaker 样本数、valid/heldout 边界和排序规则。
- 生成或复用 25 条 MFA-linear 音频，并写入完整 provenance。
- 对 natural、raw-TTS、MFA-linear 三臂运行固定 Wav2Lip + SyncNet 评测。
- 运行配对统计和完整性校验，更新实验笔记并报告跨 speaker 结论及限制
assigned_to: claude
tags:
- task
- aishell1
- mfa-linear
- multi-speaker
completed: '2026-08-13'
---

# AISHELL-1 预定义多说话人 n=25 复核

## Observations
- [description] 将 AISHELL-1 MFA-linear 下游验证扩展为预定义 5 speakers × 5 paired utterances = 25 个独立配对样本。
- [status] done
- [assigned_to] claude
- [current_step] 5
- [result] strict provenance 版本完成 75/75 fresh Wav2Lip+SyncNet cells，0 failures，所有音频/视频 SHA 校验通过。
- [result] MFA-linear 相对 natural_raw 的 Sync-C Δ=-0.0692，cluster bootstrap 95% CI [-0.6521,+0.5137]，11/25 C 正向；Sync-D Δ=+0.0687，CI [-0.1891,+0.3266]，9/25 D 正向，joint 8/25。
- [result] MFA-linear 相对 raw_tts 的 Sync-C Δ=-0.1374，cluster bootstrap 95% CI [-0.5542,+0.1976]，12/25 C 正向；Sync-D Δ=+0.0967，CI [-0.1605,+0.4095]，12/25 D 正向，joint 9/25。
- [boundary] cohort 是 S0765(valid)+S0901/S0906/S0912/S0913(train)，不含 S0770；固定单参考脸协议，不构成 heldout、真实多 speaker video 或跨模型泛化结论。
- [problem] 本地 MFA 2.2.17 对这批输入输出全段 spn，speech-phone gate 正确拒绝 25 条；最终使用冻结 strict source manifest 中已通过门控的 natural/TTS tokens，并明确记录本地重跑失败 provenance。
- [problem] 首轮 n=25 MFA-linear/75-cell 结果混用了 strict tokens 与新生成 TTS，SHA 核验发现不匹配，已标为无效；最终只采用 strict source TTS canonical + strict tokens 的 `mfa_linear_strict` / `eval_strict` 结果。

## Steps
1. [x] 核对现有 AISHELL-1 manifest、speaker 分布、自然/TTS/MFA 对齐产物和 Wav2Lip/SyncNet 入口。
2. [x] 设计并冻结 n=25 多说话人 cohort，明确 speaker 数量、每 speaker 样本数、valid/heldout 边界和排序规则。
3. [x] 生成或复用 25 条 MFA-linear 音频，并写入完整 provenance。
4. [x] 对 natural、raw-TTS、MFA-linear 三臂运行固定 Wav2Lip + SyncNet 评测。
5. [x] 运行配对统计和完整性校验，更新 Basic Memory 任务并报告跨 speaker 结论及限制。

## Context
冻结 cohort manifest：`runs/aishell1_mfa_linear_n25_20260813/data_boundary/aishell1_n25_speaker_balanced_cohort.json`。严格 source pool 为 `runs/two_stage_hubert_aishell1_20260810/data_boundary/aishell1_400_raw_mfa_faster_qwen3_heldout.json`。有效产物：`mfa/tokens.json`（frozen strict tokens）、`tts_strict/tts_meta.json`、`mfa_linear_strict/summary.json`、`eval_strict/summary.json` 和 `eval_strict/analysis.json`。旧 n=15 及首轮 mismatch `eval/` 结果不应混入最终分析。

## Relations
- project [[MFA-linear 冻结 VC 重合成与下游验证]]
