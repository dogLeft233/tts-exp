---
title: AISHELL-1 n25 audio artifact audit
type: task
permalink: tts-exp/tasks/aishell-1-n25-audio-artifact-audit
status: done
priority: high
current_step: 5
steps:
- 新增独立审计脚本和纯函数测试，不修改历史实验产物。
- 通过输入 gate，校验 25 对、5 speakers × 5、三臂音频 SHA、MFA/SYNC metadata provenance。
- 对 75 条音频提取全局与局部声学特征，检测窄带/突发噪声、重复片段和 MFA 边界异常。
- 生成 paired/speaker 汇总、事件明细、重点样本图和 Markdown 报告。
- 人工复核 S0901/S0912 重点事件，更新项目实验笔记并关闭任务。
tags:
- task
- audio
- mfa-linear
- aishell1
completed: '2026-08-14'
---

# AISHELL-1 n25 audio artifact audit

## Observations
- [description] 对冻结 AISHELL-1 n25 cohort 的 natural、raw TTS、resample-poly MFA-linear 三臂音频进行 audio-only 波形/频谱/重复片段/phone 边界伪影分析，并比较正负 speaker。
- [status] done
- [assigned_to] claude
- [current_step] 5
- [completed] 2026-08-14
- [result] 新增 `scripts/audit_aishell1_n25_audio_artifacts.py` 与 `tests/test_audio_artifact_audit.py`；新审计单测 5/5，连同 MFA/kNN-VC 相关回归测试共 14/14 通过。
- [result] 真实 audio-only 审计完成 25 paired utterances × 3 arms = 75/75；输入 SHA、speaker quota、MFA resample-poly provenance 和 fixed-face1 SyncNet join 全部通过；输出含 75 per-file rows、226 event rows、25 waveform/spectrogram plots。
- [result] 输出目录：`runs/aishell1_mfa_linear_n25_resample_poly_20260813/audio_artifact_audit_20260813/`。
- [result] 没有发现高置信度的非平凡重复片段；50/60 Hz 或高频能量没有显示 MFA-linear 独有升高。S0901/S0912 更稳定的差异是 MFA-linear peak/crest factor 与局部瞬态候选增加，部分候选在 natural/raw TTS 也存在。
- [result] S0901 speaker 的 MFA artifact-event duration 相对 natural 平均增加约 0.231 s、相对 raw TTS 约 0.337 s；S0912 相对 natural 约 +0.100 s、相对 raw TTS 约 +0.318 s。S0901/103 与 S0912/203 均有多个 impulsive-discontinuity 候选，但不能仅据此诊断为电流干扰。
- [boundary] MFA summary 没有 fallback frame 索引，审计只做 sample-level coverage/fallback 关联；right-zero-pad 尾部单独标为 padding，不作为模型伪影。固定 S0765 face 的 SyncNet 只作探索性关联。

## Steps
1. [x] 新增独立审计脚本和纯函数测试，不修改历史实验产物。
2. [x] 通过输入 gate，校验 25 对、5 speakers × 5、三臂音频 SHA、MFA/SYNC metadata provenance。
3. [x] 对 75 条音频提取全局与局部声学特征，检测窄带/突发噪声、重复片段和 MFA 边界异常。
4. [x] 生成 paired/speaker 汇总、事件明细、重点样本图和 Markdown 报告。
5. [x] 人工复核 S0901/103、S0912/203 图和数值特征；结论保持为候选伪影证据，不作电流噪声或 speaker-independent 因果诊断。

## Context
冻结输入：cohort `runs/aishell1_mfa_linear_n25_20260813/data_boundary/aishell1_n25_speaker_balanced_cohort.json`；raw TTS `runs/aishell1_mfa_linear_n25_20260813/tts_strict/tts_meta.json`；tokens `runs/aishell1_mfa_linear_n25_20260813/mfa/tokens.json`；MFA-linear 使用 `runs/aishell1_mfa_linear_n25_resample_poly_20260813/mfa_linear/summary.json`；SyncNet 关联使用 `runs/rhythm_timing/20260813_syncnet_aishell1_n25_resample_poly_face1/summary.json`。重点试听样本为 S0901/103 和 S0912/203。

## Relations
- project [[MFA-linear 冻结 VC 重合成与下游验证]]
