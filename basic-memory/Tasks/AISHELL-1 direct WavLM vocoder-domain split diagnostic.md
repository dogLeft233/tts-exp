---
title: AISHELL-1 direct WavLM vocoder-domain split diagnostic
type: task
permalink: tts-exp/tasks/aishell-1-direct-wav-lm-vocoder-domain-split-diagnostic
current_step: 5
priority: high
started: '2026-08-14'
status: done
steps:
- 冻结输入与两个 vocoder checkpoints。
- 实现 direct/warped × regular/prematched 四格。
- 运行三 speaker 小样本并 QC。
- 按 speaker 完成 feature/audio 诊断。
- 决定 vocoder adapter 或 phone trajectory route。
tags:
- task
- wavlm
- hifigan
- vocoder
- speaker
- diagnostic
completed: '2026-08-14'
---

# AISHELL-1 direct WavLM vocoder-domain split diagnostic

## Observations
- [description] 在 S0765/S0901/S0912 各 5 条上比较 natural direct、raw-TTS direct、MFA-linear 三种 WavLM feature 条件，经 regular 与 prematched frozen kNN-VC HiFi-GAN；raw-TTS direct 保留原始 feature 时长，只有 MFA-linear 使用 natural clock。
- [status] done
- [assigned_to] claude
- [current_step] 5
- [result] 90/90 audio outputs complete, 0 failures, finite, exact natural length for MFA-linear, 0 clipping；MFA-linear prematched 与既有参考 15/15 数值完全一致（max abs diff 0）。相关脚本单测 6 passed。
- [result] Prematched→regular MFA-linear mean peak：S0765 `0.0782→0.0820`，S0901 `0.3313→0.2533`，S0912 `0.4991→0.3886`；S0901/S0912 同方向降峰，说明 prematched HiFi-GAN 是真实的 speaker-dependent amplitude/resynthesis confound。
- [result] Regular vocoder 的 output→conditioning feature distance 仍在 MFA-linear 相对 raw-TTS direct 后上升：S0901 `0.0567→0.1406`（+0.0839），S0912 `0.0573→0.0924`（+0.0351）。因此 vocoder 不是唯一原因，MFA-linear trajectory mismatch 仍存在。
- [result] Prematched vocoder 在三 speaker 上 output→conditioning distance 通常高于 regular；但更低 peak 不等于更好音质、发音或 lip-sync，尚未运行 ASR/SyncNet。
- [decision] 停止把 prematched HiFi-GAN 作为唯一基线；下一步先将 regular/prematched 分叉扩展到完整 n=25，再决定是否训练 regular-vocoder adapter 或 learned phone-trajectory mapper。

## Steps
1. [x] 冻结输入、模型 revision、两个 vocoder checkpoint 和独立输出目录。
2. [x] 实现 direct/warped × regular/prematched 六条件脚本及纯函数测试。
3. [x] 在 S0765/S0901/S0912 各 5 条运行 90 个 audio-only cells。
4. [x] 完成 exact-length/finite/clipping/hash/reference QC 和 speaker-level 汇总。
5. [x] 记录分叉结论；不使用 SyncNet 选模型，转入完整 n=25 audio gate。

## Inputs and model
- Cohort: `/home/wjj/tts-audio/tts-exp/runs/aishell1_mfa_linear_n25_20260813/data_boundary/aishell1_n25_speaker_balanced_cohort.json`
- Resample-poly TTS metadata: `/home/wjj/tts-audio/tts-exp/runs/aishell1_mfa_linear_n25_resample_poly_20260813/tts_meta.json`
- Frozen tokens: `/home/wjj/tts-audio/tts-exp/runs/aishell1_mfa_linear_n25_20260813/mfa/tokens.json`
- Existing MFA-linear reference: `/home/wjj/tts-audio/tts-exp/runs/aishell1_mfa_linear_n25_resample_poly_20260813/mfa_linear/summary.json`
- Model: pinned `bshall/knn-vc` revision `c616845c4e309e24d5927f15adbdf277a3d65358`; WavLM-Large L6 1024-d; regular `g_02500000.pt`; prematched `prematch_g_02500000.pt`; both frozen.

## Outputs and code
- Output: `/home/wjj/tts-audio/tts-exp/runs/aishell1_mfa_linear_n25_resample_poly_20260814/vocoder_domain_split_20260814/`
- Generator: `scripts/run_aishell1_vocoder_domain_split.py`
- Tests: `tests/test_aishell1_vocoder_domain_split.py`
- Report: `runs/aishell1_mfa_linear_n25_resample_poly_20260814/vocoder_domain_split_20260814/report.md`

## Interpretation limits
该实验是 audio-only domain diagnostic，不能单独证明 speaker preservation、pronunciation、naturalness 或 lip-sync。regular 的峰值改善可能是振幅/声码器行为，也可能伴随 timbre/内容损失；必须在完整 n25 audio gate 后再选择训练对象。

related_to [[VC-inspired trainable speaker-robust rhythm transfer plan]]
