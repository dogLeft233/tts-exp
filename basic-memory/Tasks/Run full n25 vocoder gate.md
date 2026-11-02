---
title: Run full n25 vocoder gate
type: task
permalink: tts-exp/tasks/run-full-n25-vocoder-gate
completed: '2026-08-14'
current_step: 5
priority: high
status: done
steps:
- 参数化 speaker loader，固定五个预定义 n25 speaker。
- 运行六条件 full n25 audio-only gate。
- QC 150/150、reference 25/25、finite/exact-length/clipping。
- 计算 speaker-level peak、RMS、feature distance 与 direct-vs-MFA deltas。
- 记录 regular/prematched trade-off，转入小型可训练 vocoder adapter。
tags:
- task
- wavlm
- hifigan
- vocoder
- speaker
- n25
- diagnostic
---

# AISHELL-1 full n25 vocoder gate

## Observations
- [description] 将 regular/prematched frozen kNN-VC HiFi-GAN domain split 扩展到预定义 n25：S0765/S0901/S0906/S0912/S0913 各 5 条；natural direct、raw-TTS direct、MFA-linear 三种 WavLM feature 条件。
- [status] done
- [assigned_to] claude
- [current_step] 5
- [result] 150/150 audio cells complete, 0 failures, 0 clipping；MFA-linear exact natural length；MFA-linear prematched 与既有 reference 25/25 数值完全一致（max abs diff 0）。
- [result] Prematched→regular MFA-linear mean peak difference：S0765 `-0.0038`，S0901 `-0.0780`，S0906 `-0.0311`，S0912 `-0.1105`，S0913 `-0.0380`。regular 对 S0901/S0912 的峰值稳定作用最强。
- [result] regular MFA-linear peak 相对 raw-TTS direct 在五 speaker 都不升高（delta S0765 `-0.0019`、S0901 `-0.0218`、S0906 `-0.0082`、S0912 `-0.0063`、S0913 `-0.0054`）；prematched 在 S0906/S0913 反而升高。
- [result] prematched output→conditioning feature distance 比 regular 更低，但两者 MFA-linear 相对 raw-TTS distance 都上升；因此 prematched 更忠实 feature，regular 更稳定 amplitude，不能直接按单一指标选。
- [decision] vocoder variant 是真实 speaker-dependent confound，但并未消除 phone trajectory mismatch；下一步训练小型 vocoder input-domain/FiLM adapter，冻结 WavLM 与 HiFi-GAN 主体，不再扩大手工 residual interpolation。

## Steps
1. [x] 参数化 speaker loader，固定五个预定义 n25 speaker。
2. [x] 运行六条件 full n25 audio-only gate。
3. [x] QC 150/150、reference 25/25、finite/exact-length/clipping。
4. [x] 计算 speaker-level peak、RMS、feature distance 与 direct-vs-MFA deltas。
5. [x] 记录 regular/prematched trade-off，转入小型可训练 vocoder adapter。

## Inputs and outputs
- Inputs: `/home/wjj/tts-audio/tts-exp/runs/aishell1_mfa_linear_n25_20260813/data_boundary/aishell1_n25_speaker_balanced_cohort.json`, `/home/wjj/tts-audio/tts-exp/runs/aishell1_mfa_linear_n25_resample_poly_20260813/tts_meta.json`, `/home/wjj/tts-audio/tts-exp/runs/aishell1_mfa_linear_n25_20260813/mfa/tokens.json`, existing MFA reference summary.
- Output: `/home/wjj/tts-audio/tts-exp/runs/aishell1_mfa_linear_n25_resample_poly_20260814/vocoder_domain_split_n25_20260814/`
- Code: `scripts/run_aishell1_vocoder_domain_split.py`; tests `tests/test_aishell1_vocoder_domain_split.py` (`6 passed`).
- Model: pinned `bshall/knn-vc` `c616845c4e309e24d5927f15adbdf277a3d65358`, WavLM-Large L6, `g_02500000.pt` and `prematch_g_02500000.pt`.

## Interpretation limits
Audio-only diagnostic；不能单独证明 naturalness、speaker preservation、pronunciation 或 SyncNet。regular 峰值下降可能伴随 feature/timbre trade-off，必须用训练 adapter 的 held-out audio metrics 验证。

related_to [[AISHELL-1 direct WavLM vocoder-domain split diagnostic]]
related_to [[VC-inspired trainable speaker-robust rhythm transfer plan]]
