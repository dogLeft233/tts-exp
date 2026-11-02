---
title: AISHELL-1 residual norm and velocity constraint audio experiment
type: task
permalink: tts-exp/tasks/aishell-1-residual-norm-and-velocity-constraint-audio-experiment
current_step: 5
priority: high
status: done
steps:
- 检查 GPU/输入 provenance，固定 15 条样本、resample-poly TTS、tokens、MFA reference 和模型。
- 新增 constraint generation script/tests，定义 quantile-based residual norm and velocity
  caps，并记录每条样本的 cap、clipped fraction、trajectory statistics。
- 生成无约束、norm-only、velocity-only 和组合约束音频，完成 exact length/finite/clipping/hash QC。
- 比较 S0901/S0912 的 peak、crest、spectral flux、artifact candidates、feature cosine 和 residual
  trajectory；写报告。
- 更新项目笔记，决定是否有约束配置值得进入下游评测。
tags:
- task
- mfa-linear
- natural-anchor
- constraint
- aishell1
completed: '2026-08-14'
---

# AISHELL-1 residual norm and velocity constraint audio experiment

## Observations
- [description] 固定 alpha=1，对 S0765/S0901/S0912 各 5 条 paired utterance 比较 residual norm/velocity constraint，隔离当前 MFA-linear 的局部变质问题。
- [status] done
- [assigned_to] claude
- [completed] 2026-08-14
- [result] 90/90 音频生成成功（15 samples × 6 conditions），0 failures；全部 16 kHz、exact natural length、finite、0 clipped samples；GPU 使用 cuda，未终止任何既有进程。
- [result] 条件：`unconstrained`、`norm_q90`、`norm_q75`、`velocity_q90`、`norm_q90_velocity_q90`、`norm_q75_velocity_q90`。每条 utterance 的 cap 从其 unconstrained residual trajectory quantile 计算并记录。
- [result] unconstrained alpha=1 与已有 MFA-linear 参考 15/15 数值等价（max abs ≤1e-5）；19 项 alpha/constraint/previous regression tests 全部通过。
- [result] S0901 mean peak/crest：unconstrained `0.3313/12.449`；norm_q90 `0.3313/12.598`；norm_q75 `0.3317/13.180`；velocity_q90 `0.3337/12.545`；norm_q90+velocity `0.3337/12.679`；norm_q75+velocity `0.3313/13.159`。约束没有修复，norm cap 反而提高 crest。
- [result] S0912 mean peak/crest：unconstrained `0.4991/8.647`；norm_q90 `0.4805/8.383`；norm_q75 `0.4758/8.421`；velocity_q90 `0.4976/8.621`；norm_q90+velocity `0.4807/8.379`；norm_q75+velocity `0.4748/8.400`。S0912 有小幅改善，但不是普遍规律。
- [result] S0901/103 的 q75 norm+velocity crest 从 unconstrained `11.529` 升至 `12.012`；S0912/203 则从 `10.758` 降至 `10.656`。speaker/sample heterogeneity 明显。
- [decision] 目前不把 quantile residual norm/velocity constraint 直接推进 Wav2Lip/SyncNet；它不能稳定消除 S0901 变质。后续若继续，应改为 boundary-aware/context-aware constraint，并保留 speaker-specific audio gate。

## Steps
1. [x] 检查 GPU/输入 provenance，固定 15 条样本、resample-poly TTS、tokens、MFA reference 和模型。
2. [x] 新增 constraint generation script/tests，定义 quantile-based residual norm and velocity caps，并记录每条样本的 cap、clipped fraction、trajectory statistics。
3. [x] 生成无约束、norm-only、velocity-only 和组合约束音频，完成 exact length/finite/clipping/hash QC。
4. [x] 比较 S0901/S0912 的 peak、crest、spectral flux、feature cosine 和 residual trajectory；写报告。
5. [x] 更新项目笔记；结论是约束只对 S0912 有小幅 audio-level 改善，对 S0901 不稳定，暂不进入下游评测。

## Context
### Inputs
- Cohort: `/home/wjj/tts-audio/tts-exp/runs/aishell1_mfa_linear_n25_20260813/data_boundary/aishell1_n25_speaker_balanced_cohort.json`
- Resample-poly TTS metadata: `/home/wjj/tts-audio/tts-exp/runs/aishell1_mfa_linear_n25_resample_poly_20260813/tts_meta.json`
- TTS WAV: `/home/wjj/tts-audio/tts-exp/runs/aishell1_mfa_linear_n25_resample_poly_20260813/tts_resample_poly_16k/`
- Frozen tokens: `/home/wjj/tts-audio/tts-exp/runs/aishell1_mfa_linear_n25_20260813/mfa/tokens.json`
- Existing MFA-linear reference: `/home/wjj/tts-audio/tts-exp/runs/aishell1_mfa_linear_n25_resample_poly_20260813/mfa_linear/summary.json`

### Outputs
- Actual output directory: `/home/wjj/tts-audio/tts-exp/runs/aishell1_mfa_linear_n25_resample_poly_20260814/natural_anchor_constraints_20260814/`
- Summary: `natural_anchor_constraints_20260814/summary.json`
- Report: `natural_anchor_constraints_20260814/report.md`
- Audio arms: `audio/unconstrained`, `norm_q90`, `norm_q75`, `velocity_q90`, `norm_q90_velocity_q90`, `norm_q75_velocity_q90`

### Models/code
- Generator: `/home/wjj/tts-audio/tts-exp/scripts/run_natural_anchor_constraints.py`
- Tests: `/home/wjj/tts-audio/tts-exp/tests/test_natural_anchor_constraints.py`
- Base input/provenance loader: `/home/wjj/tts-audio/tts-exp/scripts/run_natural_anchor_alpha_blend.py`
- MFA mapping: `/home/wjj/tts-audio/tts-exp/scripts/knn_vc_retrieval.py:mfa_linear_target`
- Tail handling: `/home/wjj/tts-audio/tts-exp/scripts/pilot_generate_mfa_linear.py:extend_tokens_for_feature_tail`
- Frozen `bshall/knn-vc` revision `c616845c4e309e24d5927f15adbdf277a3d65358`; WavLM-Large L6 1024-d; prematched frozen HiFi-GAN; device cuda.

### Definition
Alpha remains 1. For residual `d = z_tts_warp - z_natural`, norm cap clips per-frame vector magnitude to the per-utterance q75/q90 norm; velocity cap clips per-frame residual increment to the per-utterance q90 velocity. Combination applies norm then velocity. No Wav2Lip/SyncNet was run.

## Relations
- project [[MFA-linear 冻结 VC 重合成与下游验证]]
