---
title: TTS style-matched reference audit
type: task
permalink: tts-exp/tasks/tts-style-matched-reference-audit
status: done
priority: high
current_step: 6
steps:
- 固定 15 条 cohort 和多 reference 选择
- 生成 multi-reference TTS variants
- 计算 content/speaker/style 指标
- 运行 exploratory Wav2Lip + SyncNet
- 汇总并记录决策
assigned_to: claude
started: '2026-08-14'
tags:
- task
- tts
- style
- mfa-linear
- speaker
completed: '2026-08-14'
---

# TTS style-matched reference audit

## Observations

- [description] 量化 raw TTS 与 natural 的 speaker/style/prosody 差距，并测试同 speaker 多 natural references 是否降低 TTS style mismatch。
- [status] done
- [assigned_to] claude
- [current_step] 6
- [result] 45/45 multi-reference FasterQwen3 TTS variants generated; repaired MFA 45/45; 75 acoustic feature rows and 60 paired style rows extracted; 60 WavLM L6 pooled proxy rows computed.
- [result] S0901 acoustic style distance improved strongly: prosody 0.508→0.302, spectral 0.765→0.262. S0765 changed little; S0912 worsened slightly.
- [result] Content-gated style-selected MFA-linear generated 15/15. Selected output had high peaks for S0901/S0912 (`0.217–0.490`) despite no clipping.
- [result] Selected downstream fixed-face Wav2Lip + SyncNet completed 45/45.
- [result] Overall natural `C=6.4689/D=7.2099`; selected raw TTS `6.4879/7.2772`; selected MFA-linear `6.0947/7.3395`.
- [result] Selected MFA-linear versus natural: Sync-C delta `-0.3742`, bootstrap CI `[-0.8607,+0.1390]`; Sync-D delta `+0.1295`, CI `[-0.2177,+0.4565]`; joint improvement `3/15`.
- [decision] Multi-reference style matching helps S0901 raw TTS but does not make MFA-linear robust. Continue with explicit target-speaker and natural-prosody/duration renderer, not reference selection or ordinary denoising.
- [constraint] S0770 excluded; original outputs preserved; fixed-face SyncNet is exploratory only.

## Steps

1. [x] Fixed 15-record cohort and deterministic three-reference policy.
2. [x] Generated 45 multi-reference TTS variants and passed audio/hash/sample-rate gate.
3. [x] Computed content/MFA, acoustic style, and WavLM representation proxy metrics.
4. [x] Ran selected fixed-face Wav2Lip + SyncNet matrix.
5. [x] Summarized speaker-level results and next model decision.
6. [x] Updated experiment report and closed task.

## Artifacts

- `scripts/generate_aishell1_multireference_tts.py`
- `runs/aishell1_mfa_linear_n25_resample_poly_20260814/multireference_tts_style_audit_qwen3_20260814/`
- `runs/aishell1_mfa_linear_n25_resample_poly_20260814/multireference_mfa_style_audit_20260814/`
- `runs/aishell1_mfa_linear_n25_resample_poly_20260814/multireference_selected_mfa_linear_20260814/`
- `runs/aishell1_mfa_linear_n25_resample_poly_20260814/multireference_selected_wav2lip_syncnet_20260814/`
- `docs/experiments/TTS style-matched reference audit results.md`

## Relations
- project [[MFA-linear 冻结 VC 重合成与下游验证]]
