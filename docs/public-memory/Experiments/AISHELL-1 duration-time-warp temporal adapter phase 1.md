---
title: AISHELL-1 duration/time-warp temporal adapter phase 1
type: note
permalink: tts-exp/experiments/aishell-1-duration-time-warp-temporal-adapter-phase-1
cohort: aishell1_n25_local_tts
date: '2026-08-15'
sample_count: 25
train_segment_count: 734
valid_segment_count: 148
status: complete
qwen_cloud_called: false
tags:
- aishell1
- temporal-adapter
- duration
- time-warp
- mfa-linear
- negative
---

# AISHELL-1 duration/time-warp temporal adapter phase 1

## Context

This experiment tested whether a small identity-initialized temporal adapter can learn a known phone-level duration transformation from local paired TTS trajectories before attempting any Qwen-specific or spontaneous-speech training. It was intentionally feature-only and did not use SyncNet for selection.

## Protocol

- Cohort: current AISHELL-1 n25 local-TTS strict cohort.
- Train: 20 utterances from `S0901/S0906/S0912/S0913`.
- Validation: five `S0765` utterances, speaker-disjoint.
- `S0770`: excluded from loading, training, selection and evaluation.
- Input: local TTS WavLM-Large layer-6 trajectories and MFA phone spans.
- Natural full WavLM features were not training targets. Natural MFA token durations supplied target frame counts.
- For each matched phone, nearest-resample was the coarse input and analytic linear resampling of the TTS trajectory was the exact known target.
- Model: bounded identity-initialized duration-conditioned residual adapter with relative position, duration ratio, source/target frame counts and silence flag controls.
- Qwen cloud was not called.

## Results

- 734 train segments and 148 validation segments.
- 1000 optimization steps completed.
- Train feature loss: `0.015938 → 0.015249`.
- Validation feature loss: baseline `0.012119`; adapter `0.012701`.
- Validation cosine distance: baseline `0.005946`; adapter `0.006309`.
- Feature loss improved on 47/148 validation segments; cosine distance improved on 33/148.
- The feature validation gate failed: the adapter fitted the training speakers but did not generalize to S0765.
- Five S0765 samples each produced coarse/adapted/oracle audio with finite values, exact natural length and zero clipping. The small audio-level change does not override the failed feature gate.

## Interpretation

This run proves the training and provenance path is executable, not that a learned temporal adapter is useful. The known linear warp is already a deterministic baseline; the adapter only sees the coarse retimed trajectory plus duration controls and learns a residual. Its training improvement did not transfer to an unseen speaker. This agrees with the earlier native-TTS trajectory-denoising failure: reducing a proxy training loss is not enough when the supervision and input representation do not identify the desired cross-duration hybrid.

## Decision

Stop this implementation without increasing capacity, steps or cloud data. Do not run SyncNet and do not apply this checkpoint to Qwen/RAMC. A future attempt must first expose the complete source phone trajectory and context to a conditional field, or use an explicit length regulator plus natural pause/prosody weak supervision and TTS content/speaker anchors. Current Qwen AISHELL n25 and RAMC n25 remain untouched test/diagnostic sets.

## Artifacts

- `scripts/train_duration_timewarp_adapter.py`
- `tests/test_train_duration_timewarp_adapter.py`
- `runs/aishell1_mfa_linear_n25_resample_poly_20260815/duration_timewarp_phase1_20260815/summary.json`
- `runs/aishell1_mfa_linear_n25_resample_poly_20260815/duration_timewarp_phase1_20260815/report.md`

## Relations

- relates_to [[第一阶段：duration/time-warp temporal adapter 机制验证]]
- relates_to [[Train learned phone trajectory adapter]]
- relates_to [[VC-inspired trainable speaker-robust rhythm transfer plan]]
- relates_to [[TTS Feature Alignment and Duration Model Research]]
- relates_to [[AISHELL-1 n25 Qwen cloud MFA-linear 2026-08-14]]
- relates_to [[RAMC n25 Qwen cloud MFA-linear 2026-08-15]]
