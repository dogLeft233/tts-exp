---
title: AISHELL-1 n25 phone-duration LR versus MFA-linear
type: report
permalink: tts-exp/experiments/aishell-1-n25-phone-duration-lr-versus-mfa-linear
date: '2026-08-15'
status: complete
paired_n: 24
s0770_used: false
tags:
- aishell1
- n25
- mfa
- length-regulator
- syncnet
- negative-result
---

# AISHELL-1 n25 phone-duration LR versus MFA-linear

## Context

Applied the MFA-token-based phone-duration/pause alignment mechanism to the strict local-TTS AISHELL-1 n=25 cohort and compared the new `phone_duration_lr` audio arm with the existing strict `mfa_linear` arm. The run used the frozen WavLM-L6/1024-D and prematched kNN-VC vocoder; no cloud TTS was called.

## Protocol

- Cohort: 25 utterances, five speakers x five (`S0765`, `S0901`, `S0906`, `S0912`, `S0913`); S0770 excluded.
- MFA tokens: `runs/aishell1_mfa_linear_n25_20260813/mfa/tokens.json`.
- New LR output: `runs/aishell1_mfa_linear_n25_resample_poly_20260815/phone_duration_lr_n25_20260815/`.
- Paired SyncNet output: `runs/aishell1_mfa_linear_n25_resample_poly_20260815/phone_duration_lr_n25_wav2lip_syncnet_20260815/`.
- Existing MFA-linear scores were reused only after audio/face/hash validation. New LR cells used independent Wav2Lip cwd/temp and SyncNet directories.
- Fixed-face SyncNet is exploratory and cannot establish speaker-matched generalization or causal lip-sync improvement.

## Results

- LR generation: 24/25. Sample 3 was rejected because MFA found natural speech `w` versus TTS speech `ʔ`; this remained fail-closed.
- Paired SyncNet comparison: n=24.
- MFA-linear mean Sync-C: 5.9040; phone-duration LR: 4.8087; LR minus linear: -1.0954.
- MFA-linear mean Sync-D: 7.3378; phone-duration LR: 7.4018; LR minus linear: +0.0640.
- Sync-C: LR better on 0/24 samples; paired p=2.26e-10 against LR improvement, with clustered bootstrap CI [-1.259, -0.929].
- Sync-D: LR better on 11/24; paired p=0.455, clustered bootstrap CI [-0.073, 0.223].
- Joint improvement: 0/24.

## Speaker groups

- S0765: n=4; mean delta Sync-C -1.0820; mean delta Sync-D +0.1325.
- S0901: n=5; mean delta Sync-C -0.8036; mean delta Sync-D -0.0526.
- S0906: n=5; mean delta Sync-C -1.0032; mean delta Sync-D -0.1760.
- S0912: n=5; mean delta Sync-C -1.2326; mean delta Sync-D +0.0904.
- S0913: n=5; mean delta Sync-C -1.3528; mean delta Sync-D +0.3394.

## Interpretation

The explicit MFA phone plan generated finite exact-length audio for 24 usable records, but the learned LR checkpoint did not improve the existing MFA-linear audio arm. Sync-C was lower for every paired sample and Sync-D was not significantly better. This is a negative downstream result for the current checkpoint, not evidence that MFA phone alignment itself is invalid. Do not select this checkpoint for further audio/Qwen experiments without a new model design or training protocol.

## Relations

- implements [[Explicit phone-duration/pause length-regulator prototype]]
- compares_to [[MFA-linear 冻结 VC 重合成与下游验证]]
