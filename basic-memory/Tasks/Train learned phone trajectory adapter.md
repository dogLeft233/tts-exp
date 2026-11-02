---
title: Train learned phone trajectory adapter
type: task
permalink: tts-exp/tasks/train-learned-phone-trajectory-adapter
completed: '2026-08-14'
current_step: 5
priority: high
status: done
steps:
- 冻结四个 train speaker、S0765 validation、regular frozen vocoder 和 no-S0770 boundary。
- 实现 identity-initialized bounded contextual trajectory adapter 和 tests。
- 修复 argument-order failure before optimizer initialization。
- 修复 crop construction so each training crop contains an actual boundary/frame corruption。
- 训练 400 steps and evaluate S0765/S0901/S0912 MFA-linear audio-only output；stop route
  after null transfer。
tags:
- task
- trajectory
- mfa-linear
- speaker
- training
- negative
---

# Train learned phone trajectory adapter

## Observations
- [description] Train a bounded identity-initialized temporal adapter on native raw-TTS WavLM-L6 phone trajectories after artificial boundary/frame interpolation corruption; apply to MFA-linear natural-clock trajectories and frozen regular HiFi-GAN.
- [status] done
- [assigned_to] claude
- [current_step] 5
- [result] First run had a deterministic call-order bug and was preserved at `phone_trajectory_adapter_20260814/`; corrected run uses crop-local corruption and is preserved at `phone_trajectory_adapter_cropfix_20260814/`.
- [result] Corrected tests `4 passed`; corrected run completed 400 steps, 659,584 trainable parameters; feature loss `0.02469→0.02289`, velocity loss `0.05017→0.04644`.
- [result] Despite a real training signal, MFA-linear audio barely changed: S0765 peak `0.0820→0.0818`, distance `0.0821→0.0821`; S0901 `0.2533→0.2536`, distance `0.1406→0.1407`; S0912 `0.3886→0.3887`, distance `0.0924→0.0924`.
- [decision] Native-TTS corruption denoising does not transfer to cross-phone-duration MFA-linear trajectory mismatch. Stop increasing this denoiser; the failure is supervision mismatch, not simply insufficient steps.

## Steps
1. [x] Freeze four train speakers, S0765 validation, regular frozen vocoder and no-S0770 boundary.
2. [x] Implement identity-initialized bounded contextual trajectory adapter and tests.
3. [x] Fix argument-order failure before optimizer initialization.
4. [x] Fix crop construction so each training crop contains an actual boundary/frame corruption.
5. [x] Train 400 steps and evaluate S0765/S0901/S0912 MFA-linear audio-only output; stop route after null transfer.

## Inputs and outputs
- Inputs: frozen n25 cohort, resample-poly TTS metadata, strict tokens, MFA reference summary.
- Old output: `/home/wjj/tts-audio/tts-exp/runs/aishell1_mfa_linear_n25_resample_poly_20260814/phone_trajectory_adapter_20260814/`
- Corrected output: `/home/wjj/tts-audio/tts-exp/runs/aishell1_mfa_linear_n25_resample_poly_20260814/phone_trajectory_adapter_cropfix_20260814/`
- Code: `scripts/train_phone_trajectory_adapter.py`; tests `tests/test_train_phone_trajectory_adapter.py` (`4 passed`).
- Model: frozen WavLM-Large L6 + regular kNN-VC HiFi-GAN, pinned revision `c616845c4e309e24d5927f15adbdf277a3d65358`.

## Conclusion
A native-TTS denoising objective does not identify the desired hybrid feature trajectory. To learn a useful mapper, future supervision must explicitly model duration-conditioned phone rendering or use a pretrained TTS/codec decoder with separate content, timing/prosody and target-speaker controls. Do not run SyncNet for this null audio-only route.

related_to [[VC-inspired trainable speaker-robust rhythm transfer plan]]
related_to [[Train frozen-vocoder input adapter]]
