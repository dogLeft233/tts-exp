---
title: Train frozen-vocoder input adapter
type: task
permalink: tts-exp/tasks/train-frozen-vocoder-input-adapter
completed: '2026-08-14'
current_step: 5
priority: high
status: done
steps:
- 冻结四个 train speaker、S0765 validation、regular/prematched teacher-vocoder pair 和 no-S0770
  boundary。
- 实现 identity-initialized bounded feature affine adapter 和 tests。
- 修复 inference-tensor/autograd boundary with `.detach().clone()`。
- 训练 250 steps on 20 train pairs × natural/raw-TTS native arms。
- 验证 S0765 native/MFA-linear outputs with audio QC；按 feature fidelity 决定下一路线。
tags:
- task
- vocoder
- adapter
- speaker
- mfa-linear
- training
---

# Train frozen-vocoder input adapter

## Observations
- [description] Train a tiny identity-initialized feature affine adapter before frozen prematched HiFi-GAN, using frozen regular HiFi-GAN as an acoustic teacher on native natural/raw-TTS features from four n25 train speakers; validate on held-out S0765 and transfer to MFA-linear.
- [status] done
- [assigned_to] claude
- [current_step] 5
- [result] First attempt failed because WavLM `@torch.inference_mode()` tensors cannot participate in adapter backward; fixed with explicit `.detach().clone()` and reran without changing permissions or bypassing autograd.
- [result] 250 steps completed; training spectral loss `0.02910→0.01809`, waveform loss `0.02458→0.01310`.
- [result] Speaker-disjoint S0765 MFA-linear: prematched peak/crest `0.0782/8.858`, adapted `0.0699/8.394`, regular `0.0820/7.812`; adapted reduces peak/crest and keeps finite/0 clipping.
- [result] S0765 MFA-linear output→conditioning distance prematched `0.1297`, adapted `0.1312`; adapter did not improve feature fidelity, so it is an acoustic/vocoder calibration only, not a trajectory solution.
- [result] S0765 raw-TTS direct adapted peak/crest `0.0747/8.708` versus prematched `0.0809/8.907`; spectral flux remained approximately unchanged. No new high-band or silence anomaly was observed in aggregate audit.
- [decision] Adapter route is numerically safe and reduces the prematched amplitude/crest confound, but cannot replace learned phone trajectory alignment. Do not run SyncNet or claim speaker/lip-sync improvement from this result.

## Steps
1. [x] Freeze four train speakers, S0765 validation, regular/prematched teacher-vocoder pair and no-S0770 boundary.
2. [x] Implement identity-initialized bounded feature affine adapter and tests.
3. [x] Fix inference-tensor/autograd boundary with `.detach().clone()`.
4. [x] Train 250 steps on 20 train pairs × natural/raw-TTS native arms.
5. [x] Validate S0765 native and MFA-linear outputs with audio QC; select next route based on feature-fidelity result.

## Inputs and outputs
- Inputs: `/home/wjj/tts-audio/tts-exp/runs/aishell1_mfa_linear_n25_20260813/data_boundary/aishell1_n25_speaker_balanced_cohort.json`, resample-poly TTS metadata, frozen MFA tokens and reference summary.
- Output: `/home/wjj/tts-audio/tts-exp/runs/aishell1_mfa_linear_n25_resample_poly_20260814/vocoder_input_adapter_20260814/`
- Code: `scripts/train_vocoder_input_adapter.py`; tests `tests/test_train_vocoder_input_adapter.py` (`4 passed`).
- Checkpoint: `adapter.pt` in output directory; 2048 bounded affine parameters.
- Model: frozen WavLM-Large L6 and regular/prematched kNN-VC HiFi-GAN at pinned revision `c616845c4e309e24d5927f15adbdf277a3d65358`.

## Conclusion
输入域/振幅 calibration 可以用很小的 learned component 部分缓解 prematched vocoder 的 speaker-dependent peak/crest，但 S0901/S0912 的主要 MFA-linear feature trajectory mismatch 仍未被训练。下一步应转向 phone-level trajectory model，而不是继续增大 vocoder adapter。

related_to [[VC-inspired trainable speaker-robust rhythm transfer plan]]
related_to [[Run full n25 vocoder gate]]
