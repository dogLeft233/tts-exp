---
title: Full phone-trajectory renderer feature gate
type: report
permalink: tts-exp/docs/experiments/full-phone-trajectory-renderer-feature-gate
tags:
- tts
- trajectory-renderer
- duration
- prosody
- aishell1
- negative-result
---

# Full phone-trajectory renderer feature gate

## Context

The fixed-K=8 phone-duration length regulator was not viable as an acoustic renderer: the n25 exploratory arm generated 24/25 exact-length waveforms, but Sync-C was lower than strict MFA-linear on all 24 paired samples. The second stage therefore moved to a feature-only, full phone-trajectory renderer rather than increasing the LR capacity or calling another cloud TTS model.

## Implementation

- Added `scripts/experiments/phone_trajectory/phone_trajectory_prosody_renderer.py`.
- Added `scripts/experiments/phone_trajectory/train_phone_trajectory_prosody_renderer.py`.
- Added `tests/test_phone_trajectory_prosody_renderer.py`.
- The renderer consumes each matched TTS phone's complete WavLM-L6 trajectory (1024-D), phone ID, local context IDs, source/target frame ratio, target position, pause flag, and four natural prosody controls: log RMS, normalized log F0, voicing, and energy derivative.
- Segment-wise linear resampling is only the full-trajectory stability baseline; the model also uses source trajectory encoding and target-to-source cross-attention. It is not the previous fixed-anchor LR.
- Native TTS self-reconstruction and TTS-only analytic phone-wise warp examples are used as supervision. Natural full WavLM features are never a target, and unmatched natural speech remains fail-closed. Unmatched pauses can be masked placeholders.
- Checkpoint interface is fixed to local kNN-VC revision, WavLM-L6, 16 kHz, 320-sample stride, 1024-D, no vocoder loaded. Source, TTS, and token hashes are saved for the successfully prepared records.

## Result

Run directory: `runs/aishell1_mfa_linear_n25_resample_poly_20260815/phone_trajectory_prosody_phase2_20260815/full/`

- 20 training utterances were prepared.
- 4/5 S0765 validation utterances were prepared.
- Sample 3 was rejected as required: natural speech token `w` had no ordered TTS speech match (`ʔ`), so it was not silently converted to silence.
- After 400 CUDA steps, validation synthetic-warp relative improvement over the global baseline was `0.999985`; validation native phone accuracy was `0.99999996`; owner coverage and finite-output checks passed.
- The final gate is `false` only because the expected five-record validation split was incomplete after the fail-closed sample-3 rejection. No vocoder/audio, SyncNet, or cloud TTS was run.
- Smoke run and strict checkpoint restore both passed.

## Verification

The new renderer plus related MFA/LR/kNN regression tests passed: `38 passed`. The temporary retry with `librosa` also stopped during collection because 12 unrelated legacy modules require the missing `pyloudnorm` package; no new renderer test failed.

## Next decision

Do not interpret the feature gate as a complete hybrid-audio success: the model learned the synthetic/self-reconstruction mechanics, but the validation cohort is incomplete and no waveform realization was authorized by the stop gate. Resolve the legal speech-replacement case or define an explicitly reviewed mapping policy before any subsequent audio/vocoder experiment. Do not use S0770, cloud TTS, or SyncNet for selection or tuning.

## Observations

- [decision] Do not enlarge the failed fixed-K LR checkpoint; expose complete source trajectories and context instead.
- [insight] Full-trajectory feature mechanics and phone identity generalize to the four valid S0765 records, while unmatched speech remains an explicit data-quality failure.
- [constraint] Natural full WavLM cannot serve as the hybrid target; the current checkpoint is feature-only and audio QC is intentionally not run.
- [result] Sample 3's `w` versus `ʔ` mismatch is a legitimate speech replacement, not a silence placeholder.

## Local listening smoke and feature ablation

With explicit user approval, generated three local exact-length listening samples using the full checkpoint and prematched local HiFi-GAN only:

- `listening_smoke_20260815/audio/1.wav` — S0765
- `listening_smoke_20260815/audio/101.wav` — S0901
- `listening_smoke_20260815/audio/201.wav` — S0912

No cloud TTS, Wav2Lip, or SyncNet was used. Feature ablation on the same three samples completed with no failures. The full model's normalized residual RMS versus the segment-linear reference was 0.00354–0.00429; removing prosody changed the output by RMS 0.00198–0.00280 and caused prosody-head MSE to rise to 2.77–5.51; removing context changed the output by RMS 0.00081–0.00116 and retained phone accuracy 0.962–1.0. This shows the controls affect the feature output, but the effect is small and does not establish waveform quality or superiority to MFA-linear.

- [result] Three bounded local audio samples generated successfully with exact natural length.
- [insight] Prosody conditioning has a larger measured effect than context in this small ablation, but the renderer remains close to its segment-linear reference.
- [constraint] Treat the samples as listening smoke only; do not generalize from three samples or run SyncNet without a separately passed feature/data gate.

## Cloud Qwen source comparison

Three cloud Qwen voice-clone samples were generated successfully with `qwen3-tts-vc-2026-01-22` using the paired natural audio as the reference. Canonical raw Qwen files are under `runs/aishell1_mfa_linear_n25_resample_poly_20260815/phone_trajectory_prosody_phase2_20260815/qwen_cloud_listen3_20260815/tts/` for sample IDs 1, 101, and 201. The API key was injected only into the generation process and was not written to disk or metadata.

The three raw Qwen samples were MFA-aligned locally using MFA 3.4.1 and then passed to the phase-two renderer. All three were rejected by the existing fail-closed speech mapping: sample 1 had unmatched natural `tʲ`, sample 101 had unmatched natural `kʷ`, and sample 201 had unmatched natural `tɕ`. Therefore no cloud-Qwen phase-two hybrid waveform was produced; this is a phone-token compatibility failure, not a cloud TTS generation failure.

- [result] Cloud Qwen raw TTS generation: 3/3 successful.
- [result] Cloud Qwen → phase-two renderer: 0/3 accepted under strict mapping.
- [constraint] Do not force these speech mismatches to silence; any broader speech-replacement policy must be evaluated separately.

## Cloud-Qwen n25 phase2 comparison

The full-trajectory renderer was run on the existing 25-sample cloud-Qwen source set (`qwen3-tts-vc-2026-01-22`) without additional cloud calls. All 25 phase2 feature outputs passed generation, exact-length, finite and owner-coverage checks. Wav2Lip and SyncNet then scored all 25 with independent per-sample work directories against the cached same-cohort MFA-linear arm.

Output directories:
- phase2 outputs: `runs/aishell1_mfa_linear_n25_resample_poly_20260815/phone_trajectory_prosody_phase2_20260815/qwen_cloud_phase2_n25_20260815/`
- paired video scores: `runs/aishell1_mfa_linear_n25_resample_poly_20260815/phone_trajectory_prosody_phase2_20260815/qwen_cloud_phase2_wav2lip_syncnet_20260815/`

Overall fixed-face exploratory comparison (n=25, no S0770):

- MFA-linear mean Sync-C: 6.56908; phase2 renderer: 6.56280; delta phase2-minus-linear: -0.00628.
- MFA-linear mean Sync-D: 7.10160; phase2 renderer: 7.10484; delta phase2-minus-linear: +0.00324.
- Sync-C paired p=0.9089; Sync-D paired p=0.9493.
- Cluster bootstrap 95% CI: Sync-C delta [-0.1142, 0.1030]; Sync-D delta [-0.1341, 0.1052].
- Phase2 was better on Sync-C for 12/25, Sync-D for 8/25, and jointly for 7/25.

Speaker means (phase2 minus MFA-linear): S0765 C +0.2058/D -0.2536; S0901 +0.0330/+0.0452; S0906 -0.0016/-0.0158; S0912 -0.1008/+0.1000; S0913 -0.1678/+0.1404. The result does not show a meaningful aggregate improvement over MFA-linear; fixed-face SyncNet remains exploratory and is not a speaker-generalization or causal claim.

- [result] Full n25 cloud-Qwen phase2 generation and paired Wav2Lip/SyncNet scoring completed 25/25.
- [insight] The new renderer is statistically indistinguishable from MFA-linear on this protocol, so it has not demonstrated practical downstream gain.
