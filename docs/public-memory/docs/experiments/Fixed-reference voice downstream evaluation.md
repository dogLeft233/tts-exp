---
title: Fixed-reference voice downstream evaluation
type: report
permalink: tts-exp/docs/experiments/fixed-reference-voice-downstream-evaluation
date: '2026-08-14'
sample_count: 15
status: done
syncnet_status: complete
tags:
- fixed-reference
- mfa-linear
- wav2lip
- syncnet
- negative-result
---

# Fixed-reference voice downstream evaluation

## Setup

2026-08-14: Fixed-reference TTS control passed repaired MFA and same-environment MFA-linear audio gates. Ran 45 fresh Wav2Lip + SyncNet cells: 15 samples × (`natural_raw`, fixed-reference `raw_tts`, same-environment `mfa_linear`). Fixed face: `runs/two_stage_hubert_aishell1_20260810/ditto_videos/natural_raw/1.mp4`; each Wav2Lip cell used isolated cwd/temp. SyncNet `min_track=50`. No S0770.

Output: `runs/aishell1_mfa_linear_n25_resample_poly_20260814/fixed_reference_wav2lip_syncnet_20260814/`.

## Results

All 45 cells succeeded; no failures.

Overall means:

- natural_raw: Sync-C `6.4821`, Sync-D `7.1928`
- fixed-reference raw_tts: Sync-C `6.1541`, Sync-D `7.3184`
- fixed-reference mfa_linear: Sync-C `6.0232`, Sync-D `7.4676`

Paired fixed-reference MFA-linear minus natural:

- Sync-C `-0.4589`, bootstrap 95% CI `[-0.9571, +0.0255]`, 6/15 higher;
- Sync-D `+0.2748`, bootstrap 95% CI `[+0.0475, +0.4863]`, 4/15 lower;
- joint improvement 4/15;
- exploratory paired p: Sync-C `0.1019`, Sync-D `0.0331`.

Speaker means:

- S0765: natural `C=5.766/D=7.525`; fixed raw TTS `6.314/7.105`; fixed MFA-linear `6.012/7.705`.
- S0901: natural `6.725/6.988`; fixed raw TTS `6.071/7.267`; fixed MFA-linear `5.753/7.389`.
- S0912: natural `6.955/7.065`; fixed raw TTS `6.077/7.583`; fixed MFA-linear `6.304/7.308`.

## Interpretation

Using one fixed S0765 reference voice did not remove the speaker-dependent pattern in this fixed-face downstream test. Raw fixed-reference TTS was already below natural overall; MFA-linear reduced Sync-C further and increased Sync-D overall. S0912 improved relative to its fixed raw TTS on Sync-C, while S0901 did not. This is exploratory fixed-face evidence only, not speaker-matched generalization or causal proof about mouth motion.

The repaired MFA setup resolves earlier whole-utterance `spn` failure. It does not make MFA-linear universally robust. The remaining failure is likely in duration-conditioned trajectory rendering, TTS/fixed-reference quality, and speaker/prosody interaction rather than MFA installation.
