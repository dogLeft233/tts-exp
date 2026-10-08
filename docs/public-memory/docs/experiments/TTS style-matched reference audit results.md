---
title: TTS style-matched reference audit results
type: report
permalink: tts-exp/docs/experiments/tts-style-matched-reference-audit-results
date: '2026-08-14'
sample_count: 15
status: done
syncnet_status: complete
decision: style_matching_insufficient_for_mfa_linear
tags:
- tts
- style
- mfa-linear
- speaker
- wav2lip
- syncnet
- negative-result
---

# TTS style-matched reference audit results

## Setup

2026-08-14: 15 AISHELL-1 target utterances (`S0765/S0901/S0912`, five each), three same-speaker non-target natural references per target, 45 multi-reference FasterQwen3 TTS variants. All variants canonicalized to mono 16 kHz, passed repaired MFA with 45/45 success, and retained full provenance.

Artifacts:

- generator: `scripts/experiments/natural_anchor/generate_aishell1_multireference_tts.py`
- TTS: `runs/aishell1_mfa_linear_n25_resample_poly_20260814/multireference_tts_style_audit_qwen3_20260814/`
- MFA: `runs/aishell1_mfa_linear_n25_resample_poly_20260814/multireference_mfa_style_audit_20260814/mfa_tokens.json`
- acoustic metrics: `.../audio_features.json`, `.../style_metrics.json`
- WavLM proxy: `.../wavlm_style_proxy.json`
- selected control: `.../selected_variants.json`
- selected MFA-linear: `runs/aishell1_mfa_linear_n25_resample_poly_20260814/multireference_selected_mfa_linear_20260814/`
- downstream: `runs/aishell1_mfa_linear_n25_resample_poly_20260814/multireference_selected_wav2lip_syncnet_20260814/`

Selection rule was fixed before downstream scoring: among each target's three candidates, require absolute MFA phone-count delta versus same-environment natural alignment <= 1 and speech phones > 0; choose minimum `prosody_distance_mean + spectral_distance_mean`. SyncNet was not used for selection. Selection is still same-cohort exploratory and not a generalization estimate.

## Acoustic style results

Mean distance to paired natural:

- S0765 self-clone TTS: prosody `0.345`, spectral `0.227`, all-feature `0.255`; multi-reference: `0.380`, `0.183`, `0.252`.
- S0901 self-clone TTS: `0.508`, `0.765`, `0.595`; multi-reference: `0.302`, `0.262`, `0.261`.
- S0912 self-clone TTS: `0.202`, `0.245`, `0.214`; multi-reference: `0.262`, `0.308`, `0.263`.

Multi-reference strongly reduced S0901 acoustic style distance. S0765 changed little. S0912 worsened slightly.

Mean pooled WavLM-Large layer-6 cosine proxy to paired natural:

- self-clone: S0765 `0.9685`, S0901 `0.9675`, S0912 `0.9798`;
- multi-reference: S0765 `0.9632`, S0901 `0.9556`, S0912 `0.9680`.

This proxy is not speaker verification and was not used for selection. Its decrease shows acoustic-style distance and pooled SSL similarity measure different factors.

## Downstream results

45 selected-control cells completed with fixed S0765 face, isolated Wav2Lip temp, SyncNet `min_track=50`, no S0770.

Overall:

- natural: Sync-C `6.4689`, Sync-D `7.2099`;
- selected multi-reference raw TTS: `6.4879`, `7.2772`;
- selected multi-reference MFA-linear: `6.0947`, `7.3395`.

Selected MFA-linear minus natural: Sync-C `-0.3742` (bootstrap CI `[-0.8607,+0.1390]`), Sync-D `+0.1295` (CI `[-0.2177,+0.4565]`), joint improvement `3/15`.

Speaker means (`C/D`):

- S0765: natural `5.750/7.554`; selected raw `6.329/7.119`; selected MFA-linear `6.477/7.197`.
- S0901: natural `6.697/7.016`; selected raw `6.380/7.441`; selected MFA-linear `5.986/7.229`.
- S0912: natural `6.959/7.060`; selected raw `6.755/7.272`; selected MFA-linear `5.821/7.592`.

## Decision

Multi-reference natural references reduce acoustic style mismatch for S0901, and selected raw TTS restores a small overall Sync-C advantage. They do not make MFA-linear robust: S0901 remains negative and S0912 becomes strongly negative. Style matching alone is insufficient; frozen WavLM/MFA-linear trajectory rendering remains the bottleneck or introduces speaker/prosody-dependent artifacts.

Next model work should condition explicit target-speaker and natural prosody/duration rendering, not continue selecting references or treating SyncNet gain as style fidelity. Do not use selected same-cohort reference choice as a heldout claim.
