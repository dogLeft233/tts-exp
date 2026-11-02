---
title: Codex checkpoint - 2026-10-12T18-30-00Z - Cross-TFG trainable boundary confirmed
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-12-t18-30-00-z-cross-tfg-trainable-boundary-confirmed
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: 2026-10-12 18:30:00+00:00
username: laptop-e133gsle\dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
trigger: compact
model: gpt-6.1-sol
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-12T18-30-00Z - Cross-TFG trainable boundary confirmed

## Summary

The evidence now covers both the executable route and the existing trainable head across three TFG development renders plus one bounded unseen-speaker check; the trainable checkpoint remains below the RMS baseline, so model-aware routing with RMS fallback is the usable method.

## Story

The active objective requires a reusable natural-audio enhancement head with original PCM restored in the final video, an RMS baseline, a trainable waveform path, real multi-TFG generation and SyncNet evidence, and explicit generalization limits. To balance cost and reliability, I did not reopen the retired center or Confirm matrix. Instead I inspected the existing fit_v4 artifacts and the completed bounded Wav2Lip holdout.

The fit_v4 checkpoint ran on Wav2Lip, MuseTalk, and Ditto development renders: 10 speakers per model, 60 paired seed rows per model, 180/180 producer rows passed. Full Cbest candidate-minus-RMS means were Wav2Lip -0.039777, MuseTalk -0.029347, Ditto -0.063774, equal-model -0.044300. This is genuine cross-model trainable execution, but it is uniformly below RMS. The separate fit-only checkpoint used in the bounded unseen check ran on ten unseen speakers with Wav2Lip only, 20/20 rows passed, and gave full +0.003285 with 5/5 wins and losses; guard20 gave +0.003601 with 4/5. The trainable route therefore stays research-only.

The production method is the frozen model-aware route: MuseTalk uses explicit11 local spectral shaping only when runtime, source pins, shape digest, and finite checks pass; Wav2Lip, Ditto, unknown models, or any provider failure use fixed RMS. The CLI generates control audio, and final mux replaces generated audio with untouched natural PCM. Existing engineering smoke proved generation, lossless mux, and official SyncNet execution.

## Working State

Route config SHA256: 886f75c03977ab7d8ac7a5caecd568421c9a46cbf455d6b9cef87a9bf24fb00.
Trainable fit_v4 checkpoint SHA256: 3b200be269089a1ca8e7afe7d3d5481ad9c45b45c18eb4c94e5d2e4dab7941f4.
Fit_v4 aggregate: runs/cross_tfg_audio_head_20261003/v4_cross_frontend_score_evidence_readback_attempt01/extracted/paired_descriptive_metrics.json.
Fit_v4 aggregate report SHA field: 4229c2f732bcc3d9db38bff218eae8abf8a51b374d75c073840f54b454bb8740.
Fit_v4 results SHA256: 0f1f009207c85087ff7eab7378221ecdfe4d6ab3a01d83442b29e211871da5eb5.
Bounded unseen bundle: .codex/scratch/trainable_w2l_holdout_20261013_attempt01/.
HANDOFF.md, scripts/audio_enhancement/README.md, the active Basic Memory task, and two experiment notes now record these boundaries.

## Verification

- Read-only inspection of fit_v4 report/status, checkpoint hash, source pins, aggregate report, and producer manifest.
- fit_v4 producer manifest: 180/180 PASS; repeat QC was not run.
- Bounded Wav2Lip unseen holdout: 20/20 PASS, all generator and SyncNet return codes zero, all mux audio matched natural PCM, video identity preserved, ten paired source checks passed.
- No implementation tests were run.
- No new GPU work was launched in this turn.

## Observations

- [result] The trainable spectral head is executable across Wav2Lip, MuseTalk, and Ditto but underperforms fixed RMS in the existing development aggregate.
- [result] The trainable checkpoint's unseen Wav2Lip result is near zero and heterogeneous.
- [decision] Deliver model-aware routing with RMS fallback as the practical method and keep trainable checkpoints research-only.
- [decision] Further GPU work requires a new hypothesis and a small explicit budget; do not repeat the old matrix.
- [next_step] Use the documented route and its explicit boundaries as the current method; only open a new experiment for a materially different trainable objective.

## Relations

- continues [[Codex checkpoint - 2026-10-12T18-12-00Z - Trainable holdout closed and route boundary]]
- relates_to [[Developing a cross-TFG natural audio enhancement head]]
- relates_to [[Fit v4 trainable enhancement head cross-TFG development result]]