---
title: 011-common-scale-mask-and-envelope-ablation
type: decision
permalink: tts-exp/docs/adr/011-common-scale-mask-and-envelope-ablation
project: tts-exp
decided: '2026-10-04T06:55:06.4976142+00:00'
repository: dogLeft233/tts-exp
status: open
git_sha: 7088c1e1a563a9b627946f313cbc324b135736ad
tags:
- audio-enhancement
- acoustic-mechanism
- protocol
---

# 011-common-scale-mask-and-envelope-ablation

## Context

The user wants an applicable cross-TFG natural-audio head and an explanation of useful local acoustic changes. The current shared-profile source preserves phase/sample clock, applies ratio0.90 local envelope compression, and centers its spectral mask over frequency at each frame before uniform per-frame scaling into ±3dB. Complete fit-only selection has frozen three profiles; their540-row dev results and untouched confirmation are not yet available.

The source-level constraint implies the spectral mask's time-only frequency-average term is zero. Local envelope compression is a separate operation before the STFT. A temporal spectral null imposed by this construction cannot serve as evidence that temporal energy is unimportant. In addition, subtracting the frequency mean over time can produce an interaction mask as large as ±6dB even when the full mask is bounded by3dB.

## Decision

Use a prospective matched2×2 spectral intervention with one common utterance scalar for all component branches. For the frozen profile mask M(f,t), define μ(f)=mean_t M(f,t), δ(f,t)=M(f,t)−μ(f), and κ=min(1,3/max(abs(M),abs(μ),abs(δ))). Apply the four masks0, κμ, κδ and κ(μ+δ) to the same envelope-compressed STFT/phase. Preserve the exact source sample count, original PCM score clock, final RMS and peak policy. Ifκ<1, retain the original full-profile output as a separate calibration control; the scaled-full factorial cell is not relabeled as the original head.

Track localcompress-only separately as the temporal envelope control. Record the actual realized envelope gain and verify spectral per-frame zero means, interaction marginal means, μ+δ reconstruction, κ bounds, final RMS/peak flags and F32 waveform seals. Name any recomputed-on-uncompressed-input variant as a separate parameterization. Do not independently clip or rescale components and then claim an exact decomposition.

## Rationale

Commonκ preserves additive mask closure and zero means while keeping all four spectral cells within the same declared bound. Four native generated-video cells can estimate whether the frequency mean, dynamic interaction, or their combination matters under the same envelope context. The preserved unscaled output makes the calibration cost observable. Final RMS normalization remains part of each intervention, so video and evaluator effects are measured empirically; linear mask decomposition does not imply additive video effects or causal mediation.

## Alternatives

1. A time-only spectral branch would measure a forced-zero formula rather than an independent temporal acoustic factor.
2. Independent component clipping would mix component removal with different nonlinear projection.
3. Allowingδ up to6dB would change the bounded-head parameterization and content risk.
4. Recomputing each mask from a different envelope input would change both the intervention and its reference.

## Consequences and Evidence Boundary

This freezes an analysis design, not a completed experiment or efficacy conclusion. Implement and qualify algebra/audio closure before any ablation videos. Perform the predeclared comparisons on exposed historical fit or the development cohort only; do not use sealed confirmation for tuning. Each new audio branch needs its own content guard and exact generated-ledger PCM join; independent mouth measures and original-audio SyncNet remain separate endpoints. Select the usable shared candidate by the existing frozen dev protocol before final confirmation.

Affected source: `scripts/audio_enhancement/shared_frequency_profile_v1.py`, especially center-then-uniform-scale mask projection. Affected specification: `openspec/changes/develop-cross-tfg-audio-head-20261003/transfer-iteration.md`. Existing fit-only selector and primary original-audio protocol remain unchanged.

## Observations

- [decision] Separate temporal envelope compression from the forced-zero temporal marginal of the centered spectral mask.
- [rationale] One commonκ preserves bounded additive component closure across a matched four-cell spectral intervention.
- [alternative] Independent component clipping or a6dB interaction branch changes the estimand and is not used as an exact decomposition.
- [consequence] Qualify audio/algebra closure and new content guards before native-video ablations; untouched confirmation stays closed.

## Relations

- affects [[开发跨 TFG 的自然音频增强头]]
- affects [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[009-shared-profile-fit-and-lag-contrast-selection]]
- relates_to [[010-audio-content-check-before-render-completion]]