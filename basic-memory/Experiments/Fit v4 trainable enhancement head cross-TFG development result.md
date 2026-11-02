---
title: Fit v4 trainable enhancement head cross-TFG development result
type: experiment
permalink: tts-exp/experiments/fit-v4-trainable-enhancement-head-cross-tfg-development-result
status: concluded
project: tts-exp
tags:
- experiment
- tts
- tfg
- trainable-head
---

# Fit v4 trainable enhancement head cross-TFG development result

## Summary

The existing fit_v4 trainable waveform checkpoint was executed on three TFG models and ran successfully, but its candidate score was below the fixed RMS baseline on all three models.

## Protocol and provenance

Checkpoint:
runs/cross_tfg_audio_head_20261003/fit_v4_native_fullcontext_readback_attempt02/artifacts/output/shared_audio_head.pt
SHA256 3b200be269089a1ca8e7afe7d3d5481ad9c45b45c18eb4c94e5d2e4dab7941f4.

Aggregate:
runs/cross_tfg_audio_head_20261003/v4_cross_frontend_score_evidence_readback_attempt01/extracted/paired_descriptive_metrics.json
Report SHA field 4229c2f732bcc3d9db38bff218eae8abf8a51b374d75c073840f54b454bb8740.
Results SHA256 0f1f009207c85087ff7eab7378221ecdfe4d6ab3a01d83442b29e211871da5eb.
Plan raw SHA256 70701ac4a83e5ea2abd737708aba50b5c5ed2a0d51c18bad38ae84b4480ef28c.
Producer manifest recorded 180/180 PASS; repeat QC was not run.

The run used a 16 kHz spectral head with global RMS target 0.0376838172675746, RMS-stage peak ceiling 0.99, post-head target RMS matching, final peak ceiling 0.99, n_fft 512, hop 128, hidden width 24, and a bounded 3 dB spectral residual. The fit objective was a Wav2Lip/SyncNet training proxy with RMS and lag guards.

## Results

Full Cbest candidate minus RMS means:
- Wav2Lip: -0.0397772233
- MuseTalk: -0.0293471972
- Ditto: -0.0637744665
- Equal-model: -0.0442996290

The common complete nine-speaker subset was -0.0225986375. Guard20 policy excluded S0022 in this aggregate.

## Conclusion

This is genuine executable cross-model trainable-head evidence, but it is development data and uniformly below RMS. The checkpoint remains research-only and is not a production replacement. The current usable method is model-aware routing with MuseTalk explicit11 when runtime/source checks pass and RMS fallback for Wav2Lip, Ditto, unknown models, or failed candidate execution. A future trainable attempt needs a new, pre-budgeted hypothesis and a small cross-model holdout.


## Relations

- relates_to [[Developing a cross-TFG natural audio enhancement head]]
- relates_to [[Model-aware natural-audio routing and lossless track replacement qualification]]
- relates_to [[Bounded unseen-speaker validation of the trainable enhancement checkpoint]]
