---
title: 25-single-encoder-waveform-enhancer
type: note
permalink: tts-exp/docs/experiments/25-single-encoder-waveform-enhancer
---

# 25 — Single-encoder HuBERT waveform enhancer feasibility

## Scope

This experiment tests a waveform-level enhancer rather than another Ditto hidden-state adapter:

```text
natural waveform → HuBERT-conditioned waveform enhancer → enhanced WAV → Ditto
```

The local training encoder is HuggingFace `facebook/hubert-base-ls960`, frozen, 768-D, 16 kHz, and 320-sample frame stride. Ditto is not given these 768-D tensors. During downstream evaluation Ditto re-encodes the enhanced WAV with its native `hubert_streaming_fix_kv.onnx` frontend, which is a separate 1024-D approximately 25-fps interface.

The historical Chinese corpus recovery and split/provenance ledger is [`26-aishell1-400-training-inventory.md`](26-aishell1-400-training-inventory.md). It identifies the 400 → 392 → 391 lineage and must be read before preparing a Chinese run. The old 13-pair `data/wav2sem_analysis_zh` corpus is not merged into this inventory.

## Model and supervision

The model is an identity-initialized, non-causal residual TCN. It receives the natural waveform and a 64-D frame-interpolated projection of HuBERT layer 6. Its output is:

```text
y = x_natural + residual_scale × tanh(residual)
```

A fresh model is exactly identity and every residual is bounded by the configured scale. The same frozen HuBERT weights are called for natural conditioning and for the enhanced-output feature loss. The latter call retains gradients through the encoder input, so HuBERT parameters remain frozen while the waveform model receives feature-loss gradients.

Targets are `weak_phone_local_tts_target` WAVs made by matching independent natural/TTS alignments, resampling matched TTS phone segments onto the natural timeline, preserving unmatched natural/pause regions, and crossfading boundaries. These are not disentangled ground truth: they can transfer speaker, phase, timing, prosody, and other acoustic factors. Missing alignments, uniform fallback, invalid spans, low coverage, missing arms, and transcript/split conflicts are excluded fail-closed.

The first fixed loss is:

```text
1.00 multi-resolution STFT
0.30 log-mel
0.10 waveform L1
0.10 HuBERT layer-6 smooth-L1
0.05 energy envelope
0.01 residual L1 to natural
```

SyncNet/Ditto metrics are not training losses.

## Debug and leakage controls

Every run records source/target hashes, alignment hashes, accepted/excluded pairs, split and speaker groups, audio statistics, feature statistics, frame counts, residual bound/statistics, individual loss terms, gradient and parameter norms, finite checks, checkpoint metadata, and environment information. JSON diagnostics use strict non-NaN serialization.

The current trainer also records train/valid `hubert_proximity` summaries: natural and enhanced HF HuBERT layer-6 features are compared with raw-TTS layer-6 features aligned through independent MFA phone spans. This reuses tensors already produced for the explicit style loss and adds `extra_encoder_calls: 0`; it is descriptive telemetry only, not a loss or checkpoint-selection criterion. It does not imply that representation proximity predicts or improves Ditto/SyncNet.

Train/valid/held-out assignment is pair- and speaker-group-level with seed 42. Validation selects checkpoints; held-out data must not select layers, thresholds, loss weights, preprocessing, or early stopping. The current study corpus is small, so any result using the available handful of pairs remains feasibility/training-style evidence unless a meaningful speaker-disjoint held-out run is available.

## Status and successor boundary

The strict Chinese single-stage run is archived in [`27-aishell1-hubert-waveform-training.md`](27-aishell1-hubert-waveform-training.md) and its confirmed objective conflict in [`28-diagnosis-chinese-hubert-training.md`](28-diagnosis-chinese-hubert-training.md). The default successor for local train/valid development is the two-stage frozen-target design in [`29-two-stage-feature-targeted-waveform-model.md`](29-two-stage-feature-targeted-waveform-model.md): it removes weak phone-local waveform reconstruction from the default objective, first predicts a bounded direct raw-TTS-directed L6 goal, then renders a bounded waveform residual toward that frozen goal. Experiment 29 is still only a local train/valid feature-space feasibility result, not a downstream result.

## Downstream evaluation boundary

After local implementation and checks, a separate authorized remote run may stage natural, enhanced, deterministic bounded-random, and raw-TTS WAVs. Ditto should run the same ordinary/identity/enhanced/random/raw-TTS matrix with the same source image, preprocessing, seed, tracking configuration, and SyncNet evaluator. Failed metrics remain failures. Report paired identity-relative Sync-C and Sync-D deltas, not an unqualified claim that HF HuBERT representation matching is a Ditto-native objective.

## Critical distinction

The experiment asks whether a HuBERT-supervised waveform change can transfer through Ditto's native re-encoding. It does **not** inject 768-D HF HuBERT features into Ditto, project them to 1024-D, or claim that a weak phone-local target is the true TTS waveform. A later success would support waveform-level transfer; a feature-only success without downstream improvement would instead show that the chosen HuBERT target is not sufficient.