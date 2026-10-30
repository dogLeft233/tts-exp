---
title: 29-two-stage-feature-targeted-waveform-model
type: note
permalink: tts-exp/docs/experiments/29-two-stage-feature-targeted-waveform-model
---

# 29 — Two-stage feature-targeted waveform renderer

## Scope and result boundary

This report archives a replacement for the failed single-stage Chinese HuBERT waveform objective diagnosed in [`28-diagnosis-chinese-hubert-training.md`](28-diagnosis-chinese-hubert-training.md). It is a local, speaker-disjoint **train/valid engineering-feasibility** result on paired AISHELL-1 natural / declared Faster-Qwen3 raw-TTS audio.

It does **not** report heldout evaluation, listening quality, ASR/PER, speaker preservation, Ditto, TFG, SyncNet, multilingual transfer, or generalization. Local 768-D HuggingFace HuBERT measurements are not Ditto's native approximately 25-fps, 1024-D frontend representation. Any later Ditto experiment must re-encode the exported WAV with Ditto's own frontend.

## Why replace the single-stage objective

The earlier single-stage waveform model mixed weak phone-local waveform reconstruction with an explicit raw-TTS HuBERT layer-6 term. The diagnostic in experiment 28 found that the weighted acoustic and style gradients were strongly opposed (cosine `-0.7366`) and that the acoustic gradient norm was about `15.7×` larger. This caused near-identity updates and no stable train/valid movement toward raw-TTS features.

The replacement separates target prediction from waveform realization:

```text
Stage 1
natural HuBERT L2 + L6
  → bounded normalized-L6 residual Δ
  → frozen raw-TTS-directed L6 goal

Stage 2
natural waveform + frozen natural L2/L6/Δ condition
  → bounded waveform residual
  → enhanced waveform
  → frozen HuBERT L2/L6 measurements
```

The output of both stages is ultimately an enhanced waveform. Stage 2 never projects local HuBERT features into, injects them into, or labels them as Ditto features.

## Strict data and provenance boundary

The stage manifest is:

```text
runs/two_stage_hubert_aishell1_20260810/data_boundary/
  two_stage_hubert_pair_eligibility.json
```

Its SHA-256 is:

```text
6216c3f98fce7f21fd83d2e5d99b1dabe1c6da96b323c983a0cd5e5dcff92c24
```

It contains only the declared paired raw-TTS arm and speaker-disjoint `train` / `valid` rows:

| Split | Paired items | Role |
|---|---:|---|
| train | 290 | normalizer, Stage-1 bound fitting, optimization, train-only calibration |
| valid | 50 | checkpoint selection and descriptive diagnostics |
| heldout | 0 loaded | excluded from all Stage-1/Stage-2 loading, fitting, optimization, selection, and diagnostics |

Natural audio is 16 kHz. Declared Faster-Qwen3 24 kHz raw-TTS is explicitly checked in its source duration domain and resampled to 16 kHz with the declared polyphase policy before local HuBERT extraction. The stage loader rejects foreign splits, duplicate pairs, cross-split speaker/path/hash/transcript leakage, unsupported TTS arms, missing hashes, non-MFA alignment, malformed spans, and manifest/provenance mismatches.

The weak phone-local waveform is deliberately not an objective or target at either stage. It remains an upstream pair-quality artifact only.

## Local HuBERT interface

Both stages use frozen HuggingFace `facebook/hubert-base-ls960` under one explicit interface:

```text
sample rate:          16,000 Hz
hidden size:          768
frame stride:         320 samples
content layer:        2
style / target layer: 6
encoder parameters:   frozen
Ditto-native:         false
```

Stage 1 extracts natural L2/L6 in one frozen no-grad encoder call and uses raw-TTS L6 only while constructing its direct aligned training target. Stage 2 extracts natural L2/L6 together once, creates the Stage-1 condition under `no_grad`, and re-encodes only the enhanced waveform with input gradients retained. Stage-1 and HuBERT parameters remain frozen; only the Stage-2 renderer is trainable.

## Stage 1: bounded direct raw-TTS feature puller

`AlignedTTSFeaturePuller` receives normalized natural L2 and L6, applies LayerNorm, a 1×1 projection, temporal residual blocks, and a zero-initialized 768-D head. It predicts a per-frame L2-capped residual in **normalized HuBERT L6 space**:

```text
z_goal_l6 = z_natural_l6 + Δz_l6
goal_l6   = natural_l6 + Δz_l6 × train_l6_std
```

The residual conversion intentionally does not re-add the normalizer mean. The hard cap is fitted once from train-only aligned natural-to-raw-TTS L6 displacement frames; for the archived seed-29 checkpoint it is the 0.95 quantile, `36.1266613`, from `47,926` target frames.

### Corrected loss-scale contract

An implementation defect was found before the archived runs: direct pull terms reduced the 768 feature dimensions by mean, while delta regularizers reduced them by sum. The latter inflated residual regularization by approximately 768 times. The corrected Stage-1 checkpoint contract is schema version 3:

```text
residual_space: normalized_hubert_layer6
loss_reduction: per_frame_feature_mean
```

All residual magnitude, unmatched-frame zero-delta, and temporal-smoothness terms now use per-frame feature means. A numerical regression test fixes the expected value for a one-feature delta and prevents the prior dimension-sum behavior from reappearing. Schema v1/v2 results are historical audit artifacts only and cannot supply a Stage-2 target.

Stage-1 losses are limited to aligned raw-TTS L6 cosine / Smooth-L1 pull, unmatched real-frame zero-delta penalty, all-real-frame delta magnitude penalty, and consecutive-real-frame delta smoothness penalty. There are no waveform reconstruction, STFT, mel, phone-prototype, Ditto, SyncNet, or weak-waveform terms.

### Stage-1 dual-seed valid-only result

Each best checkpoint was selected by the declared valid direct-pull metric; both selected epoch 1. The metrics below re-extract only the 50 valid pairs after selection.

| Stage-1 seed | Best epoch | Valid pull combined | Natural → raw-TTS combined | Goal → raw-TTS combined | Absolute improvement | Relative improvement | Goal closer |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 13 | 1 | 0.62595148 | 0.22467959 | 0.21706260 | 0.00761699 | 3.3902% | 50 / 50 |
| 29 | 1 | 0.62505467 | 0.22467959 | 0.21632764 | 0.00835195 | 3.7173% | 49 / 50 |

The corrected direction is positive but deliberately bounded and incomplete. On the same valid re-extraction, mean normalized predicted delta norms were `5.7966` (seed 13) and `5.7811` (seed 29), compared with mean required raw-TTS displacement norm `22.3282`. Mean direction cosine was `0.1761` / `0.1768`; mean target-displacement projection fraction was `0.04892` / `0.04865`.

Thus Stage 1 consistently moves in a useful local direction, but realizes only a small part of the measured raw-TTS displacement. Continued epochs over-pulled: the raw-TTS proximity improvement became negative after epoch 10 for seed 13 and after epoch 12 for seed 29. The early checkpoint choice is therefore a validation rule result, not evidence that longer training converges.

## Stage 2: frozen-target waveform renderer

`FeatureTargetedWaveformRenderer` accepts natural waveform plus frozen natural L2/L6/ΔL6. It independently projects the three feature streams, interpolates them to sample rate, applies a nearest-neighbor frame-validity gate, concatenates them, and sends them to an identity-initialized `ConditionedResidualTCN`:

```text
y = x_natural + residual_scale × tanh(renderer(x_natural, condition))
```

The output projection is zero-initialized, so a fresh renderer is exactly identity even when the frozen Stage-1 condition is nonzero. The archived real runs use `residual_scale = 0.05`; every output residual is hard-bounded by that scale. Invalid/padded frame conditioning is gated before fusion so it cannot smear across a waveform boundary.

Stage 2 accepts only a validated corrected Stage-1 `best.pt`. The checkpoint stores the required Stage-1 path, SHA-256, epoch, model/interface contract, train/valid split identities and source hashes, and source-manifest SHA-256. It stores renderer trainable weights only: trainable Stage-1 and HuBERT weights are neither copied into nor updated by the Stage-2 checkpoint.

### Objective and fixed schedule

Stage 2 derives its target solely from frozen Stage 1. Its exact loss whitelist is:

```text
realize_cosine
realize_smooth_l1
content_cosine
content_smooth_l1
energy
residual_l1
residual_smoothness
anti_clipping
```

`realize_*` compares enhanced L6 with the frozen Stage-1 goal. `content_*` compares enhanced L2 with natural L2. Energy, residual L1, residual temporal variation, and anti-clipping constrain the waveform. Raw-TTS features and weak phone-local waveform targets are not loaded, calculated, or optimized during Stage-2 training.

The schedule was fixed before training:

```text
warmup epochs: 1
epoch 0: realize + residual constraints; content and energy weights set to zero
epoch ≥1: fixed content and energy weights enabled
```

Checkpoint selection is also fixed before training:

```text
valid_realize_combined
= valid realize cosine + valid realize Smooth-L1

choose the lowest finite value among zero-clipping checkpoints;
ties choose the earlier epoch
```

All valid items were required to be eligible; a clipped output cannot become `best.pt`.

## Stage-2 dual-seed valid-only result

Both real runs used ten epochs, learning rate `1e-4`, one warmup epoch, `290` train pairs, `50` valid pairs, and the fixed architecture / residual scale. Each selected its epoch-9 checkpoint. The raw-TTS figures are post-selection, valid-only descriptive re-extractions; neither raw-TTS metric participated in Stage-2 optimization or selection.

| Seed | Bound Stage-1 best | Best epoch | Valid realization combined | Stage-1 goal improvement | Goal closer | Raw-TTS descriptive improvement | Raw-TTS closer | Max residual peak | Clipped samples |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 29 | v6 corrected seed-29, epoch 1 | 9 | 0.01733329 | 2.5762% | 50 / 50 | 1.2768% | 50 / 50 | 0.00509601 | 0 |
| 13 | v5 corrected seed-13, epoch 1 | 9 | 0.01700713 | 2.4591% | 50 / 50 | 1.1062% | 49 / 50 | 0.00556510 | 0 |

For seed 29, the selected checkpoint binds Stage-1 SHA-256:

```text
4d9ea40e3bdc71afb29088b6ddca282d8545b853ac6fe7462dbb37b765e28f31
```

Its separate diagnostic records the following explicit scope:

```json
{
  "objective_used": false,
  "checkpoint_selection_used": false,
  "training_used": false,
  "heldout_loaded": false
}
```

The two seeds therefore reproduce a small, bounded train/valid feature-space movement toward their frozen targets and a smaller raw-TTS descriptive movement. They do not establish perceptual quality or any downstream effect.

## Critical risk: realization and preservation conflict remains

The two-stage split avoids weak-waveform reconstruction as a direct target, but it does not prove content preservation is solved. Fixed train-only calibration diagnostics measure the cosine between realization and combined preservation gradients. The values are often strongly negative:

```text
seed 29: epoch 0 −0.7121; epoch 1 −0.7463; epoch 7 −0.8751; epoch 8 −0.6648
seed 13: epoch 7 −0.4428; epoch 8 −0.8325; epoch 9 −0.9153
```

Content displacement also grows later in training. Zero clipping and a small residual establish bounded numeric behavior, not naturalness, intelligibility, speaker identity, or preservation of semantic/prosodic content. This objective conflict must remain a release gate for any subsequent protocol.

## Controlled valid WAV export

Only the seed-29 Stage-1 / Stage-2 best pair was exported. The explicit export list contains exactly `50` valid natural inputs and only:

```text
paired_key
natural_path
natural_sha256
```

It rejects TTS, metadata, MFA-span, and heldout fields. The export destination is:

```text
runs/two_stage_hubert_aishell1_20260810/
  stage2_feature_targeted_20260810_v1_seed29/
    valid_enhanced_wav/
      50 *.wav
      enhanced_manifest.json
```

The manifest scope is `natural_audio_only_no_tts_or_heldout_input`. Integrity checks passed:

| Check | Result |
|---|---:|
| exported WAVs | 50 |
| sample rate / subtype | 16 kHz / FLOAT |
| natural source hash mismatches | 0 |
| output length mismatches | 0 |
| output clipped samples | 0 |
| maximum residual peak | 0.00509601 |
| Stage-2 checkpoint SHA-256 | `d53a6362b02e26e31ad3c3f204bd80136ec9eb666c2e9bccb72512573186e0a3` |

This is an integrity-checked local valid export. The WAVs have not been evaluated with Ditto, SyncNet, TFG, ASR/PER, speaker metrics, listening tests, remote services, or heldout audio.

## Engineering validation

The focused two-stage regression suite passed:

```text
60 passed in 5.30s
```

Coverage includes strict train/valid-only manifests, source-duration-aware resampling provenance, Stage-1 loss-scale regression, corrected checkpoint compatibility, frozen Stage-1 binding, identity initialization with nonzero condition, conditioning gates, gradient freezing, Stage-2 loss whitelist/schedule, synthetic CPU smoke, raw-TTS exclusion during Stage 2, export-list rejection, output length/hash checks, and fresh-output protection. The relevant Python modules compiled successfully and `git diff --check` was clean at validation time.

## What this result supports

- A strict local train/valid-only paired data contract can carry direct aligned raw-TTS feature supervision without weak waveform reconstruction.
- The corrected Stage-1 residual loss scale yields small, reproducible positive valid feature movement in two seeds.
- A frozen Stage-1 goal can be partially realized by a bounded waveform renderer in two seeds while retaining zero clipping.
- The train/valid lineage, checkpoint binding, selection rules, and natural-only WAV export are auditable.

## What it does not support

- Heldout performance or any generalization claim.
- Better perceptual quality, intelligibility, speaker preservation, or natural-content preservation.
- Better Ditto representations, lip synchronization, TFG, SyncNet, or video generation.
- A claim that 768-D HF HuBERT proximity equals or predicts Ditto's 1024-D frontend behavior.
- A multilingual conclusion.
- A claim that the raw-TTS descriptive metric was a Stage-2 training or selection target.

## Next boundary-preserving step

No additional training, selection, or tuning should use heldout material. Any downstream use of the exported WAVs requires a separately authorized, fixed protocol: re-encode the selected WAVs with Ditto's native frontend, preserve the paired identity/control matrix, evaluate the specified downstream metrics, and report negative or null results unchanged. Before that, independent content, audio-quality, speaker, and listening evaluation would be required to assess whether the bounded feature movement is practically acceptable.

## Related documents

- Historical single-stage architecture and downstream boundary: [`25-single-encoder-waveform-enhancer.md`](25-single-encoder-waveform-enhancer.md)
- AISHELL-1 inventory and split provenance: [`26-aishell1-400-training-inventory.md`](26-aishell1-400-training-inventory.md)
- First single-stage Chinese training result: [`27-aishell1-hubert-waveform-training.md`](27-aishell1-hubert-waveform-training.md)
- Confirmed single-stage objective conflict: [`28-diagnosis-chinese-hubert-training.md`](28-diagnosis-chinese-hubert-training.md)