---
title: 27-aishell1-hubert-waveform-training
type: note
permalink: tts-exp/docs/experiments/27-aishell1-hubert-waveform-training
---

# 27 — AISHELL-1 Chinese HuBERT waveform enhancer training

## Scope

This report archives the first Chinese run using the explicit raw-TTS HuBERT layer-6 objective and the new train/validation proximity telemetry. The data inventory and provenance are defined separately in [`26-aishell1-400-training-inventory.md`](26-aishell1-400-training-inventory.md).

This is a local training/feasibility result. It is not a Ditto, TFG, or SyncNet result, and it does not establish heldout generalization of the enhancer.

## Run artifacts

Fresh output directory:

```text
runs/hubert_waveform_aishell1_20260810/
├── aishell1_test_400_condition_mfa_with_silence.json
├── targets/target_manifest.json
├── training_target_manifest.json
└── training_explicit_style_3epoch_retry/
    ├── checkpoint.pt
    ├── epoch_metrics.jsonl
    └── training_debug.json
```

The original condition manifest was not overwritten. The derived condition manifest adds explicit `sil` spans for observed gaps and audio boundaries, and records its source manifest and derivation. The trainer-compatible manifest aliases dataset `test` to `heldout`; no item membership changed.

The raw Faster-Qwen3 TTS files are 24 kHz. Target construction and training make the 24 kHz → 16 kHz conversion explicit with polyphase resampling. Natural and weak-target audio remain 16 kHz. No 24 kHz file is silently interpreted as 16 kHz.

## Data used

The recovered source inventory contains 400 candidates, 392 strict-MFA pairs, and 391 historical dataset-stage accepted items. For this run, the fresh strict target rebuild accepted 351 pairs:

```text
train:   259
valid:    45
heldout:  47
```

The 41 additional exclusions in this fresh rebuild were recorded in:

```text
runs/hubert_waveform_aishell1_20260810/targets/target_manifest.json
```

Their reasons were:

```text
39  alignment does not cover the complete audio duration
 1  matched token coverage 0.590 is below threshold
 1  matched speech coverage 0.898 is below threshold
```

The heldout split was not used for optimization, validation, checkpoint selection, or loss-weight selection.

## Model and objective

The model is the identity-initialized residual waveform enhancer described in [`25-single-encoder-waveform-enhancer.md`](25-single-encoder-waveform-enhancer.md):

```text
natural waveform
    → frozen facebook/hubert-base-ls960 layer-6 conditioning
    → residual waveform TCN
    → enhanced waveform
```

The local HuBERT interface is frozen HF HuBERT, 768-D, 16 kHz, layer 6, with 320-sample frame stride. The output remains a waveform. HF 768-D features are never injected into Ditto and are never projected into Ditto's native 1024-D frontend.

Run configuration:

```text
epochs:       3
learning rate: 1e-4
seed:         42
device:       CUDA
```

The explicit style target is raw TTS HuBERT layer-6 features aligned to natural frame positions through independent MFA phone spans. The weak phone-local waveform target remains a weak pseudo-target, not disentangled ground truth.

## Training trajectory

| Epoch | Train total | Valid total |
|---:|---:|---:|
| 1 | 2.669697 | 2.452643 |
| 2 | 2.670656 | 2.440275 |
| 3 | 2.665737 | 2.451692 |

The validation loss was lowest at epoch 2 and increased slightly at epoch 3. The short run does not show a sustained monotonic loss decrease.

## HuBERT proximity telemetry

The telemetry compares masked natural/enhanced layer-6 features with aligned raw-TTS layer-6 features. It is computed from tensors already present for the explicit style objective and adds no encoder forward pass.

Combined distance:

| Epoch | Train natural → TTS | Train enhanced → TTS | Valid natural → TTS | Valid enhanced → TTS |
|---:|---:|---:|---:|---:|
| 1 | 0.276221 | 0.276385 | 0.226136 | 0.226419 |
| 2 | 0.276221 | 0.276476 | 0.226136 | 0.226181 |
| 3 | 0.276221 | 0.276371 | 0.226136 | 0.226291 |

Enhanced-closer fraction:

```text
train:   45.56% → 40.54% → 46.33%
valid:   33.33% → 40.00% → 37.78%
```

Mean combined-distance progress (`natural - enhanced`; positive would mean enhanced is closer):

```text
train:
  epoch 1: -0.00016399
  epoch 2: -0.00025550
  epoch 3: -0.00015043

valid:
  epoch 1: -0.00028373
  epoch 2: -0.00004590
  epoch 3: -0.00015572
```

The overall progress is negative in every epoch. Epoch 2 validation is numerically closest to neutral, but the improvement is too small and unstable to support a claim that the enhancer learned a general TTS-directed representation shift.

## Telemetry contract verification

`training_debug.json` records:

```json
{
  "enabled": true,
  "uses_existing_encoder_outputs": true,
  "extra_encoder_calls": 0,
  "layer": 6,
  "mask_source": "independent_MFA_phone_spans",
  "descriptive_only": true
}
```

The telemetry was not added to `LossWeights`, was not used for checkpoint selection, and was not computed on the heldout split during training.

## Interpretation and limitations

The run successfully demonstrates:

- Chinese data can be rebuilt into a strict, speaker-disjoint HuBERT waveform training manifest;
- 24 kHz Faster-Qwen3 TTS audio can be resampled explicitly under the declared 16 kHz HuBERT contract;
- explicit raw-TTS HuBERT supervision and train/valid proximity diagnostics execute end to end;
- diagnostics serialize as finite JSON without additional HuBERT encoder calls.

The run does **not** demonstrate:

- stable convergence toward TTS HuBERT features;
- improved Ditto input representations;
- improved TFG/SyncNet scores;
- heldout generalization;
- that HuBERT layer-6 proximity is sufficient for lip synchronization.

The next decision should be made from the validation trajectory and a preregistered training change, not by tuning against the 47 heldout items. Diagnosis of the failed TTS-directed update is archived in [`28-diagnosis-chinese-hubert-training.md`](28-diagnosis-chinese-hubert-training.md): the style branch has valid gradients, but the weighted acoustic branch is much larger and strongly opposed. The resulting two-stage replacement is archived in [`29-two-stage-feature-targeted-waveform-model.md`](29-two-stage-feature-targeted-waveform-model.md); it uses a corrected direct Stage-1 feature objective and a frozen-target Stage-2 renderer, but remains a separate local train/valid feasibility result. Any future enhanced WAV export or downstream Ditto/SyncNet evaluation must use a fresh output directory and a fixed checkpoint-selection rule.

## Related documents

- Data recovery and provenance: [`26-aishell1-400-training-inventory.md`](26-aishell1-400-training-inventory.md)
- Enhancer architecture and downstream boundary: [`25-single-encoder-waveform-enhancer.md`](25-single-encoder-waveform-enhancer.md)
- Earlier feature-domain MVP: [`22-tts-feature-supervision-and-mvp.md`](22-tts-feature-supervision-and-mvp.md)
- Historical Chinese downstream baseline: `runs/aishell1_strict_20260707T081223Z/05_report/report.md`