---
title: 28-diagnosis-chinese-hubert-training
type: note
permalink: tts-exp/docs/experiments/28-diagnosis-chinese-hubert-training
---

# 28 — Diagnosis: why Chinese HuBERT waveform training failed to improve proximity

## Symptom and reproducible feedback loop

The reported failure is not a process crash. The three-epoch Chinese run completed and produced a checkpoint, but enhanced HuBERT layer-6 features did not become stably closer to aligned raw-TTS features.

Reproduction artifact:

```text
runs/hubert_waveform_aishell1_20260810/training_explicit_style_3epoch_retry/
├── checkpoint.pt
├── epoch_metrics.jsonl
└── training_debug.json
```

The logged trajectory reproduces the symptom:

```text
train combined distance:
  natural → TTS:   0.276221 (identity baseline, unchanged)
  enhanced → TTS:  0.276385, 0.276476, 0.276371

valid combined distance:
  natural → TTS:   0.226136 (identity baseline, unchanged)
  enhanced → TTS:  0.226419, 0.226181, 0.226291
```

The mean progress `natural - enhanced` is negative for every epoch on both train and valid. The failure is therefore reproducible from the saved metrics and is not an interpretation based on one sample.

## Ranked hypotheses and probes

### 1. Competing acoustic and style objectives suppress the TTS-directed update — confirmed

Prediction: if acoustic losses oppose the HuBERT style update, their parameter-gradient cosine will be negative; a style-only or high-style-weight short probe will improve proximity while the full objective will not.

On the same train item, same identity initialization, and same frozen HuBERT encoder:

```text
style-only, 8 AdamW steps:
  combined proximity: 0.351773 → 0.346996
  progress: +0.004778

full objective, 8 AdamW steps:
  combined proximity: approximately 0.351773 → 0.351656
  progress: approximately +0.000117 at the best point

weighted acoustic gradient norm: 20.7221
weighted style gradient norm:     1.3168
cosine(acoustic gradient, style gradient): -0.7366
```

The style branch has a valid gradient, but its weighted gradient is about 15.7× smaller than the acoustic branch and points in a strongly opposing direction. The full objective therefore mostly follows the natural/weak-target acoustic reconstruction direction. This is the primary confirmed cause.

A separate scale probe changed only the style multiplier:

```text
style scale 1:   proximity best ≈ 0.351436
style scale 5:   proximity best ≈ 0.351229
style scale 20:  proximity      0.351773 → 0.348953
```

This is causal evidence for objective competition, not proof that scale 20 is a valid final recipe. It must not be selected using heldout data.

### 2. The HuBERT style branch has no gradient — rejected

Prediction: `hubert_tts` or `hubert_delta` would have zero/nonfinite gradient with respect to the enhancer waveform or trainable parameters.

Probe result on the first train item at identity initialization:

```text
hubert_tts parameter-gradient norm: 0.9407
hubert_delta parameter-gradient norm: 0.0870
```

The gradient flows through the frozen HuBERT input to the waveform enhancer. The encoder being frozen does not block input gradients. This hypothesis is false.

### 3. Alignment is entirely unusable — rejected as the primary cause

Prediction: an aligned raw-TTS target would not be optimizable even under style-only updates, or the alignment mask would contain no valid frames.

The first train item has 232 matched frames. The style-only probe reduces its proximity. Across the completed run, the telemetry reports:

```text
train matched frames: 43,048
train mean coverage:  0.64256
valid matched frames:  7,478
valid mean coverage:  0.71497
```

Therefore alignment is imperfect and sparse, but not empty or inherently non-differentiable. It remains a secondary quality limitation, not the primary explanation for the failed update.

### 4. Fresh target construction silently contaminated the run — rejected for the main failure, but found a data-quality issue

Prediction: the 24 kHz TTS conversion or derived silence spans would make the run use invalid sample rates or corrupted targets.

The run explicitly resampled 24 kHz TTS to 16 kHz and all accepted training items passed finite/exact-length checks. The first-item style-only probe uses the same conversion and improves proximity, so the conversion does not prevent optimization.

However, the fresh rebuild accepted 351 pairs rather than the historical 391. It excluded 41 items:

```text
39 alignment does not cover the complete audio duration
 1 matched token coverage 0.590
 1 matched speech coverage 0.898
```

The 39 duration exclusions are caused by a strict tolerance interacting with polyphase resampling: every inspected TTS duration difference is approximately `2.0833e-05 s`, i.e. one resampling rounding step, not a substantive unaligned tail. This is a real data-preparation defect in the fresh run. It reduced training coverage and should be fixed separately with a sample-domain/tolerance-aware validation test. It does not explain the opposing gradients in the 351 accepted items.

### 5. No model update occurred — rejected

Prediction: the optimizer or trainable parameters would remain unchanged, or all residuals would stay exactly zero.

The checkpoint parameter norm changes slightly across the run, train gradient norms are finite, and nonzero residuals appear after the first update. The model updates, but the updates are directed primarily by competing acoustic losses and do not produce a stable TTS proximity shift.

## Root-cause conclusion

The primary failure is **objective conflict and scale imbalance**:

```text
large acoustic reconstruction losses
    + weak natural/content preservation pressure
    + small explicit TTS HuBERT style gradient
    + strongly opposing gradient direction
    → near-identity waveform updates
    → no stable enhanced→TTS proximity improvement
```

The waveform enhancer is not failing because HuBERT gradients are absent. It is failing because the current multi-loss objective asks the same waveform to remain acoustically close to a weak phone-local splice/natural signal while also moving its HuBERT representation toward raw TTS. At the observed scale, the acoustic branch wins.

The weak phone-local waveform target is also not a clean acoustic ground truth. It can carry speaker, phase, timing, formant, prosody, and synthesis artifacts. Matching it with STFT/mel terms makes the acoustic branch an especially strong and potentially contradictory constraint.

## Secondary data issue

The fresh target builder currently rejects TTS items when the resampled duration exceeds the MFA end time by roughly `2.08e-05 s`. This is a numerical boundary artifact from 24 kHz → 16 kHz resampling. The rejected rows should not be recovered by silently padding or uniform alignment. A safe fix is to validate audio-boundary coverage in samples with a documented small tolerance, or to clip only the final derived silence boundary after confirming the difference is a resampling-rounding artifact. This requires a regression test and a fresh manifest hash.

The separate 0.590 token-coverage rejection is substantive and should remain excluded. The 0.898 speech-coverage rejection should remain excluded until its alignment is independently audited.

## What the evidence does not establish

- It does not establish that a larger style weight will improve speaker-disjoint heldout downstream performance.
- It does not establish that HuBERT layer 6 is the right causal target for Ditto.
- It does not establish a Ditto/SyncNet result.
- It does not justify selecting a checkpoint or loss weight using the 47 heldout items.

## Safe next experiment

Before another full run:

1. Add a regression test for 24 kHz resampled-duration tolerance and regenerate a fresh target manifest without recovering substantive low-coverage failures.
2. Run a train-only/validation-only ablation with a preregistered style multiplier, keeping the same data and initialization:
   - acoustic baseline;
   - style multiplier 5 or 10;
   - optionally style-first warmup followed by fixed full objective.
3. Compare proximity, content drift, residual bound, and acoustic losses on train/valid only.
4. Select a checkpoint by the declared validation rule, not by heldout proximity.
5. Only then export WAVs and run a separate fixed downstream Ditto/SyncNet evaluation.

The style-only probe demonstrates that the path is technically optimizable; the next engineering task is to design a non-conflicting objective rather than simply training more epochs with the current weights.

## Follow-up

The follow-up is archived in [`29-two-stage-feature-targeted-waveform-model.md`](29-two-stage-feature-targeted-waveform-model.md). It implements the suggested separation concretely: Stage 1 learns a bounded direct raw-TTS-directed L6 residual without weak waveform reconstruction, and Stage 2 freezes that goal while rendering a bounded waveform residual under realization and preservation constraints. It confirms small replicated local train/valid feature movement, but its calibration still exposes strongly opposed realization/preservation gradients. It therefore does not resolve the preservation question or establish any heldout/downstream result.