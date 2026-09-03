## Context

Parent run:

```text
runs/lrs3_masked_tts_retention_exploratory_20260901/
```

It contains 392 evaluation masks from 24 records/eight source groups, three seeds, three required conditions, per-cell `patch`, `velocity`, and `total` losses, and saved predictions. Its scientific status remains `EXPLORATORY_DESCRIPTIVE_ONLY` because the frozen 900/400 readiness denominator was not met.

Observed group-level token-negative cases are:

```text
6ORDQFh0Byw  token_gain = -0.08060362935066223
6yR5OUVb2gY  token_gain = -0.03605937957763672
```

The follow-up has two ordered phases. Phase B may begin once Phase A is complete and internally consistent; it does not require Phase A to produce a favorable explanation.

## Goals / Non-Goals

**Goals**

- Explain which loss component, phones, records, durations, and seeds drive the two negative groups.
- Test whether the paired-TTS branch changes frozen-Wav2Lip motion in a way that remains more compatible with untouched natural audio than phone-centroid or natural-only predictions.
- Finish quickly by reusing saved predictions and bypassing waveform decoding.

**Non-goals**

- Retrain or tune the reconstructor, Wav2Lip, or SyncNet.
- Claim waveform retention, perceptual TTS identity, audio quality, or population generalization.
- Search records, seeds, clipping rules, thresholds, or TFG checkpoints based on resulting SyncNet scores.

## Phase A — Negative-group and loss analysis

Read the parent `mask_manifest.json`, `evaluation.json`, saved `prediction.npz` cells, natural mel/normalization snapshots, and phone centroids. Do not rerun training.

For every mask and seed compute:

```text
modality_patch_gain    = patch(NAT_ONLY)       - patch(PAIRED_TTS)
modality_velocity_gain = velocity(NAT_ONLY)    - velocity(PAIRED_TTS)
modality_total_gain    = modality_patch_gain + 0.25 * modality_velocity_gain

token_patch_gain       = patch(PHONE_CENTROID) - patch(PAIRED_TTS)
token_velocity_gain    = velocity(PHONE_CENTROID) - velocity(PAIRED_TTS)
token_total_gain       = token_patch_gain + 0.25 * token_velocity_gain
```

Verify the decomposed totals match the recorded total-loss contrasts within `1e-6`. Aggregate in the parent order: mask median → record median → source-group median per seed → median across seeds.

For each group, and especially the two negative groups, report:

- patch versus weighted-velocity contribution to token gain;
- per-seed and per-record token gains;
- lexical-phone counts and per-phone token gains;
- natural core length, TTS source-frame length, and TTS/natural duration ratio;
- paired-TTS trajectory distance from its phone centroid and its within-phone temporal variation;
- paired-versus-centroid prediction RMS on the core.

Compare the two negative groups with the pooled six positive groups descriptively. Name the dominant observed pattern for each negative group (`patch`, `velocity`, `phone_mix`, `duration_alignment`, `seed_instability`, or `diffuse`) but do not turn this post-hoc attribution into a causal claim or a new gate.

Required Phase-A outputs:

```text
00_diagnosis/analysis.json
00_diagnosis/report.md
```

## Phase B — Direct-mel frozen-TFG probe

### Cohort

Use all eight evaluation source groups. Select one record per group by:

1. largest number of evaluation masks;
2. tie-break by `SHA256("masked-tts-tfg-probe-v1\0" + sample_id)`.

Selection uses mask metadata only, never loss, prediction quality, or SyncNet values. The resulting eight-record cohort is a sensitivity probe, not a population sample. Both token-negative groups must remain represented.

### Mel drivers

For each selected record, start with its full standardized natural mel. For each seed and model arm, replace only the global core interval

```text
window_start_frame + [core_start, core_end)
```

with the matching saved prediction core. If frame quantization makes cores overlap, use the arithmetic mean of contributing predictions at the overlapping frame. Leave all uncovered frames exactly equal to natural mel. Denormalize with the parent train-only statistics and clamp only to the valid mel range from the frozen Wav2Lip hparams; report the fraction clamped per driver.

Required drivers are:

```text
NATURAL_MEL                         one per record
PAIRED_TTS × 3 seeds                three per record
PHONE_CENTROID × 3 seeds            three per record
NAT_ONLY × 3 seeds                  three per record
```

This gives exactly 10 drivers and renders per record, 80 total.

### Frozen Wav2Lip and replacement scoring

Use the existing frozen Wav2Lip checkpoint and inference settings. Add a local wrapper that bypasses only audio-to-mel extraction: it shall feed precomputed full-record mel chunks using the same chunk length, frame-index rule, face preprocessing, model call, and video writer as the existing inference path. Do not modify vendor code.

Before the matrix, verify on one cohort record that `NATURAL_MEL` chunks are numerically identical to chunks produced by the normal natural-audio frontend. Then render every required driver against the record's same original face video.

For scoring, copy each rendered video stream and mux the record's untouched natural audio. Score the resulting media with the existing frozen official SyncNet V2 configuration. The driver audio/mel is used to generate motion only; the primary scoring audio is always untouched natural audio.

## Analysis

For each seed and record, positive-is-better contrasts are:

```text
token_C_gain    = C(PAIRED_TTS) - C(PHONE_CENTROID)
token_D_gain    = D(PHONE_CENTROID) - D(PAIRED_TTS)
modality_C_gain = C(PAIRED_TTS) - C(NAT_ONLY)
modality_D_gain = D(NAT_ONLY) - D(PAIRED_TTS)
```

Take the median across three seeds within each source group. Report all group values, seed medians, win counts, medians, and 10,000 whole-source-group bootstrap 95% intervals using `PCG64(20260902)` and NumPy linear quantiles.

Emit one descriptive status:

- `EXPLORATORY_TOKEN_SIGNAL` when both token-gain interval lower bounds are above zero and at least seven of eight groups are positive for both token metrics;
- `EXPLORATORY_MODALITY_ONLY` when the analogous modality rule passes but the token rule does not;
- `NO_EXPLORATORY_TFG_GAIN` when neither rule passes;
- `NOT_EVALUATED` when the required matrix is incomplete.

The natural-mel arm is a reference ceiling/damage diagnostic and is not part of promotion rules.

## Claim Boundary

A positive token result means only that, in this eight-group direct-mel sensitivity probe, paired TTS-conditioned predictions drive frozen Wav2Lip motion that is more compatible with untouched natural audio than phone-centroid-conditioned predictions. It does not establish audible TTS-feature retention or a deployable waveform replacement system.

A modality-only result means TTS-side conditioning helps downstream relative to no TTS input, but the experiment still does not distinguish fine token trajectory from phone-level information. A negative result means the feature-space modality gain did not translate into this TFG endpoint.

## Minimal Outputs

```text
runs/lrs3_masked_tts_tfg_probe_20260902/
├── 00_diagnosis/{analysis.json,report.md}
├── 01_cohort/manifest.json
├── 02_mels/
├── 03_renders/
├── 04_replacement/
├── 05_syncnet/
├── 06_analysis/{analysis.json,report.md}
├── summary.json
└── decision.json
```
