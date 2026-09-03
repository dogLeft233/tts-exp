## Context

The completed probe at:

```text
runs/lrs3_masked_tts_tfg_probe_20260902/
```

used one score-independent record from each of eight source groups and reported:

```text
modality Sync-C: 8/8 positive, median 1.714, 95% CI [1.225, 2.625]
modality Sync-D: 8/8 positive, median 1.489, 95% CI [0.710, 1.867]
token Sync-C:    4/8 positive, median 0.002, 95% CI [-0.059, 0.365]
token Sync-D:    4/8 positive, median 0.022, 95% CI [-0.469, 0.207]
```

The parent evaluation has 24 records, three per source group. The first probe used eight; the remaining 16 have not been through this downstream TFG scoring matrix.

## Goals / Non-Goals

**Goals**

- Test whether the modality gain reproduces on the two remaining records per source group.
- Re-evaluate the token contrast without selecting records by loss or score.
- Make a clear decision about whether a waveform-decoder experiment is justified.

**Non-goals**

- Retrain or tune any model.
- Search cohorts, thresholds, seeds, clipping rules, or checkpoints.
- Claim audible TTS retention, population generalization, or deployability.

## Frozen cohort and matrix

Read the parent 24-record manifest and the first probe's eight-record manifest. Exclude those exact eight records and require exactly two remaining records in each of the eight source groups. Include all 16; do not rank or filter them by reconstruction loss, prediction quality, or any TFG score.

For each record, reuse the saved mel construction rules and create:

```text
NATURAL_MEL                         1
PAIRED_TTS × 3 seeds                3
PHONE_CENTROID × 3 seeds            3
NAT_ONLY × 3 seeds                  3
```

This gives 10 drivers per record and 160 required cells. Any missing cell makes the decision `NOT_EVALUATED`.

## Frozen rendering and scoring

Reuse the existing direct-mel wrapper without modifying vendor Wav2Lip. Preserve natural mel outside evaluated cores and use the existing overlap, denormalization, clamp, chunking, and provenance rules.

Render against each record's original face video with the same frozen Wav2Lip checkpoint. Strictly replace every render's audio with that record's untouched natural audio and score with the same frozen official SyncNet V2. The replacement PCM and frozen model hashes must pass the existing checks.

Do not inspect aggregate SyncNet contrasts until the 160-cell matrix is complete.

## Analysis and decision

For each seed and record, compute the existing positive-is-better contrasts:

```text
modality_C_gain = C(PAIRED_TTS) - C(NAT_ONLY)
modality_D_gain = D(NAT_ONLY) - D(PAIRED_TTS)

token_C_gain    = C(PAIRED_TTS) - C(PHONE_CENTROID)
token_D_gain    = D(PHONE_CENTROID) - D(PAIRED_TTS)
```

Aggregate without treating seeds as independent samples:

```text
median across 3 seeds per record
→ median across 2 records per source group
→ 8 source-group values
```

Report all record and group values, positive-group counts, medians, and 10,000 whole-source-group bootstrap 95% intervals using `PCG64(20260902)` and NumPy linear quantiles.

Assign exactly one status from the confirmatory records alone:

- `CONFIRMED_TOKEN_SIGNAL`: both token interval lower bounds exceed zero and at least 7/8 groups are positive for both token metrics; the modality rule must also pass.
- `CONFIRMED_MODALITY_ONLY`: the analogous modality rule passes, but the token rule does not.
- `NO_CONFIRMATORY_TFG_GAIN`: the modality rule does not pass.
- `NOT_EVALUATED`: the frozen cohort, matrix, replacement checks, or scores are incomplete.

`NATURAL_MEL` remains a reference diagnostic and is not part of promotion. The earlier eight-record results may be shown beside the confirmation results, but they must not be pooled into the confirmatory decision.

## Next-stage gate

- `CONFIRMED_TOKEN_SIGNAL`: a small waveform-decoder feasibility experiment is justified.
- `CONFIRMED_MODALITY_ONLY`: retain the direct-mel TFG finding, but do not begin waveform decoding as evidence of fine-grained TTS retention.
- `NO_CONFIRMATORY_TFG_GAIN` or `NOT_EVALUATED`: do not advance this branch.

## Minimal outputs

```text
runs/lrs3_masked_tts_tfg_confirmation_20260902/
├── 00_cohort/manifest.json
├── 01_mels/
├── 02_renders/
├── 04_replacement/
├── 05_syncnet/
├── 05_analysis/{analysis.json,report.md}
├── summary.json
└── decision.json
```
