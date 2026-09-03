## Context

This is a direct follow-up to `prototype-lrs3-masked-tts-reconstruction`. All original tensor, alignment, leakage, model, and loss contracts remain frozen unless explicitly changed below. The prior result showed incremental predictive utility of paired TTS features, not retained TTS clarity. The new question is narrower than waveform generation: does the complete paired token trajectory add information beyond a train-only representation of phone identity?

## Goals / Non-Goals

**Goals**

- Replicate the paired-TTS versus natural-only gain with more source-group diversity.
- Separate token-level paired TTS information from a train-only phone-centroid cue.
- Produce a clear go/no-go gate before spending effort on waveform decoding and TFG evaluation.

**Non-goals**

- Prove that a waveform contains perceptible TTS traits.
- Train or select a vocoder, TFG, SyncNet, ASR, attention aligner, or larger reconstruction model.
- Search cohort size, model width, training steps, loss weights, seeds, or thresholds after results are observed.

## Frozen Design

### 1. Expanded fit-only cohort

Start from metadata whose parent `protocol_split` is exactly `train`. Never open media/features/alignment files for another parent split. Reuse the six prior train groups; keep the four prior evaluation groups entirely out of this run. Discover additional fit-only groups without reading any model output, ASR/SyncNet/TFG score, or prior downstream result.

A candidate group is input-eligible when deterministic application of the existing phone/mask contract yields at least two eligible records and 20 masks. Order fresh eligible groups by `SHA256("masked-tts-retention-scaleup-v1\0" + source_group)` and freeze:

```text
train:      six prior train groups + first six fresh groups
            up to four records/group

evaluation: next eight fresh groups
            up to three records/group
```

Within each group, choose records by the same salted SHA-256 rule over `sample_id`. Freeze all groups, records, exclusions, paths, hashes, and masks before schedule generation. No failed group or record is replaced after the lock is written.

Readiness requires exactly 12 train groups and eight evaluation groups, at least 36 train records/900 train masks, at least 16 evaluation records/400 evaluation masks, and at least 30 masks in every evaluation group after the centroid-support filter. Otherwise engineering may be `GO`, science is `INSUFFICIENT`, and training does not start.

### 2. Train-only phone centroid

Retain only lexical phone labels represented by at least 20 train mask instances across at least three train source groups. Apply this support rule once to the shared primary manifest; unsupported labels are excluded from every arm and condition before training.

For each supported normalized lexical phone `p`, compute from train groups only:

```text
C_p = mean of every standardized aligned TTS-L6 core frame
      belonging to train masks labeled p
```

`PHONE_CENTROID` repeats `C_p` over the natural target core and is exactly zero outside the core. The centroid table records phone label, contributing groups/records/masks/frames, mean hash, and train-manifest hash. No evaluation frame contributes to it. This control intentionally preserves phone identity and removes the paired token trajectory; it is not claimed to preserve average sub-phone phase.

### 3. Three matched training arms

Keep unchanged:

```text
MaskedNaturalReconstructor: 482,256 trainable parameters
window/core/guard tensor contract
phone_phase_linear_v1
train-only normalization
L_total = L_patch + 0.25 * L_velocity
CPU deterministic algorithms
AdamW, lr=1e-3, weight_decay=1e-4
batch=16, gradient_clip=1.0
seeds=(20260901, 20260902, 20260903)
```

For each seed, instantiate once and clone the byte-identical initial state into:

```text
PAIRED_TTS:     paired aligned token-level TTS-L6
PHONE_CENTROID: train-only C_p repeated over the core
NAT_ONLY:       all-zero TTS tensor
```

Precompute one shared `1,200 × 16` group→record→mask schedule per seed using the existing `PCG64(seed)` sampler. All three arms consume it byte-for-byte in the same order. Evaluate only step 1,200. No validation checkpoint selection, retry, early stopping, model change, or hyperparameter search is allowed.

### 4. Held-out conditions

For every evaluation mask and seed, require:

```text
PAIRED_TTS:     paired checkpoint + correct paired TTS
PHONE_CENTROID: centroid checkpoint + train-only phone centroid
NAT_ONLY:       natural-only checkpoint + zero TTS
```

Using the unchanged paired checkpoint, optionally evaluate:

```text
SAME_PHONE_DONOR:  different train record, same lexical phone
WRONG_PHONE_DONOR: train donor with a different lexical phone
```

Donors come only from frozen train groups, use the existing duration-ratio `[0.5,2.0]` gate, and are selected by duration mismatch then salted SHA-256 rank. Missing donors are recorded and do not remove a primary mask. These same-checkpoint interventions are diagnostic-only because they are not matched training arms.

### 5. Analysis and decisions

Aggregate each loss and contrast in the existing order:

```text
mask median → record median → source-group median per seed
→ median over three seeds within source group
```

The eight final evaluation source groups are the only inferential units. Use 10,000 whole-group `PCG64(20260901)` percentile bootstrap draws with NumPy `method="linear"`.

Primary positive-is-better contrasts are:

```text
modality_gain = L_total(NAT_ONLY) - L_total(PAIRED_TTS)
token_gain    = L_total(PHONE_CENTROID) - L_total(PAIRED_TTS)
```

`MODALITY_REPLICATED` requires:

1. `modality_gain` bootstrap 95% lower bound > 0;
2. at least seven of eight final groups have positive `modality_gain`;
3. every seed has positive overall median `modality_gain`;
4. median relative reduction versus `NAT_ONLY` is at least 5%.

`TOKEN_LEVEL_TTS_SIGNAL_SUPPORTED` requires `MODALITY_REPLICATED` and:

1. `token_gain` bootstrap 95% lower bound > 0;
2. at least seven of eight final groups have positive `token_gain`;
3. every seed has positive overall median `token_gain`;
4. median relative reduction versus `PHONE_CENTROID` is at least 2%.

Final science status is:

```text
TOKEN_LEVEL_TTS_SIGNAL_SUPPORTED  both gate sets pass
PHONE_LEVEL_ONLY_SUPPORTED        modality gates pass; token gates fail
NO_SCALEUP_SUPPORT                modality gates fail in a complete run
INSUFFICIENT                      readiness denominator fails
NOT_EVALUATED                     engineering fails
```

Donor diagnostics report loss gaps, prediction RMS differences, and coverage, but cannot promote, block, or rescue a status.

## Claim Boundary

`TOKEN_LEVEL_TTS_SIGNAL_SUPPORTED` means only that paired token-level TTS-L6 improves held-out masked natural-mel prediction beyond a train-only phone-centroid cue under this task. It does not prove that TTS information is perceptible in waveform, that natural prosody is fully preserved, or that any TFG/SyncNet/replacement endpoint improves. It authorizes a separate waveform-bridge experiment with `original natural`, `natural roundtrip`, `NAT_ONLY`, `PHONE_CENTROID`, and `PAIRED_TTS` controls.

## Minimal Artifact Changes

Extend the existing package and write a new immutable run:

```text
runs/lrs3_masked_tts_retention_scaleup_20260901/
├── 00_lock/
├── 01_masks/
├── 02_features/phone_centroids.npz
├── 03_train/<seed>/{paired_tts,phone_centroid,nat_only}/
├── 04_eval/
├── 05_analysis/
├── validation.json
├── summary.json
└── decision.json
```

Do not alter the completed v2 run, vendor code, datasets, checkpoints, or model caches.
