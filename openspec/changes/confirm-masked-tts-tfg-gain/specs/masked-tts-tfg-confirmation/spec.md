## Purpose

Defines a minimal confirmatory experiment that reuses the completed direct-mel frozen-Wav2Lip pipeline to test the exploratory modality gain on the parent run's 16 previously unscored TFG records.

## ADDED Requirements

### Requirement: Complete score-independent confirmation cohort

The experiment SHALL exclude the eight records used by `runs/lrs3_masked_tts_tfg_probe_20260902/` and include every remaining parent evaluation record. It SHALL require exactly 16 records, with two records from each of the eight source groups. No record SHALL be selected or removed using loss, prediction, rendering, or SyncNet outcomes.

#### Scenario: Confirmation cohort is frozen
- **WHEN** the parent and first-probe manifests are valid
- **THEN** the experiment writes a 16-record/eight-group manifest before downstream scoring

#### Scenario: Expected records are unavailable
- **WHEN** exclusion does not leave exactly two valid records in every source group
- **THEN** the experiment stops with status `NOT_EVALUATED`

### Requirement: Reuse the matched direct-mel matrix

For each confirmation record, the experiment SHALL build `PAIRED_TTS`, `PHONE_CENTROID`, and `NAT_ONLY` drivers for all three saved seeds, plus one unchanged `NATURAL_MEL` reference. It SHALL reuse the existing target-core patching, overlap, denormalization, clamp, chunking, and provenance rules. The required matrix SHALL contain exactly 160 cells, with no retraining or checkpoint selection.

#### Scenario: Matrix construction completes
- **WHEN** every saved prediction and natural mel required by the frozen cohort is valid
- **THEN** all 160 drivers are present and uncovered frames remain equal to natural mel

### Requirement: Frozen Wav2Lip and untouched-audio replacement

The experiment SHALL render all drivers with the existing direct-mel wrapper and frozen Wav2Lip checkpoint. It SHALL mux each rendered video stream with the corresponding untouched natural audio and score it using the same frozen official SyncNet V2. Existing checkpoint-hash and PCM-identity checks SHALL pass for every cell.

#### Scenario: A replacement or provenance check fails
- **WHEN** any cell has non-identical replacement PCM, a changed frozen-model hash, or a missing score
- **THEN** the scientific status is `NOT_EVALUATED`

### Requirement: Confirmatory group-level decision

The experiment SHALL treat `PAIRED_TTS > NAT_ONLY` as the primary modality contrast and `PAIRED_TTS > PHONE_CENTROID` as the secondary token contrast for Sync-C and Sync-D. It SHALL aggregate seed median per record, then record median per source group, and make the decision from the eight group values using the thresholds in `design.md`. Earlier probe records SHALL NOT be pooled into this decision.

#### Scenario: Modality gain reproduces without token gain
- **WHEN** both modality bootstrap lower bounds exceed zero and at least 7/8 groups are positive on both metrics, while the token rule fails
- **THEN** status is `CONFIRMED_MODALITY_ONLY` and waveform decoding is not promoted as a fine-grained TTS-retention experiment

#### Scenario: Token and modality gains both reproduce
- **WHEN** both modality and token rules pass
- **THEN** status is `CONFIRMED_TOKEN_SIGNAL` and a small waveform-decoder feasibility experiment may proceed

#### Scenario: Primary modality gain does not reproduce
- **WHEN** the complete matrix fails the modality promotion rule
- **THEN** status is `NO_CONFIRMATORY_TFG_GAIN` and this branch does not advance
