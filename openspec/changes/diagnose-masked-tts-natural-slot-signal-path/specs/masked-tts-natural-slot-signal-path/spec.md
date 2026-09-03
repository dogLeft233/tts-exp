## Purpose

Defines a bounded diagnostic that corrects mask-level reconstruction pairing, traces where occurrence-specific conditioning is attenuated, tests natural-audio WavLM features in the existing TTS slot, and conditionally evaluates a better-matched wrong-instance control before further training or waveform work.

## ADDED Requirements

### Requirement: Reconstruction controls shall be paired by mask

The experiment SHALL bind the complete independent new-record run as immutable input and SHALL pair every reconstruction control with `PAIRED_TTS` from the same sample, seed, and mask. It SHALL use the stored same-mask `reconstruction_gain` or an equivalent `(sample_id, seed, mask_sha256)` key. It SHALL NOT pair reconstruction rows only by sample and seed or overwrite parent artifacts.

#### Scenario: Corrected reconstruction remains reproducible
- **WHEN** the bound 16-record, eight-group, 222-mask reconstruction matrix is reaggregated
- **THEN** both corrected control summaries match the fixed values in `design.md` within `1e-5` and retain 8/8 positive groups

#### Scenario: Multiple masks share a sample and seed
- **WHEN** at least two masks have different paired losses for one sample and seed
- **THEN** each control is compared with its own mask's paired loss and a regression test detects the former overwrite behavior

### Requirement: Existing signal survival shall be audited without recomputation

The experiment SHALL use existing artifacts to quantify paired-versus-wrong-instance and paired-versus-reversed differences at the aligned feature core, predicted mel core, patched driver frames, rendered mouth region, and SyncNet endpoint. It SHALL use shared masks, seeds, face boxes, and cell identities, and SHALL report wrong-instance magnitude relative to reversed magnitude at every defined stage. This audit SHALL remain descriptive and SHALL NOT introduce a post-hoc pass threshold.

#### Scenario: All bound stages are compatible
- **WHEN** feature provenance, prediction hashes, driver hashes, video dimensions, frame counts, face boxes, and score identities match
- **THEN** Part A writes per-record, per-group, and aggregate signal-path tables without rerunning model inference or external evaluators

#### Scenario: A bound stage cannot be paired
- **WHEN** any required identity, hash, shape, frame, or box check fails
- **THEN** Part A reports the affected stage as `NOT_EVALUATED` instead of silently omitting cells

### Requirement: Natural reference WavLM shall be tested in the existing TTS slot

The experiment SHALL construct `NATURAL_WAVLM_IN_TTS_SLOT` by encoding each bound untouched natural audio with the locally bound `bshall/knn-vc` revision, adapter source hash, `WavLM-Large.pt` hash, layer-6 interface, 16 kHz sample rate, 320-sample stride, and frame rule specified in `design.md`. It SHALL select the target natural phone from the frozen natural alignment and align it through the existing phone interpolation path. It SHALL preserve the paired target, mask, natural-mel context, core location, and destination length, and SHALL keep features outside the core exactly zero. It SHALL NOT use a remote fallback or implicit model download.

#### Scenario: Natural-slot input is valid
- **WHEN** every mask has a finite 1024-D natural WavLM span that aligns to the paired input shape and has a distinct hash
- **THEN** the experiment freezes a complete 222-mask natural-slot manifest before prediction

#### Scenario: Natural-slot provenance is incomplete
- **WHEN** an audio, alignment, feature, or aligned-core hash is missing or any mask cannot be constructed
- **THEN** Part B reports `NOT_EVALUATED` without dropping that mask or record

### Requirement: Natural-slot evaluation shall reuse frozen endpoints

The experiment SHALL evaluate all natural-slot masks with the three existing hard-negative checkpoints, build 48 record drivers, render with the bound frozen direct-mel Wav2Lip path, strictly replace with corresponding untouched natural audio, and score with the bound frozen SyncNet. Existing `PAIRED_TTS` and `NAT_ONLY` cells SHALL be reused, not rerendered.

#### Scenario: Natural reference adds frozen-TFG value over zero input
- **WHEN** natural-slot Sync-C and Sync-D satisfy the whole-group rule against `NAT_ONLY`
- **THEN** Part B reports `NATURAL_SLOT_OVER_ZERO_TFG_PASS` and independently reports whether reconstruction supports it

#### Scenario: Natural reference exceeds paired TTS at the frozen endpoint
- **WHEN** natural-slot Sync-C and Sync-D satisfy the whole-group rule against `PAIRED_TTS`
- **THEN** Part B reports `NATURAL_SLOT_OVER_TTS_TFG_PASS` and independently reports whether reconstruction supports it

#### Scenario: Natural reference helps reconstruction only
- **WHEN** a natural-slot reconstruction contrast passes but either corresponding downstream metric does not
- **THEN** Part B reports reconstruction support with `TFG_NOT_SHOWN` and does not claim a downstream advantage

#### Scenario: Natural reference helps the frozen endpoint only
- **WHEN** both downstream metrics pass but the corresponding reconstruction contrast does not
- **THEN** Part B reports the TFG pass qualified as `DOWNSTREAM_ONLY` rather than discarding it

### Requirement: Matched wrong-instance donors shall be selected before inference

The experiment SHALL preflight same-phone donor occurrences from the bound new cohort, preferring a different occurrence in the same record and then the other record in the same source group. Distinctness SHALL be defined by `(sample_id, tts_phone_index)`, not merely by mask hash. It SHALL rank candidates only by source tier, neighboring-phone mismatch count, duration difference, sample ID, canonical index, and mask hash. It SHALL freeze the first three distinct donor occurrences and one common eligible-mask subset without consulting model or endpoint scores.

#### Scenario: Matched coverage is sufficient
- **WHEN** every record retains at least two masks with three valid donors and all eight source groups remain represented
- **THEN** Part C freezes the subset and proceeds with three donor ranks

#### Scenario: Matched coverage is insufficient
- **WHEN** any record has fewer than two eligible masks or a source group is absent
- **THEN** Part C reports `INSUFFICIENT_MATCHED_DONOR_COVERAGE` and performs no matched-control inference or rendering

### Requirement: Matched controls shall use a common-mask paired baseline

When matched coverage is sufficient, the experiment SHALL evaluate all three frozen donor ranks for reconstruction robustness and SHALL rebuild `PAIRED_COMMON_MASKS` from exactly the same eligible-mask subset. It SHALL render only the predeclared first-ranked donor and the common paired baseline, for at most 96 new cells. It SHALL NOT compare a subset wrong-instance driver with the prior all-mask paired driver or replace rank one after observing scores. Reconstruction SHALL aggregate donor rank before mask; downstream evaluation SHALL use rank one and aggregate seed before record as specified in `design.md`.

#### Scenario: Cleaner occurrence identity reaches the endpoint
- **WHEN** paired conditioning beats the rank-one matched wrong instance under both Sync-C and Sync-D whole-group rules
- **THEN** Part C reports `MATCHED_INSTANCE_TFG_SIGNAL` and independently qualifies it as reconstruction-supported or downstream-only

#### Scenario: Cleaner identity is visible only in reconstruction
- **WHEN** the three-donor reconstruction rule passes but either downstream metric fails
- **THEN** Part C reports `MATCHED_INSTANCE_RECONSTRUCTION_ONLY`

#### Scenario: Cleaner identity reaches only the frozen endpoint
- **WHEN** both downstream metrics pass but the three-donor reconstruction rule does not
- **THEN** Part C reports `MATCHED_INSTANCE_TFG_SIGNAL` qualified as `DOWNSTREAM_ONLY`

#### Scenario: Cleaner identity is not detected
- **WHEN** the complete matched matrix is valid and neither reconstruction nor the complete downstream rule passes
- **THEN** Part C reports `NO_MATCHED_INSTANCE_SIGNAL`

### Requirement: Decisions and claims shall remain bounded

The experiment SHALL apply the decision precedence in `design.md`, explicitly treating `INSUFFICIENT_MATCHED_DONOR_COVERAGE` as a valid no-signal skip, write exactly one recommendation, and leave the waveform-decoder gate closed. It SHALL describe natural-slot input as an oracle/reference condition and all matched-control findings as exploratory on inspected records. It SHALL NOT claim audible retention, text-only TTS recovery, equivalence from a failed test, deployability, or population generalization.

#### Scenario: Matched trajectory evidence survives downstream
- **WHEN** Part C reports `MATCHED_INSTANCE_TFG_SIGNAL`
- **THEN** the next permitted step is a preregistered confirmation on fresh records

#### Scenario: Natural reference is the stronger usable input
- **WHEN** Part C has no downstream signal or validly skips for coverage and Part B reports both `NATURAL_SLOT_OVER_ZERO_TFG_PASS` and `NATURAL_SLOT_OVER_TTS_TFG_PASS`
- **THEN** the recommendation is `PIVOT_TO_NATURAL_REFERENCE_CONDITIONING` regardless of the independent reconstruction qualifier

#### Scenario: No stronger occurrence-specific path is demonstrated
- **WHEN** neither matched-control nor natural-slot rules justify promotion
- **THEN** the recommendation remains modality-only or requests conditioning-objective redesign according to `design.md`
