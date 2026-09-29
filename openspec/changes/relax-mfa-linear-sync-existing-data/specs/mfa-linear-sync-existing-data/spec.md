## Purpose

This capability provides a bounded existing-data experiment for the TTS-only MFA-linear waveform adapter when the larger 200-record/source-group-heldout inventory is unavailable. It measures record-level heldout behavior without weakening the model-input, SyncNet, waveform-quality, or replacement integrity contracts.

## ADDED Requirements

### Requirement: Existing-data record manifest

The experiment SHALL freeze exactly 94 eligible fit-only training records and exactly 8 different eligible fit-only evaluation records before model training or candidate inference. Evaluation records MUST have distinct sample IDs from training records. Source groups MAY overlap between the two sets, and every report and terminal decision MUST state that this is record-heldout rather than source-group-heldout evaluation.

Selection SHALL use a fixed score-independent SHA256 rule over the full eligible inventory. Eligibility SHALL retain the existing train split, exact mono 16 kHz MFA-linear length of 61,440 samples, 96-frame tracked visual geometry, verified hashes, and unambiguous detached natural-reference coordinate checks. Validation/test records, candidate outcomes, training losses, SyncNet scores, and Wav2Lip results MUST NOT affect selection.

#### Scenario: Existing-data manifest locks
- **WHEN** at least 102 eligible fit-only records are available
- **THEN** the system writes an immutable 94-training/8-evaluation manifest before loading a trainable model and records source-group overlap as descriptive metadata

#### Scenario: Existing inventory is incomplete
- **WHEN** fewer than 102 records pass the unchanged eligibility checks
- **THEN** the system emits `BLOCKED_EXISTING_DATA_INVENTORY`, creates no checkpoint or candidate waveform, and does not fill the denominator with duplicates or sealed-split records

#### Scenario: Outcome-based selection is detected
- **WHEN** a selected row, representative, or exclusion depends on an adapter, SyncNet, Wav2Lip, or audio-quality outcome beyond the fixed preflight contract
- **THEN** validation emits `OUTCOME_SELECTION_VIOLATION` and no scientific result is available

### Requirement: Fresh TTS-only training

The experiment SHALL train from the same fresh initialization for exactly 100 steps using the equal-weight mean frozen SyncNet target-margin loss over all 94 locked training records at each step. The adapter SHALL receive only one floating `[1,1,61440]` MFA-linear TTS waveform. Natural audio, video features, identity, transcript features, target offsets, and any second waveform MUST NOT be passed as model inputs or optimization side channels. SyncNet SHALL remain frozen and outside the optimizer.

Warm starts, retries, additional seeds, checkpoint search, early stopping, threshold tuning, and record-specific weighting are forbidden. The system SHALL persist step-zero and step-100 model state, ordered training IDs, configuration, loss history, and frozen-state hashes.

#### Scenario: Fresh training completes
- **WHEN** the 94-record manifest and fixed training configuration are valid
- **THEN** the system completes exactly 100 steps from fresh initialization and writes hash-bound training artifacts

#### Scenario: Side-channel leakage occurs
- **WHEN** natural audio or any record/video/identity/transcript/offset side channel reaches the model or loss
- **THEN** the run fails closed with `NATURAL_OR_SIDE_CHANNEL_LEAKAGE`

### Requirement: Record-heldout real-video evaluation

The frozen step-100 adapter SHALL run exactly once on each of the 8 evaluation records using only the record's exact MFA-linear waveform. Candidate audio MUST pass the existing exact-length, finite, residual, normalized log-mel, saturation, and source/checkpoint provenance checks.

Each baseline/candidate pair SHALL use the same canonical real-video frames, signed PCM16 representation, 91 windows, all 31 internal shifts, raw Euclidean distance, and official offset sign convention. Metrics SHALL come only from complete official curves. A record success SHALL require both gains at least `0.003`, candidate minimum equal to its detached natural-reference target, target gap greater than `0.002`, and every engineering check passing.

#### Scenario: Paired record-heldout scoring completes
- **WHEN** all 8 records have complete valid baseline/candidate official curves and identical video frame hashes
- **THEN** the system stores all curves, gains, target checks, QC, and provenance without dropping rows based on their values

#### Scenario: A record fails engineering validation
- **WHEN** any waveform, PCM, frame, window, curve, hash, QC, or target-offset predicate fails
- **THEN** the record is engineering-invalid and the aggregate cannot be promoted or treated as a scientific success

### Requirement: Record-heldout transfer decision

The real-video stage SHALL emit exactly one of `RECORD_HELDOUT_REAL_VIDEO_TRANSFER_OBSERVED`, `NO_RECORD_HELDOUT_REAL_VIDEO_TRANSFER`, or `RECORD_HELDOUT_REAL_VIDEO_NOT_EVALUATED`. The positive status requires all 8 rows engineering-valid, at least 6 record successes, and both median gains at least `0.003`. A complete engineering-valid cohort missing those gates SHALL be a scientific negative; incomplete evidence SHALL be not evaluated.

#### Scenario: Record-heldout transfer passes
- **WHEN** all 8 records are valid, at least 6 succeed, and both median gains meet `0.003`
- **THEN** the stage emits `RECORD_HELDOUT_REAL_VIDEO_TRANSFER_OBSERVED` and authorizes strict replacement

#### Scenario: Record-heldout transfer misses
- **WHEN** all 8 records are valid but the six-success or median-gain gate fails
- **THEN** the stage emits `NO_RECORD_HELDOUT_REAL_VIDEO_TRANSFER` and records `REPLACEMENT_NOT_RUN_REAL_VIDEO_GATE_FAILED`

#### Scenario: Record-heldout evidence is incomplete
- **WHEN** manifest, training, inference, scoring, or provenance evidence is incomplete
- **THEN** the stage emits `RECORD_HELDOUT_REAL_VIDEO_NOT_EVALUATED` and makes no scientific claim

### Requirement: Conditional strict replacement

The replacement stage SHALL run only after a passing record-heldout real-video decision. It SHALL use the same 8 records, one natural/MFA/candidate Wav2Lip render per record, and exactly 72 driver/evaluation-audio matrix cells. Only the natural-audio evaluation column SHALL be authoritative; diagonal and cross-audio improvements SHALL remain diagnostics.

A replacement success SHALL require candidate-driven video evaluated with untouched natural audio to improve over the MFA-driven video evaluated with untouched natural audio on both gains by at least `0.003`, match the natural-driven oracle offset with gap greater than `0.002`, and pass all nine-cell checks. Promotion SHALL require all 72 cells, at least 6 successes, and both median gains at least `0.003`.

#### Scenario: Replacement passes
- **WHEN** all 72 cells are valid and the natural-audio column satisfies the six-success and median-gain gates
- **THEN** the stage emits `FROZEN_WAV2LIP_REPLACEMENT_TRANSFER_OBSERVED`

#### Scenario: Diagonal gain does not survive natural replacement
- **WHEN** candidate-native or cross-audio cells improve but the authoritative natural-audio column fails
- **THEN** the stage emits a scientific replacement negative and does not promote it

#### Scenario: Replacement is gated or incomplete
- **WHEN** real-video transfer fails or any matrix cell is missing or invalid
- **THEN** the stage emits the explicit gate-skip or `REPLACEMENT_NOT_EVALUATED` status without reducing the denominator

### Requirement: Immutable artifacts and bounded claims

The experiment SHALL use a new create-once run root and hash-bind the configuration, manifest, source assets, checkpoints, waveforms, videos, curves, matrix cells, decisions, and report. Resume SHALL be read-only for a complete valid terminal graph and MUST NOT repair, append, rerun, overwrite, or replace artifacts.

The report SHALL state that evaluation is record-heldout within the fit-only inventory and that source-group overlap is permitted. It MUST NOT describe a result as unseen-source-group, unseen-speaker, sealed-test, population, perceptual, content, speaker, multi-generator, or production generalization.

#### Scenario: Terminal graph validates
- **WHEN** all required artifacts and hashes are consistent
- **THEN** validation returns `valid` and read-only resume returns the stored decision without external execution

#### Scenario: Terminal graph is tampered
- **WHEN** an artifact, membership list, denominator, curve, matrix cell, or decision is changed or missing
- **THEN** validation fails closed and no scientific promotion occurs
