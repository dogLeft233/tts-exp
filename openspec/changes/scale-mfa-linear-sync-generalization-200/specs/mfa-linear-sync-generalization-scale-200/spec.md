## Purpose

This capability measures whether increasing the shared TTS-only adapter fit set to exactly 200 records produces transfer to source groups that were not used for optimization, while preserving the frozen audio, SyncNet, provenance, and downstream replacement contracts.

## ADDED Requirements

### Requirement: Expanded score-independent data lock

The experiment SHALL construct and freeze an ordered manifest containing exactly 200 eligible fit records for optimization and exactly 40 eligible evaluation records from 40 source groups. Every evaluation source group MUST be disjoint from every source group represented by the 200-record optimization set. Both sets MUST come from the permitted fit-only universe and MUST be selected before model training, candidate inference, or any downstream score is available.

Eligibility SHALL require verified source metadata, train split membership, unique sample identity, exact mono 16 kHz MFA-linear audio of 61,440 samples, exact 96-frame tracked visual support, valid hashes, and a unique detached natural-reference 31-offset coordinate with a best/second gap greater than `0.002`. Candidate outputs, adapter losses, SyncNet results, and Wav2Lip results MUST NOT affect membership or ordering.

#### Scenario: Full expanded data lock
- **WHEN** the expanded fit-only inventory contains at least 200 optimization records and 40 disjoint eligible evaluation source groups
- **THEN** the system writes one immutable manifest with both ordered sets, all exclusions, source groups, hashes, selection keys, and exact denominators before training begins

#### Scenario: Expanded inventory is too small
- **WHEN** fewer than 200 optimization records or fewer than 40 disjoint evaluation source groups pass eligibility
- **THEN** the system emits `BLOCKED_DATASET_SCALE`, creates no model checkpoint or candidate waveform, and does not reduce, duplicate, or substitute either denominator

#### Scenario: Outcome-based selection or group overlap
- **WHEN** a row is selected using an experimental score or an evaluation source group appears in both sets
- **THEN** validation emits `OUTCOME_SELECTION_VIOLATION` and no scientific result is available

### Requirement: Fresh 200-record TTS-only training

The experiment SHALL train a new shared waveform adapter from the fixed fresh initialization for exactly 100 optimization steps using the equal-weight mean of the frozen SyncNet target-margin loss over all 200 optimization records at each step. The model input SHALL be only one exact-length MFA-linear TTS waveform. Natural audio, video features, identity, transcript features, target offsets, and any second waveform MUST NOT enter the training graph. SyncNet parameters and buffers MUST remain frozen and outside the optimizer.

The system MUST persist and validate the fresh step-zero state, the step-100 state, the exact record order, the loss configuration, the optimizer configuration, and before/after SyncNet hashes. Warm starts, retries, additional seeds, checkpoint search, early stopping, threshold tuning, and record-specific weighting are forbidden.

#### Scenario: Valid expanded training
- **WHEN** the locked 200-record manifest and frozen loss configuration are valid
- **THEN** the system completes exactly 100 steps from fresh initialization and writes hash-bound step-zero and step-100 checkpoints with a reproducible training history

#### Scenario: Natural audio or side-channel enters training
- **WHEN** a training forward call receives natural audio, video, identity, transcript, offset, or a second audio tensor
- **THEN** the run fails closed with `NATURAL_OR_SIDE_CHANNEL_LEAKAGE` and cannot produce a passing prerequisite

#### Scenario: Training state or SyncNet mutates unexpectedly
- **WHEN** the fresh model contract, optimizer contract, or frozen SyncNet hash differs from the declared protocol
- **THEN** the run emits an engineering failure and no transfer decision is available

### Requirement: Source-group-heldout real-video evaluation

After training, the step-100 adapter SHALL be frozen and applied exactly once to each of the 40 evaluation records using only its `[1,1,61440]` MFA-linear waveform. Each candidate SHALL be exact length, finite, and pass the unchanged residual peak `<=0.05`, normalized log-mel distance `<=0.10`, and PCM saturation fraction `<=1e-4` checks.

For every evaluation record, the system SHALL compare MFA-linear baseline and candidate audio on the same canonical 96-frame real-video segment using complete official SyncNet V2 curves with 91 windows, all 31 internal shifts from `-15` through `+15`, raw Euclidean distance, and official offset equal to the negative internal shift. All scientific metrics MUST be recomputed from the official curves.

A record SHALL be successful only when its synchronization-distance gain and synchronization-curve-separation gain are each at least `0.003`, its candidate minimum equals the detached natural-reference target, its target minimum is unique with gap greater than `0.002`, and all engineering/provenance checks pass.

#### Scenario: Complete paired evaluation
- **WHEN** all 40 evaluation records have valid baseline and candidate audio, identical decoded visual frame hashes, complete official curves, and valid provenance
- **THEN** the system persists every full curve and recomputable record-level gain without dropping any row because of its values

#### Scenario: Candidate violates an audio or coordinate contract
- **WHEN** a candidate fails length, finite-value, waveform QC, PCM, frame, window, curve, sign, or target-offset validation
- **THEN** that row is engineering-invalid and the aggregate cannot be promoted or treated as a scientific success

### Requirement: Expanded transfer decision

The real-video stage SHALL emit exactly one of `ADAPTER_HELDOUT_REAL_VIDEO_TRANSFER_OBSERVED`, `NO_ADAPTER_HELDOUT_REAL_VIDEO_TRANSFER`, or `REAL_VIDEO_TRANSFER_NOT_EVALUATED`. The positive status requires all 40 rows to be engineering-valid, at least 30 record-level successes, and median synchronization-distance gain and median synchronization-curve-separation gain each at least `0.003`. A complete engineering-valid cohort that misses these scientific gates SHALL be a scientific negative, not an engineering failure. Any incomplete, invalid, or denominator-reduced cohort SHALL be not evaluated.

#### Scenario: Expanded transfer passes
- **WHEN** all 40 rows are valid, at least 30 satisfy every record gate, and both median gains are at least `0.003`
- **THEN** the stage emits `ADAPTER_HELDOUT_REAL_VIDEO_TRANSFER_OBSERVED` and authorizes the replacement stage

#### Scenario: Complete expanded transfer misses
- **WHEN** all 40 rows are valid but fewer than 30 records succeed or either median gain is below `0.003`
- **THEN** the stage emits `NO_ADAPTER_HELDOUT_REAL_VIDEO_TRANSFER` and records `REPLACEMENT_NOT_RUN_REAL_VIDEO_GATE_FAILED`

#### Scenario: Expanded transfer evidence is incomplete
- **WHEN** the data lock, training, candidate materialization, scoring, or artifact graph is incomplete or invalid
- **THEN** the stage emits `REAL_VIDEO_TRANSFER_NOT_EVALUATED` and makes no scientific claim

### Requirement: Gated 360-cell strict replacement

The replacement stage SHALL run only after `ADAPTER_HELDOUT_REAL_VIDEO_TRANSFER_OBSERVED`. For each of the same 40 records it SHALL render exactly one natural-driven, one MFA-linear-driven, and one candidate-driven video under the same frozen Wav2Lip protocol, then materialize exactly nine driver/evaluation-audio combinations. The complete matrix SHALL contain exactly 360 valid cells, and no cell may be removed because of its score.

Only the natural-audio evaluation column SHALL be authoritative. A strict replacement record success SHALL require the candidate-driven video evaluated with untouched natural audio to improve over the MFA-linear-driven video evaluated with untouched natural audio on both gains by at least `0.003`, match the unique natural-driven oracle offset with gap greater than `0.002`, and pass every cell and provenance check. Replacement promotion SHALL require all 360 cells, at least 30 record successes, and both median natural-column gains at least `0.003`.

#### Scenario: Complete strict replacement
- **WHEN** all 360 cells are valid and the natural-audio column satisfies the 30-of-40 and median-gain gates
- **THEN** the stage emits `FROZEN_WAV2LIP_REPLACEMENT_TRANSFER_OBSERVED`

#### Scenario: Diagonal or cross-audio improvement only
- **WHEN** candidate-native or other diagnostic cells improve but the candidate-driven natural-audio column does not pass
- **THEN** the stage emits a scientific negative and does not promote replacement transfer

#### Scenario: Replacement is skipped or incomplete
- **WHEN** the real-video gate fails, or any render, mux, score, or provenance cell is missing or invalid
- **THEN** the stage emits the corresponding explicit skip or `REPLACEMENT_NOT_EVALUATED` status and never reduces the denominator

### Requirement: Immutable validation and bounded interpretation

The experiment SHALL write every stage to a new create-once run root and bind configuration, manifests, source assets, checkpoints, executable/environment, waveforms, videos, matrix cells, curves, decisions, and reports by canonical hashes. Resume SHALL be read-only for a complete hash-valid terminal run and SHALL never repair, append, rerun, overwrite, replace, or change a decision.

Reports SHALL distinguish data-scale engineering status, training prerequisite status, heldout real-video status, and replacement status. Any positive result SHALL be described only as an observation on the preregistered source-group-heldout cohort and, when applicable, this one frozen Wav2Lip/SyncNet stack. It MUST NOT be described as sealed-test or population generalization, perceptual quality, intelligibility, content preservation, speaker preservation, multi-generator safety, or production readiness.

#### Scenario: Valid immutable terminal graph
- **WHEN** all artifacts and hashes validate and the terminal decision is internally consistent
- **THEN** validation returns `valid`, and resume returns the stored decision without executing an external command

#### Scenario: Tampered or partial terminal graph
- **WHEN** an artifact, hash, membership list, denominator, curve, cell, or decision is changed or missing
- **THEN** validation fails closed and resume performs no repair or scientific promotion
