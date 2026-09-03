## Purpose

Defines a minimal three-part experiment that distinguishes engineering value relative to unchanged natural mel, exact paired-trajectory sensitivity, and the effect of one fixed hard-negative training objective before any waveform work.

## ADDED Requirements

### Requirement: Existing natural-mel reference shall be audited without rerendering

The experiment SHALL read the complete 16-record confirmation matrix and compare `PAIRED_TTS` and `PHONE_CENTROID` separately with the existing unchanged `NATURAL_MEL` rows. It SHALL also report `PHONE_CENTROID` over `NAT_ONLY` and apply the practical-pass rule in `design.md`. It SHALL reuse the confirmation's seed-to-record-to-source-group aggregation and whole-group bootstrap. It SHALL NOT rerender cells, modify scores, or change the prior `CONFIRMED_MODALITY_ONLY` decision.

#### Scenario: Paired conditioning clears the engineering reference
- **WHEN** both paired-versus-natural confidence-interval lower bounds exceed zero and at least 7/8 source groups are positive for Sync-C and Sync-D
- **THEN** Part A reports `PAIRED_BEATS_NATURAL_REFERENCE`

#### Scenario: Paired conditioning does not clear the reference rule
- **WHEN** the complete audit fails any paired-versus-natural threshold
- **THEN** Part A reports `PAIRED_NOT_SHOWN_TO_BEAT_NATURAL_REFERENCE` without claiming that natural mel is superior

### Requirement: Two matched trajectory controls shall be deterministic

For every frozen evaluation mask, the experiment SHALL construct `SAME_PHONE_WRONG_INSTANCE` from a different training sample and source group with the same normalized phone label, using the identity-only donor rule in `design.md`. It SHALL also construct `WITHIN_PHONE_REVERSED` by reversing the paired aligned frames only inside the target core. Both controls SHALL preserve the target natural context, masks, core location, and destination length. No donor or transform SHALL be chosen using loss or downstream scores.

#### Scenario: Controls are valid
- **WHEN** every target has an eligible donor and both control feature hashes differ from the paired feature hash
- **THEN** the experiment freezes one complete control manifest before prediction

#### Scenario: A control cannot isolate the intended variable
- **WHEN** a donor is unavailable, a control equals the paired feature, or provenance is incomplete
- **THEN** Part B reports `NOT_EVALUATED` and does not drop the affected mask or record

### Requirement: Frozen reconstruction and downstream sensitivity shall be evaluated separately

The experiment SHALL reuse the three original reconstructor checkpoints and existing paired predictions. It SHALL predict both new controls for every evaluation mask, compute the existing patch, velocity, and total losses, and render exactly 96 new downstream cells. Every render SHALL use the frozen direct-mel Wav2Lip path, untouched-natural-audio replacement, and frozen official SyncNet V2 checks.

#### Scenario: Exact trajectory reaches the downstream endpoint
- **WHEN** paired conditioning beats both controls under the reconstruction, Sync-C, and Sync-D group rules in `design.md`
- **THEN** Part B reports `TFG_TRAJECTORY_SIGNAL`

#### Scenario: Exact trajectory is visible only in reconstruction
- **WHEN** paired conditioning beats both controls in reconstruction but not under the complete downstream rule
- **THEN** Part B reports `RECONSTRUCTION_TRAJECTORY_SIGNAL_ONLY`

#### Scenario: Reconstruction does not identify the exact trajectory
- **WHEN** all required cells are valid but paired conditioning fails at least one hard-control reconstruction rule
- **THEN** Part B reports `NO_TRAJECTORY_SIGNAL`

### Requirement: Hard-negative training shall be single-arm and conditional

The experiment SHALL run hard-negative training only after a complete Part B result of `NO_TRAJECTORY_SIGNAL`. It SHALL keep the model architecture, data split, masks, three seeds, initial states, sampler schedules, optimizer settings, and 600-step budget unchanged. Its only objective change SHALL be the fixed same-phone ranking term specified in `design.md`; it SHALL NOT tune the ranking weight or margin or select checkpoints by evaluation outcomes.

#### Scenario: Frozen model is already trajectory-sensitive
- **WHEN** Part B reports `TFG_TRAJECTORY_SIGNAL`
- **THEN** Part C reports `SKIPPED_ALREADY_SENSITIVE`

#### Scenario: Reconstruction is sensitive but the downstream endpoint is not
- **WHEN** Part B reports `RECONSTRUCTION_TRAJECTORY_SIGNAL_ONLY`
- **THEN** Part C reports `SKIPPED_ENDPOINT_LIMITED`

#### Scenario: Training is eligible
- **WHEN** Part B reports `NO_TRAJECTORY_SIGNAL`
- **THEN** Part C trains exactly one hard-negative model per original seed and evaluates reconstruction before rendering

#### Scenario: Reconstruction gate fails
- **WHEN** the trained model fails either hard-control reconstruction rule or exceeds the paired-loss guard in `design.md`
- **THEN** Part C reports `TRAINING_NO_GO` and performs no downstream render

#### Scenario: Training reaches the downstream endpoint
- **WHEN** the reconstruction gate passes and paired conditioning beats both hard controls while retaining the paired-over-zero-TTS modality rule
- **THEN** Part C reports `TRAINING_TFG_GO`

### Requirement: Claims and next steps shall remain bounded

The experiment SHALL report that its records were previously inspected and its results are exploratory. It SHALL select exactly one final recommendation by applying the precedence order in `design.md`. It SHALL NOT claim audible TTS retention, waveform reachability, natural-prosody preservation, population generalization, or deployability, and it SHALL NOT open the waveform-decoder gate.

#### Scenario: A trajectory-sensitive mel result is found
- **WHEN** Part B reports `TFG_TRAJECTORY_SIGNAL` or Part C reports `TRAINING_TFG_GO`
- **THEN** the next permitted step is confirmation on new independent records

#### Scenario: No trajectory-sensitive result is found
- **WHEN** both the executed frozen and training paths lack a trajectory-sensitive downstream result
- **THEN** the final recommendation selects the natural-reference or simpler phone-control path according to the completed audits
