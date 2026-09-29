## Purpose

This capability produces descriptive official SyncNet evidence for the already fixed record-heldout cohort when one candidate barely exceeds the original audio-trust limit. It exposes the relaxed tolerance and post-hoc status instead of converting the diagnostic into a pre-registered scientific transfer claim.

## ADDED Requirements

### Requirement: Explicit diagnostic tolerance

The diagnostic SHALL use exactly the locked 94-record training checkpoint and 8-record evaluation manifest from the corrected existing-data run. It SHALL raise only the normalized log-mel limit to `0.11`; exact length, finite values, residual peak `<=0.05`, saturation fraction `<=1e-4`, PCM16, official SyncNet, target-offset, and matrix requirements SHALL remain unchanged.

#### Scenario: Candidate is within diagnostic tolerance
- **WHEN** a locked candidate has normalized log-mel distance `<=0.11` and passes all unchanged checks
- **THEN** it is materialized once and may receive official real-video SyncNet scoring

#### Scenario: Candidate exceeds diagnostic tolerance
- **WHEN** a candidate has normalized log-mel distance `>0.11` or fails any unchanged check
- **THEN** the row is recorded as engineering-invalid and is not silently removed from the eight-row denominator

### Requirement: Fixed cohort and checkpoint provenance

The diagnostic SHALL read the exact parent manifest and step-100 checkpoint, verify their hashes and completed training artifacts, and SHALL NOT retrain, warm-start, search checkpoints, select new records, or change the manifest. Natural audio SHALL remain outside adapter inference and training; it MAY be used only for existing detached coordinates and official evaluation.

#### Scenario: Locked parent validates
- **WHEN** the parent manifest and step-100 state are hash-valid and record order is unchanged
- **THEN** the diagnostic proceeds with exactly those eight evaluation IDs

#### Scenario: Parent is changed or incomplete
- **WHEN** the parent manifest, checkpoint, training history, or hashes are missing or changed
- **THEN** the diagnostic emits `DIAGNOSTIC_PARENT_INVALID` and performs no candidate or video scoring

### Requirement: Complete descriptive official scoring

The diagnostic SHALL attempt official paired real-video scoring for all eight fixed rows using complete 31-offset curves, 91 windows, identical decoded video frames, and unchanged sign conventions. It SHALL report every row's QC and official metrics, including rows that are engineering-invalid under the diagnostic tolerance. It MUST label all aggregate movement as exploratory and post-hoc.

#### Scenario: All diagnostic rows score
- **WHEN** all eight candidates satisfy the diagnostic engineering checks and official scoring completes
- **THEN** the system stores all paired curves and emits descriptive gains without claiming pre-registered transfer

#### Scenario: Some rows are invalid or missing
- **WHEN** one or more rows fail tolerance, scoring, or provenance checks
- **THEN** the terminal status is `DIAGNOSTIC_NOT_EVALUATED`, the denominator remains eight, and no scientific transfer status is emitted

### Requirement: Conditional exploratory replacement

Wav2Lip replacement SHALL run only if all eight official real-video rows pass the unchanged real-video gate. If it runs, it SHALL materialize all 72 cells and use only the natural-audio column as authoritative, but the terminal report SHALL remain exploratory because the tolerance was post-hoc.

#### Scenario: Real-video gate passes
- **WHEN** the complete eight-row real-video evidence passes the unchanged six-success and median-gain gate
- **THEN** the diagnostic may run the complete 72-cell strict replacement matrix

#### Scenario: Real-video gate fails
- **WHEN** the real-video gate fails or is incomplete
- **THEN** replacement is skipped explicitly and no diagonal or cross-audio cell can promote a claim

### Requirement: Immutable diagnostic report

The diagnostic SHALL write to a new create-once run root, bind parent/candidate/video/curve artifacts by hash, and support read-only resume only after terminal validation. The report SHALL state the exact `0.11` tolerance, why it was chosen, and that all findings are exploratory and not a pre-registered or population result.

#### Scenario: Diagnostic graph validates
- **WHEN** all written artifacts and hashes are internally consistent
- **THEN** validation returns `valid` and resume returns the stored decision without external execution

#### Scenario: Diagnostic graph is tampered
- **WHEN** a parent, manifest, candidate, score, matrix, or terminal decision changes
- **THEN** validation fails closed and no diagnostic promotion occurs
