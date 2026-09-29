## ADDED Requirements

### Requirement: Bounded fresh pretrain acquisition

The implementation SHALL follow design.md for one new run root, a frozen source plan, at most 24 fresh groups, and complete historical exclusion before reading candidate media. It SHALL preserve historical runs and sealed media, and enforce the documented byte/time limits.

#### Scenario: Existing pool cannot supply enough groups

- **WHEN** Existing eligible pretrain sources are insufficient
- **THEN** Acquire only from recorded accessible pretrain sources within budget; missing access or exhausted budget produces an explicit blocked artifact, never a relaxed cohort gate

#### Scenario: Acquisition inventory is scanned as prior usage

- **WHEN** P prepares the new cohort
- **THEN** Acquisition artifacts live under that same current run root and are excluded with it; prior run directories remain in the history scan

### Requirement: Real visual evidence and valid rejection records

The implementation SHALL bind actual MediaPipe model identity, source clip identity, frame arrays, geometry and input-review evidence. PASS SHALL satisfy the unchanged input thresholds and fixed clip ordering. Normal quality failures SHALL remain valid auditable records without admitting their clips.

#### Scenario: One clip fails detection

- **WHEN** A clip has insufficient valid frames or no first-frame face
- **THEN** Preserve a FAIL and its measurements, continue the prescribed clip order, and do not reject unrelated valid audit entries or fabricate passing fields

#### Scenario: Audit evidence belongs to another clip

- **WHEN** A supplied PASS has a different clip SHA, missing model asset, nonfinite geometry or modified arrays
- **THEN** Validation rejects the unsupported admission even if the JSON self-hash was recomputed

### Requirement: Separate cohort readiness from frozen experiment inputs

The implementation SHALL provide independent cohort-only validation and retain full inputs validation. It SHALL freeze 12 formal and 2 disjoint smoke groups using the original ordering and keep science closed until candidate and freeze validation finish.

#### Scenario: Exactly fourteen groups pass input screening

- **WHEN** All historical, media, visual and integrity checks pass for fourteen groups
- **THEN** Cohort validation may report COHORT_READY with inputs=PENDING_CANDIDATE, but A/B/C generation remains closed until INPUTS_FROZEN

#### Scenario: Counts pass but historical coverage is incomplete

- **WHEN** At least fourteen groups pass quality checks but historical coverage is unresolved
- **THEN** Preserve BLOCKED_HISTORY_COVERAGE as a valid blocked outcome and do not report COHORT_READY

#### Scenario: A candidate fails after cohort readiness

- **WHEN** direct_v1 construction or complete input validation fails
- **THEN** Keep the selected cohort immutable, report the engineering blocker and do not substitute another group

### Requirement: Reviewable handoff and resource closure

The implementation SHALL record stage counts, evidence paths, validation identity and the actual terminal state, then update the existing BM entity. Acquisition and input audit SHALL run on local CPU; resumed GPU experiments SHALL follow the original budget, transfer and shutdown contract.

#### Scenario: Only the unblock stage is complete

- **WHEN** COHORT_READY is independently validated but no scientific cells have run
- **THEN** Report input readiness and the next candidate/freeze commands without claiming experimental completion or a replacement benefit
