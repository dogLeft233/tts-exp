## ADDED Requirements

### Requirement: Immutable finite-cohort cache audit

The implementation SHALL bind the inputs and hashes in design.md, retain all 133 records and 23 source groups, and operate only on approved cached JSON/NPZ data without new media extraction, model inference, training, or SyncNet scoring.

#### Scenario: Input cache is corrupt

- **WHEN** a required record, feature archive, or frozen hash fails validation
- **THEN** the audit SHALL report engineering BLOCKED and diagnostic not_available, rather than treating the corruption as scientific missingness.

### Requirement: Shared natural-clock visual comparison

The implementation SHALL reproduce the legacy exact-time pairing and eligibility, then compare real/natural/candidate mouth features on the same valid real-frame intersection using the fixed coverage and normalized-distance formulas in design.md.

#### Scenario: Pairwise valid frames differ

- **WHEN** natural and candidate have different valid matches to the real video
- **THEN** both new distances SHALL use the three-arm intersection and the coverage denominator SHALL retain the longest original sequence length.

#### Scenario: Common support is insufficient

- **WHEN** old eligibility fails or common coverage is below 0.85
- **THEN** the record SHALL remain in its original group denominator with an unknown bounded contribution, even if its cached distance is finite.

### Requirement: Explicit missingness bounds and decision limits

The implementation SHALL calculate design.md's worst/best bounds using all original records within groups and equal weights across all 23 groups. It SHALL label them finite-cohort sensitivity bounds, not confidence intervals or confirmation evidence.

#### Scenario: Entire group is unobserved

- **WHEN** no record in an original group is observed
- **THEN** its bounds SHALL be [-1,1] and the group SHALL retain its full equal weight.

#### Scenario: Bounds include zero

- **WHEN** the bounds fail both fixed directional rules
- **THEN** the only completed diagnostic SHALL be INCONCLUSIVE_UNDER_MISSINGNESS; no observed-only result may promote it.

### Requirement: Parent decision and downstream authority remain separate

Every diagnostic outcome SHALL preserve the parent's BLOCKED state and all five authorization/confirmation fields listed in design.md as false. Positive cached visual evidence SHALL authorize no training or old Stage02 execution.

#### Scenario: Worst-case bound is positive

- **WHEN** L exceeds the fixed positive tolerance and independent validation passes
- **THEN** the audit SHALL report VISUAL_SIGNAL_ROBUST_IN_CACHED_COHORT and recommend only a separately designed independent visual confirmation.

### Requirement: Reproducible handoff and independent validation

The implementation SHALL provide the CLI, artifacts, tests, independent numerical validator, and phase self-review specified in design.md and update the same Basic Memory experiment entity through execution.

#### Scenario: Producer result disagrees with independent calculation

- **WHEN** reconstructed support, values, bounds, or terminal labels disagree
- **THEN** final acceptance SHALL fail with a nonzero exit code and no scientific diagnostic conclusion.

#### Scenario: Audit concludes without positive evidence

- **WHEN** artifacts and independent checks pass but the directional rule is negative or inconclusive
- **THEN** the run SHALL be marked completed with its actual diagnostic, preserve zero-model budget, and record the limitation in Basic Memory.
