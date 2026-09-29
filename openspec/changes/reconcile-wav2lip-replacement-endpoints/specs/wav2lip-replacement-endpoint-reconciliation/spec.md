## ADDED Requirements

### Requirement: Reconciliation uses a fixed paired cohort and immutable caches
The implementation SHALL use design.md's hash-bound H/S inputs and exactly 22 paired records/source groups. It SHALL select exactly 66 H and 110 S natural-audio score matrices by explicit keys, verify track clocks and freeze cache provenance. It MUST NOT combine the 23-record discovery cohort with this paired analysis.

#### Scenario: Complete inputs are available
- **WHEN** all root hashes, per-asset bindings, joins, single zero-start tracks and fixed window support pass
- **THEN** audit freezes the 176-matrix protocol and code/spec hashes before computing reconciliation outcomes

#### Scenario: Cache or cohort integrity is incomplete
- **WHEN** a hash, asset, track, timing support or unique pairing is invalid
- **THEN** the run records engineering BLOCKED and a concrete discrepancy without substituting records or rerunning models

### Requirement: Audio and processing differences remain explicit
The implementation SHALL distinguish container bytes from decoded PCM, compare the bound N/M and existing BRIDGE_075/MAG samples, and record generation/scoring differences with their provenance as specified in design.md.

#### Scenario: Candidate waveforms differ
- **WHEN** decoded BRIDGE_075 and MAG PCM are unequal while the N/M pairing is valid
- **THEN** the run reports integer-safe sample differences and retains audio construction as a confound rather than correcting the candidates

#### Scenario: A historical hash field is misleading
- **WHEN** the H audio metadata calls a container hash pcm_sha256
- **THEN** the audit computes decoded PCM identity independently and does not mistake container differences for waveform differences

### Requirement: Fixed endpoints reproduce their historical anchors
The implementation SHALL compute FULL, COMMON_INTERIOR and U exactly as defined in design.md, preserving legacy offset padding only in FULL. It SHALL produce 528 unique endpoint rows with complete distance curves and pass both historical parity checks before interpreting cross-endpoint differences.

#### Scenario: Endpoint aggregation is valid
- **WHEN** each finite matrix supports the fixed rows
- **THEN** C/D/offset are derived from the mean 31-column curve with the registered precision and tie rule, never by averaging row-wise confidences

#### Scenario: Historical parity fails
- **WHEN** H log/summary or S U/summary recomputation exceeds design.md's tolerance
- **THEN** engineering is BLOCKED with the exact discrepant cells and no tolerance adjustment, model forward or scientific reinterpretation

### Requirement: Statistical comparisons are paired and diagnostic
The implementation SHALL report all registered comparisons and the per-record decomposition using shared source-group bootstrap draws from design.md. It SHALL distinguish descriptive 95% intervals from the parents' registered scientific intervals and preserve uncertainty and processing confounds.

#### Scenario: A sign changes between windows
- **WHEN** a candidate benefit changes direction across FULL, I or U
- **THEN** the report states the means and paired uncertainty, distinguishes support from processing-chain differences, and does not claim a causal padding, ROI or phonetic mechanism

#### Scenario: One recomputed endpoint appears favorable
- **WHEN** any descriptive endpoint has a positive benefit
- **THEN** the run still records NOT_A_CONFIRMATION and does not select a new window, alpha, cohort or training route

### Requirement: Execution remains CPU-only and independently reviewable
The implementation SHALL perform zero model forwards and create zero new audio/video media. It SHALL use a new run directory, independent matrix/statistical validation, source-bound review, a self-contained report and the same BM experiment note throughout execution.

#### Scenario: Reconciliation completes
- **WHEN** the independent validator accepts all counts, bindings, calculations and final-state consistency
- **THEN** engineering is GO, diagnostic_decision is RECONCILED, scientific_decision is NOT_A_CONFIRMATION and next_action is STOP_CURRENT_SPECTRAL_CONSTRUCTION
- **AND** all training/generalization/historical-gate flags remain false and BM records a concluded diagnostic with result pointers

#### Scenario: Work cannot finish within the cache-only contract
- **WHEN** completing the analysis would require regenerating media, new scoring, changing support or modifying historical artifacts
- **THEN** the run ends BLOCKED/INCOMPLETE with scientific_decision=not_available and next_action=REVIEW_INPUT_OR_CACHE_DISCREPANCY
