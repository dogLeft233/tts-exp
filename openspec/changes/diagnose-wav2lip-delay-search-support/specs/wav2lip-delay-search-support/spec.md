## ADDED Requirements

### Requirement: Diagnostic inputs preserve parent provenance
The implementation SHALL bind design.md's immutable parent hashes and all sixteen records in eight groups, using exactly the N/N and N/A_DELAY cached cells. It SHALL verify media identity, exact delayed PCM, embeddings and real temporal support before analysis. New model forwards, video generation and GPU execution MUST remain zero.

#### Scenario: An embedding is missing or changed
- **WHEN** a required cached embedding is absent or its bound hash differs
- **THEN** execution records BLOCKED and preserves the evidence without substituting a record or regenerating the missing asset

### Requirement: Legacy gate is independently reproduced
The implementation SHALL reproduce the original distance contract, U metrics, natural anchor and thirteen-of-sixteen offset result within design.md's tolerances. It SHALL compute the expected shifted column and censoring flag for every record.

#### Scenario: The expected peak lies beyond the last legacy column
- **WHEN** natural k_N is 27 and the known audio delay is five frames
- **THEN** expected_column is 32 and expected_in_legacy_domain is false
- **AND** this observation alone is not labeled a recovered control or replacement benefit

### Requirement: Known delay uses physically matched search domains
The implementation SHALL apply the single paired-domain construction in design.md to every record: natural lags -15 through 15 and delayed lags -10 through 20, on the unchanged U rows. It SHALL compute new distances from actual cached embeddings and report physical offset from each domain's lag coordinates.

#### Scenario: Matched-domain minima have equal array indices
- **WHEN** the natural and delayed paired-domain minima occur at the same column j
- **THEN** their reported offsets are 15-j and 10-j respectively and their difference is -5 frames

#### Scenario: A requested audio index is outside real support
- **WHEN** a diagnostic cell requires an unavailable embedding or MFCC window
- **THEN** the implementation records BLOCKED instead of padding, wrapping indices or moving the scoring window

### Requirement: Sensitivity retains the uncompensated natural anchor
The implementation SHALL compute anchor damage from the original delayed curve at the frozen natural physical lag. It SHALL retain the original group bootstrap, anchor thresholds and fourteen-of-sixteen offset criterion, and separately evaluate full boundary explanation as defined in design.md.

#### Scenario: The matched delay curve returns to the natural curve
- **WHEN** known-delay compensation aligns the two distance curves
- **THEN** anchor damage is still read from the original uncompensated delayed curve
- **AND** compensation is not reported as a replacement gain

#### Scenario: Fourteen records pass but an original anomaly remains
- **WHEN** corrected_control_pass is true but boundary_explanation_complete is false
- **THEN** the terminal decision is CONTROL_UNRESOLVED and content_probe_revision_eligible is false

### Requirement: Diagnosis has bounded downstream authority
The implementation SHALL emit design.md's terminal decision only after independent validation. SEARCH_SUPPORT_RECOVERED MAY make a separately versioned content-probe amendment eligible; it MUST NOT authorize candidate execution in this change or rewrite the parent CONTROL_FAILED result. All replacement, historical-shift-repair, waveform and generalization flags SHALL remain false.

#### Scenario: All sixteen records recover and the anchor rule passes
- **WHEN** both diagnostic predicates and independent validation pass
- **THEN** the result is SEARCH_SUPPORT_RECOVERED with content_probe_revision_eligible true and stage_b_authorized false

### Requirement: Outputs are reproducible and retained in project memory
The implementation SHALL deliver hash-bound protocol, all-record diagnostics, matrices, independently recomputed validation, truthful self-review and final report. It SHALL update and read back the same BM experiment note at material stages and completion, preserving its changelog.

#### Scenario: Producer output is tampered with
- **WHEN** matrix values, lag labels, record membership, hashes or terminal flags disagree with independent reconstruction
- **THEN** validation rejects finalization and the result cannot claim SEARCH_SUPPORT_RECOVERED
