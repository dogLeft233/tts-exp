## ADDED Requirements

### Requirement: Read-only evidence reconstruction
The auditor MUST bind the fixed A/B/C and P/Q/D artifacts and independently reconstruct candidate arrays, support, score curves, grouped intervals and original decisions without importing producer science functions.

#### Scenario: An old validator reports PASS
- **WHEN** an old validator reports PASS
- **THEN** the audit MUST still recompute the scientific values and separately report original specification compliance

### Requirement: Matched delay coordinates
The auditor MUST calculate the F46 delay control with natural lags -15..15 and delayed lags -10..20, physical offsets 15-j and 10-j, retaining uncompensated anchor damage.

#### Scenario: The delayed peak lies beyond the legacy search domain
- **WHEN** the delayed peak lies beyond the legacy search domain
- **THEN** the auditor MUST use actual delayed embeddings at the matched lags and MUST NOT copy, roll or pad the natural curve

### Requirement: Honest limits and counterexamples
The audit MUST test joint group counts, every zero-exposure row, replay content, and analysis tampering; all original runs MUST remain immutable.

#### Scenario: The corrected F46 control passes
- **WHEN** the corrected F46 control passes
- **THEN** the report MUST leave F46 interaction unmeasured and MUST NOT generate candidate videos or authorize replacement

### Requirement: Reviewable completion
The implementation MUST follow design.md and the shared handoff, retain immutable parents, independently recompute numerical decisions, record actual resource usage, and update its unique BM entity.

#### Scenario: A signed report contains incorrect statistics
- **WHEN** report values are changed and its JSON hash is recomputed
- **THEN** independent validation MUST reject the report using the original arrays, and the runner MUST NOT mark completion
