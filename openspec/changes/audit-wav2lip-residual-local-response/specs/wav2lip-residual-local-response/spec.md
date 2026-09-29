## ADDED Requirements

### Requirement: CPU-only frozen support audit
The implementation MUST use only the shared contract's cached JSON and numerical arrays, mapping official sixteen-column mel chunks through five-frame visual windows to exposure on fixed U without video decoding or new model calls.

#### Scenario: Overlapping mel chunks
- **WHEN** several visual frames share mel columns
- **THEN** exposure MUST use the union of unique columns rather than counting shared columns repeatedly

### Requirement: Natural-anchor local association
The analysis MUST use the full-U natural k0 for every row, center exposure and response within each record, and compute the design's group-pooled slope and exposure-weighted gain before equal weighting all eight groups.

#### Scenario: No exposure variation
- **WHEN** a record has no exposure but its paired group member is informative
- **THEN** the unexposed record remains in the full-U and locality audit, while the group uses the frozen pooled formulas without imputation

#### Scenario: Entire group is unidentifiable
- **WHEN** any group has zero pooled exposure variance or no exposure
- **THEN** all groups remain in the audit and the decision is LOCAL_SUPPORT_NOT_IDENTIFIABLE without window changes

### Requirement: Independent cache and locality validation
An independent validator MUST reconstruct cached matrices, parent full-U contrasts, support mappings and local statistics, and check available zero-exposure rows within the frozen tolerance.

#### Scenario: Unexposed visual input yields a changed embedding
- **WHEN** a zero-exposure row violates the embedding or matrix tolerance
- **THEN** the result is engineering BLOCKED rather than a scientific nonlocal effect

### Requirement: Preserve the parent negative conclusion
The runner MUST use the design's conjunctive 99% group criteria, record retrospective status in its unique BM entity, and keep all replacement/training flags false.

#### Scenario: Local association passes
- **WHEN** both slope and local_gain satisfy the frozen criteria
- **THEN** only LOCAL_RESPONSE_ASSOCIATION_TO_CONFIRM is reported and the original full-U NO_INCREMENT_ESTABLISHED conclusion remains unchanged
