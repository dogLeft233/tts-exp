## ADDED Requirements

### Requirement: Frozen natural-only temporal intervention
The implementation MUST follow this change's design and the shared parallel contract, constructing exactly N, SMOOTH and SHARP from the locked natural mel with one fixed symmetric temporal kernel and shared amplitude.

#### Scenario: Constant input or wrong axis
- **WHEN** synthetic constant mel is processed
- **THEN** both transforms are identity, and tests MUST reject convolution along the frequency axis or temporal index displacement

### Requirement: Controlled natural-audio evaluation
The runner MUST validate the shared controls before candidate generation and preserve original N PCM, F0, U and k0 for both candidates.

#### Scenario: Bypassing the control stage
- **WHEN** candidates or resume is requested without current hash-bound control validation
- **THEN** no candidate forward is allowed

### Requirement: Complete two-direction decision
The analysis MUST report both frozen contrasts using the shared 99% group intervals and the ordered decisions in design.md, within 34 fresh videos and 36 fresh score cells.

#### Scenario: Free-offset gain without anchor gain
- **WHEN** C improves but the fixed-anchor criterion fails
- **THEN** that direction MUST NOT pass or authorize training

### Requirement: Independent handoff evidence
The implementation MUST provide independent numerical validation, targeted tests, self-review and updates to its unique BM entity without modifying another branch or parent evidence.

#### Scenario: Scientific negative
- **WHEN** controls and validation pass but neither candidate passes
- **THEN** the experiment concludes with NO_TEMPORAL_CONTRAST_GAIN_ESTABLISHED and all confirmation flags remain false
