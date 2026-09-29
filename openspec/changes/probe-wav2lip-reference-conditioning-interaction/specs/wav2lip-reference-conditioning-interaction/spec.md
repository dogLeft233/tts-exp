## ADDED Requirements

### Requirement: Fixed factorial inputs
The implementation MUST use the shared contract's sixteen records and exactly N/CORRECT crossed with static source frames zero and forty-six, using their corresponding frozen boxes without score-dependent selection.

#### Scenario: Missing frame or box
- **WHEN** source frame index 46 or its bound box is absent or invalid
- **THEN** the branch is BLOCKED without another reference or sample substitution

### Requirement: Reference-specific control qualification
The runner MUST independently validate F0 replay/parity and F46 natural-repeat and matched-delay controls before generating F46_C, within 36 fresh videos and 54 scores.

#### Scenario: New reference lacks sensitivity
- **WHEN** the valid F46 delay control fails the frozen sensitivity threshold
- **THEN** the branch reports CONTROL_FAILED and generates no F46_C

### Requirement: Shared-anchor interaction endpoint
The primary endpoint MUST be the paired difference-in-differences in design.md with one F0 natural anchor per record and the shared 99% group interval.

#### Scenario: Reference improves both arms equally
- **WHEN** F46 improves raw scores but does not change the N-to-CORRECT benefit
- **THEN** the primary interaction is zero and MUST NOT be reported as audio replacement gain

### Requirement: Limited diagnostic conclusion
The implementation MUST preserve parent negative conclusions, independently validate inputs/statistics, and update only its own BM entity and owned files.

#### Scenario: Positive interaction but negative within-reference benefit
- **WHEN** I passes but g1 remains negative
- **THEN** only reference-dependent response is reported, with replacement and training flags false
