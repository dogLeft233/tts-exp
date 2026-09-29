## ADDED Requirements

### Requirement: Frozen factorial construction
The implementation MUST use the same fixed content increment on natural and reconstructed bases, with the specified bounded base shift and same-checkpoint zero-auxiliary predictions.

#### Scenario: An implementation changes the content amplitude on the reconstructed base
- **WHEN** an implementation changes the content amplitude on the reconstructed base
- **THEN** independent validation MUST reject the run

### Requirement: Absolute replacement and conditional repair separation
The primary decision MUST compare BASE_CONTENT with the complete natural baseline using the shared gain rule and anchor, while reporting base interaction separately.

#### Scenario: The content increment repairs BASE but BASE_CONTENT still fails against natural
- **WHEN** the content increment repairs BASE but BASE_CONTENT still fails against natural
- **THEN** the scientific decision MUST be NO_BASE_CONTENT_GAIN_ESTABLISHED even when the interaction is positive

### Requirement: Bounded reuse and independent evidence
The experiment MUST independently verify reused N and N_CONTENT scores, then produce at most 50 fresh videos and 52 fresh score cells with complete wrong-instance and norm diagnostics.

#### Scenario: A reused score or one factorial cell is missing or mismatched
- **WHEN** a reused score or one factorial cell is missing or mismatched
- **THEN** the run MUST block without substituting another arm, seed or source

### Requirement: Reviewable completion
The implementation MUST follow design.md and the shared handoff, retain immutable parents, independently recompute numerical decisions, record actual resource usage, and update its unique BM entity.

#### Scenario: A signed report contains incorrect statistics
- **WHEN** report values are changed and its JSON hash is recomputed
- **THEN** independent validation MUST reject the report using the original arrays, and the runner MUST NOT mark completion
