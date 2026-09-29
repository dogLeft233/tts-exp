## ADDED Requirements

### Requirement: Phone support without outcome selection
The implementation MUST derive disjoint phone cores from the hash-bound used_masks and natural-core metadata before reading new outcomes, retaining all cohort records.

#### Scenario: A core is shorter than five columns
- **WHEN** a core is shorter than five columns
- **THEN** its natural mel MUST remain unchanged and no replacement boundary may be selected

### Requirement: Matched structural control
The implementation MUST construct PHONE_CORE and GENERIC_CORE on the same frozen support, with preclip equal norms, shared amplitude and explicit postclip norm validity.

#### Scenario: Clipping breaks norm comparability
- **WHEN** clipping breaks norm comparability
- **THEN** the experiment MUST report the issue and MUST NOT claim phone-specific gain or retune the transforms

### Requirement: Natural baseline first
PHONE_CORE MUST pass the shared original-natural gain rule before a positive conclusion; superiority to GENERIC_CORE is only a secondary conjunctive mechanism test.

#### Scenario: PHONE_CORE beats the generic control but fails against natural
- **WHEN** pHONE_CORE beats the generic control but fails against natural
- **THEN** the decision MUST be NO_PHONE_CORE_GAIN_ESTABLISHED

### Requirement: Reviewable completion
The implementation MUST follow design.md and the shared handoff, retain immutable parents, independently recompute numerical decisions, record actual resource usage, and update its unique BM entity.

#### Scenario: A signed report contains incorrect statistics
- **WHEN** report values are changed and its JSON hash is recomputed
- **THEN** independent validation MUST reject the report using the original arrays, and the runner MUST NOT mark completion
