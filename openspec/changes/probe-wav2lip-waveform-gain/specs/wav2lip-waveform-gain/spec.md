## ADDED Requirements

### Requirement: Frozen waveform intervention
The implementation MUST construct exactly the specified reciprocal positive and negative gains from original PCM, using headroom limiting and ties-to-even integer rounding without normalization or clipping.

#### Scenario: The source has no amplification headroom
- **WHEN** the source has no amplification headroom
- **THEN** the complete cohort MUST stop as INPUT_DEGENERATE without dropping that source or changing the rule

### Requirement: Original natural evaluation audio
Both gain-driven videos MUST be scored against the byte-identical original natural PCM on the shared U and natural anchor after independent controls.

#### Scenario: A candidate gain changes the evaluation PCM
- **WHEN** a candidate gain changes the evaluation PCM
- **THEN** validation MUST fail before any scientific claim

### Requirement: Complete bounded decision
Both frozen contrasts MUST be reported with the shared grouped 99 percent intervals and gain criteria, within 34 videos and 36 score cells.

#### Scenario: Only free-offset confidence improves
- **WHEN** only free-offset confidence improves
- **THEN** the experiment MUST NOT declare a positive signal unless distance and fixed-anchor criteria also pass

### Requirement: Reviewable completion
The implementation MUST follow design.md and the shared handoff, retain immutable parents, independently recompute numerical decisions, record actual resource usage, and update its unique BM entity.

#### Scenario: A signed report contains incorrect statistics
- **WHEN** report values are changed and its JSON hash is recomputed
- **THEN** independent validation MUST reject the report using the original arrays, and the runner MUST NOT mark completion
