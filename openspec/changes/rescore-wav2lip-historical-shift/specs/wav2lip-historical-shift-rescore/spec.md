## ADDED Requirements

### Requirement: Historical media and cohort are immutable
The experiment SHALL bind design.md's eight root inputs and all referenced assets, and use exactly the historical 23 records and 23 source groups. N and SHIFT_200 videos MUST remain the existing generated outputs; scoring audio MUST be the complete untouched canonical N PCM.

#### Scenario: Inputs are joined in a different order
- **WHEN** a source manifest has a different row order
- **THEN** records and cells are joined by their declared keys and retain the frozen cohort
- **AND** duplicate, missing or hash-mismatched inputs block execution rather than changing samples

#### Scenario: A SHIFT video contains its original driver audio
- **WHEN** a historical SHIFT_200 video is prepared for replacement scoring
- **THEN** its embedded audio is explicitly replaced with the complete N PCM and byte identity is checked before and after scoring

### Requirement: Rescoring reproduces the actual v4 scoring entrance
The experiment SHALL use design.md's FULL_FRAME_V4 path: all original 224×224 frames, lossless derived video, unchanged frame timing, and the frozen CPU SyncNetScorer. It MUST distinguish generation ROI from scoring crop and MUST NOT add detection, ROI cropping, temporal alignment, or TFG generation.

#### Scenario: A downstream implementation tries to use v4 ROI boxes
- **WHEN** it attempts to crop historical scoring frames with v4 generation boxes
- **THEN** validation rejects this as a different scoring entrance

#### Scenario: Scoring parity fails
- **WHEN** either of the two fixed v4 parity cells differs beyond design.md's tolerances
- **THEN** the 46 target scoring jobs do not begin and the run records engineering BLOCKED

#### Scenario: All target media are scored
- **WHEN** parity and media integrity pass
- **THEN** exactly 46 target cells plus two separate parity cells are available, with zero new TFG-generated videos, zero CUDA jobs and zero training

### Requirement: Frontend comparisons share absolute frame support and a fixed anchor
The experiment SHALL freeze design.md's J before inspecting new scores, map absolute frames through each historical track's actual origin, and compute both scoring paths on the same J. It SHALL compute mean distance curves before C/D/offset and use the shared historical-natural J anchor for all four matrices.

#### Scenario: Track origins differ
- **WHEN** N and SHIFT historical tracks start at different nonzero frames
- **THEN** the implementation maps g to g-f0 separately, preserves both local and absolute row coordinates, and reports remaining track-context limitations

#### Scenario: The common window is short
- **WHEN** a record has a nonempty J with only seven rows
- **THEN** it remains one of 23 equally weighted source groups and its support length is reported
- **AND** an empty or unsupported J blocks the run rather than permitting extrapolation or filtering

#### Scenario: Free-offset C improves without anchor improvement
- **WHEN** a candidate raises free-offset C while fixed-anchor distance benefit does not improve
- **THEN** it is reported as a score observation, not established improvement at the original natural timing

### Requirement: Statistics and conclusions respect the diagnostic scope
The experiment SHALL implement design.md's shared paired group-bootstrap draws, per-path benefits, paired frontend differences, descriptive flags and terminal states. FULL, historical I_H and new J results SHALL remain separately labeled.

#### Scenario: Only one scoring path has a positive confidence interval
- **WHEN** one path's C interval excludes zero and the other's does not
- **THEN** a frontend-change claim depends on the paired difference interval, not the difference in significance labels

#### Scenario: The complete diagnostic has any scientific direction
- **WHEN** independent validation passes for all planned artifacts
- **THEN** the final scientific label is NOT_A_CONFIRMATION and next_action is CLOSE_SHIFT_DIAGNOSTIC
- **AND** no outcome authorizes delay searches, training, generalization or historical gate repair

### Requirement: Execution is independently verifiable and handed off through memory
The implementation SHALL provide design.md's staged CLI, focused tests, immutable run artifacts, independent validator, truthful review and Basic Memory update. Completion MUST depend on verified artifacts rather than planned counts.

#### Scenario: A producer output is tampered with
- **WHEN** matrix, waveform, pixel, support, hash, statistic or decision reconstruction disagrees
- **THEN** independent validation rejects finalization and retains the discrepancy

#### Scenario: The downstream agent finishes
- **WHEN** the implementation and all required checks are complete
- **THEN** it updates the existing experiment BM note with results and boundaries, reads it back, checks completed tasks and returns the report and final artifact paths
