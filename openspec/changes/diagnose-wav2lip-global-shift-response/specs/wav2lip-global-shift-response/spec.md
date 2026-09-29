## ADDED Requirements

### Requirement: Historical and new inputs are separately frozen
The experiment SHALL use design.md's hash-bound 23-record historical cache and fixed 12-record seen-fit cohort. It MUST join by record and cell identity, preserve both cohorts separately, and freeze all input, source and specification bindings before fresh scoring.

#### Scenario: Inputs are ready
- **WHEN** the registered files, selected IDs, PCM, ROI and time support pass audit
- **THEN** prepare freezes protocol and the exact matrix plan without accessing sealed media

#### Scenario: Provenance or historical parity fails
- **WHEN** any input is missing, ambiguous, changed, or historical FULL scores fail registered parity
- **THEN** the run records engineering BLOCKED and does not start GPU generation or substitute records

#### Scenario: Historical tracks start at different frames
- **WHEN** a historical track has a nonzero or condition-dependent start frame
- **THEN** the experiment maps the common absolute-frame support to each matrix's local rows and reports the crop confound, without dropping the record or assuming all tracks start at zero

### Requirement: Global audio shifts and face settings are exact
The experiment SHALL implement N, independent N_REPEAT, zero-padded DELAY_200 and ADVANCE_200 with exact original PCM length and design.md's signs. DYNAMIC SHALL preserve the original frames and boxes; STATIC SHALL repeat frame zero and box zero together through the same lossless input path.

#### Scenario: Static generation is prepared
- **WHEN** STATIC inputs are constructed
- **THEN** every decoded source frame and every box match their corresponding frame-zero reference, with no score-based frame selection

#### Scenario: An intervention changes the contract
- **WHEN** execution rolls samples, interpolates audio, uses dynamic boxes with static pixels, shifts mux timestamps, or copies N video as N_REPEAT
- **THEN** validation rejects the run

### Requirement: The complete bounded diagnostic matrix is executed
The experiment SHALL follow design.md's two face modes and eight score cells per mode per record, with at most 96 fresh generated videos and 192 fresh scored cells. Replacement cells MUST use untouched N PCM; own-audio and same-video mismatches MUST remain separate diagnostics.

#### Scenario: Controls are scientifically unexpected
- **WHEN** repeat or timing diagnostics are weak but media and scoring integrity pass
- **THEN** the run completes the already registered shifted-driver cells and reports uninterpretability where required, without changing controls or adding arms

#### Scenario: GPU or media integrity fails
- **WHEN** host CUDA, PCM identity, frame count, ROI, or media binding fails
- **THEN** execution stops as engineering BLOCKED with actual completed counts and preserved artifacts

### Requirement: Scores distinguish free-offset and natural-coordinate behavior
The experiment SHALL produce full 31-point mean curves on FULL and design.md's fixed interior supports. It SHALL compute C, D, median, offset, baseline-anchored distance and curve-separation exactly as specified, and verify shift-sign predictions with a synthetic fixture.

#### Scenario: C improves without anchored improvement
- **WHEN** free-offset Sync-C improves but baseline-anchored distance does not
- **THEN** the result is reported as a score-pattern observation rather than improvement at the original natural time coordinate

#### Scenario: Timing response is not identifiable
- **WHEN** repeatability, same-video audio-shift detection, or peak clarity does not support interpretation
- **THEN** response fields retain the full denominator and mark the relevant result uninterpretable instead of asserting generator insensitivity

### Requirement: Comparisons and conclusions are bounded
The experiment SHALL report every registered face-mode and shift comparison with shared paired group-bootstrap draws, the lightweight visual-response diagnostic and design.md's flags. All new intervals SHALL be labeled exploratory. The system MUST NOT infer pure mouth leakage, visual quality, universal replacement failure or generalization from this assay.

#### Scenario: One face mode has a positive mean
- **WHEN** only DYNAMIC or STATIC has a positive result
- **THEN** the report includes the paired interaction interval and visual-conditioning confounds, without treating significance in only one group as proof of interaction

#### Scenario: The diagnostic finishes
- **WHEN** complete artifacts pass independent validation
- **THEN** final.json records engineering GO, diagnostic COMPLETE, scientific NOT_A_CONFIRMATION and next_action STOP_AND_REVIEW regardless of the diagnostic flag values
- **AND** training, generalization and historical-gate-repair flags remain false

### Requirement: The downstream handoff is reproducible
The implementation SHALL provide the minimal CLI, independent validator, focused contract tests, self-review and BM update described in design.md. It MUST preserve historical runs and only resume hash-matching completed cells, never retrying based on scientific outcome.

#### Scenario: Artifacts or producer calculations are tampered with
- **WHEN** the independent validator detects a binding, matrix, statistic, denominator or decision discrepancy
- **THEN** finalization cannot claim a completed valid diagnostic

#### Scenario: Delivery is complete
- **WHEN** implementation, execution and independent acceptance finish
- **THEN** the agent updates the same BM note with actual results and limitations, checks tasks, and supplies result and validation paths without starting a semantic or waveform head
