## ADDED Requirements

### Requirement: Inputs and scoring scope are frozen before outcomes
The experiment SHALL use exactly the 22 seen-fit records, byte-hash-bound inputs, model identities, and target-time U masks defined in design.md. All joins SHALL use sample_id and verify group/PCM/length/asset identity. It MUST NOT claim independent replication or replace U with source-time Q.

#### Scenario: Inputs match
- **WHEN** all frozen files and assets match and all 31 offset columns have valid support on U for every registered cell
- **THEN** Stage 00 freezes protocol, code/spec bindings, candidates and diagnostics before new scoring

#### Scenario: Inputs or support differ
- **WHEN** a hash, group, length, join, asset or support audit differs
- **THEN** execution records engineering BLOCKED without dropping or replacing records

### Requirement: Audio interventions are exact and nonadaptive
The experiment SHALL construct N, N_REPEAT, RT, MAG and ENV using design.md's exact STFT formulas, alpha=0.75, scaling and PCM contracts. ENV SHALL broadcast the time-mean log-spectrum difference over natural's original time-varying log spectrum. P SHALL be the bound plateau PCM and only be used as a mux control.

#### Scenario: Construction is valid
- **WHEN** exact-length finite candidates and pre-score mel diagnostics pass the independent construction audit
- **THEN** the run freezes all candidate bytes and scaling metadata

#### Scenario: A construction shortcut changes the experiment
- **WHEN** execution copies N as RT, copies N video as N_REPEAT, flattens natural's time structure in ENV, adjusts alpha, or generates a P-driven video
- **THEN** validation rejects the run

### Requirement: Fresh media share geometry and natural reference audio
The experiment SHALL use fixed parent face/boxes, independent per-arm generation workspaces, batch=4 host CUDA generation and CPU SyncNet as defined in design.md. All replacement cells SHALL contain unchanged N PCM. N/P SHALL share N/N's identical video stream. All scored artifacts SHALL be fresh in this run with immutable bindings.

#### Scenario: Complete two-stage matrix
- **WHEN** Stage A completes
- **THEN** exactly 66 videos and 88 cells exist for N/N, N_REPEAT/N, RT/N and N/P across 22 records
- **AND** only independently validated, scientifically passing Stage A permits MAG/N and ENV/N, adding exactly 44 videos and 44 cells

#### Scenario: Resume after an engineering interruption
- **WHEN** --resume finds cells with matching input/config/output hashes
- **THEN** it reuses only those complete cells and preserves the original frozen protocol
- **AND** it never retries a completed cell because of its score or resumes Stage B after scientific control failure

### Requirement: Endpoint controls are local and explicit
The experiment SHALL compute mean distance curves on frozen U, PLUS and MINUS before deriving C, D and offset. It SHALL apply design.md's symmetric 95% CI repeat/RT equivalence gates, baseline clarity, local P offset predictions and damage gates. These controls MUST NOT be presented as validating generated-audio temporal equivariance or repairing historical gates.

#### Scenario: A control fails
- **WHEN** repeatability, RT equivalence, baseline or mismatched-audio sensitivity fails
- **THEN** the scientifically complete decision is CONTROL_FAILED, Stage B count is zero and the report identifies each failure

### Requirement: Benefits and equivalence use prespecified comparisons
The experiment SHALL use paired source-group bootstrap with fixed shared draws and both CI levels exactly as in design.md. Candidate decisions SHALL use 97.5% intervals and require gain versus both N and RT. It SHALL distinguish MAG superiority from MAG/ENV equivalence and report all registered comparisons regardless of outcome.

#### Scenario: Temporal spectral increment is supported
- **WHEN** gain(MAG), the MAG versus ENV increment and all required safety/offset conditions hold
- **THEN** the decision is TEMPORAL_SPECTRAL_INCREMENT_SUPPORTED, scoped to this construction and U endpoint

#### Scenario: Average spectrum suffices within the registered margin
- **WHEN** the prior decision does not apply, gain(ENV) holds and both MAG−ENV intervals lie strictly inside the registered equivalence margins with valid offset agreement
- **THEN** the decision is AVERAGE_SPECTRUM_SUFFICIENT_IN_SCOPE

#### Scenario: Mechanism remains unresolved
- **WHEN** at least one candidate has gain but neither explanatory decision applies
- **THEN** the decision is GAIN_WITH_MECHANISM_UNRESOLVED without claiming equality from a nonsignificant difference

#### Scenario: Useful gain is not established
- **WHEN** controls pass but neither candidate meets gain
- **THEN** the decision is NO_USEFUL_GAIN_ESTABLISHED and this construction stops

### Requirement: Independent acceptance and bounded handoff are mandatory
The experiment SHALL produce design.md's artifacts, independent candidate/matrix/statistical validation, a self-contained result report and an updated BM experiment note. Engineering BLOCKED takes precedence over scientific decisions; complete scientific failure is a valid delivery. All outcomes SHALL retain training_authorized=false, generalization_established=false, historical_gate_repaired=false and legacy_bridge_executed=false.

#### Scenario: Experiment is handed off
- **WHEN** actual-stage counts, identities, independent calculations and terminal-state consistency pass
- **THEN** final.json records exactly one registered scientific decision, its failed/passed gates, budget and result pointers
- **AND** BM records the same outcome, the seen-fit limitation and the appropriate stop or independent-confirmation recommendation

#### Scenario: Work expands beyond the protocol
- **WHEN** execution attempts new TTS, alignment, training, sealed-data access, adaptive strength/window/seed search, extra arms, or parent overwrite
- **THEN** it stops as BLOCKED and does not rewrite a historical decision
