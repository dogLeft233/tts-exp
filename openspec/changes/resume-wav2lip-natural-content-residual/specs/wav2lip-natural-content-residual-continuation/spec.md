## ADDED Requirements

### Requirement: Continuation preserves a versioned parent protocol
The implementation SHALL bind design.md's immutable parent and diagnostic hashes in a new run and inherit the original 16-record, eight-group, four-arm protocol. It MUST independently verify cached drivers and provenance before using them.

#### Scenario: The parent experiment has a failed terminal gate
- **WHEN** preparing the continuation
- **THEN** the original CONTROL_FAILED and diagnostic stage_b_authorized=false remain unchanged
- **AND** the continuation records its own revision and evidence without resuming either parent run

### Requirement: Only the known delayed control uses translated search coordinates
The implementation SHALL use natural lags -15..15 and known A_DELAY lags -10..20 on U=range(30,58), with offsets 15-j and 10-j respectively. It MUST calculate fixed-anchor damage from the uncompensated legacy delay curve and independently reproduce the bound diagnostic.

#### Scenario: A natural peak lies at legacy column 27 or 28
- **WHEN** evaluating the known five-frame audio delay
- **THEN** genuine embeddings provide the translated search support without padding or curve copying
- **AND** the correct physical offset difference is evaluated against the unchanged [-6,-4] rule

#### Scenario: A candidate receives the translated delay domain
- **WHEN** any CORRECT, WRONG or SHUFFLE score uses translated lags or a new anchor
- **THEN** validation rejects the candidate result

### Requirement: Every candidate entry point requires independently validated controls
The implementation SHALL recompute the registered controls, run the two fixed current-environment scorer parity cells and two N_REPLAY cells, and independently validate all evidence before any candidate generation. Direct stages and resume MUST enforce the same prerequisite as all.

#### Scenario: A true authorization flag lacks matching validation
- **WHEN** all, candidates or resume is invoked with missing, stale or failing control validation
- **THEN** zero candidate forwards occur and the command reports an engineering failure

#### Scenario: Current N replay differs from the cached baseline
- **WHEN** pixel identity, media identity or registered score tolerances fail
- **THEN** the continuation is BLOCKED without relaxing tolerances or replacing the parent baseline

### Requirement: Candidate execution is bounded and reproducible
The implementation SHALL generate exactly the original 48 candidates after controls pass, score each against its complete untouched natural PCM, and retain original support, generation settings and score definitions. New work SHALL stay within 50 TFG videos and 52 scoring cells, including continuation checks, with no training or new TTS.

#### Scenario: A completed candidate exists during engineering resume
- **WHEN** its complete protocol, code, media and score bindings still match
- **THEN** it is reused and counted as reused rather than generated again
- **AND** a changed binding blocks reuse instead of silently accepting stale output

### Requirement: Scientific conclusions depend on absolute natural-baseline benefit
The implementation SHALL use the inherited group bootstrap, three C/D/anchor contrasts and ordered decisions in design.md. It MUST preserve exploratory scope and all four unestablished confirmation, waveform, generalization and historical-shift flags.

#### Scenario: Correct conditioning only beats mismatched conditioning
- **WHEN** CORRECT/N fails its registered joint rule even though another contrast passes
- **THEN** the result is NO_INCREMENT_ESTABLISHED and this fixed construction stops

#### Scenario: Correct conditioning beats natural and both controls
- **WHEN** all registered tests pass and realized perturbation norms are valid
- **THEN** the result is CONTENT_RESIDUAL_SIGNAL and only independent source-group confirmation is recommended
- **AND** sentence semantics and deployable replacement are not claimed

### Requirement: Handoff includes independent artifact validation and project memory
The implementation SHALL independently reconstruct candidate and control calculations, validate media and parent immutability, report actual fresh/reused work, save a truthful self-review and update the same continuation BM entity after material stages. A failed validator MUST prevent scientific finalization.

#### Scenario: The run stops before candidate completion
- **WHEN** an engineering or control failure prevents complete candidate evaluation
- **THEN** the report and memory state the real stopping stage and keep candidate benefit unavailable
- **AND** unfinished execution tasks remain unchecked
