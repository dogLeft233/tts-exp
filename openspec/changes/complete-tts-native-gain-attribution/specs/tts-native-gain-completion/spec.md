## ADDED Requirements

### Requirement: Immutable parent evidence and explicit continuation
The implementation MUST create a new continuation run, verify the parent evidence transitively, and reuse valid A artifacts without altering their bytes or production identity. It MUST inherit the original scientific protocol and bind this completion specification separately.

#### Scenario: Parent A artifacts are reused
- **WHEN** parent files and their transitive hashes are valid
- **THEN** all 180 scientific and 18 control identities remain auditable with original producer hashes and new validator hashes
- **AND** current code is not written into the parent snapshot to pretend the old artifacts were newly produced

#### Scenario: An imported file has changed
- **WHEN** a PCM, video, feature, matrix or support binding differs
- **THEN** import fails with the affected identities and cannot silently trust the old validation status

### Requirement: Executable and provenance-bound LeapTalk backend
The implementation MUST provide and exercise an actual LeapTalk adapter and frozen-ROI scoring path, verify all loaded model components and runtime settings, and distinguish new configuration inference from historical replication.

#### Scenario: Complete dependencies are available
- **WHEN** official-family source, revisions, all loaded weights and runtime are bound
- **THEN** the generator and crossed scorer execute without placeholder exceptions or requiring an unimplemented crop command
- **AND** the adapter records actual consumed PCM, frontend tensors, initialization and chunk-to-frame timing evidence

#### Scenario: Only a base model or command template is supplied
- **WHEN** LeapTalk LoRA, audio projection, required decoder/config or official provenance is unverified
- **THEN** the backend cannot claim ready or complete and cannot substitute Wav2Lip

### Requirement: Independent repeat controls and resumable identities
The implementation MUST preserve 144 scientific and four repeat video identities with separate immutable outputs, logs and sidecars, and MUST commit and resume cells by validated content identity.

#### Scenario: A deterministic repeat matches the main video
- **WHEN** a separate invocation produces identical pixels
- **THEN** both execution records and separate paths are retained, no baseline is overwritten, and the repeat is not a cache alias of the main cell

#### Scenario: Execution is interrupted
- **WHEN** a cell has incomplete media or missing evidence
- **THEN** it remains uncommitted and the next run retries it while skipping only fully verified completed cells

### Requirement: Complete crossed experiment and common geometry
The implementation MUST execute 336 scientific and eight control scores on the exact cohort and seeds, preserve same-clock audio, freeze baseline ROI trajectories, and use shared valid support across both seeds and all seven scientific cells per seed.

#### Scenario: A candidate has unusable ROI or padding
- **WHEN** the frozen baseline trajectory does not support a candidate or the model pads its output
- **THEN** invalid ROI is explicit, padded frames are excluded from support, and candidate-specific tracking or repeated-frame support is forbidden

#### Scenario: Score curves and controls are evaluated
- **WHEN** real features and complete lag matrices are available
- **THEN** time is averaged before min/median, repeat tolerances and actual +200ms PCM controls are verified, and failed or uninformative controls prevent automatic completion

### Requirement: Registered statistics and generation claims
The implementation MUST independently calculate all six registered contrasts, 96 four-cell decompositions and required secondary diagnostics using paired seed averaging, twelve source groups and the shared 20000-draw bootstrap. It MUST base generation claims on B evidence and valid controls.

#### Scenario: A is positive but B is inconclusive
- **WHEN** A shows evaluator sensitivity but the B generation intervals cross zero without equivalence
- **THEN** H2 remains inconclusive, no generation benefit is inferred from A, and native attribution requires a positive fresh native baseline

#### Scenario: A required B cell is missing
- **WHEN** any registered source, seed or cell is absent
- **THEN** the relevant primary contrast is INCOMPLETE, the correction family remains six, and subset analyses are labelled exploratory

### Requirement: Resource-aware execution and honest status
The implementation MUST separate GPU peak memory from disk budgets, use a shared device lease with live foreign-process checks, preserve typed errors and return nonzero for incomplete requested work.

#### Scenario: Resources become unavailable
- **WHEN** a foreign process, insufficient VRAM or insufficient filesystem space is detected
- **THEN** no new GPU cell starts, completed outputs are retained, and the stage reports RESOURCE_WAIT without killing other processes or deleting historical data

#### Scenario: The all-stage command contains a blocked stage
- **WHEN** aggregation encounters blocked or unimplemented mandatory work
- **THEN** stdout, exit code, final status and completed counts consistently report incompletion instead of default COMPLETE

### Requirement: Playable blinded assessment and real ratings support
The implementation MUST provide 96 plus ten hidden sync pairs and 48 plus five hidden quality pairs with verified playable media, random presentation and private-only condition and repetition mappings. Sync alternatives MUST actually play the same A0 audio.

#### Scenario: B videos are missing
- **WHEN** only pair identities exist
- **THEN** the sync package is PARTIAL_MEDIA and cannot satisfy the complete-package gate

#### Scenario: No human ratings are available
- **WHEN** playable packages are valid but ratings are empty
- **THEN** perception and quality remain explicitly not assessed and no synthetic human result is created

#### Scenario: Human ratings arrive
- **WHEN** at least three common raters supply valid formal ratings
- **THEN** direction mapping, ties, missing responses, within-source aggregation, source bootstrap and hidden-repeat consistency are computed without counting repeats as new observations
- **AND** rebuilding packages does not overwrite received ratings

### Requirement: Independent success-path validation and adversarial checks
The validator MUST independently recompute B curves, controls, all six primary intervals, decompositions and evidence states from the bound artifacts rather than trusting summary fields, self-hashes or producer decision functions.

#### Scenario: Valid-looking metadata contains exchanged cells
- **WHEN** q01 and q10 are swapped and JSON self-hashes regenerated
- **THEN** transitive video, PCM, ROI, seed and matrix identity checks reject the run

#### Scenario: A complete result is forged
- **WHEN** a seed, matrix, distinct repeat invocation, valid control, model component or playable media is missing or altered
- **THEN** validation fails even if all counts and stage statuses say COMPLETE

#### Scenario: All automatic work is actually valid
- **WHEN** parent A, 148 generated videos, 344 B scores, controls, six contrasts, diagnostics and playable assessment packages pass independent checks
- **THEN** AUTOMATIC_COMPLETE is allowed regardless of whether the scientific findings are positive, negative or inconclusive
- **AND** human findings remain separate from automatic completion
