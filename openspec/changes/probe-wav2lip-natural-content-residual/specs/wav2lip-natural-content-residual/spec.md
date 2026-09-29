## ADDED Requirements

### Requirement: Inputs and experimental scope are frozen
The experiment SHALL use design.md's 16 previously inspected records, eight source groups, three cached seeds and bound parent assets. It SHALL distinguish the 61440-sample unmasked natural feature support from the complete natural replacement audio. Missing or inconsistent assets MUST block execution without sample substitution.

#### Scenario: A parent recording exceeds the cached feature duration
- **WHEN** the original natural recording is longer than 3.84 seconds
- **THEN** the implementation verifies the cached mel against its exact first 61440 samples and retains the full recording in the replacement audio
- **AND** it labels conclusions as a short-support probe, not whole-record improvement

### Requirement: Auxiliary residuals preserve the natural baseline
The experiment SHALL construct exactly N, CORRECT, WRONG and SHUFFLE using design.md's cached same-model prediction contrasts, seed averaging, deterministic permutation, norm scaling and common bounded strength. N MUST bypass masked reconstruction and all candidates MUST equal N outside the frozen mask union.

#### Scenario: The auxiliary contribution is zero
- **WHEN** the residual is identically zero
- **THEN** the constructed driver is exactly the original unmasked natural mel
- **AND** the implementation does not substitute the degraded NAT_ONLY prediction

#### Scenario: Clipping invalidates matched perturbation norms
- **WHEN** either control's realized norm is outside the predefined ratio bounds
- **THEN** norm_control_valid is false and a content-specific conclusion is prohibited
- **AND** the agent does not tune strengths or replace controls after seeing outcomes

### Requirement: Generation and replacement have auditable media identity
The experiment SHALL render the same static face with the frozen Wav2Lip direct-mel interface, 93 frames at 25fps and lossless output. It SHALL score all target videos against their complete untouched natural PCM with the frozen full-frame CPU SyncNet entrance.

#### Scenario: Parent boxes use a different coordinate order
- **WHEN** a parent xyxy box is used to construct the static face
- **THEN** it is validated in the original frame and converted explicitly before any generation interface is called

#### Scenario: The scoring audio is a timing control
- **WHEN** V_N is scored against A_DELAY
- **THEN** media and scorer source_audio bindings both reference the exact delayed PCM
- **AND** this cell is excluded from replacement efficacy contrasts

### Requirement: Controls precede candidate execution
The experiment SHALL finish and independently validate design.md's parity, repeated-forward and fixed-anchor timing controls before candidate generation. Complete execution SHALL remain within 66 TFG videos and 84 scoring cells, without training or new TTS.

#### Scenario: A free-offset score compensates the audio delay
- **WHEN** the delayed audio retains its freely optimized C or D
- **THEN** sensitivity is judged by the registered natural-anchor damage and expected offset response rather than requiring free-score degradation

#### Scenario: Stage A fails
- **WHEN** a required engineering or sensitivity check fails
- **THEN** the implementation records the appropriate BLOCKED or CONTROL_FAILED status and does not run Stage B

### Requirement: Claims depend on gains over unmasked natural
The experiment SHALL compute design.md's shared U support, natural anchor, positive benefit signs, eight-group paired bootstrap and hierarchical terminal decision. It MUST keep exploratory replacement signals distinct from independent confirmation and content specificity.

#### Scenario: Correct conditioning only beats a degraded or mismatched control
- **WHEN** CORRECT fails the predeclared comparison against unmasked N
- **THEN** the result is NO_INCREMENT_ESTABLISHED regardless of its advantage over WRONG or SHUFFLE

#### Scenario: Free C and D improve but the natural anchor does not
- **WHEN** fixed-anchor benefit fails the registered rule
- **THEN** replacement superiority is not established and the agent does not search a new offset or window

#### Scenario: All registered contrasts pass
- **WHEN** CORRECT beats N and both controls, with valid norm controls and independent artifact validation
- **THEN** the outcome may be CONTENT_RESIDUAL_SIGNAL within the declared exploratory scope
- **AND** waveform training, historical shift gate repair and cross-model generalization remain unestablished

### Requirement: Execution is independently checkable and retained in project memory
The implementation SHALL provide staged resumable execution, hash-bound artifacts, focused regression tests, an independent numerical validator and a truthful self-review. It SHALL update and read back the same experiment BM note after each material stage and at completion.

#### Scenario: Cached output or claimed decision is inconsistent
- **WHEN** a hash, reconstructed driver, matrix, media identity, statistic or gate disagrees
- **THEN** independent validation rejects finalization and preserves the mismatch evidence
