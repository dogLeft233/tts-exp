## ADDED Requirements

### Requirement: Frozen historical evidence and cohort
The implementation MUST bind the completed review_v15 artifacts and exactly the 12 LRS3 source groups with IDs 151–162 before new inference. It MUST distinguish historical record-weighted summaries from source-group inference and from fresh experiments.

#### Scenario: Historical evidence is loaded
- **WHEN** the downstream runner starts audit
- **THEN** it verifies input-bindings hashes, v15 completed validation, original manifest mapping and 24 historical matrix identities
- **AND** it does not treat v13 RESOURCE_WAIT or v14 metadata failure as the final experiment result

#### Scenario: A required record is unavailable
- **WHEN** one ID or source artifact is missing
- **THEN** the fixed denominator remains 12, the missing artifact is reported, and the relevant stage cannot claim complete
- **AND** no higher-scoring or more convenient record is substituted

### Requirement: Same-clock interventions and measured headroom
The implementation MUST implement the exact P1 waveform algorithms, preserve sample counts and timestamps, and use a single common headroom factor per ID across N/T/R and all variants. It MUST retain ORIGINAL and A0 as separate logical conditions.

#### Scenario: Noise exceeds PCM headroom
- **WHEN** the unquantized noise mixture has a peak above 0.98
- **THEN** the shared factor is frozen before scoring, baseline and every variant use it, and ORIGINAL→A0 effects are reported
- **AND** no per-arm limiter, clipping, loudnorm or target-Sync-C parameter search is used

#### Scenario: An acoustic operation is not valid
- **WHEN** the identity reconstruction, SNR, gain, sample clock or manipulation checks fail
- **THEN** the failure and raw measurements remain in the manifest, the relevant hypothesis is limited, and the sample is not silently discarded

### Requirement: Fixed-video evaluator experiment
The implementation MUST complete A's 180 scientific and 18 control logical cells using the three specified video types and five audio conditions, with frozen pixels, ROI and PTS within each video family.

#### Scenario: Evaluator-input effect is estimated
- **WHEN** A0 and its processed audio are scored on an identical video
- **THEN** their contrast is labelled evaluator-input effect
- **AND** it is not labelled improvement in generated lip motion

#### Scenario: Cached identical conditions exist
- **WHEN** ORIGINAL and A0 have identical PCM or features are shared
- **THEN** logical manifest rows remain complete, cache provenance is recorded, and no unnecessary inference is launched

### Requirement: Same-generator crossed experiment
The implementation MUST complete B with a provenance-bound LeapTalk configuration, 144 scientific videos, four repeat videos, 336 scientific score cells and eight control score cells. It MUST use seeds 42 and 43, a fixed portrait and same-clock A0/An/Ad audio.

#### Scenario: Crossed effects are computed
- **WHEN** q00, q01, q10 and q11 are available
- **THEN** G=q10−q00, E=q01−q00 and I=q11−q10−q01+q00 are calculated and their sum equals q11−q00
- **AND** the two transformations share only the correctly identified q00

#### Scenario: Historical generator cannot be recovered
- **WHEN** an official but different LeapTalk configuration is available
- **THEN** it is declared and frozen before generation as NEW_LEAPTALK_CONFIGURATION, with a fresh native baseline
- **AND** exact historical replication is not claimed

#### Scenario: A different generator is available locally
- **WHEN** LeapTalk dependencies are unavailable but Wav2Lip is installed
- **THEN** B remains DEPENDENCY_BLOCKED, A can progress, and Wav2Lip is not substituted as a completed B experiment

### Requirement: Complete scoring geometry and control interpretation
The implementation MUST save actual SyncNet audio/video embeddings and complete lag matrices, use common valid INTERIOR support, average time before min/median, and keep official and normalized diagnostic metrics distinct.

#### Scenario: A confidence change is decomposed
- **WHEN** C changes
- **THEN** B, D, D0, offset and unit-norm diagnostics are reported with the official C identity
- **AND** background change alone is not called perceptual improvement or evaluator bias
- **AND** raw C and C_unit are not numerically subtracted across scales; their descriptive comparison uses paired direction counts and the predeclared within-metric standardized effects

#### Scenario: A known delay is injected
- **WHEN** actual PCM is shifted by plus/minus five frames
- **THEN** MFCC/features are recomputed, offset direction and fixed-column distance are checked on common valid support, and successful detection is labelled DELAY_DETECTED
- **AND** a retained high searched confidence does not count as a failed delay detector

#### Scenario: Wrong-content retrieval is computed
- **WHEN** A's 396 planned wrong-source feature pairings are evaluated
- **THEN** the correct and wrong pairs use the same fixed zero-lag quantile rule, transcript identity is checked, and source-level summaries are used
- **AND** these pairs are not counted as independent samples or new model forwards

### Requirement: Fixed inferential family and honest limits
The implementation MUST use the six P4 primary contrasts, paired seed averaging, source-group means, common 20000-draw bootstrap indices and six-test Bonferroni intervals. Missing stages MUST NOT reduce the correction family.

#### Scenario: A primary interval crosses zero
- **WHEN** the corrected interval is wide and crosses zero
- **THEN** the finding is inconclusive rather than proof of no effect
- **AND** equivalence is only described when the entire corrected interval lies inside the predeclared ±0.200 range

#### Scenario: Fresh native advantage is not reestablished
- **WHEN** B_NATIVE_FRESH does not meet its positive-evidence rule
- **THEN** B intervention effects are still reported but the link to historical native gain is NATIVE_NOT_REESTABLISHED

### Requirement: Human assessment and training evidence boundaries
The implementation MUST build the 96-pair same-audio synchronization package and 48-pair audio-quality package, blind identities, provide empty scoring templates and implement source-level analysis. It MUST distinguish measured facts, mechanistic compatibility and causal claims.

#### Scenario: Human ratings are absent
- **WHEN** no qualifying human panel has returned ratings
- **THEN** perception and quality remain NOT_ASSESSED while the packages and automated results are delivered
- **AND** an agent does not fabricate ratings or equate Sync-C with human quality

#### Scenario: A training preference is discussed
- **WHEN** training data or checkpoint-selection evidence is incomplete
- **THEN** missing facts remain UNKNOWN, H3 stays descriptive and H4 remains causally unidentified
- **AND** paper-level statements are not automatically assigned to every actual model checkpoint

### Requirement: Resource-safe execution and independent completion audit
The implementation MUST follow P6's shared GPU lease, foreign-process checks, memory/disk reserve, bounded transient outputs and hash-complete cache. An independent validator MUST recompute primary results and reject corrupted identities or false completion claims.

#### Scenario: Resource capacity is insufficient
- **WHEN** free resources are below the frozen peak-plus-reserve plan or another GPU user is active
- **THEN** no new GPU cell starts, status is RESOURCE_WAIT, and CPU work and valid partial results are preserved
- **AND** historical data or unrelated processes are not removed to force progress

#### Scenario: Only A has completed
- **WHEN** A is valid and B has not run
- **THEN** engineering_status is PARTIAL with separate stage and scientific states
- **AND** neither tasks nor final report state that all specified automated experiments completed

#### Scenario: All automated work is valid
- **WHEN** A/B, diagnostic calculations, human packages and independent validation are complete
- **THEN** engineering_status can be AUTOMATIC_COMPLETE, with per-hypothesis evidence and explicit human/training limitations
- **AND** this status does not assert that every proposed mechanism was proved
