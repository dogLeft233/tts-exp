## Purpose

Defines a concise two-phase follow-up that first diagnoses the failed token-level masked-reconstruction contrast and then tests its downstream effect through direct-mel frozen-Wav2Lip rendering and untouched-natural-audio SyncNet replacement scoring.

## ADDED Requirements

### Requirement: Read-only loss decomposition and negative-group diagnosis

The experiment SHALL read the completed exploratory parent run without modifying or retraining it. It SHALL decompose every modality and token contrast into patch and `0.25 × velocity` contributions, verify that the decomposition reproduces recorded total-loss contrasts, and preserve the parent's mask→record→group→seed aggregation. It SHALL compare `6ORDQFh0Byw` and `6yR5OUVb2gY` with the six token-positive groups by seed, record, phone, core duration, TTS/natural duration ratio, paired-to-centroid feature distance, feature temporal variation, and paired-to-centroid prediction RMS.

#### Scenario: A token-negative group is diagnosed
- **WHEN** all required parent cells and predictions are present
- **THEN** the report identifies the observed patch/velocity and data-composition contributors without presenting the post-hoc diagnosis as causal evidence

#### Scenario: Loss arithmetic does not reproduce
- **WHEN** `patch_gain + 0.25 * velocity_gain` differs from the recorded total gain by more than `1e-6`
- **THEN** Phase A is incomplete and TFG execution does not start

### Requirement: Small score-independent eight-group cohort

The TFG probe SHALL contain one record from each of the parent's eight evaluation source groups, selected by maximum evaluation-mask count and the salted sample-ID hash tie-break in `design.md`. Selection SHALL NOT use losses, predictions, Phase-A labels, or SyncNet scores. Both observed token-negative groups SHALL remain in the cohort.

#### Scenario: Cohort is frozen
- **WHEN** the parent mask manifest is valid
- **THEN** exactly eight records/eight groups are written before any TFG score exists

### Requirement: Natural-context mel patching with three matched arms

For every selected record and seed, the experiment SHALL create `PAIRED_TTS`, `PHONE_CENTROID`, and `NAT_ONLY` full-record mel drivers by replacing only evaluated target-core frames with the corresponding saved predictions and leaving all other frames natural. It SHALL also create one unchanged `NATURAL_MEL` driver per record. Overlaps, denormalization, Wav2Lip-range clamping, and clamp reporting SHALL follow `design.md`. No reconstructor training or checkpoint selection is allowed.

#### Scenario: A model-arm driver is built
- **WHEN** all saved prediction cores for a record/seed/arm are available
- **THEN** uncovered mel frames equal the natural source, core provenance is recorded, and the driver belongs to the complete 80-cell matrix

### Requirement: Direct-mel frozen-Wav2Lip replacement evaluation

The experiment SHALL use a local wrapper to provide precomputed mel chunks to the existing frozen Wav2Lip model while retaining its normal chunk indexing, face preprocessing, inference settings, and video output path. Vendor code SHALL remain unchanged. A natural-mel chunk parity check SHALL pass before rendering. Every rendered video SHALL then be scored after its video stream is muxed with the corresponding untouched natural audio using the existing frozen official SyncNet V2.

#### Scenario: Direct-mel parity passes
- **WHEN** natural audio and its stored natural mel are processed for the parity record
- **THEN** the direct and ordinary Wav2Lip mel chunks are numerically identical and the 80-cell render may begin

#### Scenario: Replacement audio differs
- **WHEN** the decoded scoring audio is not identical to the untouched natural source under the existing PCM contract
- **THEN** that cell is invalid and the scientific status is `NOT_EVALUATED`

### Requirement: Group-level exploratory TFG interpretation

The experiment SHALL compute positive-is-better paired token and modality gains for Sync-C and Sync-D, take the median across seeds within each source group, and report eight-group medians, wins, seed summaries, and deterministic whole-group bootstrap intervals. It SHALL assign exactly one status using the rules in `design.md`: `EXPLORATORY_TOKEN_SIGNAL`, `EXPLORATORY_MODALITY_ONLY`, `NO_EXPLORATORY_TFG_GAIN`, or `NOT_EVALUATED`.

#### Scenario: Paired TTS beats the phone centroid robustly
- **WHEN** both token-gain interval lower bounds exceed zero and at least seven of eight groups are positive on both Sync metrics
- **THEN** status is `EXPLORATORY_TOKEN_SIGNAL` with the direct-mel and non-waveform claim boundary

#### Scenario: Only modality gain is robust
- **WHEN** the token rule fails but the corresponding paired-versus-`NAT_ONLY` rule passes
- **THEN** status is `EXPLORATORY_MODALITY_ONLY` and no fine-grained token-retention claim is made
