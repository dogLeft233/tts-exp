## Purpose

Defines a fail-closed evaluation of a frozen shared MFA-linear TTS waveform adapter on source-group-disjoint records and a strict natural-audio-replacement audit through a frozen talking-face generator.

## ADDED Requirements

### Requirement: Validated shared-checkpoint prerequisite
The evaluator SHALL accept only a checkpoint produced by a fresh `P2_SHARED_FOUR` execution of `prototype-mfa-linear-real-video-sync` whose immutable run validates successfully, whose terminal status is `FIXED_DATA_SHARED_TTS_ONLY_SYNC_CONSTRUCTED`, and whose step-100 checkpoint hash, four ordered fit-record IDs, four source groups, configuration, model initialization, and frozen SyncNet hash agree with that run's artifacts. It MUST NOT accept the P1 step-20 checkpoint, a warm start, an intermediate checkpoint, a manually copied state dict without provenance, or a P2 run with modified constants.

#### Scenario: Passing P2 prerequisite
- **WHEN** the evaluator receives a hash-valid P2 run with the required terminal status and internally consistent bindings
- **THEN** it records those bindings and may proceed to cohort locking

#### Scenario: Missing or invalid P2 prerequisite
- **WHEN** the P2 run is absent, failed, unvalidated, modified, or provenance-inconsistent
- **THEN** the evaluator emits `BLOCKED_P2_NOT_PASSED`, creates no heldout candidate waveform, and performs no downstream render

### Requirement: Score-independent adapter-heldout cohort
Before loading the adapter checkpoint or generating candidate audio, the evaluator SHALL freeze exactly eight evaluation records from exactly eight source groups disjoint from every P2 fit source group. “Heldout” SHALL mean adapter-heldout within the existing fit-only LRS3 asset universe, not a sealed dataset validation or test split.

Eligibility SHALL be determined only from locked source metadata and natural-reference preflight: `protocol_split=train`, unique sample and source-group identity, mono 16 kHz finite exact-length MFA-linear waveform, verified 96-frame/61,440-sample mapping, verified frozen tracked-video geometry, and an unambiguous detached natural-reference 31-offset curve whose best/second gap exceeds `2q` for `q=0.001`. Candidate outputs, adapter losses, generated videos, and baseline/candidate SyncNet scores MUST NOT influence eligibility.

Among eligible records, the evaluator SHALL select one record per source group and then eight groups by ascending bytewise SHA256 keys derived from the fixed salt `mfa-linear-sync-transfer-v1`, source group, and sample ID. It SHALL persist the complete eligible universe, exclusions, keys, selected IDs, source groups, paths, hashes, and denominator before candidate inference. Missing or failed selected records MUST NOT be replaced after the manifest is frozen.

#### Scenario: Cohort freezes successfully
- **WHEN** at least eight source-group-disjoint records satisfy every predeclared eligibility rule
- **THEN** the evaluator writes an immutable ordered eight-record manifest before loading the adapter checkpoint

#### Scenario: Cohort denominator cannot be met
- **WHEN** fewer than eight distinct non-P2 source groups are eligible or a selected asset cannot be hash-verified
- **THEN** the evaluator emits `BLOCKED_COHORT_LOCK`, performs no candidate inference, and does not reduce or substitute the denominator

#### Scenario: Score-based selection is attempted
- **WHEN** a record is ranked, filtered, substituted, or excluded using adapter, baseline/candidate SyncNet, or talking-face results
- **THEN** artifact validation fails with `OUTCOME_SELECTION_VIOLATION`

### Requirement: Frozen TTS-only inference and natural-audio isolation
For each selected record, the evaluator SHALL run the fixed P2 adapter exactly once on only the exact MFA-linear waveform and SHALL produce one exact-length candidate waveform. It MUST NOT optimize, backpropagate, update model or optimizer state, select a checkpoint, retry a seed, ensemble outputs, tune a threshold, or expose video, natural waveform/features, identity, transcript features, target offset, or any second audio tensor to the adapter.

Natural audio MAY be loaded outside the adapter only to construct the detached coordinate artifact, to serve as the natural driver/evaluation arm in the replacement matrix, and to verify provenance. It MUST NOT enter adapter inference, a preservation objective, or any model-selection path. The adapter parameter/buffer hash SHALL remain identical before and after all eight records.

#### Scenario: Valid heldout inference
- **WHEN** the frozen adapter receives a selected record
- **THEN** its only model argument is the record's `[1,1,61440]` MFA-linear waveform and its output has exactly the same shape and sample count

#### Scenario: Natural or record-specific side channel reaches the adapter
- **WHEN** natural audio, video features, identity, transcript features, target labels, or a second waveform are present in the adapter payload or forward call
- **THEN** the evaluator emits `NATURAL_OR_SIDE_CHANNEL_LEAKAGE` and invalidates the run

#### Scenario: Checkpoint state changes during inference
- **WHEN** the adapter parameter/buffer hash differs after evaluation
- **THEN** the evaluator emits `FROZEN_ADAPTER_MUTATION` and no scientific decision is available

### Requirement: Heldout real-video comparison on fixed coordinates
The evaluator SHALL compare the exact MFA-linear baseline and frozen-adapter candidate against the same canonical 96-frame real-video segment for each record. Baseline and candidate conditions SHALL differ only in signed PCM16 audio; they SHALL decode to identical visual frame hashes and use the same 91 SyncNet windows, all 31 internal shifts `[-15,+15]`, raw Euclidean distances, and sign convention `official AV offset = -internal shift`.

The differentiable/proxy curve is optional for evaluation, but any persisted proxy curve SHALL match the independent official file-level SyncNet V2 curve at every offset within `q=0.001`. Scientific metrics SHALL be recomputed from the official 31-point curves:

```text
D(x) = min(curve_x)
C(x) = median(curve_x) - min(curve_x)
D_gain = D(MFA) - D(candidate)
C_gain = C(candidate) - C(MFA)
```

Each candidate SHALL also pass exact length, finite waveform/PCM/embedding/curve values, pointwise residual peak `<=0.05`, normalized candidate-to-MFA log-mel distance `<=0.10`, and PCM saturation fraction `<=1e-4`. The detached natural curve SHALL define `s_ref`; a record-level target match requires the candidate's unique official minimum to occur at `s_ref` with a best/second gap greater than `2q`.

#### Scenario: Valid paired real-video scoring
- **WHEN** both audio conditions preserve the locked video and frontend contracts
- **THEN** the evaluator persists both full official curves, D/C values, offsets, gaps, QC, hashes, and positive-is-better D/C gains

#### Scenario: Coordinate or official-path mismatch
- **WHEN** frame hashes, PCM hashes, window counts, offset signs, checkpoint hashes, or proxy/official values violate the frozen contract
- **THEN** the evaluator emits an engineering failure and does not classify the record as a transfer success

### Requirement: Adapter-heldout transfer decision
A record SHALL count as a heldout real-video success only when `D_gain >= 3q`, `C_gain >= 3q`, the candidate target match is true, and every waveform/provenance/official-score predicate passes. The stage SHALL emit exactly one scientific status:

- `ADAPTER_HELDOUT_REAL_VIDEO_TRANSFER_OBSERVED` when all eight records complete, at least six of eight are record-level successes, and the across-record medians of both `D_gain` and `C_gain` are at least `3q`;
- `NO_ADAPTER_HELDOUT_REAL_VIDEO_TRANSFER` when all eight records are engineering-valid but the preceding scientific gate does not pass;
- `REAL_VIDEO_TRANSFER_NOT_EVALUATED` when the cohort, inference, scoring, or artifact graph is incomplete or invalid.

The evaluator SHALL report all record values and descriptive medians. It MUST NOT treat eight records as a population estimate or use resampling significance to enlarge the claim.

#### Scenario: Transfer gate passes
- **WHEN** the complete eight-record evidence satisfies the six-of-eight and both median gates
- **THEN** the decision is `ADAPTER_HELDOUT_REAL_VIDEO_TRANSFER_OBSERVED` and the strict replacement stage may run

#### Scenario: Complete evidence misses the transfer gate
- **WHEN** all records are valid but fewer than six jointly pass or either median gain is below `3q`
- **THEN** the decision is `NO_ADAPTER_HELDOUT_REAL_VIDEO_TRANSFER` and the replacement stage is recorded as `REPLACEMENT_NOT_RUN_REAL_VIDEO_GATE_FAILED`

### Requirement: Frozen talking-face render arms
The replacement stage SHALL run only after `ADAPTER_HELDOUT_REAL_VIDEO_TRANSFER_OBSERVED`. For each of the same eight records, it SHALL render exactly three videos from the same hash-locked canonical tracked-face segment using one frozen Wav2Lip GAN checkpoint and fixed invocation settings:

- `G_N`: driven by the untouched exact natural segment;
- `G_B`: driven by the exact MFA-linear baseline;
- `G_C`: driven by the frozen-adapter candidate.

All three driver waveforms SHALL be mono 16 kHz, exactly 61,440 samples, and independently hash-verified. The render checkpoint, code, environment, source frames, frame rate, crop geometry, padding, smoothing policy, and inference parameters SHALL be identical across arms. Rendered videos SHALL have equal nonzero frame counts and matching geometry. No arm may be rerendered after scores are observed.

#### Scenario: Three render arms complete
- **WHEN** all frozen inputs and the Wav2Lip environment validate
- **THEN** exactly one immutable `G_N`, `G_B`, and `G_C` video is produced per record with complete hashes and invocation provenance

#### Scenario: One render fails or changes protocol
- **WHEN** an arm is missing, rerun, uses a different source frame/checkpoint/setting, or has incompatible geometry or frame count
- **THEN** the stage emits `RENDER_MATRIX_INCOMPLETE` and no replacement claim is made

### Requirement: Complete strict 3×3 replacement matrix
For each record, the evaluator SHALL discard any renderer-muxed audio and create all nine combinations of driver video `G ∈ {G_N,G_B,G_C}` and evaluation audio `E ∈ {E_N,E_B,E_C}` by stream-copying the exact same video bytes within each driver row and muxing signed PCM16 audio:

```text
G_N_E_N  G_N_E_B  G_N_E_C
G_B_E_N  G_B_E_B  G_B_E_C
G_C_E_N  G_C_E_B  G_C_E_C
```

`E_N`, `E_B`, and `E_C` SHALL be the same natural, MFA-linear, and candidate waveforms bound to that record's render arms. Every mux SHALL verify demuxed PCM identity, decoded frame identity to its driver video, duration/window contract, and frozen official SyncNet V2 checkpoint before storing the full 31-point curve. All 72 cells SHALL be present and valid; cells MUST NOT be dropped because of their values.

`G_B_E_N` and `G_C_E_N` SHALL be the authoritative replacement cells. `G_N_E_N` SHALL be an oracle/reference control. The six cells evaluated with `E_B` or `E_C`, including the native diagonals `G_B_E_B` and `G_C_E_C`, SHALL be diagnostics only and MUST NOT contribute to promotion.

#### Scenario: Full matrix is valid
- **WHEN** three videos and three audios per record produce all nine verified conditions for all eight records
- **THEN** the evaluator stores 72 uniquely keyed official score rows and permits replacement analysis

#### Scenario: Diagonal-only improvement occurs
- **WHEN** candidate-native or other diagnostic cells improve but `G_C_E_N` does not beat `G_B_E_N`
- **THEN** the evaluator reports the diagnostic movement but does not count a replacement success

#### Scenario: Matrix cell is missing or provenance-invalid
- **WHEN** any expected cell, hash, curve, or scorer binding is absent or inconsistent
- **THEN** the replacement status is `REPLACEMENT_NOT_EVALUATED`

### Requirement: Strict natural-audio-replacement decision
For each record, authoritative positive-is-better gains SHALL be computed only from the natural-audio column:

```text
replacement_D_gain = D(G_B_E_N) - D(G_C_E_N)
replacement_C_gain = C(G_C_E_N) - C(G_B_E_N)
```

The record's oracle downstream internal shift SHALL be the unique minimum of `G_N_E_N`, requiring a best/second gap greater than `2q`. A record SHALL count as a strict replacement success only when both replacement gains are at least `3q`, the unique minimum of `G_C_E_N` equals that oracle shift with a gap greater than `2q`, and every render/mux/scorer predicate passes.

The stage SHALL emit exactly one scientific status:

- `FROZEN_WAV2LIP_REPLACEMENT_TRANSFER_OBSERVED` when all 72 cells are valid, at least six of eight records are strict replacement successes, and median replacement D and C gains are each at least `3q`;
- `NO_FROZEN_WAV2LIP_REPLACEMENT_TRANSFER` when all 72 cells are valid but the scientific gate does not pass;
- `REPLACEMENT_NOT_EVALUATED` when execution or provenance is incomplete or invalid.

#### Scenario: Strict replacement gate passes
- **WHEN** the natural-audio-column evidence satisfies the six-of-eight, median-gain, and oracle-offset gates
- **THEN** the terminal scientific result is `FROZEN_WAV2LIP_REPLACEMENT_TRANSFER_OBSERVED`

#### Scenario: Native-pair gain does not survive replacement
- **WHEN** `G_C_E_C` improves over a diagnostic comparator but the natural-audio-column gate fails
- **THEN** the terminal scientific result is `NO_FROZEN_WAV2LIP_REPLACEMENT_TRANSFER`

### Requirement: Immutable artifacts and fail-closed validation
The evaluator SHALL write to a new create-once run root and SHALL bind every parent artifact, source asset, model, executable, configuration, waveform, video, matrix row, curve, and decision by canonical content hash. A resume operation MAY read a fully complete hash-valid terminal run but MUST NOT append missing stages, overwrite files, change decisions, or retry failed cells. Partial, stale, duplicate, non-finite, or conflicting artifacts SHALL fail validation.

Engineering status, real-video transfer status, and replacement status SHALL be represented separately. Scientific failure SHALL not be mislabeled as engineering failure, and engineering incompleteness SHALL never be converted into a negative scientific result.

#### Scenario: Hash-valid terminal resume
- **WHEN** a complete run's artifact graph and terminal decision hashes validate
- **THEN** resume returns the stored decision without executing inference, rendering, muxing, or scoring

#### Scenario: Partial or corrupted run is resumed
- **WHEN** any required artifact is missing, changed, duplicated, or inconsistent
- **THEN** resume fails closed and does not repair or overwrite the run

### Requirement: Bounded interpretation
A real-video transfer pass SHALL support only the statement that the one frozen P2 adapter improved the frozen official SyncNet metrics on the preregistered eight adapter-heldout source groups under this protocol. A replacement pass SHALL additionally support only that this improvement transferred through the one hash-locked Wav2Lip/SyncNet stack under strict natural-audio replacement on the same cohort.

Neither status SHALL imply population or sealed-test generalization, performance on another TTS system, architecture superiority, audible synchronization, perceptual quality, intelligibility, content or speaker preservation, production readiness, replacement safety across talking-face generators, or causal proof that SyncNet gain reflects human-visible lip synchronization.

#### Scenario: Report is generated
- **WHEN** any terminal result is written
- **THEN** the report includes the applicable bounded statement and explicitly lists the unsupported claims
