## Purpose

Defines an auditable same-cohort LRS3 experiment for deciding whether MFA-constrained phone-local hard-DTW improves frozen TFG behavior over MFA-linear and, only after that improvement is established, whether it changes strict natural-audio replacement behavior.

## ADDED Requirements

### Requirement: The experiment uses the frozen 133-record LRS3 comparison cohort
The experiment SHALL use exactly the ordered 133-record face-ready cohort bound by the completed MFA-linear strict-replacement protocol manifest, with ordered sample-id SHA-256 `61f8c982041cdfdded8daf8850d31e127943386ba2cb7035aa742e94cea9a973`. It SHALL bind every record to its source group, face video, untouched natural PCM, raw TTS PCM, paired MFA3 natural/TTS alignment rows, historical MFA-linear candidate, render, mux, and score artifacts by path and SHA-256. It MUST reject missing, reordered, duplicated, altered, or out-of-cohort records.

#### Scenario: Frozen cohort loads successfully
- **WHEN** all 133 ordered records and their parent artifacts match the registered paths, hashes, counts, and identities
- **THEN** the experiment accepts the cohort without selecting records from any observed score or candidate quality metric

#### Scenario: Parent identity differs
- **WHEN** a parent path, hash, sample identity, source group, order, or cohort count differs from the registered contract
- **THEN** the experiment stops before candidate generation or scoring and records an engineering failure

### Requirement: DTW is constrained to matched instances of the same phone
For each matched non-silence MFA3 phone instance, the experiment SHALL compute a deterministic hard-DTW path between natural and TTS WavLM-L6 frames using cosine distance, endpoints at the first and last local frames, monotonic horizontal/vertical/diagonal moves, and normalized-coordinate band ratio `0.5`. It SHALL use the existing ordered phone-instance matching policy and MUST NOT match across phone boundaries, labels, or instances. Each natural frame in a matched phone SHALL receive a source coordinate derived only from TTS frame indices visited for that natural frame. Natural WavLM values SHALL participate only in path-cost calculation and MUST NOT be copied, averaged, or otherwise injected into output conditioning. Silence handling SHALL remain identical to the historical MFA-linear policy.

#### Scenario: Same-phone path is valid
- **WHEN** a matched phone has one or more natural and TTS WavLM frames and admits a complete path within the fixed band
- **THEN** the output records the path, mean path cost, step counts, frame counts, source coordinates, and verifies endpoint, monotonicity, band, same-instance, and complete-natural-frame coverage invariants

#### Scenario: A phone cannot produce a valid constrained path
- **WHEN** any matched speech phone lacks source or target frames or no complete path satisfies the fixed constraints
- **THEN** the record is rejected without cross-phone, unconstrained, linear, or natural-feature fallback

#### Scenario: Repeated construction
- **WHEN** the same bound features, phone rows, and configuration are processed more than once
- **THEN** the path traces, conditioning hashes, and canonical candidate PCM hashes are identical

### Requirement: MFA-DTW and MFA-linear candidates differ only in the alignment map
The DTW candidate SHALL use the same pinned WavLM-L6 extraction interface, TTS-feature interpolation contract, HiFi-GAN decoder, no-loudness-normalization policy, exact-natural-sample-count adjustment, PCM16 canonicalization, and waveform quality checks as the historical MFA-linear candidate. The output conditioning SHALL contain only values sampled from the paired TTS WavLM sequence. The experiment SHALL report per-record DTW-versus-linear source-coordinate displacement and conditioning/candidate hash equality diagnostics.

#### Scenario: Candidate satisfies the common audio contract
- **WHEN** DTW mapping succeeds for a cohort record
- **THEN** its candidate is finite 16 kHz mono PCM16, has exactly the untouched natural waveform's sample count, passes the unchanged waveform checks, and carries complete extractor, decoder, mapping, and source provenance

#### Scenario: A non-alignment variable changes
- **WHEN** model assets, decoding policy, normalization, output-length handling, PCM encoding, or waveform gates differ from the historical MFA-linear contract
- **THEN** the experiment rejects the comparison as confounded

### Requirement: Frozen-TFG advantage is evaluated before replacement
The first scientific stage SHALL render exactly one deterministic frozen-Wav2Lip video per DTW candidate and score the `DTW video + DTW audio` cell with the pinned official file-level SyncNet V2 pipeline. It SHALL compare those 133 scores only against the same-record historical `MFA-linear video + MFA-linear audio` cells. It SHALL define paired benefits as `delta_C = SyncC_DTW - SyncC_LINEAR` and `delta_D = SyncD_LINEAR - SyncD_DTW`, cluster bootstrap by source group with 10,000 draws, and use a fixed recorded seed. The DTW TFG advantage gate SHALL pass only if the complete engineering matrix is valid and the lower bound of each two-sided 95% cluster-bootstrap confidence interval is strictly greater than zero.

#### Scenario: Both diagonal endpoints establish advantage
- **WHEN** all 133 DTW diagonal cells are valid and both paired-benefit 95% confidence intervals have lower bounds greater than zero
- **THEN** the decision is `DTW_TFG_ADVANTAGE` and the strict replacement stage becomes eligible

#### Scenario: The dual endpoint does not pass
- **WHEN** either paired-benefit interval has a lower bound less than or equal to zero
- **THEN** the scientific decision is `NO_DTW_TFG_ADVANTAGE`, strict replacement remains sealed, and no replacement mux or replacement score is created

#### Scenario: The diagonal matrix is incomplete
- **WHEN** any candidate, render, mux, score, or provenance check is incomplete or invalid
- **THEN** the decision is engineering `BLOCKED` rather than a scientific no-advantage conclusion, and replacement remains sealed

### Requirement: Strict replacement runs only after authorized promotion
The experiment SHALL require a self-hashed diagonal decision artifact recording `DTW_TFG_ADVANTAGE` before constructing replacement media. If promoted, it SHALL reuse the already-bound DTW video stream without re-rendering and replace its audio with the exact untouched natural PCM for that record. Replacement muxing SHALL copy the video stream, encode audio as PCM s16le, and verify decoded PCM byte equality against the bound natural source before SyncNet scoring.

#### Scenario: Promotion artifact is valid
- **WHEN** the diagonal gate passed and the decision artifact, parent hashes, cohort hash, and rendered-video hashes are unchanged
- **THEN** the experiment creates and scores exactly 133 `DTW video + untouched natural audio` cells

#### Scenario: Replacement is requested without promotion
- **WHEN** the diagonal decision is absent, altered, blocked, or does not equal `DTW_TFG_ADVANTAGE`
- **THEN** the experiment refuses replacement before opening an output mux and records the sealed status

#### Scenario: Natural PCM differs after muxing
- **WHEN** decoded replacement PCM differs in bytes, sample count, channels, sample rate, or sample width from the bound untouched natural source
- **THEN** the affected cell and complete replacement matrix fail engineering validation and no scientific replacement decision is issued

### Requirement: Replacement reporting separates relative improvement from replacement safety
If replacement is promoted, the experiment SHALL report three paired contrasts on the same 133 records: DTW replacement versus natural baseline, MFA-linear replacement versus natural baseline, and DTW replacement versus MFA-linear replacement. For each contrast it SHALL report mean Sync-C benefit, mean Sync-D benefit, source-group cluster-bootstrap 95% intervals, C/D/joint win counts, and per-record values. `REPLACEMENT_SAFE_GO` SHALL require both DTW-versus-natural confidence-interval lower bounds to be strictly greater than zero; superiority over MFA-linear alone MUST NOT be reported as replacement safety.

#### Scenario: DTW beats MFA-linear but not the natural baseline
- **WHEN** DTW replacement has positive paired evidence relative to MFA-linear but either DTW-versus-natural confidence interval includes or falls below zero
- **THEN** the report states that DTW improved over MFA-linear but did not establish replacement-safe gain

#### Scenario: DTW establishes replacement-safe gain
- **WHEN** both DTW-versus-natural benefit intervals have lower bounds greater than zero and all engineering checks pass
- **THEN** the report issues `REPLACEMENT_SAFE_GO` while retaining all three contrasts

### Requirement: Experiment boundaries and terminal artifacts are explicit
The experiment MUST NOT train or tune a model, alter the frozen cohort, select a path band or threshold after observing TFG scores, access sealed validation/test media, overwrite historical MFA-linear artifacts, or substitute candidate self-consistency for replacement evidence. Every attempted stage SHALL write a machine-readable summary, decision, failure ledger, parent bindings, media-access ledger, and next-stage authorization state to a distinct run root.

#### Scenario: Diagonal failure ends the experiment
- **WHEN** the diagonal stage reaches valid `NO_DTW_TFG_ADVANTAGE`
- **THEN** terminal artifacts state that LRS3 phone-local hard-DTW did not establish frozen-TFG advantage over MFA-linear and that replacement was intentionally not run

#### Scenario: Engineering failure occurs
- **WHEN** any invariant, asset, media, or execution check fails
- **THEN** the terminal artifact distinguishes engineering `BLOCKED` from scientific `NO_GO`, preserves the failure details, and authorizes no downstream stage
