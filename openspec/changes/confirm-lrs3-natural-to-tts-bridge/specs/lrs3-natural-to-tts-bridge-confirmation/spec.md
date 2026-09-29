## Purpose

Defines a small, control-calibrated confirmation of whether a natural-timed audio bridge can move toward a frozen TTS acoustic target, drive Wav2Lip differently enough to improve the primary sync endpoint, and remain compatible with untouched natural audio.

## ADDED Requirements

### Requirement: Confirmation uses a fixed cohort disjoint from discovery
The experiment SHALL bind the ordered 133-record parent protocol with SHA-256 `e8c50a459c8197c86ec11cf06a01c000e3f62def981d16d8ed9275a38a473784`, the matching replacement manifest with SHA-256 `157cd678f5f03be6eae9986a82babfdfb1f1a44e611bb0b86476c6e19b7eb272`, and the discovery final artifact with file SHA-256 `9bab2a853edda0cdb16e79e2b9a8c6d32ce3e36d721e4a1c1140306d9ffac8c1`. It SHALL select the second ordered occurrence of each source group from the parent cohort, yielding exactly 22 records from 22 groups with ordered sample-ID SHA-256 `5999f6bae8ba6b17c2e316b5f1e1f9e6d73b402a34cc8e4cfe482bcc714f5b72`. The selected IDs SHALL be disjoint from the 23-record discovery cohort.

#### Scenario: Fixed confirmation cohort is accepted
- **WHEN** all parent hashes, ordered records, source groups, bound media, disjointness, and the registered selected-ID hash match
- **THEN** Stage 00 accepts the 22-record cohort before decoding media or reading confirmation outcomes

#### Scenario: Cohort identity differs
- **WHEN** any parent, record, order, group, asset, disjointness, count, or hash check differs
- **THEN** the experiment records engineering `BLOCKED` and generates no candidate media

### Requirement: Four audio arms are deterministic and fixed
For each record the experiment SHALL materialize exactly four 16-kHz mono PCM16 arms in this order: `N`, `N_REPEAT`, `LOCAL_SWAP`, and `BRIDGE_075`. `N` and `N_REPEAT` SHALL have decoded PCM bytes exactly equal to the bound untouched natural audio. `N_REPEAT` SHALL be rendered independently from `N` using a distinct work directory.

For natural PCM `x` of length `L`, `LOCAL_SWAP` SHALL use boundaries `floor(L/4)`, `floor(L/2)`, and `floor(3L/4)` and concatenate the first quarter, third quarter, second quarter, and fourth quarter without changing sample values or total length.

`BRIDGE_075` SHALL use the frozen discovery STFT contract and `alpha=0.75` to combine 25% natural log magnitude and 75% bound MFA-linear log magnitude while retaining natural phase and exact natural length. It SHALL apply exactly one global RMS match, at most one global peak attenuation to `0.999`, and one PCM16 canonicalization.

#### Scenario: Complete arm set is generated
- **WHEN** the bound natural and MFA-linear inputs satisfy the registered format, common-length, and hash contracts
- **THEN** Stage 01 writes all four arms with input hashes, output hashes, construction metadata, and waveform QC

#### Scenario: An arm or transform differs
- **WHEN** an arm, order, local-swap boundary, bridge strength, STFT option, scale rule, output length, sample format, or input identity differs
- **THEN** the complete scientific experiment is engineering `BLOCKED`

### Requirement: Diagnostics are frozen before confirmation scores
Before reading any confirmation SyncNet output, the experiment SHALL compute official Wav2Lip mel tensors for natural, the bound MFA-linear target, and `BRIDGE_075`. It SHALL report directional progress, mel distances, orthogonal residual, RMS, peak, clipping, DC offset, and all applied scale factors. Candidate diagnostics MUST NOT be used to repair, filter, or rerun any record.

A bridge SHALL satisfy movement only when at least 20 of 22 records have directional progress of at least `0.15` and the lower two-sided percentile 95% confidence bound of mean progress is strictly greater than `0.15`.

#### Scenario: Diagnostics freeze successfully
- **WHEN** all 22 records produce finite, shape-compatible official Wav2Lip mel diagnostics before Stage 03
- **THEN** scoring may proceed with the frozen candidates and diagnostics

#### Scenario: Diagnostics are incomplete or outcome-adaptive
- **WHEN** any diagnostic is missing, non-finite, shape-incompatible, computed after score access, or used to alter a candidate
- **THEN** the experiment records engineering `BLOCKED`

### Requirement: The media matrix is narrow and complete
The experiment SHALL freshly render all four arms for all 22 records using the frozen Wav2Lip checkpoint, command, face video, and one shared face geometry per record, producing exactly 88 videos. It SHALL score exactly six cells per record: `V_N/A_N`, `V_N_REPEAT/A_N`, `V_LOCAL_SWAP/A_N`, `V_LOCAL_SWAP/A_LOCAL_SWAP`, `V_BRIDGE_075/A_N`, and `V_BRIDGE_075/A_BRIDGE_075`, producing exactly 132 unique cells.

Every mux SHALL copy the video stream, encode audio as 16-kHz mono PCM s16le, and verify decoded PCM byte equality and video elementary-stream identity before pinned official SyncNet V2 scoring.

#### Scenario: Complete matrix is accepted
- **WHEN** all 88 videos and 132 registered cells pass model, command, geometry, media-identity, and scorer checks
- **THEN** Stage 03 accepts the matrix for registered analysis

#### Scenario: A media cell is invalid
- **WHEN** any required cell is absent, duplicated, cross-record, extra, score-retried, or fails a frozen binding or bitstream check
- **THEN** the experiment records engineering `BLOCKED` and issues no scientific result

### Requirement: Positive and negative controls validate the endpoint
The experiment SHALL use 10,000 deterministic bootstrap draws over the 22 source groups with seed `20260904`. All confidence intervals SHALL be two-sided percentile 95% intervals and all threshold comparisons SHALL be strict.

`N_REPEAT` SHALL pass repeatability when the lower confidence bounds of both positive-is-better gaps relative to `V_N/A_N` are greater than `-0.10` and at least 20 of 22 offsets differ by no more than one frame.

`LOCAL_SWAP` own-audio validity SHALL pass when `V_LOCAL_SWAP/A_LOCAL_SWAP` is non-inferior to `V_N/A_N` under the same two `-0.10` bounds and 20-of-22 offset rule. Define `damage_C = SyncC(V_LOCAL_SWAP,A_LOCAL_SWAP) - SyncC(V_LOCAL_SWAP,A_N)` and `damage_D = SyncD(V_LOCAL_SWAP,A_N) - SyncD(V_LOCAL_SWAP,A_LOCAL_SWAP)`. Local sensitivity SHALL pass only when the lower confidence bounds of both damage metrics are greater than `0.10` and at least 18 of 22 records have both damage values greater than zero.

#### Scenario: Both controls pass
- **WHEN** `N_REPEAT` is stable and `LOCAL_SWAP` is valid with its own audio and detectably worse with natural replacement
- **THEN** the experiment may interpret `BRIDGE_075`

#### Scenario: Either control fails
- **WHEN** repeatability, local-swap own-audio validity, or local-swap sensitivity fails any registered gate
- **THEN** the terminal scientific decision is `CONTROL_FAILED` and bridge results remain descriptive

### Requirement: Bridge confirmation requires movement, compatibility, and primary gain
For `BRIDGE_075`, the experiment SHALL compute `gap_C = SyncC(V_BRIDGE_075,A_N) - SyncC(V_N,A_N)` and `gap_D = SyncD(V_N,A_N) - SyncD(V_BRIDGE_075,A_N)`. Replacement compatibility SHALL require lower 95% confidence bounds strictly greater than `-0.10` for both gaps and at least 20 of 22 offsets within one frame of `V_N/A_N`.

The discovery-selected primary useful-effect endpoint SHALL require the lower 95% confidence bound of `gap_C` to be strictly greater than `0`. `gap_D` SHALL remain a non-inferiority safety endpoint. Own-audio scores SHALL be descriptive and MUST NOT replace the natural-audio replacement endpoint.

#### Scenario: Natural-to-TTS bridge is confirmed
- **WHEN** both controls pass and `BRIDGE_075` passes movement, replacement compatibility, and primary SyncC gain
- **THEN** the terminal decision is `NATURAL_TO_TTS_BRIDGE_CONFIRMED`

#### Scenario: Bridge does not meet a registered gate
- **WHEN** controls pass but movement, replacement compatibility, or primary SyncC gain fails
- **THEN** the terminal decision is `NATURAL_TO_TTS_BRIDGE_NOT_CONFIRMED` with each failed sub-gate reported

### Requirement: Terminal scope and artifacts are immutable
The experiment SHALL write only under `runs/lrs3_natural_to_tts_bridge_confirmation_20260904/` and SHALL produce exactly one self-hashed terminal decision from `BLOCKED`, `CONTROL_FAILED`, `NATURAL_TO_TTS_BRIDGE_CONFIRMED`, or `NATURAL_TO_TTS_BRIDGE_NOT_CONFIRMED`. Only the confirmed decision SHALL set `reference_conditioned_audio_head_spec_eligible=true`.

A confirmed result MAY authorize writing a separate OpenSpec for a reference-conditioned audio-head prototype. It MUST NOT directly authorize training, identity-transfer claims, deployment, or held-out evaluation. The run SHALL NOT generate TTS, run MFA/DTW, train or fine-tune, search strengths, filter records by outcomes, retry based on scores, modify parent/discovery artifacts, or access sealed validation/test media.

#### Scenario: Registered scope is respected
- **WHEN** execution uses only the fixed fit-only cohort, four arms, six cells, frozen tools, and isolated run root
- **THEN** the terminal artifact records the decision, every sub-gate, and zero forbidden activity

#### Scenario: A forbidden operation is requested
- **WHEN** execution attempts model training, new TTS/alignment, adaptive search, outcome-based filtering/retry, parent overwrite, or sealed-data access
- **THEN** execution stops and records engineering `BLOCKED`
