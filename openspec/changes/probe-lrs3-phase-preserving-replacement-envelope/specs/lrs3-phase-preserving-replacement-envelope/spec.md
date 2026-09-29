## Purpose

Defines a small, provenance-locked LRS3 proof-of-concept for determining whether natural-phase audio can move materially toward an aligned TTS spectral trajectory while remaining compatible with untouched natural audio after frozen Wav2Lip rendering.

## ADDED Requirements

### Requirement: The experiment uses one fixed record per existing source group
The experiment SHALL derive its cohort from the ordered 133-record LRS3 MFA-linear comparison cohort bound by protocol manifest SHA-256 `e8c50a459c8197c86ec11cf06a01c000e3f62def981d16d8ed9275a38a473784` and replacement manifest SHA-256 `157cd678f5f03be6eae9986a82babfdfb1f1a44e611bb0b86476c6e19b7eb272`. It SHALL select the first ordered record from each source group, yielding exactly 23 records and ordered sample-ID SHA-256 `c56e420d9ade7e04f5558f37fbf68ee68463d3817060baab79e8e6e0bf7e8fbd`. Each selected record SHALL bind its source group, untouched natural PCM, exact-length MFA-linear candidate PCM, face video, sample count, and asset hashes.

#### Scenario: Fixed cohort is accepted
- **WHEN** both parent manifests, all 23 first-per-group records, their order, and every bound asset match the registered identities
- **THEN** Stage 00 accepts the cohort without consulting new candidate, render, or SyncNet outcomes

#### Scenario: Cohort identity differs
- **WHEN** a parent or asset hash, record count, source group, selection order, sample count, or path differs
- **THEN** the experiment records engineering `BLOCKED` before candidate generation

### Requirement: Candidate arms are deterministic and fixed before scoring
For every record, the experiment SHALL produce exactly seven driver-audio arms: untouched natural `N`; polarity-inverted natural `INV`; natural-phase/TTS-magnitude blends `MAG_025`, `MAG_050`, `MAG_075`, and `MAG_100`; and delayed natural `SHIFT_200`. Every output SHALL be 16-kHz mono PCM16 with exactly the natural sample count. Candidate generation MUST NOT depend on any render or SyncNet result.

The four `MAG` arms SHALL use the registered blend strengths `0.25`, `0.50`, `0.75`, and `1.00` between the natural and bound MFA-linear log-STFT magnitudes while retaining natural phase. `SHIFT_200` SHALL prepend exactly 3,200 zero samples and remove exactly 3,200 samples from the end. `INV` SHALL negate the natural waveform without any time, gain, or spectral correction.

#### Scenario: Complete candidate set is generated
- **WHEN** a selected record has valid natural and MFA-linear PCM with the registered sample format and exact common length
- **THEN** the experiment writes all seven arms with their construction parameters, input hashes, output hashes, and waveform QC

#### Scenario: Candidate construction is changed or incomplete
- **WHEN** an arm, strength, STFT contract, delay, input, output length, sample format, or deterministic post-scaling rule differs from the registered protocol
- **THEN** the record and complete scientific experiment are engineering `BLOCKED`

### Requirement: Audio controls validate equivalence and timing sensitivity
The `INV` arm SHALL serve as the non-identical positive control: its official Wav2Lip mel shall match natural within mean absolute error `1e-5`, and its replacement result SHALL satisfy the same non-inferiority and offset-agreement gates as a blend arm. The `SHIFT_200` arm SHALL serve as a sensitivity control and SHALL demonstrate either an absolute replacement-offset change of at least 3 frames on at least 18 of 23 records or group mean degradation beyond `-0.10` on both replacement benefits.

#### Scenario: Both controls behave as intended
- **WHEN** `INV` passes mel equivalence and replacement compatibility and `SHIFT_200` demonstrates registered timing sensitivity
- **THEN** the experiment may interpret the blend envelope

#### Scenario: A control fails
- **WHEN** `INV` fails equivalence or compatibility, or `SHIFT_200` fails to demonstrate timing sensitivity
- **THEN** the terminal scientific decision is `CONTROL_FAILED` and no blend-envelope claim is issued

### Requirement: Every arm uses one frozen rendering and scoring contract
The experiment SHALL freshly render one video per record and driver arm with one pinned Wav2Lip checkpoint, command, face video, and shared per-record face geometry. For each arm `X`, it SHALL score only `V_X/A_N` and `V_X/A_X`, with `V_N/A_N` represented once, yielding exactly 161 videos and 299 unique scored cells. Every mux SHALL copy the video stream, use PCM s16le audio, and verify decoded PCM byte equality and video elementary-stream identity before pinned official file-level SyncNet V2 scoring.

#### Scenario: Rendering and scoring complete
- **WHEN** all 161 videos and 299 required cells pass model, geometry, media-identity, and scorer validation
- **THEN** the experiment accepts the complete matrix for frozen analysis

#### Scenario: A media cell is missing or changed
- **WHEN** any required video or score is absent, duplicated, cross-record, score-retried, rendered with different geometry, or fails PCM/video identity checks
- **THEN** the experiment records engineering `BLOCKED` and issues no scientific result

### Requirement: TTS-direction movement is measured before replacement outcomes are read
For every `MAG` candidate, the experiment SHALL compute the exact frozen Wav2Lip mel and report its directional projection from natural toward the bound MFA-linear mel, mel distance from natural, orthogonal residual, and waveform QC. A blend level SHALL be considered nontrivial only when at least 21 of 23 records have directional progress of at least `0.15` and the bootstrap lower 95% confidence bound of mean progress is greater than `0.15`. These diagnostics MUST NOT be used to alter, repair, or selectively rerun a candidate.

#### Scenario: Candidate demonstrates measurable TTS-direction movement
- **WHEN** a fixed `MAG` arm passes both registered directional-progress conditions
- **THEN** that arm is eligible for the nontrivial replacement-envelope decision

#### Scenario: Candidate movement is insufficient
- **WHEN** either directional-progress condition fails
- **THEN** that arm is reported as acoustically too close to natural for a nontrivial-envelope claim

### Requirement: Replacement compatibility uses paired non-inferiority
For each arm `X`, the experiment SHALL compute positive-is-better paired benefits `gap_C(X) = SyncC(V_X,A_N) - SyncC(V_N,A_N)` and `gap_D(X) = SyncD(V_N,A_N) - SyncD(V_X,A_N)`. It SHALL use 10,000 deterministic bootstrap draws over the 23 fixed source groups with seed `20260904` and report means, two-sided percentile 95% confidence intervals, record wins, and per-record values.

A `MAG` arm SHALL be replacement-compatible only when the lower 95% confidence bounds of both `gap_C` and `gap_D` are strictly greater than `-0.10` and at least 21 of 23 records have replacement offsets within one frame of their `V_N/A_N` baseline. Own-audio scores and own-audio preference SHALL be descriptive and MUST NOT substitute for `V_X/A_N` compatibility.

#### Scenario: A nontrivial blend arm is compatible
- **WHEN** an arm passes directional movement, both replacement non-inferiority bounds, and offset agreement
- **THEN** it qualifies as a nontrivial compatible point on this fixed cohort

#### Scenario: Diagonal score is high but replacement fails
- **WHEN** `V_X/A_X` is strong but any registered `V_X/A_N` compatibility gate fails
- **THEN** the arm is not replacement-compatible

### Requirement: The terminal decision is narrow and reproducible
If both controls pass and at least one `MAG` arm is nontrivial and replacement-compatible, the terminal decision SHALL be `PHASE_PRESERVING_REPLACEMENT_ENVELOPE_FOUND` and the report SHALL name the largest qualifying registered blend strength. If controls pass but no `MAG` arm qualifies, the decision SHALL be `ONLY_TRIVIAL_REPLACEMENT_EQUIVALENCE_FOUND`. Engineering incompleteness SHALL produce `BLOCKED`, not a scientific negative result.

The result SHALL be scoped to this fixed fit-only cohort and this exact phase-preserving construction. It MUST NOT be described as target-speaker identity transfer, general LRS3 performance, a deployable replacement model, or evidence that arbitrary TTS audio is compatible. A positive result may only authorize a separate OpenSpec for identity characterization or reference-conditioned voice conversion.

#### Scenario: Phase-preserving envelope is found
- **WHEN** all artifacts are complete, both controls pass, and one or more `MAG` arms satisfy every registered gate
- **THEN** the final artifact records `PHASE_PRESERVING_REPLACEMENT_ENVELOPE_FOUND`, the largest qualifying strength, all endpoint values, and the narrow follow-up eligibility

#### Scenario: Only trivial equivalence is found
- **WHEN** all artifacts and controls pass but no `MAG` arm satisfies every registered gate
- **THEN** the final artifact records `ONLY_TRIVIAL_REPLACEMENT_EQUIVALENCE_FOUND` and authorizes no identity-model experiment

### Requirement: Prior runs and sealed data remain untouched
The experiment SHALL write only under `runs/lrs3_phase_preserving_replacement_envelope_20260904/`, preserve all parent artifacts unchanged, and access no sealed validation/test media. It MUST NOT train or fine-tune a model, generate new TTS, run MFA or DTW, search transformation families or strengths, filter records by outcomes, or retry media based on scores.

#### Scenario: Protocol boundaries are respected
- **WHEN** execution uses only the locked fit-only inputs, seven fixed arms, frozen evaluators, and isolated run root
- **THEN** the terminal artifact records zero sealed-data access, zero training, and zero outcome-based selection or retry

#### Scenario: A forbidden path is requested
- **WHEN** execution attempts new TTS generation, alignment, training, strength search, score-based filtering, parent overwrite, or sealed-data access
- **THEN** the experiment stops and records engineering `BLOCKED`
