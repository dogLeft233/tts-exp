## Purpose

Defines a provenance-locked natural-only LRS3 experiment for determining whether a native-16-kHz acoustic neural codec reduces the strict natural-audio replacement incompatibility caused by the current WavLM-L6/HiFi-GAN reconstruction path, and whether it reaches a pre-registered practical identity-compatibility margin.

## ADDED Requirements

### Requirement: The experiment uses the frozen 50-record LRS3 codec cohort
The experiment SHALL use exactly the ordered 50-record, 38-source-group cohort bound by `runs/lrs3_wavlm_hifigan_direct_20260826/summary.json`, whose whole-file SHA-256 is `96a29d7355ca3916ef2f60935147d9f20a123a8a7aefa12db130bbcf07eb08d5` and whose ordered sample-ID SHA-256 is `8e47514fdf877e9a57533e8719bd16fab1b96a99d8cb47cfa007c96689326b94`. It SHALL bind each record's source group, untouched natural PCM, face video, historical WavLM-L6/HiFi-GAN reconstruction PCM, sample count, and all parent hashes. It MUST reject missing, reordered, duplicated, changed, or substituted records.

#### Scenario: Frozen codec cohort loads successfully
- **WHEN** the parent summary, source manifest, 50 ordered records, 38 source groups, and all per-record assets match their registered identities
- **THEN** Stage 00 accepts the complete cohort without consulting DAC outputs, render outcomes, fidelity values, or SyncNet scores

#### Scenario: Parent identity differs
- **WHEN** any parent hash, sample identity, source group, path, order, count, natural sample count, or bound media hash differs
- **THEN** the experiment stops before codec inference or scoring and records engineering `BLOCKED`

### Requirement: DAC configuration is frozen before cohort media is processed
The new codec arm SHALL use the official Descript Audio Codec source tag `0.0.5`, commit `408235a9dcd2983684c87615a1bc2a8954f6eb47`, native `16khz` model, `8kbps` checkpoint, and all checkpoint-provided quantizers. Stage 00 SHALL record and freeze the source revision, dependency lock, checkpoint URL, local path, byte size, and SHA-256 before opening any cohort audio. Every downstream stage MUST reject a changed asset or configuration.

#### Scenario: DAC asset lock succeeds
- **WHEN** the source checkout, environment, official release checkpoint, model type, bitrate, sample rate, and all-quantizer configuration are complete and hashable
- **THEN** the experiment writes an immutable model-binding artifact and authorizes natural-audio reconstruction

#### Scenario: DAC identity or configuration is unresolved
- **WHEN** the checkpoint is missing, its hash is not frozen, the source revision differs, or a quantizer count other than all available quantizers is requested
- **THEN** the experiment records engineering `BLOCKED` before any cohort audio is decoded

### Requirement: DAC reconstruction does not use corrective preprocessing or postprocessing
For every record, the experiment SHALL decode the exact untouched 16-kHz mono PCM16 natural waveform, convert it deterministically to model float input, and invoke the pinned DAC model directly at 16 kHz with all quantizers. It SHALL bypass the convenience CLI loudness-normalization path. Internal right-padding to the codec hop length is permitted, but decoded output SHALL be cropped only to the original sample count and canonicalized once to 16-kHz mono PCM16. It MUST NOT apply resampling, loudness normalization, gain matching, temporal shift, delay compensation, lag alignment, interpolation, filtering, denoising, phase correction, trimming, or output repair.

#### Scenario: Exact DAC reconstruction succeeds
- **WHEN** model inference produces a finite mono waveform covering the original sample count
- **THEN** the experiment crops only model-internal right-padding, writes exactly the original sample count as canonical PCM16, and records input, latent/configuration, raw-output, and canonical-output provenance

#### Scenario: Corrective operation is requested
- **WHEN** any operation would align, normalize, repair, resample, pad after decode, or otherwise move DAC output closer to natural after observing it
- **THEN** the experiment rejects the output as protocol-incompatible

#### Scenario: DAC output fails engineering QC
- **WHEN** output is non-finite, clipped under the frozen clipping rule, has a wrong sample count/rate/channel/sample width, or cannot be deterministically canonicalized
- **THEN** the affected record and complete scientific matrix are engineering `BLOCKED` without record substitution

### Requirement: Audio fidelity is measured on the untouched sample grid
The experiment SHALL compare the DAC and historical WavLM/HiFi-GAN reconstructions with untouched natural audio using the frozen Wav2Lip mel frontend, multi-resolution log-STFT distance, SI-SDR, zero-lag waveform correlation, RMS/peak ratio, pinned F0 and voicing errors, short-time energy-envelope correlation, and available onset/phone-boundary-window errors. All metrics SHALL use the unshifted exact sample grid and MUST NOT modify a candidate. The pre-registered mechanistic contrast SHALL be `mel_improvement = mel_L1(N,W) - mel_L1(N,D)`, where positive is better for DAC.

#### Scenario: Fidelity panel is complete
- **WHEN** all 50 natural, WavLM, and DAC waveforms pass provenance and engineering checks
- **THEN** the experiment writes every per-record metric, source-group association, aggregate, and deterministic metric configuration before any TFG score is inspected

#### Scenario: A metric attempts lag optimization
- **WHEN** a fidelity implementation searches or applies a per-record delay, phase, gain, or alignment to improve a reported value
- **THEN** the fidelity stage fails protocol validation

### Requirement: All driver arms use one frozen rendering contract
For each record, the experiment SHALL define `A_N` as untouched natural PCM, `A_W` as the bound historical WavLM-L6/HiFi-GAN reconstruction, and `A_D` as the new DAC reconstruction. It SHALL freshly render `V_N`, `V_W`, and `V_D` with the same pinned Wav2Lip checkpoint, command, face video, and one hash-bound face geometry shared across all three driver arms. It MUST NOT train, fine-tune, select, retry by score, or change face handling between audio arms.

#### Scenario: Three driver videos render successfully
- **WHEN** all three valid audio arms and one valid shared face geometry exist for a record
- **THEN** the experiment produces exactly one deterministic video per driver arm with complete input, model, command, face-geometry, and output hashes

#### Scenario: One driver arm or face geometry fails
- **WHEN** any required driver render is absent, invalid, or uses different face geometry
- **THEN** the complete matrix is engineering `BLOCKED`; the record is not removed and no subset conclusion is issued

### Requirement: The experiment scores a complete strict 3-by-3 matrix
The experiment SHALL construct and score all nine combinations of `V_N`, `V_W`, and `V_D` with `A_N`, `A_W`, and `A_D` for every one of the 50 records. Every mux SHALL copy the video stream, encode audio as PCM s16le, and verify decoded PCM byte equality, sample count, rate, channels, and sample width against the selected source audio before running pinned official file-level SyncNet V2. All 450 cells SHALL be valid for a scientific decision.

#### Scenario: Strict matrix is complete
- **WHEN** 150 driver videos and all 450 mux/score cells pass video-stream, decoded-PCM, provenance, and scorer checks
- **THEN** the experiment accepts the matrix for frozen statistical analysis

#### Scenario: Media identity or matrix completeness fails
- **WHEN** any video stream changes during mux, decoded audio differs from its source, a cell is missing/duplicated, or a scorer/model binding differs
- **THEN** the experiment records engineering `BLOCKED` and issues no scientific codec conclusion

### Requirement: Codec improvement and identity compatibility are separate decisions
The experiment SHALL compute paired benefits `DAC_over_W_C = SyncC(V_D,A_N) - SyncC(V_W,A_N)`, `DAC_over_W_D = SyncD(V_W,A_N) - SyncD(V_D,A_N)`, `DAC_identity_C = SyncC(V_D,A_N) - SyncC(V_N,A_N)`, and `DAC_identity_D = SyncD(V_N,A_N) - SyncD(V_D,A_N)`. It SHALL use 10,000 source-group cluster-bootstrap draws with seed `20260904` and report two-sided percentile 95% confidence intervals, means, per-record wins, and group summaries.

`DAC_REDUCES_CODEC_REPLACEMENT_PENALTY` SHALL require the lower 95% confidence bounds for `DAC_over_W_C`, `DAC_over_W_D`, and `mel_improvement` all to be strictly greater than zero. `DAC_IDENTITY_COMPATIBLE` SHALL additionally require the lower confidence bounds for `DAC_identity_C` and `DAC_identity_D` both to be strictly greater than the fixed non-inferiority margin `-0.10` SyncNet units.

#### Scenario: DAC improves and reaches identity compatibility
- **WHEN** all three improvement lower bounds are greater than zero and both identity lower bounds are greater than `-0.10`
- **THEN** the terminal decision is `DAC_CODEC_IDENTITY_COMPATIBLE` and `future_tts_alignment_experiment_eligible=true`

#### Scenario: DAC improves but misses identity compatibility
- **WHEN** all three improvement lower bounds are greater than zero but either identity lower bound is less than or equal to `-0.10`
- **THEN** the terminal decision is `DAC_CODEC_PARTIAL_IMPROVEMENT` and no TTS/alignment experiment is authorized by this protocol

#### Scenario: DAC does not establish improvement
- **WHEN** any of the three improvement lower bounds is less than or equal to zero
- **THEN** the terminal decision is `DAC_CODEC_HYPOTHESIS_NOT_SUPPORTED` for this exact codec configuration and no TTS/alignment experiment is authorized

### Requirement: Codec self-consistency is not treated as replacement evidence
The experiment SHALL report diagonal scores, own-audio preference for `V_W` and `V_D`, all cross-codec cells, and their grouped intervals as descriptive diagnostics. It MUST NOT use high `V_D/A_D` scores, subjective audio quality, WavLM similarity, or any crossed cell other than the pre-registered endpoints to issue identity compatibility or future-stage authorization.

#### Scenario: DAC diagonal is strong but replacement is not compatible
- **WHEN** `V_D/A_D` is high or prefers `A_D` but the registered `V_D/A_N` identity conditions fail
- **THEN** the report identifies codec-specific audio/video co-adaptation and does not promote the codec

### Requirement: Experiment boundaries and terminal artifacts are explicit
The experiment MUST NOT use TTS audio, MFA alignments, DTW, Soft-DTW, residual injection, masking, trainable parameters, checkpoint search, bitrate search, alternate quantizer counts, post-outcome retries, record substitution, or sealed validation/test media. Every attempted stage SHALL write append-only machine-readable manifests, complete parent/model/media bindings, a media-access ledger, failure details, and one terminal decision in the distinct run root `runs/lrs3_dac16k_codec_identity_20260904/`.

#### Scenario: Scientific result is complete
- **WHEN** the codec, fidelity, rendering, strict matrix, and registered statistical analysis all complete
- **THEN** the terminal artifact states exactly one scientific decision, its scope and limitations, and whether a separate future TTS/alignment OpenSpec is eligible

#### Scenario: Engineering failure occurs
- **WHEN** any invariant, asset, media, execution, matrix, or provenance check fails
- **THEN** the terminal artifact distinguishes engineering `BLOCKED` from a scientific negative result and authorizes no downstream experiment
