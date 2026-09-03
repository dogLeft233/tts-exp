## Purpose

Defines a reproducible, leakage-safe experiment for measuring whether uncorrected ASR recognition errors coincide in time with poor local SyncNet V2 audio-visual synchronization in paired natural and TTS LRS3 audio.

## ADDED Requirements

### Requirement: Frozen fit-only paired cohort
The experiment SHALL use the ordered 24-record `fresh_confirmation` cohort frozen by `runs/lrs3_mfa_linear_replacement_20260824/00_protocol_lock_retry1/manifest.json`. It SHALL create exactly two arm-records for every sample: `natural` using the canonical natural audio and `tts` using the canonical raw TTS audio, both paired with the same source video and reference transcript. It SHALL preserve the 24 source groups, SHALL treat `source_group` as a grouping identifier rather than a verified speaker identity, and SHALL NOT read media or derived scores from sealed internal-dev, validation, or test splits.

#### Scenario: Cohort is prepared successfully
- **WHEN** all parent manifests, hashes, files, split labels, and natural/TTS joins match the frozen cohort
- **THEN** preparation emits 24 paired samples and 48 arm-records in frozen order with `sealed_splits_accessed=false`

#### Scenario: A parent asset or split does not match
- **WHEN** a sample is missing an arm, a content hash differs, a group is not fit-only, or a record belongs to a sealed split
- **THEN** preparation fails before opening experiment media and records the exact validation failure

### Requirement: Reproducible local model deployment
The experiment SHALL run locally without a model-serving API. Preflight SHALL verify the main Python environment, FFmpeg/FFprobe, the dedicated SyncNet Python environment, CUDA visibility, free disk space, package versions, and the SyncNet V2 checkpoint whose expected SHA-256 is `961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442`. The ASR model SHALL be `facebook/wav2vec2-large-960h-lv60-self`; acquisition MAY resolve `main` once, but SHALL record and lock the resolved immutable Hugging Face commit, file inventory, and hashes before experiment inference. All later stages SHALL load that locked snapshot with network access disabled.

#### Scenario: Online acquisition followed by offline execution
- **WHEN** the operator explicitly runs model acquisition and the model is not cached
- **THEN** the model is downloaded into the configured repository-local cache, an immutable lock is written, and the inference smoke test succeeds before data processing begins

#### Scenario: Locked model is already cached
- **WHEN** the model lock and all locked files are present and valid
- **THEN** preflight performs no network request and loads only the locked local snapshot

#### Scenario: Required runtime is invalid
- **WHEN** an executable, required package/API, checkpoint, model file, model revision, or smoke test is missing or mismatched
- **THEN** preflight emits `decision=NO_GO`, gives one actionable remediation command, and no cohort audio or video is processed

### Requirement: Single-device deterministic ASR execution
A run SHALL select exactly one ASR device before inference. `auto` SHALL choose CUDA when available and otherwise CPU; explicit `cuda` SHALL fail if CUDA is unavailable. Inference SHALL use evaluation mode, batch size one, float32, no automatic mixed precision, and a recorded random seed. The run SHALL record Python, PyTorch, torchaudio, Transformers, device, GPU, driver, model revision, tokenizer vocabulary hash, and command-line configuration. It SHALL NOT silently switch devices after an arm-record failure.

#### Scenario: CUDA execution is available
- **WHEN** device selection is `auto` or `cuda` and a usable GPU passes the smoke test
- **THEN** every ASR arm-record in the run uses the same recorded CUDA device in float32

#### Scenario: CPU fallback is selected
- **WHEN** device selection is `auto` and CUDA is unavailable
- **THEN** all ASR arm-records run on CPU in float32 and the summary labels the run as CPU fallback

### Requirement: PCM-preserving audio-video inputs
For each arm-record, the experiment SHALL use the exact canonical 16 kHz mono audio identified by the parent manifests and SHALL pair it with the unchanged source video using the repository's strict mux contract: copied video stream and mono 16 kHz `pcm_s16le` audio, with no time stretch, crop, pad, `-shortest`, duration cap, or video re-encode. The decoded mux audio SHALL be byte-identical to decoding the arm source audio under the same PCM contract. The ASR and SyncNet stages SHALL consume audio proven to be the same PCM content.

#### Scenario: Natural and TTS muxes are valid
- **WHEN** both canonical audio files and the source video match their parent hashes
- **THEN** the experiment writes two strict muxes with identical video provenance and arm-specific PCM hashes that pass exact decoded-PCM comparison

#### Scenario: A mux changes content or duration policy
- **WHEN** decoded PCM differs or the output stream violates the strict mux contract
- **THEN** that arm-record fails at input preparation and is not scored

### Requirement: Uncorrected Wav2Vec2-CTC recognition
The ASR result SHALL come from argmax over frame-level CTC logits followed only by standard CTC collapse: merge adjacent repeated token IDs, remove blank IDs, and convert the tokenizer word delimiter to spaces. No beam search, lexicon, language model, spell correction, inverse text normalization, prompt, or reference transcript SHALL influence the recognized token sequence. Each arm-record SHALL export logits metadata, collapsed tokens, normalized words, word start/end times, per-word acoustic confidence, and the final uncorrected transcript.

#### Scenario: Greedy decoding is performed
- **WHEN** a valid 16 kHz mono arm waveform is inferred
- **THEN** the stored transcript and words reproduce exactly from the stored frame argmax IDs, tokenizer, blank ID, delimiter ID, and collapse rules

#### Scenario: A decoder correction path is requested
- **WHEN** configuration enables beam search, an external language model, a lexicon, or transcript-conditioned correction
- **THEN** validation rejects the configuration before inference

### Requirement: Per-arm reference word timing without ASR correction
The experiment SHALL normalize the canonical transcript with one versioned normalizer and SHALL create a separate CTC forced alignment of that transcript against each arm's acoustic emissions. Forced alignment SHALL be used only to locate reference words; it SHALL NOT replace, insert, delete, re-rank, or otherwise alter greedy ASR words. The natural-arm forced alignment SHALL be checked against the official LRS3 word-timing table, and both the official and forced timings SHALL be retained. Unsupported transcript characters, failed forced alignment, non-monotonic spans, or transcript/timing token disagreement SHALL make the arm-record unanalyzable rather than invoking an implicit fallback.

#### Scenario: TTS reference timing is created
- **WHEN** raw TTS timing differs from the natural clip but every normalized reference token is representable by the CTC tokenizer
- **THEN** the TTS arm receives monotonic reference word spans on its own audio clock while its greedy ASR transcript remains unchanged

#### Scenario: Natural timing provides a sanity check
- **WHEN** natural-arm forced alignment succeeds and the official LRS3 timing table normalizes to the manifest transcript
- **THEN** the output records boundary-error diagnostics between forced and official timings without substituting official words into ASR output

#### Scenario: Reference alignment is invalid
- **WHEN** the normalized reference cannot be represented or aligned monotonically
- **THEN** the arm-record fails at reference alignment with no fabricated deletion timestamps

### Requirement: Deterministic word-error localization
The experiment SHALL align normalized greedy ASR words to normalized reference words using unit-cost global word-level edit distance with a documented deterministic tie order. Equal words SHALL be non-errors. A deletion SHALL use the forced-aligned reference word interval, an insertion SHALL use the greedy ASR word interval, and a substitution SHALL use the union of its reference and greedy intervals. Contiguous non-equal operations SHALL be grouped into edit blocks; overlapping or touching error intervals SHALL be merged. Every emitted error interval SHALL retain its contributing operation IDs and word indices.

#### Scenario: Mixed edit operations are localized
- **WHEN** an alignment contains substitutions, deletions, and insertions
- **THEN** all operations receive reproducible arm-clock spans from the specified timing source and the merged error spans cover their union

#### Scenario: Recognition is exact
- **WHEN** normalized greedy words equal normalized reference words
- **THEN** the arm-record has an empty error-span list and `word_error_rate=0`

### Requirement: Local SyncNet score follows upstream semantics
The experiment SHALL derive local scores from the existing SyncNet V2 distance matrix rather than treating an arbitrary five-frame crop as an independent clip-level Sync-C. For each selected face track it SHALL reproduce the upstream calculation with `vshift=15`: `mdist[j]=mean_t(dists[t,j])`, `j*=argmin_j(mdist[j])`, `av_offset_frames=vshift-j*`, `sync_d=mdist[j*]`, and `sync_c=median(mdist)-sync_d`. The local raw confidence SHALL be `local_c_raw[t]=median(mdist)-dists[t,j*]`; the analysis confidence SHALL apply the upstream odd-width median filter of 9 values. Higher local confidence SHALL mean better local synchronization.

#### Scenario: Local scores pass upstream parity
- **WHEN** SyncNet emits a finite distance matrix for a track
- **THEN** recomputed global offset, Sync-D, and Sync-C match the existing evaluator within declared numeric tolerances before local scores are accepted

#### Scenario: A shortcut score is attempted
- **WHEN** an implementation averages separately evaluated five-frame clips or labels raw embedding distance as Sync-C
- **THEN** conformance tests fail because the required distance-matrix derivation and parity fields are absent

### Requirement: Score-independent face-track selection and timeline mapping
SyncNet face tracking SHALL use `min_track=50`. If multiple tracks exist, the experiment SHALL select the track with the greatest frame count and then the smallest track index, without consulting SyncNet or ASR scores. It SHALL record all track start/end frames and the selection reason. Local score row `t` SHALL map to the original audio clock using the selected track start frame, the chosen global offset, and the center of the five-frame embedding window. Rows whose shifted audio index refers to SyncNet padding, whose mapped center lies outside both the arm audio and selected track interval, or whose 9-value median filter support reaches beyond the local-score sequence SHALL be excluded and counted.

#### Scenario: The selected track begins after clip start
- **WHEN** the longest face track starts at source-video frame `s>0`
- **THEN** every local score timestamp includes `s/25` and aligns to the original arm audio clock rather than restarting at zero

#### Scenario: Multiple tracks have equal length
- **WHEN** two or more tracks share the maximum frame count
- **THEN** the smallest track index is selected deterministically and the other tracks remain diagnostic only

#### Scenario: No eligible track exists
- **WHEN** SyncNet produces no face track longer than `min_track`
- **THEN** the arm-record fails at local-sync scoring and no score-based track substitution occurs

### Requirement: Shared valid grid and low-sync mask
Each arm-record SHALL be analyzed on the finite local-score points mapped to the arm audio clock at 25 Hz. An error grid cell SHALL be true when its local-score center lies inside a merged ASR error interval. The low-sync threshold SHALL be computed independently per arm-record as `mean(local_c)-1.0*population_std(local_c)`, and a cell SHALL be low-sync only when `local_c<threshold`. Threshold `k=1.0`, strict comparison, population standard deviation, filter width, grid rate, and valid time bounds SHALL be frozen before aggregate results are computed. Duration outside the video/selected-track overlap SHALL be reported and excluded from both masks.

#### Scenario: Masks are rasterized
- **WHEN** a scored arm-record has finite local confidence and localized error spans
- **THEN** it emits equal-length timestamps, confidence, error-mask, and low-sync-mask arrays plus counts and threshold provenance

#### Scenario: Local confidence is constant
- **WHEN** population standard deviation is zero
- **THEN** the low-sync mask is empty, the record is marked `degenerate_low_mask`, and undefined metrics are represented as JSON `null` rather than NaN or infinity

### Requirement: Per-record association metrics
For every analyzable arm-record, the experiment SHALL report error/low-sync intersection and union cell counts, IoU, error recall, low-sync precision, Spearman correlation between the binary ASR-error mask and continuous sync badness `-local_c`, and the mean sync-badness contrast between error and correct cells. Precision SHALL be null when there are no low-sync cells; recall SHALL be null when there are no error cells; IoU SHALL be null when the union is empty; Spearman and contrast SHALL be null when their required variables do not vary. Null metrics SHALL include a machine-readable reason.

#### Scenario: Both masks contain events
- **WHEN** error and low-sync masks each contain at least one true cell and required variables vary
- **THEN** all per-record metrics are finite, bounded where applicable, and reproducible from exported arrays

#### Scenario: ASR makes no word errors
- **WHEN** the error mask is empty
- **THEN** recall, Spearman, and badness contrast are null with reason `no_asr_error_cells`, while count fields remain valid

### Requirement: Arm-level and paired reporting
The experiment SHALL aggregate natural and TTS arms separately and SHALL join cross-arm comparisons only by frozen `sample_id`. It SHALL report macro distributions of per-record metrics, pooled count-based IoU/precision/recall, WER, analyzable/excluded counts, source-group counts, and paired TTS-minus-natural differences for Spearman and badness contrast. Uncertainty intervals SHALL resample paired samples, not individual 25 Hz cells, using a recorded seed and 10,000 bootstrap draws. Frame-level p-values that treat autocorrelated cells as independent SHALL NOT be reported.

#### Scenario: Complete paired aggregation
- **WHEN** both arms are analyzable for a sample
- **THEN** the sample contributes one paired difference and remains in the same bootstrap resampling unit

#### Scenario: One arm is missing
- **WHEN** only one arm of a sample is analyzable
- **THEN** it may contribute to its arm's descriptive summary but is excluded from paired differences with an explicit count and reason

### Requirement: Explicit engineering and scientific decisions
The run SHALL emit separate engineering and scientific decisions. Engineering SHALL be `GO` only when preflight passes, the frozen 24-by-2 manifest is complete, all 48 arm-records finish, all parent and PCM hashes validate, all SyncNet parity checks pass, and all required JSON/array/plot outputs validate; otherwise it SHALL be `NO_GO`. Scientific status SHALL be evaluated per arm: `INSUFFICIENT` when fewer than 12 records have defined Spearman and badness contrast or fewer than 50 error cells occur across at least 8 source groups; otherwise `SUPPORT` only when both the sample-bootstrap 95% interval for median Spearman and the interval for median badness contrast have lower bounds greater than zero, and `NO_SUPPORT` otherwise. Overall status SHALL preserve the two arm statuses rather than collapsing them into a misleading single binary claim.

#### Scenario: Evidence supports one arm only
- **WHEN** the TTS arm meets the frozen support rule and the natural arm does not
- **THEN** the decision reports `tts=SUPPORT` and the natural arm's exact `NO_SUPPORT` or `INSUFFICIENT` status without claiming universal support

#### Scenario: Pipeline is incomplete
- **WHEN** any required arm-record or parity check fails
- **THEN** engineering is `NO_GO` and scientific decisions are `NOT_EVALUATED`

### Requirement: Immutable, resumable, inspectable outputs
A new run SHALL refuse a non-empty output directory unless `--resume` is given. Every stage SHALL write atomically, validate its own schema, bind outputs to input/config/code/model hashes, and maintain a failure ledger containing stage, sample, arm, exception type, and log path without secrets. Resume SHALL reuse a cell only when its success marker and all bound hashes match; otherwise it SHALL recompute that cell without deleting unrelated valid cells. JSON SHALL reject NaN and infinity. The final run SHALL contain a manifest, environment/model lock, per-arm ASR and alignment records, strict-mux provenance, SyncNet distance/local-score arrays, per-record metrics, aggregate summary, decision, logs, and minimal timeline plots.

#### Scenario: Interrupted run is resumed
- **WHEN** a prior run stopped after some valid arm-cells completed and `--resume` is used with unchanged bindings
- **THEN** valid cells are reused, missing or invalid cells are recomputed, and the summary records reused/recomputed counts

#### Scenario: Existing output is not explicitly resumed
- **WHEN** the target directory is non-empty and `--resume` is absent
- **THEN** the experiment exits before overwriting any file

### Requirement: Minimal diagnostic visualization
The experiment SHALL create one non-interactive timeline plot per paired sample, showing natural and TTS arms with the same time scale where possible. Each arm SHALL show local confidence, its frozen low-score threshold, ASR error spans, excluded timeline regions, and arm/global identifiers. Plots SHALL be diagnostics only and SHALL NOT affect cohort selection, thresholds, track selection, or scientific decisions.

#### Scenario: Paired plot is generated
- **WHEN** at least one arm of a sample has valid local scores
- **THEN** a deterministic plot is written and linked from the sample result

#### Scenario: Plotting fails after numeric analysis
- **WHEN** numeric outputs are valid but a required plot cannot be produced
- **THEN** engineering is `NO_GO`, the numeric artifacts remain intact, and the plotting failure is recorded for resume
