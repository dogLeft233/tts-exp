## Purpose

Defines a small, reproducible, fit-only causal prototype for testing whether duration-normalized TTS waveform patches placed at natural-ASR error blocks improve frozen Wav2Lip videos under strict untouched-natural-audio SyncNet replacement scoring, beyond comparable patches placed at ASR-correct regions.

## ADDED Requirements

### Requirement: Frozen fit-only parents and score-blind selection
The prototype SHALL use only the ordered 24 fit-only records and immutable ASR/model/config artifacts from `runs/lrs3_asr_sync_error_correlation_20260831`. It SHALL validate parent paths, hashes, source groups, split locks, natural/TTS audio, face video, transcripts, ASR records, and success markers before opening cohort media. Cohort, target-block, and control-block selection SHALL consume an explicit score-free projection containing only sample/source-group identifiers, fit provenance, transcript words, operation identities, forced-reference start/end spans, audio identity, and quantized bounds. Selection SHALL NOT consume existing SyncNet distances, local confidence, ASR confidence, global Sync-C/Sync-D, visual metrics, plots, or scientific decisions. It SHALL NOT read media or derived results from sealed internal-dev, validation, or test splits. The inferential unit SHALL be `unit_id=source_group`; this frozen cohort SHALL assert exactly one retained sample per source group before analysis and SHALL require at least 12 distinct source groups.

#### Scenario: Source-group unit is not one-to-one
- **WHEN** retained samples do not map one-to-one to distinct source groups
- **THEN** engineering fails before analysis with reason `source_group_unit_mismatch`, and no duplicated source group is resampled as independent evidence

#### Scenario: Frozen parents reproduce
- **WHEN** every locked parent and ASR artifact matches and all records remain fit-only
- **THEN** preflight emits the same ordered 24 records, records every consumed parent field and hash, and reports `sealed_splits_accessed=false`

#### Scenario: A downstream score is offered to selection
- **WHEN** configuration or code attempts to select a sample, target block, or control using SyncNet or visual-score content
- **THEN** conformance validation fails before candidate audio or video is created

#### Scenario: A parent or split lock differs
- **WHEN** a required hash, marker, transcript, source group, fit label, or sealed-split boundary differs
- **THEN** engineering is `NO_GO`, science is `NOT_EVALUATED`, and no cohort media is processed

### Requirement: Reference-mappable natural-error targets with ASR-clean TTS donors
The prototype SHALL deterministically rebuild word-level edit alignments from the frozen natural and TTS greedy ASR records. A target block SHALL contain at least one natural substitution or deletion, a non-empty contiguous reference-word range, finite monotonic forced-reference spans on both arms, and a complete TTS operation interval from the first through last mapped reference operation containing only `equal` operations (there SHALL be no TTS substitution, deletion, or insertion inside that interval). Pure insertion-only blocks, mixed natural insertion blocks, non-contiguous reference ranges, unsupported spans, and TTS-error donor blocks SHALL be excluded with reason codes. Selection SHALL occur before any prototype render or score exists.

#### Scenario: Eligible error block is found
- **WHEN** a natural substitution/deletion block has a contiguous reference range and every corresponding TTS word is ASR-equal
- **THEN** the manifest binds its natural destination span, TTS donor span, operation IDs, reference indices, timing provenance, and hashes

#### Scenario: Natural error is insertion-only
- **WHEN** an error block has no reference-side word index
- **THEN** it is excluded from primary targeting with reason `no_reference_side_donor_mapping`

#### Scenario: TTS donor is also recognized incorrectly
- **WHEN** any corresponding TTS reference word is substitution/deletion, the complete mapped TTS operation interval contains an insertion, or a mapped word is not represented by an equal operation
- **THEN** that block is excluded with reason `tts_donor_not_asr_clean`

### Requirement: Two score-blind matched non-error controls
For every retained target block, the prototype SHALL select two deterministic control blocks from the same sample where both natural and TTS reference words are ASR-equal and the complete TTS operation interval over each control contains only `equal` operations. Control selection SHALL use the score-free projection's reference-word count, quantized natural-clock duration, quantized TTS donor/destination duration ratio, stable sample/block identifiers, and seed `20260901` only. A control block SHALL NOT overlap a target block or another block in the same control replicate. Its word-count difference from its target SHALL be at most one and its natural duration SHALL lie between one half and twice the target duration. Each complete control replicate's total edited duration SHALL lie in `[0.8,1.25]` times the targeted total edited duration. The two control replicates SHALL be distinct assignments; a sample SHALL be excluded if no distinct pair can be constructed. Assignment SHALL use lexicographic backtracking over targets in ascending target-block identifier order, candidate control runs represented by maximal ASR-equal runs and their admissible contiguous windows in ascending `(start_reference_index, end_reference_index)` order, and the frozen ranking tuple.

#### Scenario: Both controls can be assigned
- **WHEN** a sample has two complete, admissible, deterministic control assignments for every target block
- **THEN** it remains eligible and the manifest records candidate pools, ranking keys, selected blocks, edit budgets, and non-overlap checks

#### Scenario: Matched controls are incomplete
- **WHEN** either control replicate cannot satisfy all matching, duration-budget, and non-overlap constraints
- **THEN** the sample is excluded before rendering with reason `incomplete_matched_controls` and no threshold is relaxed

#### Scenario: Existing SyncNet data changes
- **WHEN** old SyncNet artifacts change while all frozen ASR/reference inputs remain identical
- **THEN** target and control assignments remain byte-for-byte identical

#### Scenario: Target/control contrast is interpreted
- **WHEN** the two matched controls are compared with the targeted condition
- **THEN** those contrasts are reported as predeclared descriptive comparators, not as randomized causal estimates of ASR targeting; only the natural-baseline estimands support the conditional intervention claim

### Requirement: Deterministic exact-length local PCM intervention
The prototype SHALL decode validated natural and TTS audio as mono 16 kHz PCM, convert each finite timing interval to half-open sample bounds with `start=floor(start_s*16000)` and `end=ceil(end_s*16000)` followed by clipping to the corresponding waveform, and reject non-positive intervals. It SHALL duration-normalize each TTS donor interval to its natural destination sample count with one frozen finite `scipy.signal.resample` operator at SciPy version `1.18.0`, and blend it only inside the destination using an in-segment raised-cosine boundary ramp of `min(320, floor(destination_samples/4))` samples. It SHALL apply disjoint blocks in chronological order and write mono 16 kHz PCM16 with exactly the canonical natural sample count. It SHALL NOT loudness-normalize, limit, globally shift, globally stretch, pad, crop, denoise, or modify samples outside destination intervals. Non-finite values or pre-write clipping SHALL fail the candidate rather than invoke a fallback.

#### Scenario: Targeted candidate is constructed
- **WHEN** all selected donor and destination intervals are valid
- **THEN** the output has the exact natural length, finite unclipped samples, fixed resampling/crossfade provenance, and zero pre-quantization difference outside recorded destinations

#### Scenario: Natural identity passes through the writer
- **WHEN** the natural condition is decoded and written without patches
- **THEN** its decoded PCM is byte-identical to canonical natural decoded PCM

#### Scenario: Candidate would clip or require fallback
- **WHEN** any patched float sample is non-finite or outside the valid unclipped range
- **THEN** the cell fails and no normalization, limiter, alternate resampler, or partial candidate is used

### Requirement: Exactly four required conditions per eligible sample
Every eligible sample SHALL have exactly the ordered conditions `natural`, `asr_targeted`, `target_control_0`, and `target_control_1`. The natural condition SHALL be the identity driver, the targeted condition SHALL contain every eligible target patch, and each control condition SHALL contain its complete matched-control patch set. No full-TTS, low-sync oracle, high-confidence, learned, mel-seam, alternate-crossfade, alternate-seed, or additional dose condition SHALL enter the prototype decision.

#### Scenario: Required condition matrix is complete
- **WHEN** `N` samples pass frozen eligibility
- **THEN** the run manifest contains exactly `4*N` unique sample-condition cells in fixed order

#### Scenario: An exploratory arm is requested
- **WHEN** an unregistered condition or intervention variant is supplied
- **THEN** configuration validation rejects it before candidate construction

### Requirement: Frozen and isolated Wav2Lip rendering
The prototype SHALL render every required condition with the same frozen face input, Wav2Lip code, checkpoint, flags, batch settings, device policy, and recorded runtime. Every render cell SHALL use a unique working directory and unique `temp/` path; a shared Wav2Lip working directory SHALL be forbidden. Selection SHALL not depend on render success or output scores, and a failed cell SHALL not be replaced by another sample.

#### Scenario: Isolated rendering succeeds
- **WHEN** a valid four-condition sample is rendered
- **THEN** four videos are produced with condition-specific logs, work paths, input hashes, code/checkpoint hashes, commands, and output hashes

#### Scenario: A shared temporary path is detected
- **WHEN** two cells resolve to the same Wav2Lip work or temporary directory
- **THEN** preflight fails before either render starts

#### Scenario: One condition fails to render
- **WHEN** Wav2Lip fails or produces an invalid video for any required condition
- **THEN** that sample remains incomplete, no substitute is selected, and engineering cannot be `GO`

### Requirement: Strict untouched-natural-audio replacement endpoint
For every rendered condition video, the prototype SHALL create scoring media by copying that rendered video stream and muxing the sample's untouched canonical natural audio as mono 16 kHz `pcm_s16le`. It SHALL use no crop, pad, stretch, duration cap, `-shortest`, audio substitution, or video re-encode. Decoded scoring-media audio SHALL be byte-identical to decoding canonical natural audio under the same PCM contract. Candidate-driver audio SHALL NOT be used for the primary SyncNet score.

#### Scenario: Replacement media is valid
- **WHEN** a rendered condition is prepared for scoring
- **THEN** its video stream is copied, its decoded audio equals untouched canonical natural PCM, and its provenance binds both sources

#### Scenario: Candidate audio remains in the scoring file
- **WHEN** decoded scoring-media audio differs from canonical natural PCM
- **THEN** the cell fails before SyncNet and cannot contribute to a replacement claim

### Requirement: Official SyncNet V2 scoring and fixed-coordinate local diagnostic
The prototype SHALL run the existing frozen SyncNet V2 with `vshift=15`, `min_track=50`, score-independent longest-track selection, and the locked checkpoint. It SHALL export each full distance matrix and reproduce official whole-track offset, Sync-D, and Sync-C with the existing exact PyTorch float32 parity semantics. Global Sync-C and Sync-D SHALL be primary. For every sample, all four conditions SHALL have identical selected track index, frame count, start/end range, and scorer-input support; any mismatch SHALL make the sample incomplete and engineering `NO_GO`, rather than retaining a primary score from unlike track support. As a secondary diagnostic, each non-natural condition SHALL be compared with natural using the explicit natural matrix column `j_natural = argmin_j mean_t(D_natural[t,j])`, raw distances `D[t,j_natural]`, `a=t+j_natural-vshift`, and timestamp `(track_start_frame+a+2)/fps` over common valid rows. Valid rows SHALL exclude SyncNet shift padding, the four median-filter edge rows, rows outside the selected track, and rows outside natural audio. Edited-region support SHALL be the union of destination intervals expanded by 0.20 seconds and clipped to common support. The diagnostic SHALL report distance improvement inside and outside support and SHALL NOT alter official global scores.

#### Scenario: Global score passes parity
- **WHEN** SyncNet returns a finite distance matrix
- **THEN** recomputed offset, Sync-D, and Sync-C pass the locked upstream parity tolerances before the cell is accepted

#### Scenario: Conditions share track support
- **WHEN** natural and a candidate condition select identical track metadata
- **THEN** fixed-coordinate inside/outside distance diagnostics are computed at the natural offset on common valid rows

#### Scenario: Track metadata differs
- **WHEN** any condition selects different track metadata or scorer-input support
- **THEN** the sample is incomplete with reason `paired_track_mismatch`, no score-based track substitution occurs, and its official scores do not contribute to engineering or scientific decisions

### Requirement: Paired estimands and whole-sample uncertainty
For every complete sample, the prototype SHALL compute positive-is-better ASR-targeted gains against natural identity and descriptive contrasts against the mean of the two matched controls:

```text
baseline_C_gain = C(asr_targeted) - C(natural)
baseline_D_gain = D(natural) - D(asr_targeted)
control_C_adv   = C(asr_targeted) - mean(C(control_0), C(control_1))
control_D_adv   = mean(D(control_0), D(control_1)) - D(asr_targeted)
```

It SHALL report each sample's values, means, medians, win counts, condition scores, and edit-budget diagnostics. It SHALL compute 95% intervals for the medians with 10,000 deterministic `PCG64(20260901)` bootstrap draws that resample distinct `unit_id=source_group` rows, using NumPy's `method="linear"` percentile convention. It SHALL NOT treat frames, patches, words, or condition cells as independent inferential units and SHALL NOT report frame-independent p-values.

#### Scenario: Complete sample is analyzed
- **WHEN** all four condition scores exist for a sample
- **THEN** it contributes exactly one row containing all four estimands to every bootstrap resample in which it appears

#### Scenario: A condition is missing
- **WHEN** any required condition score is absent or invalid
- **THEN** the sample contributes to no primary estimand and engineering cannot be `GO`

### Requirement: Conservative prototype decisions and claim boundary
The prototype SHALL emit separate engineering and scientific decisions. Engineering SHALL be `GO` only when the frozen lock, candidate/control matrix, all required candidate PCM, all renders, all strict replacement muxes, SyncNet parity, identical four-condition track support, primary metrics, hashes, and schemas validate. If engineering is not `GO`, science SHALL be `NOT_EVALUATED`. With engineering `GO`, science SHALL be `INSUFFICIENT` when fewer than 12 complete distinct `unit_id=source_group` rows remain. Otherwise science SHALL be `PROTOTYPE_SUPPORT` only when the bootstrap 95% lower bound is greater than zero for the two natural-baseline median estimands and the two descriptive control contrasts; all other complete outcomes SHALL be `NO_PROTOTYPE_SUPPORT`.

#### Scenario: All four promotion intervals are positive
- **WHEN** at least 12 distinct source-group rows exist and every frozen median interval has lower bound greater than zero
- **THEN** science is `PROTOTYPE_SUPPORT` and the report limits authorization to designing a fresh fit-only confirmation or bounded follow-up head

#### Scenario: A promotion interval crosses or falls below zero
- **WHEN** the denominator is sufficient but at least one frozen interval has lower bound at or below zero
- **THEN** science is `NO_PROTOTYPE_SUPPORT` and the report distinguishes intervals crossing zero from intervals whose upper bound is at or below zero

#### Scenario: Denominator is too small
- **WHEN** fewer than 12 complete distinct source-group rows remain
- **THEN** science is `INSUFFICIENT`, no matching rule is relaxed, and no replacement sample is added

#### Scenario: Engineering is incomplete
- **WHEN** any required engineering contract fails
- **THEN** engineering is `NO_GO`, science is `NOT_EVALUATED`, and no partial scientific conclusion is emitted

### Requirement: Immutable, resumable, and minimally scoped execution
The prototype SHALL write a new immutable run directory containing lock, candidates, renders, replacement media, distance matrices, logs, per-sample metrics, compact plots, summary, and decisions. It SHALL refuse a non-empty directory without explicit `--resume`; resume SHALL reuse only cells whose input/config/code/model/output bindings and hashes validate. JSON SHALL reject NaN and infinity, arrays SHALL reject object dtype, and failures SHALL retain stage/sample/condition/log provenance. Network access MAY install a pinned missing package during preflight but SHALL NOT download or switch to a mutable inference model. The prototype SHALL add no training loop, model service, general workflow framework, database, dashboard, or vendor-code modification.

#### Scenario: Interrupted run resumes safely
- **WHEN** a previous prototype run has valid and invalid cells and `--resume` is given
- **THEN** valid hash-bound cells are reused, invalid cells are recomputed, and unrelated outputs are not deleted or overwritten

#### Scenario: Non-empty output is not resumed
- **WHEN** the target run directory is non-empty and `--resume` is absent
- **THEN** execution stops before writing or deleting any run artifact

#### Scenario: A new model dependency is proposed
- **WHEN** execution attempts to download or select an inference model not present in the frozen lock
- **THEN** preflight rejects the run and no cohort media is processed

### Requirement: Minimal diagnostics do not influence decisions
The prototype SHALL produce one compact diagnostic plot per complete sample showing the four global Sync-C/Sync-D values, target/control edit budgets, and available fixed-coordinate local distance changes. Plots SHALL be generated only after numeric artifacts are frozen and SHALL NOT influence eligibility, control selection, thresholds, score inclusion, or scientific status.

#### Scenario: Diagnostic plot is generated
- **WHEN** a sample has four complete numeric condition records
- **THEN** one deterministic plot is written and linked from the sample record without changing any numeric hash or decision input
