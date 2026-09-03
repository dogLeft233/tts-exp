## Purpose

Defines a small, reproducible, fit-only LRS3 feasibility prototype for testing whether deterministic phone-aligned paired-TTS WavLM features improve reconstruction of masked natural-phone log-mel patches on held-out source groups beyond an identically initialized and scheduled natural-context-only model. Same-checkpoint zero/wrong-phone inputs are sensitivity diagnostics only.

## ADDED Requirements

### Requirement: Frozen fit-only parents and source-group split
The prototype SHALL bind the existing Exp31 split metadata, policy-cohort source manifest/records, cached WavLM-L6 arrays, and MFA3 phone JSON under the configured local asset root by path and SHA-256. It SHALL first project sample ID, source group, and parent `protocol_split` from metadata and SHALL open record media, cached feature arrays, and phone JSON only for IDs whose parent `protocol_split` is exactly `train`. It SHALL reproduce exactly 30 selected records and ten groups, partitioned into the six frozen train groups/19 records and four frozen evaluation groups/11 records listed in `design.md`, with no group overlap. It SHALL NOT open or process parent validation/test media, features, phone files, videos, or downstream results. Existing Exp31 checkpoints, losses, attention values, validation results, and TFG/SyncNet scores SHALL NOT be model inputs or selection fields.

#### Scenario: Frozen fit-only projection reproduces
- **WHEN** all local parents match the frozen metadata and hashes
- **THEN** `00_lock` contains exactly the 30 permitted records, ten groups, frozen 19/11 split, consumed-field ledger, and `sealed_splits_accessed=false`

#### Scenario: A parent record is not fit/train
- **WHEN** a candidate ID has parent `protocol_split` other than `train`
- **THEN** its record JSON, feature arrays, alignment JSON, audio, and media are not opened and it cannot enter either prototype split

#### Scenario: A group or hash differs
- **WHEN** a required hash, ID, source group, split assignment, or one-to-one ID binding differs
- **THEN** engineering is `NO_GO`, science is `NOT_EVALUATED`, and no model training starts

### Requirement: Deterministic score-blind matched-phone masks
The prototype SHALL normalize every MFA phone label only as `unicodedata.normalize("NFC", str(label).strip())`, with no case folding for lexical equality, pronunciation remapping, IPA substitution, or aliasing. A label SHALL be administrative and excluded when its normalized case-folded value is one of `{"", "sil", "sp", "spn", "<eps>", "<sil>", "silence"}`. It SHALL deterministically edit-align the remaining ordered natural and TTS labels using equal cost for non-equal operations and tie order `equal/substitute`, `delete`, then `insert`, retaining only `equal` operations with byte-identical normalized lexical labels. It SHALL exclude administrative, insertion/deletion/substitution, non-finite/non-monotonic, boundary-collapsed, and out-of-support pairs with explicit reason codes. Canonical ordering SHALL use the explicit group order from `design.md`, UTF-8 `sample_id` order within group, and mask order `(natural_phone_operation_index,natural_core_start_frame,natural_core_end_frame,tts_phone_operation_index)` within record; ordered lists and canonical JSON hashes SHALL freeze before schedule generation. Natural support SHALL be the first 61,440 mono 16 kHz samples. An eligible target SHALL contain 4–40 natural mel frames and at least two TTS WavLM frames. Every example SHALL use the fixed 96-frame window formula in `design.md`. Input masking SHALL cover the target phone plus exactly four natural mel frames of guard on each side, clipped to that window; reconstruction loss SHALL use only the unexpanded target core. The aligned TTS input SHALL have shape `[96,1024]`, contain mapped TTS features only at core positions, and be zero elsewhere. Separate `[96,1]` binary channels SHALL mark the core and masked support. Selection SHALL NOT read reconstruction outputs, model losses, attention, ASR/SyncNet confidence, visual scores, or plots.

#### Scenario: A lexical phone pair is eligible
- **WHEN** natural and TTS phone labels are equal, timings are valid, and both frame-count gates pass
- **THEN** the mask manifest binds the phone operation, source timings, natural core/guard frames, TTS feature frames, hashes, and exclusion-free provenance

#### Scenario: A phone is unsupported
- **WHEN** a phone is silence/`spn`, mismatched, non-monotonic, outside 3.84 s natural support, or below/above a frame gate
- **THEN** it is excluded before training with one deterministic reason and no threshold relaxation

#### Scenario: Readiness denominator is insufficient
- **WHEN** there are fewer than 12 training records/five train groups/80 train masks or fewer than eight evaluation records/four frozen evaluation groups/40 evaluation masks/five masks per evaluation group
- **THEN** preflight engineering may be `GO`, science is `INSUFFICIENT`, no final model is promoted, and no sample, phone, or group is substituted

### Requirement: Fixed feature extraction, normalization, and alignment
The prototype SHALL compute natural target log-mel with the hash-locked `third_party/Wav2Lip/audio.py::melspectrogram`/`hparams.py` contract frozen in `design.md`, including 16 kHz, pre-emphasis, FFT/window/hop, librosa centering/padding, 80-bin frequency range, dB floor/reference, and output normalization. Natural mel frame `m` SHALL use center time `m*200/16000` seconds and enter a phone core iff that center lies in `[phone_start,phone_end)`. It SHALL use cached finite 1024-D TTS WavLM-L6 arrays at 320-sample stride. TTS phone rows SHALL use the exact Exp31/Python-round boundary convention and clamped half-open selection in `design.md`; fewer than two rows SHALL be rejected. `phone_phase_linear_v1` SHALL use the exact destination phase, clipped source coordinate, left/right indices, and interpolation equation in `design.md`, then insert mapped rows at target-core positions of a length-96 tensor and zero every other row. Every frame center, boundary, source coordinate/index, clip, and weight SHALL be recorded. The prototype SHALL estimate mel per-band and TTS-L6 normalization statistics from frozen train groups only, apply positive finite std floors, bind the statistics, and reuse them unchanged for evaluation. No waveform resampling, decoding, splice, learned attention, DTW, alignment fallback, or alternate frame-time convention is allowed.

#### Scenario: A phone trajectory is aligned
- **WHEN** an eligible phone has finite natural mel and TTS-L6 support
- **THEN** the output has one finite 1024-D TTS vector per natural target frame, exact recorded interpolation provenance, and no feature outside the matched TTS phone

#### Scenario: Evaluation normalization is requested
- **WHEN** evaluation tensors are built
- **THEN** only the immutable train-derived normalization statistics are applied and no evaluation statistic is fitted or updated

#### Scenario: Alignment would need a fallback
- **WHEN** frame support cannot satisfy the frozen interpolation contract
- **THEN** the mask is excluded before training rather than using a different aligner, waveform warp, clipping, extrapolation, or repeated boundary frame

### Requirement: Natural target content is hidden from the trainable input
The standardized natural `[96,80]` window SHALL be zeroed over target-plus-guard before entering the trainable context encoder and SHALL be accompanied by separate `[96,1]` masked-support and target-core channels. The model input SHALL NOT include clean target-core mel, natural WavLM/contextual embedding, transcript text, phone label/ID, reference token, or any downstream score. Train and evaluation code SHALL expose a leakage-audit function that traces every tensor field and verifies that changing clean target-core values after the masked input is built changes only the supervision target, not either model input branch. The report SHALL describe this as reconstruction beyond residual unmasked natural-context predictability, not as proof that natural context contains no target information.

#### Scenario: Leakage audit passes
- **WHEN** clean target-core mel is perturbed after masked inputs and aligned TTS are frozen
- **THEN** masked natural input, masked-support/core channels, and aligned TTS input remain byte-identical while the target changes

#### Scenario: A forbidden natural-content input is supplied
- **WHEN** a model batch includes natural WavLM, target phone ID/text, unmasked target mel, or a score-bearing field
- **THEN** schema validation fails before forward execution

### Requirement: One compact architecture and fixed training protocol
The prototype SHALL implement one `MaskedNaturalReconstructor` with exact inputs `[96,80]` masked mel, `[96,1]` masked-support, `[96,1]` target-core, and `[96,1024]` aligned TTS; the exact two-branch TCN structure, width, kernels, dilations, and sub-1.5M trainable-parameter limit are frozen in `design.md`. It SHALL contain no modality-present flag, attention, recurrence, pretrained trainable module, waveform decoder, TFG, or evaluator. For each seed `20260901`, `20260902`, and `20260903`, it SHALL instantiate once, hash the initial state, clone that byte-identical state into `FULL_CORRECT` and `NAT_ONLY`, and train FULL with paired aligned TTS while NAT_ONLY receives a zero TTS tensor. Before training, it SHALL create one shared 600-step schedule per seed using `PCG64(seed)`, sampling group uniformly with replacement, then record within group uniformly, then mask within record uniformly, for each of 600×16 positions; both arms SHALL consume the identical recorded schedule. Both SHALL run CPU-only with deterministic algorithms and recorded thread configuration, AdamW constants, batch size 16, gradient clip 1.0, exactly 600 steps, and target-core loss defined exactly as: `L_patch` is mean absolute mel error over all core frames/bins; `L_velocity` is mean absolute first-difference error only for adjacent frame pairs whose two endpoints both lie inside the unexpanded core; `L_total=L_patch+0.25*L_velocity`. No core/guard or window-boundary difference SHALL enter `L_velocity`. Only step 600 SHALL be eligible for evaluation; there SHALL be no validation selection, early stopping, seed retry, device switch, loss-weight search, or hyperparameter sweep.

#### Scenario: Fixed training completes
- **WHEN** one arm/seed reaches step 600 with finite losses and gradients
- **THEN** its checkpoint binds code/config/data/mask/normalization hashes, optimizer step, seed, frozen sampler sequence, and model parameter count

#### Scenario: A scientific constant is overridden
- **WHEN** a run requests alternate steps, seed set, width, loss weights, mask guard, learning rate, or architecture
- **THEN** frozen-config validation rejects the run before training

#### Scenario: A checkpoint is selected by evaluation loss
- **WHEN** an intermediate checkpoint would outperform step 600 on held-out masks
- **THEN** it remains diagnostic and cannot replace the frozen final checkpoint

### Requirement: One primary held-out comparison and two sensitivity diagnostics
For every eligible evaluation mask and seed, the prototype SHALL require `FULL_CORRECT`, `FULL_ZERO`, and `NAT_ONLY` exactly as defined in `design.md`, sharing target, fixed 96-frame natural context, masks, support, normalization, initial-state seed, and batch unit. FULL_ZERO SHALL use the unchanged FULL checkpoint and SHALL not enter FULL training. There SHALL be no modality-present flag. After the independent primary mask manifest freezes, the prototype SHALL attempt the optional `FULL_SHUFFLED` diagnostic using a donor selected only from frozen training-group masks, with different lexical phone label, duration ratio in `[0.5,2.0]`, and deterministic duration-mismatch/SHA-256 rank. Because the donor pool is train-only, no evaluation group's features SHALL enter another evaluation unit. If no donor exists, the primary mask SHALL remain and its shuffled condition SHALL record `diagnostic_missing`; this SHALL not change readiness, engineering, or science. Required predictions and `L_patch`, `L_velocity`, and `L_total` SHALL be finite. FULL_ZERO/FULL_SHUFFLED losses and prediction differences SHALL be reported only as out-of-distribution input-sensitivity diagnostics and SHALL NOT promote, block, or rescue feasibility. The sole primary comparison SHALL be FULL_CORRECT against the separately trained, identically initialized and scheduled NAT_ONLY.

#### Scenario: Complete paired evaluation occurs
- **WHEN** a frozen evaluation mask and seed are scored
- **THEN** the three required condition records share all non-TTS inputs and report predictions, losses, and input/output hashes; an available fourth shuffled diagnostic additionally binds its training-pool donor provenance and sensitivity

#### Scenario: Wrong-phone diagnostic donor is unavailable
- **WHEN** no training-group different-phone donor satisfies the frozen duration range
- **THEN** the primary target remains unchanged, FULL_SHUFFLED records `diagnostic_missing`, and no evaluation-group, same-phone, or score-selected fallback is used

#### Scenario: An ablation retrains the model
- **WHEN** zero-input or shuffled TTS is proposed as an additional trained checkpoint
- **THEN** conformance validation rejects it; those are same-checkpoint input interventions only

### Requirement: Hierarchical paired analysis and whole-group uncertainty
For every complete mask and seed, the prototype SHALL compute the sole positive-is-better primary `nat_gain=L_total(NAT_ONLY)-L_total(FULL_CORRECT)`. It SHALL also report non-promoting `zero_gap` and `shuf_gap` sensitivity diagnostics from the same FULL checkpoint. It SHALL aggregate median mask values within record/seed, median record values within source-group/seed, and median seed values within source group. It SHALL report every lower-level value but SHALL treat only the four final evaluation source-group `nat_gain` rows as inferential units. It SHALL compute deterministic 95% percentile intervals for median `nat_gain` with 10,000 whole-group `PCG64(20260901)` bootstrap draws and NumPy `method="linear"`, plus observed median relative loss reduction against NAT_ONLY. It SHALL NOT bootstrap, gate on, or causally interpret zero/shuffled diagnostics. Masks, records, conditions, frames, mel bins, and seeds SHALL NOT be resampled or reported as independent evidence.

#### Scenario: Complete group is analyzed
- **WHEN** all primary masks, three required conditions, and seeds for an evaluation source group are valid
- **THEN** it contributes exactly one final row to each bootstrap draw in which that source group is sampled

#### Scenario: A required condition or seed is incomplete
- **WHEN** any frozen mask lacks FULL_CORRECT, FULL_ZERO, or NAT_ONLY for any required seed
- **THEN** engineering is `NO_GO`, the group is not silently reduced, and no partial scientific decision is emitted

### Requirement: Conservative feasibility decision and claim boundary
The prototype SHALL emit separate engineering and science decisions. Engineering SHALL be `GO` only when all locks, fit-only checks, primary masks, normalization, leakage audits, tensors, six training runs, checkpoints, three required-condition evaluations, hierarchy, hashes, and schemas validate; optional shuffled-diagnostic availability SHALL not affect engineering. Engineering failure SHALL yield science `NOT_EVALUATED`. A complete preflight below the frozen readiness denominator SHALL yield science `INSUFFICIENT`. Otherwise science SHALL be `ALGORITHM_FEASIBLE` only when: (a) the bootstrap lower bound exceeds zero for `nat_gain`; (b) all four evaluation groups have positive final `nat_gain`; (c) median relative `L_total` reduction is at least 5% against NAT_ONLY; and (d) every seed has positive overall median `nat_gain` before seed aggregation. Every other complete outcome SHALL be `NO_ALGORITHM_FEASIBILITY_SUPPORT`. FULL_ZERO/FULL_SHUFFLED diagnostics SHALL not alter this decision.

`ALGORITHM_FEASIBLE` SHALL be described only as evidence that training and inference with paired phone-aligned TTS features improve masked natural log-mel reconstruction relative to an identically initialized/scheduled natural-context-only model under this frozen fit-only task. It SHALL NOT be described as proof of correct-content causality, TTS clarity retention, semantic enhancement, natural-prosody disentanglement, waveform reachability, candidate audio quality, visual gain, SyncNet gain, replacement effect, cross-TFG transfer, or population generalization.

#### Scenario: Every feasibility gate passes
- **WHEN** engineering is complete and all interval, group, practical-effect, and seed gates pass
- **THEN** science is `ALGORITHM_FEASIBLE` and only a later factorized natural-anchor/TTS-content prototype is authorized

#### Scenario: A complete gain gate fails
- **WHEN** denominators are sufficient but any interval, group, practical-effect, or seed gate fails
- **THEN** science is `NO_ALGORITHM_FEASIBILITY_SUPPORT`, with each failed gate and directional result reported explicitly

#### Scenario: A downstream claim is requested
- **WHEN** the result is used to claim waveform, TFG, SyncNet, or replacement benefit
- **THEN** the report rejects the claim as outside this prototype's authority

### Requirement: Immutable, resumable, and minimally scoped execution
The prototype SHALL write a new immutable run tree with lock, masks, features, per-seed FULL/NAT_ONLY checkpoints, three required-condition predictions, optional shuffled diagnostics, analysis, logs, validation, summary, and decision. A non-empty run path SHALL be rejected unless `--resume` is explicit. Resume SHALL reuse only artifacts whose input/config/code/output hashes and schemas validate and SHALL not overwrite unrelated or historical artifacts. JSON SHALL reject NaN/infinity; arrays SHALL reject object dtype. The implementation SHALL add no model service, framework, registry, database, dashboard, vendor-code modification, model download, Wav2Lip render, SyncNet run, or waveform output.

#### Scenario: Interrupted run resumes
- **WHEN** valid and incomplete stage artifacts exist and `--resume` is supplied
- **THEN** valid hash-bound artifacts are reused, incomplete cells resume deterministically, and prior experiments remain untouched

#### Scenario: Non-empty output is not resumed
- **WHEN** the requested run path contains files and `--resume` is absent
- **THEN** execution stops before writing, deleting, or training
