## Purpose

Defines a modest fit-only scale-up that tests whether paired token-level TTS-L6 provides masked natural-mel reconstruction information beyond both natural context and a train-only phone-centroid representation.

## ADDED Requirements

### Requirement: Score-blind expanded fit-only cohort
The experiment SHALL reuse the six prior train groups, exclude the four prior evaluation groups from the run, and deterministically select six additional train groups plus eight fresh evaluation groups from parents whose `protocol_split` is exactly `train`, using the eligibility and salted SHA-256 rules in `design.md`. It SHALL freeze source groups, records, masks, exclusions, paths, and hashes before schedule generation and SHALL NOT inspect or select on model outputs, prior losses, ASR/SyncNet/TFG scores, or sealed media. Readiness SHALL require exactly 12 train/eight evaluation groups, at least 36 train records/900 train masks, at least 16 evaluation records/400 evaluation masks, and at least 30 masks per evaluation group after phone-support filtering.

#### Scenario: Expanded lock is ready
- **WHEN** all fit-only locks and frozen denominators pass
- **THEN** training may start with `sealed_splits_accessed=false` and no source-group overlap

#### Scenario: Expanded denominator is unavailable
- **WHEN** any frozen group/record/mask minimum fails
- **THEN** science is `INSUFFICIENT`, no group or record is substituted, and training does not start

### Requirement: Train-only phone-centroid control
The experiment SHALL retain only phones supported by at least 20 train mask instances across at least three train groups. For every retained phone it SHALL compute one 1024-D centroid by averaging all standardized aligned TTS-L6 core frames from frozen train masks only. `PHONE_CENTROID` SHALL repeat that centroid over the target core and be exactly zero elsewhere. The centroid table SHALL bind contributing groups, records, masks, frames, and hashes; no evaluation value may contribute. Unsupported phones SHALL be removed once from the shared primary manifest rather than differently by arm.

#### Scenario: A supported phone is encoded
- **WHEN** a primary mask has a train-supported lexical phone
- **THEN** paired, centroid, and zero arms share the same mask while the centroid arm receives only the bound train-derived phone representation

#### Scenario: Evaluation data reaches a centroid
- **WHEN** any centroid contribution traces to an evaluation group
- **THEN** engineering is `NO_GO` and science is `NOT_EVALUATED`

### Requirement: Three identically controlled training arms
The experiment SHALL retain the completed prototype's model, tensors, masking, `phone_phase_linear_v1`, normalization, loss, optimizer, batch size, CPU determinism, and three seeds. It SHALL change only training duration to exactly 1,200 steps. For each seed it SHALL instantiate once and clone a byte-identical initial state into `PAIRED_TTS`, `PHONE_CENTROID`, and `NAT_ONLY`. All arms SHALL consume the same recorded `1,200×16` group→record→mask schedule. Only step 1,200 may be evaluated; validation selection, early stopping, retry, model scaling, and hyperparameter search are forbidden.

#### Scenario: One seed trains
- **WHEN** the three arms complete for a seed
- **THEN** their checkpoints bind the same initial-state hash, sampler hash, model/data/mask/normalization hashes, and final step

#### Scenario: An arm receives a different schedule
- **WHEN** any sampled batch position differs across arms for a seed
- **THEN** engineering is `NO_GO` and no scientific contrast is emitted

### Requirement: Required contrasts and diagnostic donors
Every evaluation mask/seed SHALL have finite `PAIRED_TTS`, `PHONE_CENTROID`, and `NAT_ONLY` predictions and losses with identical non-TTS inputs. The experiment SHALL compute `modality_gain=L_total(NAT_ONLY)-L_total(PAIRED_TTS)` and `token_gain=L_total(PHONE_CENTROID)-L_total(PAIRED_TTS)`. It MAY evaluate same-phone and wrong-phone train-only donors through the unchanged paired checkpoint using the deterministic duration/rank rules in `design.md`; donor cells are diagnostic-only and missing donors SHALL NOT change the primary denominator or decision.

#### Scenario: Primary evaluation is complete
- **WHEN** all three required arms exist for a frozen mask and seed
- **THEN** both primary contrasts bind shared target/context/mask hashes and separate checkpoint/input hashes

#### Scenario: A donor diagnostic changes direction
- **WHEN** same-phone or wrong-phone diagnostics appear favorable or unfavorable
- **THEN** their coverage/loss/output differences are reported but they cannot promote, block, or rescue science

### Requirement: Hierarchical token-signal decision
The experiment SHALL aggregate masks within record/seed, records within source-group/seed, and three seeds within source group by median, treating only the eight final evaluation groups as inferential units. It SHALL run 10,000 deterministic whole-group percentile bootstrap draws. It SHALL apply the modality and token gate sets exactly as frozen in `design.md` and emit one of `TOKEN_LEVEL_TTS_SIGNAL_SUPPORTED`, `PHONE_LEVEL_ONLY_SUPPORTED`, `NO_SCALEUP_SUPPORT`, `INSUFFICIENT`, or `NOT_EVALUATED`.

`TOKEN_LEVEL_TTS_SIGNAL_SUPPORTED` SHALL mean only that paired token-level TTS-L6 improves this mel reconstruction task beyond a train-only phone centroid. It SHALL NOT be reported as perceptible TTS-feature retention, waveform reachability, natural-prosody preservation, TFG/SyncNet gain, or replacement effect.

#### Scenario: Both gate sets pass
- **WHEN** modality and token interval, group-sign, seed-sign, and practical-effect gates all pass
- **THEN** science is `TOKEN_LEVEL_TTS_SIGNAL_SUPPORTED` and a separately specified waveform bridge is authorized

#### Scenario: Only modality gates pass
- **WHEN** paired TTS beats NAT_ONLY but does not pass every token-gain gate against PHONE_CENTROID
- **THEN** science is `PHONE_LEVEL_ONLY_SUPPORTED` and waveform/TFG promotion is not authorized

### Requirement: Immutable minimal execution
The experiment SHALL extend the existing package and write a new hash-bound run without modifying the completed v2 run, old checkpoints, datasets, caches, or vendor code. Resume SHALL reuse only schema/hash-valid artifacts. It SHALL add no waveform decoder, vocoder training, TFG/SyncNet run, service, framework, registry, or dashboard.

#### Scenario: Existing prototype artifacts are present
- **WHEN** the scale-up executes or resumes
- **THEN** prior runs remain byte-unchanged and all new outputs stay under the new run path
