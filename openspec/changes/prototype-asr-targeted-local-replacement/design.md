## Context

See `proposal.md` for motivation and `specs/asr-targeted-local-replacement-prototype/spec.md` for the behavior contract.

The prototype can reuse all expensive recognition work from `runs/lrs3_asr_sync_error_correlation_20260831/`. That run binds the ordered 24-record fit-only cohort, immutable Wav2Vec2 revision, per-arm greedy words, per-arm forced-reference word spans, and deterministic edit alignment. A metadata-only scan of those artifacts finds 71 natural error blocks: 52 have reference-side words, and after excluding mixed insertions and any non-equal TTS operation inside the mapped interval, 21 strict TTS-clean target blocks occur in 13 samples/source groups. Preflight must reproduce those facts from locked inputs; the scan is readiness evidence, not an alternative manifest.

The existing natural/TTS `05_alignment` files also contain SyncNet outcomes. Selection must not read those fields, because using low-sync scores to choose patches would turn the ASR-targeted experiment into a score-targeted oracle. Selection is therefore rebuilt from the locked `03_asr` records and the existing deterministic word-error alignment code. Existing video results are not reused; every required condition is rendered and muxed through one standardized chain.

Prior project evidence constrains the design:

- full MFA-linear and direct-resynthesis replacement did not retain a robust gain;
- global duration, phone-duration, and pause transfer were negative;
- feature improvement did not reliably transfer to Wav2Lip/SyncNet;
- Wav2Lip jobs sharing its hard-coded temporary directory can corrupt one another;
- candidate-driven audio/video self-consistency is not the endpoint.

This prototype therefore uses one transparent PCM patch operator, matched local-edit controls, isolated Wav2Lip work directories, and strict replacement with untouched natural PCM. It is a mechanism screen on one frozen Wav2Lip/SyncNet pipeline, not a prosody model or a generalization experiment.

## Goals / Non-Goals

**Goals:**

- Determine whether the specified ASR-targeted local PCM intervention shows a strict replacement gain over both natural identity and comparable non-error interventions.
- Make selection, candidate construction, rendering, muxing, scoring, and decisions reproducible from immutable inputs.
- Keep failures interpretable: distinguish insufficient eligible data, candidate-construction failure, render/scoring failure, and absence of promotion evidence.
- Reuse existing code and assets so implementation and execution remain small.

**Non-Goals:**

- Establish a population-wide or cross-TFG causal conclusion from this discovery cohort.
- Train a prosody encoder, residual head, selector, ASR model, Wav2Lip, or SyncNet.
- Search patch widths, crossfades, seeds, decoders, thresholds, donor rules, or statistical gates after observing scores.
- Use current low-sync values, real-video visual distances, or any downstream score to select samples, target blocks, or controls.
- Claim that the intervention isolates prosody; duration normalization, TTS acoustics, and boundary blending remain part of the tested operator.

## Decisions

### 1. Reuse the frozen 24-record cohort and ASR emissions, but recompute selection

The prototype reads:

- `runs/lrs3_asr_sync_error_correlation_20260831/01_manifest/manifest.json`;
- `03_asr/records/{natural,tts}/<sample_id>.json`;
- their success markers and model/config locks;
- canonical natural/TTS audio and source video already bound by that run.

It validates all hashes and reproduces greedy/reference edit operations with the existing `word_errors` implementation. Selection projects only identifiers, fit/source-group provenance, transcript words, operation identities, forced start/end spans, audio identity, and floor/ceil quantized bounds. It does not use `04_sync`, `05_alignment`, `06_analysis`, or plots during cohort and block selection, and does not read downstream score-bearing QA fields. After the manifest is frozen, no old downstream artifact is loaded: reconstruction is the sole selection source.

A sample is eligible only if it has at least one target block and two complete matched-control assignments. The prototype asserts exactly one retained sample per `source_group`; at least 12 distinct source groups are required for a scientific prototype decision. Otherwise the run reports `INSUFFICIENT` without substituting samples.

Alternative considered: use all natural error spans directly from `05_alignment`. Rejected because those files mix ASR-derived spans with downstream SyncNet fields and make score-blind selection harder to audit.

### 2. Target only reference-mappable natural errors with ASR-clean TTS donors

Start from contiguous natural edit blocks. A target block must:

1. contain a non-empty, strictly increasing, contiguous set of reference word indices;
2. have finite monotonic natural and TTS forced-reference spans for those words;
3. contain at least one natural substitution or deletion;
4. have every corresponding TTS reference word aligned as `equal` by the frozen TTS greedy-ASR edit alignment, with the complete TTS operation interval from the first through last mapped reference operation containing only `equal` operations;
5. contain at least one natural substitution/deletion and no natural insertion operation (mixed insertion blocks are excluded);
6. lie fully inside both decoded waveforms after sample-boundary quantization.

Pure natural insertions and mixed insertion blocks are excluded from the primary prototype because they have no unambiguous reference-side donor interval. The natural target interval is the union from the first to last forced-reference word, not the union with unrelated prediction timing. The TTS donor interval is the corresponding TTS forced-reference interval. Timing is quantized with `start=floor(start_s*16000)` and `end=ceil(end_s*16000)`, clipped to the waveform, and rejected if empty.

This makes the estimand conditional: it asks whether replacing natural-ASR mistakes with TTS segments that the same recognizer gets right is useful. It does not estimate effects for every ASR error.

Alternative considered: allow any TTS donor. Rejected because a donor that is itself ASR-wrong does not test the proposed “clearer TTS articulation at a natural error” mechanism.

### 3. Use two matched correct-region controls rather than a low-sync oracle

For every target block, build a pool of maximal contiguous reference-word runs in the same sample where both natural and TTS greedy alignments are `equal` and the complete TTS operation interval contains only `equal` operations. Admissible contiguous windows are generated in ascending `(start_reference_index, end_reference_index)` order. Candidate controls are ranked without downstream scores by:

1. absolute reference-word-count difference;
2. absolute log duration-ratio difference on the natural clock;
3. absolute log TTS donor/destination duration-ratio difference;
4. a stable hash derived from sample ID, target block ID, control replicate, and seed `20260901`.

A control candidate is admissible only when its word-count difference is at most one, its natural duration is between one half and twice the target duration, and it does not overlap any target block. Within each replicate, selected blocks must not overlap each other. Two distinct complete assignments are selected with lexicographic backtracking over target blocks in ascending target-block identifier order; the second assignment forbids first-assignment block IDs for the same target. Failure to assign two distinct complete control sets makes the sample ineligible before rendering.

Each control patch uses its own same-text TTS reference block and its own natural destination. The total target/control edit durations need not be byte-identical, so analysis records their quantized ratio and requires each replicate's total edited duration to remain within `[0.8, 1.25]` of the targeted total. Outside that range the sample is ineligible. `control_*_adv` is a predeclared descriptive comparator, not a randomized causal estimate of targeting; the conditional intervention estimands are the natural-baseline gains.

Alternative considered: circularly move the target donor waveform to a random time. Rejected because it changes linguistic content and would make ASR targeting appear favorable for a trivial reason. A low-sync oracle arm is also excluded to keep the prototype narrow and score-blind.

### 4. Freeze one exact-length PCM patch operator

All audio is decoded as mono 16 kHz PCM16 and represented as finite float32 for patching after exact hash validation. Every timing interval uses half-open bounds with `start=floor(start_s*16000)` and `end=ceil(end_s*16000)`, clipped to its waveform and rejected if empty. SciPy `1.18.0` and `scipy.signal.resample` are frozen. For a source interval of `S` samples and a destination interval of `L` samples:

1. extract the TTS source interval using half-open sample bounds;
2. apply `scipy.signal.resample` once to produce exactly `L` finite samples;
3. construct an in-segment raised-cosine mask whose boundary ramp is `r=min(320, floor(L/4))` samples;
4. write `(1-mask)*natural + mask*resampled_tts` into a copy of natural;
5. apply all disjoint blocks in chronological order;
6. reject non-finite values or clipping instead of normalizing, limiting, or changing gain;
7. write mono 16 kHz PCM16 with exactly the natural sample count.

The natural condition is decoded and written through the same PCM writer with no patch. Its decoded output must be byte-identical to the canonical natural decoded PCM. Candidate differences outside each destination interval must be exactly zero before PCM quantization; the output records changed-sample bounds and counts.

This operator intentionally tests a concrete duration-normalized waveform patch, not “prosody transfer” in isolation. Matched controls share the same operator, limiting but not eliminating processing confounding.

Alternatives considered: a phase vocoder adds short-segment failure/fallback behavior; a mel seam requires a modified Wav2Lip input path and would not be a plug-compatible waveform driver. Both are deferred unless this waveform prototype produces interpretable evidence.

### 5. Render four conditions and score only strict natural-audio replacement

Required conditions per eligible sample are fixed and ordered:

```text
natural
asr_targeted
target_control_0
target_control_1
```

Every condition uses the same frozen face input, Wav2Lip checkpoint, flags, batch sizes, and deterministic runtime. Each cell gets an isolated working directory containing its own `temp/`; concurrent use of a shared Wav2Lip cwd is forbidden. One pre-run identity smoke cell must demonstrate that rerunning the natural condition produces the expected decoded PCM and stable score within the existing evaluator's numeric tolerance.

For condition `k`, Wav2Lip creates `V_k` from its driver audio. The scoring media is always:

```text
R_k = strict_mux(copy_video=V_k, audio=untouched_canonical_natural)
```

The mux must contain copied video plus mono 16 kHz `pcm_s16le` natural audio, with no crop, pad, stretch, duration cap, `-shortest`, or video re-encode. Decoding `R_k` audio must reproduce canonical natural PCM exactly. Candidate-driver audio is never used as the final SyncNet audio.

Existing Wav2Lip renders are not reused because same-chain rendering is part of the causal contrast. Valid per-cell markers may be reused by hash-aware `--resume` inside this run.

### 6. Preserve official global scores and add a fixed-coordinate local diagnostic

Official SyncNet V2 runs unchanged with `vshift=15`, `min_track=50`, the locked checkpoint, score-independent longest-track selection, and full distance-matrix export. Global Sync-C and Sync-D are primary. For every sample, all four conditions must select identical track index, frame count, start/end range, and scorer-input support; a mismatch makes the sample incomplete and engineering `NO_GO`. The fixed local coordinate is explicit: `j_natural = argmin_j mean_t(D_natural[t,j])`, `a=t+j_natural-vshift`, and timestamp `(track_start_frame+a+2)/fps`. Compare raw `D[t,j_natural]` on common rows after excluding shift padding, four median-filter edge rows, rows outside the selected track, and rows outside natural audio. Edited support is the union of destination intervals expanded by 0.20 seconds and clipped to common support. Local diagnostics remain secondary and cannot alter official scores.

### 7. Use paired sample/source-group estimands and one frozen decision rule

For sample `i`:

```text
baseline_C_gain_i = C(asr_targeted) - C(natural)
baseline_D_gain_i = D(natural) - D(asr_targeted)
control_C_adv_i   = C(asr_targeted) - mean(C(control_0), C(control_1))
control_D_adv_i   = mean(D(control_0), D(control_1)) - D(asr_targeted)
```

Positive values always favor ASR targeting. Report every per-sample value, medians, means, win counts, and exact edit-budget diagnostics.

Use whole-sample/source-group bootstrap with 10,000 `PCG64(20260901)` draws, where `unit_id=source_group` and exactly one retained sample is required per unit. Use NumPy's `method="linear"` percentile convention. Each draw resamples complete four-condition rows and recomputes all four medians. No frame or patch is an independent inferential unit.

Decision logic:

- any required engineering failure: engineering `NO_GO`, science `NOT_EVALUATED`;
- fewer than 12 complete distinct source-group rows: science `INSUFFICIENT`;
- otherwise `PROTOTYPE_SUPPORT` only if the 95% lower bound is positive for the two natural-baseline median estimands and the two descriptive control contrasts;
- otherwise `NO_PROTOTYPE_SUPPORT`.

`NO_PROTOTYPE_SUPPORT` means the promotion rule was not met; the report must separately identify intervals crossing zero and intervals entirely at or below zero. `PROTOTYPE_SUPPORT` authorizes a fresh fit-only confirmation design or a bounded learned head; it is not a cross-TFG or population-level conclusion.

### 8. Keep the implementation and artifact tree narrow

Use one operator entry point and six small stages:

```text
scripts/experiments/asr_targeted_local_replacement/
├── config.py
├── protocol.py
├── patch_audio.py
├── evaluate.py
├── analyze.py
└── run.py

runs/lrs3_asr_targeted_local_replacement_prototype_20260901/
├── 00_lock/
├── 01_candidates/
├── 02_renders/
├── 03_replacement/
├── 04_sync/
├── 05_analysis/
├── logs/
├── summary.json
└── decision.json
```

Reuse strict JSON/hash/marker helpers, edit alignment, strict mux, Wav2Lip invocation patterns, SyncNet adapter, and local-score parity code. Do not create a base-stage framework, registry, dashboard, service, database, or new model abstraction. One compact per-sample plot may show the four global scores and fixed-coordinate distance change; plots are diagnostic only.

CLI flags are limited to run path, device, stage, and resume. Scientific constants are code/config values, not casual flags. Network access is permitted only during preflight to install a pinned missing package; inference assets must be local and hash-locked before cohort media is opened.

## Risks / Trade-offs

- **[Discovery-cohort reuse]** The same 24 records motivated the intervention, so prototype support is not fresh confirmation. → Label the result as prototype-only and require a later source-group-separated fit-only confirmation before a broader causal claim.
- **[Small eligible denominator]** Strict donor and control matching leaves 13 eligible samples in the current metadata scan. → Freeze `INSUFFICIENT <12`; do not relax donor/control rules or substitute samples after scores exist.
- **[ASR-selected conditional estimand]** Results apply only where natural is wrong and TTS is correct under one recognizer. → State this condition in every summary and do not generalize to all low-sync or all speech regions.
- **[Duration normalization changes pitch/timbre]** The operator does not isolate “TTS clarity” or “prosody.” → Name the intervention exactly, use the same operator in matched controls, and reserve component attribution for later experiments.
- **[Boundary artifacts]** Crossfades and resampling can affect Wav2Lip beyond the linguistic error. → Keep the operator fixed, report edit budgets and local guard regions, and reject clipping/non-finite outputs.
- **[Weak ASR–sync overlap]** Most ASR-error time is not low-sync, so the prototype may be underpowered or diluted. → Treat an inconclusive/no-support outcome as a valid stop signal; do not add a low-sync oracle after seeing results.
- **[Global Sync-C is nonlinear]** Sparse local changes may not move whole-track confidence measurably. → Retain official global C/D as endpoint and use fixed-coordinate local distance only as a predeclared mechanism diagnostic.
- **[Wav2Lip/SyncNet specificity]** A gain may not survive another TFG or evaluator. → Do not claim plug-and-play success; cross-TFG work is downstream of a fresh confirmation.
- **[Wav2Lip temp collision]** Shared working directories can corrupt renders. → Require per-cell cwd/temp isolation and bind video hashes to cell markers.
- **[Score leakage]** Existing records colocate ASR-derived and SyncNet-derived data. → Build selection from `03_asr` only, log every field/path consumed, and test that changing old SyncNet artifacts does not change the manifest.

## Migration Plan

This is an additive prototype with no migration of existing code or results.

1. Add the narrow package and synthetic unit tests.
2. Run metadata-only lock/eligibility preflight and stop if fewer than 12 complete samples can be assigned two controls.
3. Run one isolated identity/candidate smoke sample and validate PCM, render, strict mux, and SyncNet parity.
4. Execute all four conditions for the locked eligible cohort with hash-aware resume.
5. Validate artifacts, compute the frozen decision once, and run `openspec validate prototype-asr-targeted-local-replacement --strict`.
6. If engineering fails, preserve the immutable partial run and retry only in a new run directory. Existing runs, checkpoints, model caches, and sealed split locks remain untouched.
