## Context

See `proposal.md` for motivation and `specs/mfa-linear-sync-transfer-and-replacement-evaluation/spec.md` for the behavior contract.

The predecessor change has a validated one-record P1 result at `runs/lrs3_mfa_linear_real_video_sync_20260903_v4/`, but P2 was deliberately not requested. Its implementation already contains a fresh four-record shared fit, exact-length waveform generation, fixed real-video materialization, official-equivalent SyncNet curves, QC, create-once outputs, and machine validation. This change must consume those seams rather than introduce another trainer.

The central distinction is between three questions:

1. Can one record be optimized against itself? P1 already says yes.
2. Can one shared adapter improve untouched source groups on fixed real video? This change's real-video stage answers that.
3. Does that audio change improve motion produced by a frozen talking-face generator when both generated videos are evaluated against untouched natural audio? Only the strict replacement stage answers that.

A diagonal pair such as candidate-driven video scored with candidate audio cannot answer question 3 because the video and scorer audio may co-adapt. The authoritative comparison therefore holds evaluation audio fixed to natural while changing only the driver that produced the generated video.

## Goals / Non-Goals

**Goals:**

- Consume exactly one validated fresh P2 shared checkpoint without training or checkpoint search in the follow-up evaluator.
- Pre-register an eight-record, eight-source-group adapter-heldout cohort before candidate inference.
- Reuse the predecessor's fixed coordinates and full official SyncNet curves to make real-video transfer directly comparable with the MFA baseline.
- Test downstream transfer with a complete matrix that clearly separates driver-video effects from scorer-audio effects.
- Keep the follow-up small enough for rapid execution while making every denominator, threshold, condition, and terminal decision machine-recomputable.

**Non-Goals:**

- Reopening or appending P2 to the sealed P1 `v4` run; P2 requires its own fresh create-once run.
- Additional optimization, fine-tuning, architecture changes, hyperparameter search, multiple adapter seeds, or checkpoint selection.
- A sealed LRS3 test-set claim or population inference. The eight records are held out only from adapter optimization.
- Evaluation of multiple talking-face generators. The first downstream gate uses one frozen Wav2Lip GAN checkpoint.
- ASR/PER, speaker verification, MOS, listening tests, perceptual lip-sync judgment, or adversarial-robustness claims.
- A claim that natural audio is unavailable during evaluation. It is intentionally needed as detached coordinate truth and strict replacement audio, but never as adapter input.

## Decisions

### 1. Treat P2 as a hard upstream gate, not part of this evaluator

Run the predecessor command in a new immutable output root with its explicit P2 opt-in. It must repeat P0/P1 and then create a fresh P2 model exactly as already specified; it must not append to `runs/lrs3_mfa_linear_real_video_sync_20260903_v4/`. Bind:

```text
terminal status             FIXED_DATA_SHARED_TTS_ONLY_SYNC_CONSTRUCTED
P2 steps                    100
fit records                 exact frozen four, exact frozen order
initial model hash          same fresh seed-20260903 identity state
step-100 model hash         from P2 artifact
SyncNet hash                before == after == locked V2 hash
validation                  artifact_graph_valid == true
```

The follow-up loader validates the whole parent graph and then loads only the final P2 state dict. If this gate fails, the follow-up writes a blocked decision and stops before cohort candidate generation.

Alternative considered: evaluate the successful P1 step-20 model immediately. Rejected because a one-record checkpoint would make any apparent transfer dominated by one-record fitting and would skip the already-designed shared-model gate. Alternative considered: warm-start P2 from P1. Rejected because the predecessor protocol explicitly requires a fresh shared model.

### 2. Freeze one representative from each of eight untouched source groups

Build an eligibility table from the same hash-locked policy/MFA asset universe as the predecessor while excluding all four P2 source groups. Structural eligibility is evaluated before adapter state loading. Natural-reference offset calibration is then performed as a predeclared label-quality check, still before candidate inference.

For each eligible record compute:

```text
record_key = sha256(
  "mfa-linear-sync-transfer-v1\0" + source_group + "\0" + sample_id
)
```

Within each source group, retain the record with the lowest bytewise `record_key`. Sort those group representatives by `(record_key, source_group bytes, sample_id bytes)` and take the first eight. Persist the full structural universe, calibration failures, all representatives, and selected rows. This gives exactly one evaluation unit per source group without using baseline, candidate, or downstream outcomes.

The manifest is written create-once before opening the P2 state dict. If eight groups cannot be locked, the denominator is not reduced and no later record may substitute for a failed selected record.

Alternative considered: use the three unused P2 records plus several new records. Rejected because any P2 record was exposed to optimizer gradients. Alternative considered: select records whose MFA baseline scores are poor, leaving more room for improvement. Rejected as outcome selection.

### 3. Separate structural/calibration preflight from scientific scoring

Natural audio is required for two evaluation roles:

- compute one detached natural/real-video curve and `s_ref` for each selected record;
- form `N` driver and evaluation arms in the replacement audit.

The adapter-facing record remains exactly:

```text
{"mfa_linear_tts_waveform": Tensor[1,1,61440]}
```

Use a dedicated adapter wrapper whose signature accepts only that tensor. Candidate inference runs under `torch.inference_mode()`, and adapter parameter/buffer hashes are checked before and after all records. The wrapper rejects kwargs and record dictionaries so paths, labels, or natural tensors cannot leak accidentally.

Alternative considered: use natural mel distance as a quality gate. Rejected because it turns evaluation into natural-acoustic reconstruction and changes the tested hypothesis. Candidate-to-MFA QC remains the only acoustic trust measure.

### 4. Reuse exact P1/P2 waveform and real-video scoring seams

Promote no duplicate MFCC or curve formula. Reuse the current package's signed PCM16 writer, official JPEG/BGR scoring-frame extraction, 5-frame visual/20-MFCC audio windows, raw Euclidean 31-offset curve, mux verification, curve metrics, and waveform QC. The evaluation package adds cohort orchestration and paired analysis only.

For each record:

```text
B = exact MFA-linear input
C = frozen P2 adapter(B)
V = canonical 96-frame real tracked-face segment

score(V,B) and score(V,C)
D = min(curve)
C_sync = median(curve) - min(curve)
```

Both condition files stream-copy the same FFV1 visual segment. Demuxed PCM and every official JPEG/BGR scoring frame are verified. The candidate must respect the predecessor's `0.05` residual bound, `0.10` normalized log-mel limit, and `1e-4` PCM saturation fraction.

No optimizer, autograd smoke, or differentiable proxy is necessary in this stage. If a proxy is retained for diagnostics, per-offset official parity remains mandatory and only official file curves feed the decision.

Alternative considered: score only official summary D/C values. Rejected because full curves are needed to recompute D/C, detect ties, check natural-target offsets, and audit sign conventions.

### 5. Use a descriptive six-of-eight real-video gate

For record `r`:

```text
D_gain_r = D(B_r) - D(C_r)
C_gain_r = C_sync(C_r) - C_sync(B_r)
record_success_r =
    D_gain_r >= 0.003
    and C_gain_r >= 0.003
    and argmin(curve_C_r) == s_ref_r
    and candidate best/second gap > 0.002
    and all engineering/QC gates pass
```

Promotion requires at least six record successes plus median D and C gains each at least `0.003`. All eight rows must be engineering-complete; an execution failure is `NOT_EVALUATED`, not a scientific miss.

Six of eight is a preregistered pilot robustness rule: it requires agreement across most independent source groups while tolerating at most two scientific counterexamples. It is not a significance threshold. Do not bootstrap or report a p-value as evidence of population generalization.

Alternative considered: require all eight records. Rejected as too brittle for a first shared-checkpoint transfer gate. Alternative considered: median-only promotion. Rejected because a few large gains could hide broad regressions.

### 6. Gate Wav2Lip execution on real-video transfer

The 3×3 replacement matrix runs only after the real-video gate passes. This avoids an expensive downstream matrix when the adapter does not transfer even to untouched real-video coordinates. It does not alter the cohort: P4 uses the exact same manifest and checkpoint.

A P3 scientific failure seals the run with `REPLACEMENT_NOT_RUN_REAL_VIDEO_GATE_FAILED`. It does not trigger retraining or a weaker cohort. A new hypothesis would require a new OpenSpec change.

Alternative considered: always run replacement for diagnostic completeness. Rejected for the rapid staged experiment because it spends substantially more compute after the necessary upstream transfer condition has failed.

### 7. Render natural, MFA baseline, and candidate driver videos once

Use the existing hash-locked Wav2Lip GAN checkpoint:

```text
ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8
```

and SyncNet V2 checkpoint:

```text
961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442
```

For every selected record, pass the already canonical 224×224 tracked-face segment as the Wav2Lip face input for all driver arms. Freeze the invocation, including `--nosmooth`, batch sizes, frame rate, code hash, Python executable, CUDA/device provenance, and working-directory isolation. The only arm-varying input is driver PCM:

```text
G_N <- natural PCM
G_B <- MFA-linear PCM
G_C <- candidate PCM
```

Ignore the audio stream produced by Wav2Lip. Decode each generated video once, validate equal frame count/geometry across the three arms, and rematerialize its frames as a canonical video-only lossless FFV1 stream with BGR frame hashes. Because the input is already a tracked crop, do not run `run_pipeline.py` or independently redetect a face. This preserves each generated visual row exactly across its three audio-evaluation columns.

Before the eight-record run, add a P0 render seam using one disposable/predeclared record to verify that the Wav2Lip output can be treated as the direct tracked crop, all arms produce the same frame support, FFV1 round-trip preserves BGR hashes, and the unmodified `run_syncnet.py` scorer accepts the prepopulated crop without tracking. This is an engineering fixture only and must not be used to tune scientific thresholds.

Alternative considered: rerun face detection/tracking on each of nine condition files. Rejected because tracking differences and repeated lossy encoding could masquerade as audio replacement effects. Alternative considered: use only baseline and candidate driver arms. Rejected because the natural driver is needed to define the frozen-generator oracle offset and ceiling diagnostic.

### 8. Materialize a complete 3×3 driver/evaluation matrix

Use notation `G_X_E_Y`, where `X` is the waveform that drove Wav2Lip and `Y` is the waveform muxed for SyncNet evaluation:

```text
             E_N       E_B       E_C
G_N        G_N_E_N   G_N_E_B   G_N_E_C
G_B        G_B_E_N   G_B_E_B   G_B_E_C
G_C        G_C_E_N   G_C_E_B   G_C_E_C
```

For each row, stream-copy one canonical generated FFV1 video into all three cells and mux the exact signed PCM16 evaluation arm. Verify output video frame hashes against the row parent and demuxed PCM bytes against the column parent. Require the same computed window count in all nine cells of a record and store all 31 curve values.

Only the natural-audio column answers replacement:

- `G_B_E_N`: baseline generated motion after strict natural replacement;
- `G_C_E_N`: candidate generated motion after strict natural replacement.

`G_N_E_N` defines the downstream oracle shift and reference ceiling. The remaining six cells diagnose audio-domain sensitivity and diagonal co-adaptation. They are displayed but never enter promotion.

Alternative considered: a 2×2 natural-versus-candidate matrix. Rejected because it cannot directly answer whether the adapter improves over its own MFA-linear input. The 3×3 adds the required baseline while retaining the standard natural/candidate controls.

### 9. Require replacement gain at a fixed natural-audio column and oracle offset

For each record:

```text
D_repl_gain = D(G_B_E_N) - D(G_C_E_N)
C_repl_gain = C_sync(G_C_E_N) - C_sync(G_B_E_N)
s_oracle = unique argmin(curve_G_N_E_N)
```

A replacement success requires both gains at least `0.003`, `G_C_E_N` to have a unique best shift equal to `s_oracle` with gap greater than `0.002`, and all nine cells to be valid. Promotion uses the same six-of-eight and both-median-at-least-`0.003` rule as P3.

Matching the natural-driven oracle offset prevents a lower minimum at a different lag from being called successful replacement. The natural ceiling itself is reported but not required to be reached; the primary question is improvement over MFA at fixed natural evaluation audio.

Alternative considered: promote on `G_C_E_C > G_B_E_B`. Rejected because this is a diagonal comparison and permits complete driver/scorer co-adaptation. Alternative considered: compare `G_C_E_N` only with `G_N_E_N`. Rejected because failure to equal a natural oracle does not answer whether the adapter improved over MFA.

### 10. Keep engineering and scientific decisions orthogonal

Use a terminal object with independent fields:

```text
engineering_status
p2_prerequisite_status
real_video_transfer_status
replacement_status
claim_scope
```

Representative explicit states include:

```text
BLOCKED_P2_NOT_PASSED
BLOCKED_COHORT_LOCK
OUTCOME_SELECTION_VIOLATION
NATURAL_OR_SIDE_CHANNEL_LEAKAGE
FROZEN_ADAPTER_MUTATION
REAL_VIDEO_SCORE_INCOMPLETE
ADAPTER_HELDOUT_REAL_VIDEO_TRANSFER_OBSERVED
NO_ADAPTER_HELDOUT_REAL_VIDEO_TRANSFER
REPLACEMENT_NOT_RUN_REAL_VIDEO_GATE_FAILED
RENDER_MATRIX_INCOMPLETE
REPLACEMENT_NOT_EVALUATED
FROZEN_WAV2LIP_REPLACEMENT_TRANSFER_OBSERVED
NO_FROZEN_WAV2LIP_REPLACEMENT_TRANSFER
```

An engineering-invalid run has no scientific result. A complete matrix that misses its promotion threshold is a valid scientific negative. Never collapse both into a generic `NO_GO`.

### 11. Use create-once stage artifacts and a recomputing validator

Proposed implementation and run layout:

```text
scripts/experiments/mfa_linear_sync_transfer/
├── config.py          # fixed salts, arms, thresholds, parent hashes
├── protocol.py        # P2/cohort locks and inference schema
├── real_video.py      # baseline/candidate conditions and paired metrics
├── replacement.py     # Wav2Lip renders, FFV1 rows, 3×3 mux/scoring
├── evaluate.py        # record/aggregate gates
├── validate.py        # recompute graph and decisions
└── run.py             # staged orchestration and read-only resume

tests/experiments/mfa_linear_sync_transfer/
├── test_protocol.py
├── test_real_video.py
├── test_replacement_matrix.py
├── test_decision.py
└── test_validation.py

runs/lrs3_mfa_linear_sync_transfer_20260903/
├── 00_parent_lock/
├── 01_cohort_lock/
├── 02_p0_seams/
├── 03_real_video/{records,decision.json}/
├── 04_replacement/{renders,matrix,decision.json}/  # gated
├── validation.json
├── decision.json
└── report.md
```

Every stage writes immutable per-record artifacts before its aggregate decision. The validator rereads source files, recomputes canonical hashes, exact membership, curve metrics, gains, counts, medians, and status precedence. Resume is read-only and only allowed after complete terminal validation. A partial run is preserved for diagnosis but never repaired in place.

Do not build a generic evaluator framework. Reuse current helpers directly or move narrowly shared code to a stable current-tree module with regression coverage; never import runtime code from a historical worktree.

### 12. Stop claims at the exact evidence boundary

A P3 pass is an empirical source-group-disjoint transfer observation for the frozen cohort, not a dataset-heldout or population claim. A P4 pass adds one frozen Wav2Lip/SyncNet strict-replacement observation. It does not establish transfer to Ditto or other TFGs, and it says nothing about human-visible lip sync, intelligibility, speaker identity, acoustic naturalness, or deployability.

If P4 passes, the appropriate next change is a separately preregistered multi-TFG replication and content/identity/perceptual audit. If P4 fails while P3 passes, the correct conclusion is that real-video SyncNet optimization did not survive this downstream replacement test; do not salvage it using diagonal cells.

## Risks / Trade-offs

- **[Eight adapter-heldout groups are still a small fit-only sample]** → Use them only as a fixed empirical gate, report every record, and forbid population language.
- **[Natural-reference eligibility can favor records with clean SyncNet geometry]** → Freeze and disclose the full eligibility denominator and describe the cohort as calibration-eligible rather than representative.
- **[The adapter may exploit SyncNet while staying inside waveform QC]** → Require downstream natural-audio replacement and keep perceptual claims out of scope.
- **[Wav2Lip may produce a different frame count than its 96-frame input]** → Require identical nonzero output support across all three driver arms, derive one recorded window count, and fail rather than trim or pad an arm.
- **[Generated-video compression can change SyncNet scores]** → Canonicalize each rendered visual row once to lossless FFV1 and reuse identical frames across columns.
- **[Direct scoring of generated tracked crops differs from the common detector/tracker pipeline]** → Use already tracked 224×224 inputs, lock direct-crop semantics in P0, and describe the result as this fixed-crop Wav2Lip protocol.
- **[Conditional P4 execution hides replacement behavior after a P3 failure]** → Record the skipped status explicitly; the staged design intentionally answers replacement only for an adapter that first transfers on real video.
- **[Six-of-eight and 0.003 are engineering-scientific pilot gates, not inferential guarantees]** → Label them preregistered descriptive thresholds and do not attach population significance.
- **[Natural evaluation audio and candidate driver may differ acoustically]** → That mismatch is intentional: holding natural audio fixed is what removes diagonal co-adaptation from the authoritative contrast.

## Migration Plan

This is additive experimental evaluation; no production migration is required.

1. Finish regression tests and strict validation for the predecessor P2 path without changing its frozen protocol.
2. Run P2 once in a fresh immutable run root and stop immediately if its validated shared-fit status does not pass.
3. Implement parent/cohort locks and freeze the eight-record manifest before candidate inference.
4. Add the real-video paired evaluator and machine-recomputable transfer decision.
5. Add the fixed-crop Wav2Lip/FFV1/matrix P0 seam.
6. Run P3 once; run P4 only on its declared pass.
7. Validate the complete artifact graph and generate the bounded report.
8. Preserve all negative and engineering-failure outputs. Rollback means leaving the additive evaluator unused; prior runs and third-party code are never rewritten.
