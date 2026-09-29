## Context

See `proposal.md` and `specs/mfa-linear-sync-generalization-scale-200/spec.md` for the motivation and observable contract. The existing eight-record evaluator is intentionally frozen at its old denominator and cannot be edited in place. The repository's current eligible scan yielded 102 rows, so the scale experiment must consume a newly expanded fit-only inventory and fail before training unless its exact 200/40 split can be locked.

The implementation should reuse the current waveform model, target-margin SyncNet loss, signed PCM16 writer, official scoring-frame extraction, official 31-offset curve, FFV1 and Wav2Lip seams. A new scale-specific package and run root isolate the changed data-size protocol from the old four-record prerequisite and eight-record transfer artifacts.

## Goals / Non-Goals

**Goals:**

- Pre-register and hash-lock exactly 200 optimization records and 40 source-group-disjoint evaluation records before model training.
- Measure the effect of fit-set scale while holding architecture, loss, initialization seed, step count, thresholds, and official scoring conventions fixed.
- Train from a fresh identity state for exactly 100 steps using the equal-weight mean over all 200 records and no natural-audio model input.
- Produce a machine-recomputable real-video transfer decision and, only on transfer pass, a complete 360-cell strict natural-audio replacement audit.
- Preserve enough provenance to distinguish insufficient expanded data, invalid engineering execution, and a complete scientific negative.

**Non-Goals:**

- Modifying or reinterpreting `evaluate-mfa-linear-sync-transfer-and-replacement` or any sealed earlier run.
- Warm-starting from the four-record checkpoint, adding model capacity, changing the SyncNet loss, or tuning steps and thresholds.
- Testing sealed validation/test data, making population claims, or treating source-group holdout as proof of broad generalization.
- Adding a generic trainer/evaluator framework or running multiple seeds in this isolated data-scale comparison.
- Running Wav2Lip replacement after a failed real-video transfer gate.

## Decisions

### 1. Use a new scale-specific package and immutable run

Implement under `scripts/experiments/mfa_linear_sync_generalization_200/` with focused tests under `tests/experiments/mfa_linear_sync_generalization_200/`. Keep `scripts/experiments/mfa_linear_sync_transfer/` unchanged because its constants, eight-record manifest, and decisions are part of an earlier frozen protocol. Use a new run name such as `lrs3_mfa_linear_sync_generalization_200_20260903` and reject any existing non-empty output root.

Alternative: parameterize the old eight-record package. Rejected because it would make old constants and validators ambiguous and would risk changing the interpretation of an already sealed experiment.

### 2. Partition source groups before selecting records

Build one structural/calibration inventory from an expanded fit-only policy/MFA asset root. Exclude the four source groups used by the predecessor shared fit and any sealed split. Compute a fixed group key from a new literal salt, select the first 40 source groups as evaluation groups, and exclude them before selecting exactly 200 training records from the remaining groups using a separate fixed record key. Within each group, use deterministic bytewise ordering for representatives and preserve the entire eligible universe and exclusions.

This makes source-group disjointness an enforceable property rather than an after-the-fact report. Calibration is allowed only as the preregistered detached coordinate-eligibility check; no candidate or downstream outcome can affect either partition.

Alternative: select 200 rows first and take any remaining eight or forty groups. Rejected because it can leave an unregistered and outcome-sensitive evaluation denominator and does not make the group split explicit.

### 3. Reuse the exact loss path with a full-record mean

Load the existing waveform model at the same fresh seed and the same frozen SyncNet checkpoint. For each optimization step, calculate the target-margin loss independently for every locked training record and optimize the arithmetic mean across all 200 records. Preserve the existing 31-offset coordinate, window, target, sign, and positive-margin semantics. Use 100 steps, save step zero and step 100, and record model/optimizer/SyncNet hashes and the ordered record list.

If memory makes one simultaneous 200-record batch impractical, use deterministic sequential record evaluation with gradient accumulation that is mathematically equivalent to the declared mean, without changing record weights or order. Do not use early stopping or retry after a failed run.

Alternative: random minibatches or increased steps. Rejected because either changes the exposure/weighting comparison or confounds data scale with optimization budget.

### 4. Keep adapter input and evaluation audio roles separate

The training and inference adapter interface accepts exactly one floating tensor shaped `[1,1,61440]`. Natural audio is loaded only for detached reference-offset calibration and, after the real-video gate, the natural driver/evaluation arm. It is never passed to the model, loss, optimizer, checkpoint selection, or candidate QC. Hash the adapter state before and after the 40-record inference pass.

Use the exact candidate-to-MFA QC from the prior protocol, including residual peak `0.05`, normalized log-mel distance `0.10`, and saturation fraction `1e-4`. Generate each candidate once after the manifest is sealed.

Alternative: add natural-audio reconstruction or preservation loss. Rejected because it changes the TTS-only synchronization hypothesis and can conceal side-channel leakage.

### 5. Scale the real-video gate proportionally

For each of the 40 records, score MFA baseline and candidate on the identical canonical video with the official 31-point curve. Compute distance improvement as baseline minimum minus candidate minimum and separation improvement as candidate separation minus baseline separation. Keep per-record thresholds at `0.003` and target gap at `0.002`.

Require all 40 engineering-valid and at least 30 complete record-level successes, plus both across-record medians at least `0.003`, for `ADAPTER_HELDOUT_REAL_VIDEO_TRANSFER_OBSERVED`. A complete 40-row engineering-valid cohort that misses the gate is `NO_ADAPTER_HELDOUT_REAL_VIDEO_TRANSFER`; missing or invalid evidence is `REAL_VIDEO_TRANSFER_NOT_EVALUATED`.

The 30-of-40 rule preserves the prior 75% descriptive pilot gate while increasing the denominator. It is not a statistical significance threshold.

### 6. Preserve strict natural-column replacement as a conditional stage

Only after real-video transfer promotion, render one video per driver arm (`G_N`, `G_B`, `G_C`) for each evaluation record with the same fixed-crop Wav2Lip invocation. Materialize all nine driver/evaluation combinations, yielding exactly 360 cells. Use only `G_B_E_N` versus `G_C_E_N` for the authoritative replacement comparison; all diagonal and cross-audio cells are diagnostics.

Require all cells, at least 30 strict replacement successes, both median natural-column gains at least `0.003`, and the natural-driven oracle-offset checks. If any cell is absent or invalid, return `REPLACEMENT_NOT_EVALUATED`; never shrink the denominator.

Alternative: run the matrix after a real-video scientific miss. Rejected because the staged protocol intentionally prevents an expensive downstream result from being mistaken for transfer of a non-transferring adapter.

### 7. Validate and report the entire graph

Write create-once manifests, checkpoints, candidates, conditions, official curves, renders, cells, decisions, and report. A validator rereads all bound files and recomputes membership, disjointness, counts, hashes, waveform/video contracts, curve metrics, gains, status precedence, and claim scope. Resume is read-only and only returns a complete hash-valid terminal decision.

The report must explicitly call the evaluation source-group-heldout within fit-only data, distinguish engineering invalidity from scientific failure, and disclaim population/sealed-test, perceptual, content, speaker, multi-generator, and production conclusions.

## Risks / Trade-offs

- **The expanded fit-only asset universe may still contain fewer than 240 usable rows/groups.** → Perform the lock before training and emit `BLOCKED_DATASET_SCALE`; never duplicate rows or open sealed splits.
- **A 200-record mean may exceed GPU memory.** → Use deterministic gradient accumulation equivalent to the same arithmetic mean, and record the accumulation contract; do not switch to outcome-dependent minibatching.
- **More fit records can improve fit without improving source-group transfer.** → Keep the 40-group disjoint evaluation cohort fixed before training and report every record, not only medians.
- **Forty evaluation groups may still not represent the LRS3 population.** → Use only the bounded source-group-heldout claim and do not attach inferential significance.
- **Running 360 replacement cells is expensive.** → Keep it strictly gated on the real-video transfer pass and fail closed on any missing cell.
- **Reuse of current helpers could accidentally inherit an eight-record constant.** → Add scale-specific constants and tests for 200/40/30/360 at every stage; do not call old aggregate decision functions with implicit denominators.
- **New asset preparation could leak evaluation outcomes.** → Bind the full inventory and manifests before candidate inference; record selection keys and all exclusions.

## Migration Plan

1. Extend or prepare the fit-only policy/MFA inventory without opening sealed validation/test data.
2. Implement the new scale-specific package, tests, and immutable artifact validator.
3. Run focused tests and strict OpenSpec validation before any GPU execution.
4. Lock the 200-record training set and 40-source-group evaluation set; stop if `BLOCKED_DATASET_SCALE`.
5. Run the fresh 100-step shared training once; stop on any prerequisite or engineering failure.
6. Run candidate inference and complete 40-record official real-video scoring once.
7. Run the 360-cell Wav2Lip replacement only if the real-video transfer status is positive.
8. Validate and report the terminal graph. Rollback is additive: leave old runs untouched and remove no prior artifacts.
