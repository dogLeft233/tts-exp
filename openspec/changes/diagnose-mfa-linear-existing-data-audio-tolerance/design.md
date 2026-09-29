## Context

The corrected existing-data run has a valid immutable manifest and completed 94-record fresh training artifacts, but its candidate stage stopped at the original normalized log-mel limit for `lrs3_7DLzXAjscXk_00006`. Read-only inspection found a distance of `0.1005600542`; residual, saturation, shape, and finite-value checks passed. This follow-up is explicitly diagnostic and post-hoc.

## Goals / Non-Goals

**Goals:**

- Reuse exactly the completed 94-record training state and fixed 8-row manifest.
- Permit the single narrowly over-limit candidate and score all eight rows under official real-video SyncNet.
- Preserve all synchronization and replacement gates, provenance checks, and natural-audio isolation.
- Make the changed audio tolerance and non-preregistered status impossible to miss in artifacts and report.

**Non-Goals:**

- Presenting the result as a pre-registered scientific transfer or changing the 94/8 cohort.
- Retraining, checkpoint search, extra seeds, candidate repair, waveform tuning, or selection by score.
- Relaxing residual, saturation, length, target-offset, official curve, or replacement thresholds.
- Claiming unseen-source-group, sealed-test, population, perceptual, content, speaker, or production generalization.

## Decisions

### 1. Continue from the completed training artifact, not a new fit

Validate the parent run's manifest, training history, step-100 checkpoint, and model hash. Copy their JSON objects into a new diagnostic root with parent hashes, then load the step-100 state only for inference. This is not a new training run and avoids a second optimization that would confound the narrow tolerance diagnostic.

Alternative: retrain with a changed tolerance. Rejected because the tolerance is an inference QC rule and should not alter the trained weights.

### 2. Relax only normalized log-mel distance to 0.11

Use a fixed diagnostic limit of `0.11`, chosen because the observed failing value was `0.1005600542`. Keep residual peak at `0.05`, saturation at `1e-4`, exact length, finite values, and every SyncNet threshold unchanged. Store both original and diagnostic limits in the configuration and report.

Alternative: accept all candidate outputs regardless of audio deviation. Rejected because large spectral deviations could make SyncNet movement uninterpretable.

### 3. Score all fixed rows without denominator reduction

Generate each fixed candidate once if it passes the diagnostic limit, create the same baseline/candidate real-video conditions, and score complete official curves. Preserve all eight rows and mark any failed row invalid. Because the limit was chosen after seeing one candidate value, aggregate movement is descriptive only; do not emit the standard pre-registered transfer status.

### 4. Keep replacement conditional and exploratory

Run the existing strict natural-audio replacement only if all eight real-video rows pass the unchanged six-of-eight gate. Require all 72 cells and natural-column comparison exactly as before, but carry the diagnostic/non-preregistered label into the terminal report even if metrics pass.

### 5. Use a separate immutable root

Use `runs/lrs3_mfa_linear_existing_data_audio_tolerance_20260903/`. The earlier data-inventory failure and corrected candidate-QC failure roots remain unchanged. Resume reads a valid terminal graph and never repairs or appends it.

## Risks / Trade-offs

- **The tolerance was selected after observing a failing row.** → Emit diagnostic-only status and prohibit pre-registered scientific interpretation.
- **The one accepted row may still change SyncNet meaningfully.** → Keep all other QC and official curve requirements unchanged and report its exact distance.
- **Parent training decision mislabeled the candidate-stage failure as training not run.** → Validate the actual step-100 history/checkpoint directly and bind both parent artifacts rather than trusting only the terminal summary.
- **A diagnostic run could be mistaken for a successful transfer.** → Use distinct status names and a report section explicitly saying no scientific transfer claim is available.

## Migration Plan

1. Validate the parent manifest and completed step-100 checkpoint read-only.
2. Create the diagnostic configuration and new immutable root.
3. Generate all fixed candidates under the `0.11` diagnostic limit and score all real-video pairs.
4. Stop with diagnostic-only evidence if any row is invalid or if the real-video gate misses.
5. Run the complete 72-cell replacement only after a complete real-video pass.
6. Validate the graph and write the post-hoc exploratory report.
