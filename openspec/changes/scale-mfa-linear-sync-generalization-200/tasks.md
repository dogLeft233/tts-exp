## 1. Frozen protocol and expanded data lock

- [ ] 1.1 Add the scale-specific configuration with exactly 200 training records, 40 evaluation source groups, 30 minimum successes, 100 fresh-training steps, 360 replacement cells, new selection salt, existing `q=0.001`/`3q`/`2q` thresholds, checkpoint hashes, and forbidden controls; verify serialization rejects denominator changes, extra seeds, threshold tuning, warm starts, retries, and old-run mutation.
- [ ] 1.2 Implement the expanded fit-only inventory loader and structural/MFA/geometry/hash preflight; verify the current 102-row inventory returns `BLOCKED_DATASET_SCALE` rather than duplicating records or accessing sealed validation/test data.
- [ ] 1.3 Implement deterministic source-group partitioning and exact 200-record training/40-group evaluation selection; verify manifests are bytewise reproducible, score-independent, source-group-disjoint, ordered, complete with exclusions, and create-once before training.
- [ ] 1.4 Add focused data-lock tests for insufficient rows/groups, duplicate identities, P2-group leakage, train/evaluation overlap, outcome fields, malformed assets, detached natural-reference ambiguity, and post-lock substitution; verify every failure is fail-closed.

## 2. Fresh 200-record TTS-only training

- [ ] 2.1 Implement the scale-specific fresh trainer using the existing waveform model and frozen SyncNet target-margin loss; verify each step uses the equal-weight mean over all 200 locked records, natural audio never reaches the model or loss, and SyncNet remains frozen.
- [ ] 2.2 Persist and bind step-zero/step-100 checkpoints, ordered training manifest, optimizer/configuration, loss history, model hashes, and SyncNet before/after hashes; verify exactly 100 steps, fresh initialization, finite artifacts, and no warm start or checkpoint search.
- [ ] 2.3 Add training smoke tests with a small deterministic fixture that prove full-record mean equivalence, identity step zero, gradient direction, frozen SyncNet, and rejection of natural/video/identity/transcript/offset/second-audio side channels.

## 3. Heldout candidate inference and real-video scoring

- [ ] 3.1 Implement the frozen adapter wrapper and one-pass candidate materialization for all 40 evaluation records; verify exact `[1,1,61440]` input/output, candidate-to-MFA residual/log-mel/saturation QC, source/checkpoint hashes, and unchanged adapter state.
- [ ] 3.2 Reuse the signed PCM16, FFV1, official scoring-frame, 91-window, 31-offset, and official-offset-sign seams for paired MFA baseline/candidate conditions; verify identical visual frame hashes, correct PCM hashes, complete finite curves, and per-offset proxy parity when a proxy is persisted.
- [ ] 3.3 Implement record evidence with positive-is-better distance and separation gains, detached target-offset equality, unique best/second gap, and engineering predicates; verify boundary cases just below/equal/above thresholds, ties, sign inversions, QC failures, and missing curves.
- [ ] 3.4 Implement the 40-row aggregate decision with all-row engineering completeness, at least 30 successes, and both median gains; verify synthetic fixtures distinguish observed transfer, complete scientific miss, incomplete evaluation, and correct replacement skip precedence.

## 4. Conditional frozen Wav2Lip replacement

- [ ] 4.1 Generalize the fixed-crop Wav2Lip P0 seam for the scale cohort without changing the frozen checkpoint, invocation, geometry, frame, or scorer contracts; verify one fixture renders all three arms with equal nonzero support and survives FFV1 round-trip.
- [ ] 4.2 Implement exactly one natural, MFA-linear, and candidate driver render per evaluation record with isolated working directories and immutable provenance; verify rerender/overwrite attempts fail and renderer audio is discarded.
- [ ] 4.3 Materialize and validate all nine driver/evaluation-audio cells for each of 40 records; verify exactly 360 unique cells, row-identical frame hashes, column-correct PCM hashes, complete 31-point curves, and no denominator reduction.
- [ ] 4.4 Implement natural-column replacement evidence and the 30-of-40 aggregate gate; verify diagonal/cross-audio gains cannot promote, oracle offset ties/mismatches fail, and incomplete cells return `REPLACEMENT_NOT_EVALUATED`.

## 5. Validation, reporting, and execution

- [ ] 5.1 Implement create-once artifact graph validation and read-only terminal resume for the scale run; verify tampering, stale parents, partial stages, duplicate cells, changed manifests, and conflicting decisions fail closed while a complete run executes no external command on resume.
- [ ] 5.2 Generate a bounded report separating data-scale engineering, fresh training, real-video transfer, and replacement statuses; verify it states source-group-heldout fit-only scope and disclaims population/sealed-test, perceptual/content/speaker, multi-generator, and production claims.
- [ ] 5.3 Run the new focused suite, predecessor SyncNet/MFA/replacement regressions, and `openspec validate scale-mfa-linear-sync-generalization-200 --strict`; persist commands, versions, hashes, and results before GPU execution.
- [ ] 5.4 Execute the new immutable scale run once with the locked 200/40 protocol; verify it stops at the first declared gate, never mutates earlier runs, and produces a validated terminal decision and report.
