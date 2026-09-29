## 1. Relaxed immutable manifest

- [ ] 1.1 Add frozen configuration for exactly 94 training records, 8 record-heldout evaluation records, six required successes, 100 steps, unchanged thresholds, and 72 replacement cells; verify serialization rejects source-group-heldout claims, threshold tuning, extra seeds, retries, and old-run mutation.
- [ ] 1.2 Implement score-independent record selection over the existing 102 eligible rows, preferring one evaluation representative per source group and filling the remaining training set deterministically; verify IDs are disjoint, source-group overlap is recorded, train/validation/test split rules remain closed, and the manifest is create-once.
- [ ] 1.3 Add tests for the 94/8 denominator, deterministic selection, selected-ID tampering, outcome fields, missing inventory, and explicit source-group-overlap claim scope.

## 2. Fresh shared training and candidate inference

- [ ] 2.1 Implement fresh 94-record training using the existing waveform model and frozen SyncNet target-margin loss; verify the equal-weight mean, exact 100 steps, exact waveform-only model input, and unchanged SyncNet state.
- [ ] 2.2 Persist step-zero/step-100 checkpoints, ordered IDs, configuration, loss history, and hashes; verify fresh initialization, no warm start/checkpoint search/retry, identity at step zero, and finite artifacts.
- [ ] 2.3 Materialize one candidate waveform for each of 8 evaluation IDs under the existing residual, log-mel, saturation, PCM, exact-length, and provenance checks; verify adapter state is unchanged and natural audio never reaches its forward call.

## 3. Record-heldout real-video gate

- [ ] 3.1 Reuse the signed PCM16, FFV1, official scoring-frame, 91-window, 31-offset, and official sign-convention seams for all 8 baseline/candidate pairs; verify identical visual hashes and complete official curves.
- [ ] 3.2 Implement record-heldout evidence and gains with candidate target-offset matching, unique gap, and unchanged `0.003`/`0.002` thresholds; verify boundary, tie, sign, QC, and missing-curve cases.
- [ ] 3.3 Implement the six-of-eight aggregate decision with explicit record-heldout status names; verify complete scientific misses are negative while incomplete evidence is not evaluated and replacement gating is correct.

## 4. Conditional strict replacement

- [ ] 4.1 Reuse the fixed-crop Wav2Lip P0 seam and render exactly natural, baseline, and candidate drivers for each selected row only after a real-video pass; verify equal frame support, FFV1 round-trip, and rerender refusal.
- [ ] 4.2 Materialize all nine driver/evaluation-audio cells per row and verify exactly 72 unique cells with row-identical frames, column-correct PCM, complete official curves, and no denominator reduction.
- [ ] 4.3 Implement natural-column replacement gains and the six-of-eight aggregate gate; verify diagonal/cross-audio improvements cannot promote replacement and incomplete cells return `REPLACEMENT_NOT_EVALUATED`.

## 5. Validation, reporting, and execution

- [ ] 5.1 Implement create-once artifact validation and read-only terminal resume for the relaxed run; verify manifest, checkpoint, curve, matrix, hash, and decision tampering fails closed.
- [ ] 5.2 Generate a report that says record-heldout within fit-only data and lists source-group overlap; verify it disclaims unseen-source-group, sealed-test, population, perceptual, content/speaker, multi-generator, and production claims.
- [ ] 5.3 Run the new focused tests, predecessor regressions, and `openspec validate relax-mfa-linear-sync-existing-data --strict`; persist commands, versions, hashes, and results.
- [ ] 5.4 Execute the new immutable 94/8 run once and validate its terminal graph; verify the old 200/40 blocked run and all prior artifacts remain unchanged.
