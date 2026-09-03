## 1. Bind inputs and correct reconstruction analysis

- [ ] 1.1 Freeze the parent cohort, feature, alignment, mask, reconstruction, driver, render, face-box, SyncNet, checkpoint, local kNN-VC revision, adapter source, WavLM checkpoint, and extractor interface paths and hashes; verify the required 16-record, eight-group, 222-mask, three-seed, 192-score matrix.
- [ ] 1.2 Fix reconstruction pairing to use the stored same-mask gain or `(sample_id, seed, mask_sha256)`, and add a regression test with two masks sharing one sample and seed.
- [ ] 1.3 Reaggregate the existing reconstruction rows, write the correction report in the new run, and verify both fixed summaries against `design.md` within `1e-5`.

## 2. Trace the existing signal path

- [ ] 2.1 Reconstruct paired, original wrong-instance, and reversed aligned inputs from bound provenance and compute input-core and predicted-core RMS plus first-difference RMS.
- [ ] 2.2 Compute covered-frame driver RMS and lower-half shared-face-box RGB response from existing paired and control artifacts; fail closed on identity, hash, shape, frame-count, or box mismatch.
- [ ] 2.3 Aggregate and report wrong-instance versus reversed magnitudes through input, prediction, driver, video, and existing SyncNet stages without adding a post-hoc gate.

## 3. Run the natural-slot oracle

- [ ] 3.1 Extract and hash natural-audio WavLM-L6 with the frozen TTS feature extractor, then build all 222 `NATURAL_WAVLM_IN_TTS_SLOT` inputs from natural MFA phone spans through the existing interpolation path.
- [ ] 3.2 Add focused tests for frame conversion, core-only support, paired-shape equality, paired-distinct hashes, and complete provenance.
- [ ] 3.3 Run the three frozen checkpoints, write natural-slot reconstruction rows and 48 drivers, and verify outside-core natural-mel equality.
- [ ] 3.4 Render 48 cells with frozen Wav2Lip, strictly replace each with its bound untouched natural audio, score with frozen SyncNet, and verify the complete matrix.
- [ ] 3.5 Reuse paired and zero-input results, aggregate natural-slot-over-zero and natural-slot-over-TTS contrasts, and report frozen-TFG and reconstruction axes independently.

## 4. Run the conditional matched-instance diagnostic

- [ ] 4.1 Build the score-free donor preflight using the fixed source-tier, neighboring-phone, duration, and identity ranking; freeze three donors per eligible mask and report coverage.
- [ ] 4.2 Apply the coverage gate. Record `INSUFFICIENT_MATCHED_DONOR_COVERAGE` and skip the remaining Part C tasks unless every record retains at least two masks and all groups remain represented.
- [ ] 4.3 If eligible, evaluate all three donor ranks for reconstruction robustness and build a `PAIRED_COMMON_MASKS` baseline plus one predeclared rank-one matched-wrong driver from exactly the same frozen mask subset.
- [ ] 4.4 Render, strictly replace, and score the at-most-96-cell rank-one matched matrix; aggregate reconstruction and frozen-TFG axes separately and assign one Part C status with its qualifier.

## 5. Decide and validate

- [ ] 5.1 Write stage statuses, per-record and per-group results, signal-path tables, oracle warning, inspected-cohort warning, bounded claims, one final recommendation, and explicit waveform-gate closure.
- [ ] 5.2 Add focused tests for source binding, mask pairing, natural-slot construction, donor determinism, common-mask baselines, aggregation order and direction, conditional skips, and incomplete-matrix handling.
- [ ] 5.3 Run the focused tests and `openspec validate diagnose-masked-tts-natural-slot-signal-path --strict`; do not mark the change complete while any required executed stage or validation fails.
