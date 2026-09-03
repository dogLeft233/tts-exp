## 1. Bind and audit existing results

- [x] 1.1 Freeze the parent reconstruction and 16-record confirmation manifests, hashes, conditions, seeds, source groups, and previously inspected-data warning.
- [x] 1.2 Compute the no-rerender natural-reference audit for paired and phone-centroid conditions, plus the descriptive phone-centroid-over-zero-TTS contrast.

## 2. Build the trajectory controls

- [x] 2.1 Build and test the deterministic same-phone donor map from frozen training masks only.
- [x] 2.2 Build and test the within-phone reversed features; require complete provenance, paired-distinct hashes, unchanged target masks, and zero values outside each core.

## 3. Diagnose the frozen model

- [x] 3.1 Run both controls through all evaluation masks and three original checkpoints; record patch, velocity, and total losses.
- [x] 3.2 Build 96 new mel drivers, render with frozen Wav2Lip, strictly replace with untouched natural audio, and score with frozen SyncNet V2.
- [x] 3.3 Aggregate mask, seed, record, and source-group results exactly as specified and assign one Part B status.

## 4. Run the conditional training feasibility

- [x] 4.1 Apply the Part C gate. Record the specified skip status unless Part B is `NO_TRAJECTORY_SIGNAL`.
- [x] 4.2 If eligible, train one fixed hard-negative arm for each original seed with no architecture, schedule, checkpoint, or hyperparameter search.
- [x] 4.3 Evaluate reconstruction first and stop as `TRAINING_NO_GO` if the hard-control or paired-quality gate fails.
- [x] 4.4 Only after the reconstruction gate passes, render and score the four-condition 192-cell matrix and assign the Part C status.

## 5. Decide and validate

- [x] 5.1 Write the per-record and per-group results, stage statuses, bounded claims, final recommendation, and explicit waveform-gate closure.
- [x] 5.2 Add focused tests for donor eligibility and determinism, reversal scope, aggregation direction, stage gates, and incomplete-matrix handling.
- [x] 5.3 Run the focused tests and `openspec validate diagnose-masked-tts-trajectory-specificity --strict`.
