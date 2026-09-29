## 1. Diagnostic protocol and parent validation

- [ ] 1.1 Add frozen diagnostic configuration with original log-mel limit `0.10`, diagnostic limit `0.11`, fixed 94/8 manifest, parent checkpoint hash, unchanged SyncNet thresholds, and explicit post-hoc claim boundary; verify no training or selection controls are configurable.
- [ ] 1.2 Validate and bind the corrected parent manifest, training history, step-100 checkpoint, and hashes; verify changed/missing parent artifacts emit `DIAGNOSTIC_PARENT_INVALID` before inference.
- [ ] 1.3 Add tests for the narrow tolerance, parent hash binding, fixed IDs, no natural adapter input, and explicit exploratory status.

## 2. Candidate and official real-video diagnostic

- [ ] 2.1 Materialize all eight fixed candidate PCM artifacts once, accepting normalized log-mel distance `<=0.11` but retaining all other QC checks; verify the previously failing `0.1005600542` row is accepted and no row is substituted.
- [ ] 2.2 Reuse the official paired real-video scorer for all eight rows and persist complete curves, frame/PCM hashes, gains, target offsets, and exact QC values; verify all rows remain in the denominator.
- [ ] 2.3 Emit diagnostic-only aggregate evidence or `DIAGNOSTIC_NOT_EVALUATED` when any row is invalid; verify no pre-registered transfer status is emitted.

## 3. Conditional replacement and integrity

- [ ] 3.1 Run the fixed-crop Wav2Lip three-arm render only after all eight unchanged real-video gates pass; verify exactly 72 cells and natural-column-only authoritative comparison.
- [ ] 3.2 Add immutable terminal validation and read-only resume for diagnostic parent, candidate, curves, renders, cells, and report; verify tampering or missing cells fails closed.
- [ ] 3.3 Write a report with original/diagnostic limits, post-hoc reason, record-heldout source-group overlap, and all claim disclaimers.

## 4. Verification and execution

- [ ] 4.1 Run diagnostic focused tests, predecessor regressions, and `openspec validate diagnose-mfa-linear-existing-data-audio-tolerance --strict`; persist command and environment metadata.
- [ ] 4.2 Execute the diagnostic continuation once in a new immutable root and validate its terminal graph without changing either prior run.
