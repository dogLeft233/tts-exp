## 1. Protocol and Cohort

- [ ] 1.1 Create `scripts/experiments/lrs3_natural_to_tts_bridge_confirmation/` with frozen constants, canonical hashes, atomic writes, run-root guards, and focused tests for changed or incompatible configuration.
- [ ] 1.2 Bind the exact parent protocol, parent replacement, and discovery final hashes; select the second ordered record per source group and verify 22 records, 22 groups, disjointness from discovery, and ordered-ID SHA-256 `5999f6bae8ba6b17c2e316b5f1e1f9e6d73b402a34cc8e4cfe482bcc714f5b72`.
- [ ] 1.3 Write Stage 00 provenance and media-access ledgers; verify selection reads no scores, decodes no cohort media, uses no outcome-based filtering, and accesses no sealed path.

## 2. Audio and Diagnostics

- [ ] 2.1 Implement exact 16-kHz mono PCM16 `N`, byte-identical `N_REPEAT`, and quarter-boundary `LOCAL_SWAP`; verify synthetic fixtures cover exact bytes, exact length, boundary arithmetic, deterministic hashes, and rejection of malformed inputs.
- [ ] 2.2 Implement `BRIDGE_075` with the frozen CPU-float64 STFT, natural phase, `alpha=0.75`, one RMS match, optional peak attenuation, and one PCM16 canonicalization; verify deterministic endpoint, scale, clipping, and forbidden-option fixtures.
- [ ] 2.3 Generate exactly 88 audio cells for 22 records with per-cell provenance and hash-validated resume; compute and freeze official Wav2Lip mel progress and QC before any confirmation SyncNet score is read.

## 3. Frozen Media Matrix

- [ ] 3.1 Bind the same frozen Wav2Lip/SyncNet checkpoints and runtime tools as discovery, then freshly render exactly 88 videos with one shared geometry per record and separate work directories for `N` and `N_REPEAT`.
- [ ] 3.2 Enumerate exactly six registered cells per record and reject missing, duplicate, cross-record, or extra cells; require 132 unique cells in total.
- [ ] 3.3 Strict-mux and score all 132 cells with video stream copy, decoded PCM equality, video elementary-stream identity, and pinned official SyncNet V2; allow only hash-identical resume and no score-based retry.

## 4. Registered Analysis and Validation

- [ ] 4.1 Implement deterministic 10,000-draw source-group bootstrap with seed `20260904` and unit tests for endpoint signs, strict confidence-bound thresholds, record counts, and repeatability.
- [ ] 4.2 Evaluate `N_REPEAT` stability and `LOCAL_SWAP` own-audio validity/sensitivity before bridge interpretation; verify either failure yields exactly `CONTROL_FAILED`.
- [ ] 4.3 Evaluate frozen bridge movement, replacement non-inferiority, offset agreement, and primary SyncC gain; write exactly one of `BLOCKED`, `CONTROL_FAILED`, `NATURAL_TO_TTS_BRIDGE_CONFIRMED`, or `NATURAL_TO_TTS_BRIDGE_NOT_CONFIRMED`.
- [ ] 4.4 Add an independent Stage 00-04 validator with one-field corruption fixtures; run focused pytest, relevant render/mux regressions, Ruff, Python compilation, and `openspec validate confirm-lrs3-natural-to-tts-bridge --strict` before media execution.
- [ ] 4.5 Execute Stages 00-03 with hash-validated resume, finalize Stage 04 once, validate the terminal artifact independently, and report only the registered fit-only conclusion and follow-up eligibility.
