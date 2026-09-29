## 1. Protocol and Cohort

- [x] 1.1 Create `scripts/experiments/lrs3_phase_preserving_replacement_envelope/` with frozen constants, stage paths, canonical hashing, atomic writes, and run-root guards; verify focused tests reject changed constants and incompatible existing roots.
- [x] 1.2 Implement Stage 00 binding to the two registered parent manifests and first-record-per-source-group selection; verify exactly 23 records, 23 groups, ordered-ID SHA-256 `c56e420d9ade7e04f5558f37fbf68ee68463d3817060baab79e8e6e0bf7e8fbd`, and all natural/MFA-linear/video hashes.
- [x] 1.3 Add failure and media-access ledgers; verify Stage 00 reads no prior scores, decodes no cohort media, and accesses no sealed validation/test paths.

## 2. Candidate Audio and Diagnostics

- [x] 2.1 Implement exact PCM16 loading/canonicalization plus `N`, `INV`, and `SHIFT_200`; verify synthetic tests cover sign inversion, the `-32768` rejection, exact 3,200-sample delay, exact output length, and deterministic hashes.
- [x] 2.2 Implement the frozen CPU-float64 natural-phase log-STFT blend and deterministic RMS/peak scaling for `MAG_025`, `MAG_050`, `MAG_075`, and `MAG_100`; verify reconstruction, blend endpoints, scale recording, clipping rejection, repeatability, and forbidden-option tests.
- [x] 2.3 Generate all seven arms for all 23 records with per-cell provenance and hash-validated resume; verify Stage 01 contains exactly 161 valid audio cells and no substitutions.
- [x] 2.4 Implement official Wav2Lip mel directional progress, orthogonal residual, mel distances, and waveform QC; verify hand-computable fixtures and freeze all 23-record diagnostics before any new SyncNet result is read.

## 3. Frozen Rendering and Narrow Matrix

- [x] 3.1 Bind the frozen Wav2Lip/SyncNet checkpoints, runtime tools, face videos, and one shared geometry per record; verify any model, command, face, or geometry drift is rejected.
- [x] 3.2 Freshly render all 161 videos with hash-validated resume and no score-based retry; verify one valid video exists for every record/arm pair.
- [x] 3.3 Enumerate exactly the 299 unique `V_X/A_N` and `V_X/A_X` cells with `V_N/A_N` deduplicated; verify a unit test rejects missing, duplicate, cross-record, or extra cells.
- [x] 3.4 Strict-mux and score all cells with video stream copy, decoded PCM byte equality, and pinned official SyncNet V2; verify the complete 299-cell manifest before analysis.

## 4. Registered Decisions

- [x] 4.1 Implement 10,000-draw deterministic bootstrap analysis with seed `20260904` for directional progress and positive-is-better `gap_C`/`gap_D`; verify endpoint signs, intervals, wins, and repeatability on synthetic fixtures.
- [x] 4.2 Implement the `INV` equivalence/compatibility gate and `SHIFT_200` sensitivity gate; verify boundary fixtures produce `CONTROL_FAILED` exactly when either control fails.
- [x] 4.3 Implement per-level movement, non-inferiority, and offset-agreement gates plus `max_compatible_alpha`; verify zero-touching confidence bounds fail strict gates and only registered strengths can be selected.
- [x] 4.4 Write one self-hashed terminal artifact yielding exactly `BLOCKED`, `CONTROL_FAILED`, `PHASE_PRESERVING_REPLACEMENT_ENVELOPE_FOUND`, or `ONLY_TRIVIAL_REPLACEMENT_EQUIVALENCE_FOUND`; verify only the envelope-found result enables a separate identity-characterization spec.

## 5. Validation and Execution

- [x] 5.1 Add an independent Stage 00-05 validator covering parent/cohort bindings, candidate formulas, controls, media identities, complete counts, statistics, decisions, and sealed-scope claims; verify one-field corruption fixtures fail each stage.
- [x] 5.2 Run focused pytest, relevant strict-mux/render regressions, Ruff, Python compilation, and `openspec validate probe-lrs3-phase-preserving-replacement-envelope --strict`; require all checks to pass before media execution.
- [x] 5.3 Execute Stages 00-04 with hash-validated resume, then finalize Stage 05 exactly once; independently validate the terminal artifact and report the largest qualifying registered blend strength without broader identity or generalization claims.
