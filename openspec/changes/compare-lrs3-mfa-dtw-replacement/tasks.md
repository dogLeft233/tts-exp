## 1. Protocol and Parent Bindings

- [ ] 1.1 Create the isolated `lrs3_mfa_dtw_replacement` package, stage configuration, immutable parent paths/hashes, protocol ID, seed, band ratio, and run-root guard; verify imports and frozen constants in a focused config test.
- [ ] 1.2 Implement the Stage 00 three-parent join from the strict 133-record protocol, replacement manifest, and 146-record MFA3 alignment manifest; verify tests reject wrong hashes, reordered/duplicate/missing IDs, source-group mismatches, and any cohort hash other than `61f8c982041cdfdded8daf8850d31e127943386ba2cb7035aa742e94cea9a973`.
- [ ] 1.3 Bind per-record natural/TTS audio, face video, MFA3 token/TextGrid identities, historical MFA-linear candidate, and required historical score cells; verify every path and SHA-256 is checked and no score value participates in record selection.
- [ ] 1.4 Add canonical JSON hashing, atomic writes, append-only stage decisions, failure ledger, media-access ledger, and hash-validated resume helpers; verify corrupted or partial sidecars cannot be resumed.

## 2. Same-Phone Hard-DTW Mapping

- [ ] 2.1 Implement deterministic float64 cosine-cost hard-DTW with endpoint, monotonic-step, normalized-band `0.5`, and fixed tie-order constraints; verify exact paths for diagonal, unequal-length, singleton, tied-cost, and no-admissible-path fixtures.
- [ ] 2.2 Implement matched-phone frame extraction using the existing token-instance matcher and WavLM center-time ownership convention; verify repeated phone labels remain instance-local and cross-phone/cross-instance indices are rejected.
- [ ] 2.3 Convert each path's per-natural-frame visited TTS indices to a mean continuous source coordinate and existing left/right/alpha interpolation rows; verify natural-frame coverage, within-phone monotonicity, endpoints, bounds, and reconstruction from saved traces.
- [ ] 2.4 Preserve the historical MFA-linear silence rows exactly and forbid every speech fallback; verify synthetic unmatched speech fails while valid silence rows are byte-equivalent to linear mapping rows.
- [ ] 2.5 Add path/displacement diagnostics and the no-natural-feature-injection contract; verify disjoint-range synthetic features prove all emitted conditioning values are TTS interpolants and repeated inputs yield identical trace/conditioning hashes.

## 3. DTW Candidate Generation

- [ ] 3.1 Implement Stage 01 extraction and candidate generation by reusing the pinned `WavLMKNNVCAdapter`, existing interpolation/vocoder path, exact-length adjustment, canonical PCM16, and unchanged waveform QC; verify a mocked end-to-end record differs from MFA-linear only in mapping rows.
- [ ] 3.2 Record model/tool/input/feature/path/conditioning/raw-waveform/canonical-PCM provenance for every candidate and refuse incompatible historical contracts; verify tampered assets or changed normalization/length/PCM policies fail before output acceptance.
- [ ] 3.3 Add per-cell temporary outputs and hash-validated resume so the frozen models load once and completed candidates are not silently regenerated; verify interruption/resume tests preserve order and reject stale audio or traces.

## 4. Frozen-TFG Diagonal Evaluation

- [ ] 4.1 Implement Stage 02 frozen-Wav2Lip rendering for one DTW video per record with the historical checkpoint, flags, face input, and tool bindings; verify command construction and render provenance match the parent contract.
- [ ] 4.2 Implement strict `DTW video + DTW audio` muxing, decoded candidate-PCM byte equality, and pinned official file-level SyncNet scoring; verify channel/rate/width/sample/byte mismatches invalidate the complete diagonal matrix.
- [ ] 4.3 Reuse historical `G_M_E_M` score rows through hash-bound references and join them to DTW rows by ordered sample ID/source group; verify no historical media or result file is written or modified.

## 5. Promotion Statistics and Gate

- [ ] 5.1 Implement deterministic 10,000-draw source-group cluster bootstrap with seed `20260903`, paired Sync-C/Sync-D benefits, percentile 95% intervals, and C/D/joint wins; verify against a hand-computable grouped fixture and repeated-run equality.
- [ ] 5.2 Implement Stage 03 engineering validation and the dual lower-confidence-bound promotion rule; verify promotion occurs only when both lower bounds are strictly positive and incomplete matrices produce `BLOCKED` rather than `NO_DTW_TFG_ADVANTAGE`.
- [ ] 5.3 Write a canonical self-hashed diagonal decision with explicit `replacement_authorized`; verify either endpoint touching zero seals replacement and no secondary metric can rescue the decision.

## 6. Conditional Strict Replacement

- [ ] 6.1 Implement a separate Stage 04 entry point that validates the Stage 03 self-hash and refuses to create its output directory unless `replacement_authorized=true`; verify absent, altered, blocked, and negative decisions all leave replacement sealed.
- [ ] 6.2 Reuse each existing DTW video without re-rendering, mux untouched natural PCM with video stream copy and PCM s16le, verify decoded PCM byte identity, and run pinned SyncNet once per record; verify all 133 cells are required for engineering GO.
- [ ] 6.3 Implement the three replacement contrasts—DTW versus natural, MFA-linear versus natural, and DTW versus MFA-linear—with grouped intervals and wins; verify the terminal decision distinguishes relative improvement from `REPLACEMENT_SAFE_GO` versus natural.

## 7. Validation and Reproducibility

- [ ] 7.1 Add an artifact validator that independently recomputes cohort joins, path invariants, file hashes, PCM equality, score counts, statistics, stage authorization, and sealed-media claims; verify it detects one-field corruption at every stage.
- [ ] 7.2 Run focused pytest, the relevant historical MFA-linear regression tests, Ruff, Python compilation, and `openspec validate compare-lrs3-mfa-dtw-replacement --strict`; verify every command passes before media execution.
- [ ] 7.3 Run a read-only Stage 00 preflight and independent artifact validation; verify exactly 133 ordered fit records, all parent hashes, zero sealed validation/test access, and no GPU/media output.

## 8. Gated LRS3 Execution

- [ ] 8.1 During the registered 08:00–23:00 GPU window and on an idle GPU, generate all 133 Stage 01 DTW candidates with resume enabled; verify exact natural lengths, unchanged QC, complete traces, deterministic hashes, and an engineering-GO candidate artifact.
- [ ] 8.2 Independently validate Stage 01 and summarize DTW path geometry, trivial-phone rate, path costs, and displacement from MFA-linear without inspecting TFG outcomes; verify no path configuration is changed after this summary.
- [ ] 8.3 During the allowed GPU window, render, strict-mux, and score all 133 Stage 02 diagonal cells; verify complete provenance, candidate PCM equality, and 133/133 official SyncNet scores.
- [ ] 8.4 Finalize Stage 03 exactly once from the frozen diagonal matrix and historical linear controls; verify the self-hashed decision reports both paired means/intervals and either authorizes replacement or records `NO_DTW_TFG_ADVANTAGE`.
- [ ] 8.5 If and only if Stage 03 authorizes replacement, run and independently validate all 133 Stage 04 strict natural-audio cells; otherwise verify `04_replacement/` is absent and record `sealed_not_run`.
- [ ] 8.6 Produce the final machine-readable and human-readable comparison, including prior direct-resynthesis context, MFA-linear baseline, DTW diagonal conclusion, conditional replacement contrasts, limitations, and exact artifact paths; verify reported numbers recompute from bound per-record scores.
