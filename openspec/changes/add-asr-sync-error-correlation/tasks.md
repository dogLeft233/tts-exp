## 1. Scaffold and Runtime Contract

- [x] 1.1 Create `scripts/experiments/asr_sync_error_correlation/` and `tests/experiments/asr_sync_error_correlation/` with the module layout from `design.md`, and verify every module imports without model or media access.
- [x] 1.2 Add a minimal pinned requirements/constraints file for the audited main environment plus Matplotlib, without changing `~/.venvs/syncnet`, and verify a clean dependency check reports PyTorch, torchaudio, Transformers, Hugging Face Hub, SciPy, NumPy, SoundFile, and Matplotlib versions.
- [x] 1.3 Implement frozen configuration constants and CLI validation in `config.py`, including rejection of scientific overrides and correction-enabled decoders, and verify focused tests cover valid defaults and every forbidden override.
- [x] 1.4 Implement strict finite-JSON, SHA-256, canonical-hash, atomic-write, NPZ `allow_pickle=False`, failure-ledger, and success-marker helpers in `io.py`, and verify tests reject NaN/infinity, corrupt outputs, and stale bindings.

## 2. Preflight and Wav2Vec2 Deployment

- [x] 2.1 Implement environment preflight for both Python executables, FFmpeg/FFprobe, disk space, CUDA/device selection, package APIs, Matplotlib `Agg`, and the expected SyncNet checkpoint hash, and verify failures produce `NO_GO` plus one actionable remediation command before media access.
- [x] 2.2 Implement explicit Hugging Face snapshot acquisition for `facebook/wav2vec2-large-960h-lv60-self`, immutable commit resolution, used-file hashing, tokenizer-vocabulary hashing, and `asr_model_lock.json`, and verify a mocked acquisition cannot write a lock for a partial or mutable snapshot.
- [x] 2.3 Implement locked offline processor/model loading with `local_files_only=True`, float32 evaluation mode, batch size one, and one fixed run device, and verify tests reject revision/file/hash mismatch and explicit CUDA on a non-CUDA fixture.
- [x] 2.4 Add a preflight smoke test for Wav2Vec2 inference, CTC frame stride, `torchaudio.functional.forced_align`, tokenizer character coverage over the frozen transcripts, and offline reload, and verify the opt-in local-model smoke command exits successfully without cohort media.

## 3. Frozen Protocol and Strict Inputs

- [x] 3.1 Implement canonical LRS3 `.txt` parsing for `Text:`, confidence, and `WORD START END ASDSCORE` rows plus the versioned normalizer, and verify fixtures cover whitespace, apostrophes, unsupported characters, non-monotonic times, and transcript/table disagreement.
- [x] 3.2 Implement parent manifest/test-lock validation and the exact ordered 24-record `fresh_confirmation` join to canonical source and TTS metadata, and verify tests produce 24 source groups and reject a changed hash, missing arm, transcript mismatch, or sealed-split record.
- [x] 3.3 Expand the frozen records into adjacent `natural`/`tts` arm-records with parent, path, transcript, timing, audio, video, and split provenance, and verify `01_manifest/manifest.json` contains exactly 48 unique arm keys in deterministic order.
- [x] 3.4 Reuse the repository strict mux contract for both arms and write per-cell PCM/video provenance, and verify a fixture mux copies video, emits mono 16 kHz PCM, exactly preserves decoded arm PCM, and uses no crop/pad/stretch/shortest policy.
- [x] 3.5 Implement paired input validation that requires natural/TTS video identity and reports audio duration outside video support without truncation, and verify mismatched video hashes or changed PCM fail only the affected cell and populate `02_inputs/failures.json`.

## 4. Uncorrected CTC ASR and Reference Timing

- [x] 4.1 Implement one-arm float32 Wav2Vec2 emission export with model/runtime/input bindings and compressed numeric NPZ output, and verify a tiny mocked model produces reproducible frame counts, argmax IDs, and hashes.
- [x] 4.2 Implement pure greedy CTC collapse and token/word timestamps from `inputs_to_logits_ratio`, including repeated tokens separated by blanks and delimiter handling, and verify unit fixtures reconstruct the exact expected uncorrected transcript and half-open spans.
- [x] 4.3 Implement diagnostic geometric-mean word confidence from contributing argmax posteriors without filtering words, and verify confidence changes never change token IDs, words, timestamps, or WER inputs.
- [x] 4.4 Implement per-arm reference-token encoding and `torchaudio.functional.forced_align` grouping into monotonic reference word spans, and verify a fabricated emission aligns every reference word while leaving greedy output byte-for-byte unchanged.
- [x] 4.5 Implement natural-arm official-versus-forced timing diagnostics and strict token-count checks, and verify start/end/midpoint error summaries reproduce from a known LRS3 fixture while TTS never consumes natural official times as its reference timing.
- [x] 4.6 Implement the `03_asr` cell runner, schema validators, success markers, and failure ledger, and verify an interrupted two-cell fixture reuses the valid ASR cell and recomputes only the invalid cell on `--resume`.

## 5. Word Errors and Time Localization

- [x] 5.1 Implement unit-cost global word edit distance with the frozen exact-match/substitution/deletion/insertion tie order and stable operation IDs, and verify fixtures cover repeated words and ambiguous minimum-cost paths.
- [x] 5.2 Implement WER counts and operation timing assignment from forced-reference and greedy-prediction spans, and verify equal, insertion, deletion, and substitution cases use exactly the required timing sources.
- [x] 5.3 Implement contiguous edit-block grouping, `1e-9` touching/overlap merging, and operation/word-index provenance, and verify many-to-many edit fixtures lose no atomic operation or covered time.
- [x] 5.4 Implement alignment-record validation that refuses unsupported, missing, empty, out-of-audio, or non-monotonic spans with no fallback timestamps, and verify each invalid fixture receives a machine-readable failure reason.

## 6. SyncNet Distance Export and Local Score

- [x] 6.1 Implement `syncnet_adapter.py` for the dedicated SyncNet interpreter to load `tracks.pckl`, select the longest then lowest-index track without scores, run existing `SyncNetInstance.evaluate`, and export numeric distances plus all track ranges, and verify track-selection tests cover no-track and equal-length ties.
- [x] 6.2 Implement the main-environment subprocess wrapper for `run_pipeline.py` and the adapter using unique per-cell work/reference paths and `min_track=50`, and verify one opt-in fit-sample smoke run produces logs, track metadata, and a finite `[T,31]` distance matrix.
- [x] 6.3 Implement exact `mdist`, `j_star`, global offset, Sync-D, Sync-C, raw local confidence, and 9-value median-filter formulas, and verify hand-built matrices plus the opt-in smoke run meet exact/`1e-6`/three-decimal parity tolerances.
- [x] 6.4 Implement original-audio-clock mapping `a=t+j_star-vshift` and `(track_start+a+2)/25`, excluding shift padding, unsupported time, and four median-filter edge rows, and verify tests cover nonzero track starts and both positive and negative offsets.
- [x] 6.5 Enforce paired natural/TTS video and selected-track metadata identity and write `04_sync` cell markers/failures, and verify a score-independent track mismatch causes engineering incompleteness instead of track substitution.

## 7. Grids, Metrics, Bootstrap, and Decisions

- [x] 7.1 Implement the retained 25 Hz center grid, half-open error mask, per-arm `mean-1*population_std` threshold, strict low mask, and excluded-region counts, and verify boundary equality, threshold equality, and constant-confidence fixtures.
- [x] 7.2 Implement per-record intersection/union counts, IoU, recall, precision, Spearman of error mask versus continuous negative local confidence, badness contrast, and null reason codes, and verify all finite and degenerate cases against hand-calculated expectations.
- [x] 7.3 Implement arm-level descriptive/micro aggregation and sample-ID-only paired joins, and verify a missing arm contributes only to its valid arm summary and never to paired deltas.
- [x] 7.4 Implement deterministic 10,000-draw whole-sample bootstrap intervals using `PCG64(20260831)` for arm medians and paired TTS-minus-natural medians, and verify repeat runs are identical and no frame-level resampling path exists.
- [x] 7.5 Implement engineering `GO/NO_GO` and per-arm `SUPPORT/NO_SUPPORT/INSUFFICIENT` gates exactly at the predeclared record/error-cell/source-group/confidence-bound thresholds, and verify below/equal/above boundary tests plus `NOT_EVALUATED` on any required engineering failure.
- [x] 7.6 Write and validate `05_alignment` grids/records and `06_analysis` per-record, summary, and decision artifacts, and verify every reported metric can be recomputed from exported NPZ arrays and strict JSON contains no NaN/infinity.

## 8. Visualization, Orchestration, and End-to-End Validation

- [x] 8.1 Implement deterministic Matplotlib `Agg` paired timeline plots with local confidence, threshold, ASR-error spans, excluded regions, identifiers, and shared time scale, and verify a fixture plot plus `07_plots/index.json` are produced without affecting numeric outputs.
- [x] 8.2 Implement `run.py` stage ordering, predecessor-marker checks, single-device enforcement, non-empty-directory refusal, cell-local continuation, nonzero incomplete exit, and hash-aware `--resume`, and verify orchestration tests cover fresh, interrupted, stale, corrupt, and stage-only runs.
- [x] 8.3 Implement final root `summary.json` and `decision.json` validation requiring 24 samples, 48 successful arms, all parity/hash/plot checks, and separate engineering/scientific statuses, and verify deleting or altering any required artifact changes engineering to `NO_GO`.
- [x] 8.4 Run the network-free unit suite for the new package and record the exact passing command/output; verify no unit test downloads a model, opens sealed media, or invokes S3FD.
- [x] 8.5 Run the opt-in locked-model ASR smoke test and one-sample SyncNet parity smoke test, and verify both bind the same versions/checkpoint/model lock expected by full execution.
- [x] 8.6 Execute preflight and manifest stages for the intended run directory, inspect the frozen 24x2 manifest/test lock, and verify no cohort media is processed until both stages report `GO`.
- [x] 8.7 Execute the full offline 48-arm experiment with `--resume`, then run the artifact validator and verify engineering/scientific decisions, all 24 plots, failure counts, excluded durations, and reproducibility bindings are present.
- [x] 8.8 Run `openspec validate add-asr-sync-error-correlation --strict` and verify the change, delta spec, design, and completed task checklist remain mutually consistent before archive consideration.
