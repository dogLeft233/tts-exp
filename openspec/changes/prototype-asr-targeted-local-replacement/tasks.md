## 1. Scaffold and Frozen Lock

- [ ] 1.1 Create the six-module `asr_targeted_local_replacement` package and focused test directory, reuse existing experiment helpers rather than introducing base classes/frameworks, and verify every module imports without opening media or loading models.
- [ ] 1.2 Implement the metadata-only parent/config/asset lock for the frozen 24-record fit-only cohort and `03_asr` records, including a consumed-field ledger and sealed-split rejection; verify tests reject changed hashes, score-bearing selection inputs, and any non-fit record before media access.

## 2. Target and Control Manifest

- [ ] 2.1 Rebuild natural/TTS greedy edit alignments from the locked ASR records and select only contiguous substitution/deletion reference blocks with finite per-arm spans and ASR-clean TTS donors; verify fixtures exclude pure insertions, non-contiguous ranges, and TTS-error donors with exact reason codes.
- [ ] 2.2 Implement the two deterministic within-sample matched-control assignments using only ASR/reference word count, duration, identifiers, and seed `20260901`; verify controls are non-overlapping, satisfy per-block and total-budget bounds, and remain unchanged when old SyncNet artifacts are altered.
- [ ] 2.3 Write and validate the immutable eligibility/condition manifest, require two complete control replicates, and emit `INSUFFICIENT` without GPU work when fewer than 12 samples/source groups remain; verify the real metadata preflight reproduces the readiness counts and exactly four ordered cells per retained sample.

## 3. Exact-Length Candidate Audio

- [ ] 3.1 Implement the frozen finite donor resampling and in-segment raised-cosine blend with exact natural length, chronological disjoint patches, no fallback/normalization, and pre-quantization outside-mask identity; verify synthetic tests cover length changes, short ramps, multiple blocks, clipping, non-finite input, and unchanged samples.
- [ ] 3.2 Generate and validate mono 16 kHz PCM16 audio for `natural`, `asr_targeted`, `target_control_0`, and `target_control_1`, bind every interval and hash, and verify natural identity decodes byte-identically to canonical natural PCM while every candidate satisfies its recorded edit budget.

## 4. Frozen Rendering and Replacement Scoring

- [ ] 4.1 Implement the frozen Wav2Lip cell runner with unique cwd/temp paths and hash-aware markers; verify an isolated one-sample smoke run produces all four videos and a test rejects any shared work path before launch.
- [ ] 4.2 Reuse the strict mux contract to pair each rendered video with untouched canonical natural PCM, never driver audio, and verify copied video provenance plus exact decoded-PCM equality with no crop/pad/stretch/shortest/video-reencode policy.
- [ ] 4.3 Reuse the SyncNet adapter and exact local-score parity code to export every full distance matrix and official global C/D/offset, then compute natural-offset fixed-coordinate inside/outside diagnostics on common support; verify one smoke cell passes upstream parity and track mismatch yields a null local diagnostic without score-based substitution.

## 5. Analysis, Orchestration, and Prototype Run

- [ ] 5.1 Implement the four positive-is-better per-sample estimands, descriptive summaries, win counts, edit-budget diagnostics, and deterministic 10,000-draw whole-sample `PCG64(20260901)` bootstrap; verify synthetic fixtures reproduce hand-calculated gains and never expose frame/patch resampling.
- [ ] 5.2 Implement engineering and scientific decision logic exactly at the frozen boundaries (`NOT_EVALUATED`, `<12` `INSUFFICIENT`, four-lower-bounds-positive `PROTOTYPE_SUPPORT`, otherwise `NO_PROTOTYPE_SUPPORT`) and verify below/equal/above boundary tests plus explicit interval interpretation.
- [ ] 5.3 Implement the single stage/resume entry point, strict finite artifacts, failure ledger, final validator, and one compact post-analysis plot per sample; verify interrupted, stale, corrupt, non-empty, and plotting-failure fixtures preserve immutable valid work and never alter numeric decisions.
- [ ] 5.4 Run the network-free focused tests, metadata preflight, one-sample Wav2Lip/strict-mux/SyncNet smoke, and—only after those pass—the locked eligible-cohort prototype; verify final artifacts recompute from source arrays, report the exact claim boundary, leave sealed splits untouched, and pass `openspec validate prototype-asr-targeted-local-replacement --strict`.
