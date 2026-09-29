## 1. Protocol and Asset Lock

- [x] 1.1 Create `scripts/experiments/lrs3_dac16k_codec_identity/` with frozen protocol constants, stage paths, run-root guards, canonical JSON hashing, append-only decisions, and atomic per-cell writes; test changed constants and populated incompatible roots are rejected.
- [x] 1.2 Pin Descript Audio Codec source tag `0.0.5` / commit `408235a9dcd2983684c87615a1bc2a8954f6eb47`, native `16khz` / `8kbps`, and all available quantizers; lock the dependency environment, release URL, checkpoint byte size, and computed SHA-256 before cohort access.
- [x] 1.3 Implement Stage 00 binding to parent summary SHA-256 `96a29d7355ca3916ef2f60935147d9f20a123a8a7aefa12db130bbcf07eb08d5`; verify exactly 50 ordered IDs, 38 source groups, ordered ID hash `8e47514fdf877e9a57533e8719bd16fab1b96a99d8cb47cfa007c96689326b94`, and all per-record natural/face/WavLM artifact hashes.
- [x] 1.4 Add failure and media-access ledgers; verify Stage 00 performs no cohort audio decode, no GPU inference, no score read for selection, and no sealed validation/test access.

## 2. DAC Reconstruction

- [x] 2.1 Implement direct pinned-model inference from exact 16-kHz mono PCM16 float input with `n_quantizers=None`, bypassing the DAC CLI loudness normalization; test the call contract and reject every resampling, gain/loudness normalization, temporal correction, filtering, or output-repair option.
- [x] 2.2 Permit only model-internal right-padding and crop decoded output to the original length; canonicalize once with the audited PCM16 helper and test exact sample count, rate, channels, sample width, finite output, clipping, and deterministic repeated hashes.
- [x] 2.3 Write per-record DAC provenance including source/checkpoint/config hashes, original and padded lengths, latent/code shapes and counts, raw output diagnostics, canonical PCM hash, and QC; test interruption/resume and tampered-sidecar rejection.
- [x] 2.4 Run Stage 01 for all 50 records and independently verify a complete exact-length DAC audio manifest; an incomplete set must produce engineering `BLOCKED` without substitution.

## 3. Frozen Audio-Fidelity Panel

- [x] 3.1 Implement exact Wav2Lip mel parity and multi-resolution log-STFT metrics with frozen parameters; verify against the repository's official Wav2Lip preprocessing and deterministic synthetic fixtures.
- [x] 3.2 Implement zero-lag waveform correlation, SI-SDR, RMS/peak ratio, pinned F0/voicing errors, energy-envelope correlation, and available onset/phone-boundary-window diagnostics; verify no metric searches or applies lag, phase, gain, or alignment.
- [x] 3.3 Compute Stage 02 metrics for natural versus WavLM and natural versus DAC on all 50 records before TFG outcome inspection; freeze per-record rows and `mel_improvement = mel_L1(N,W) - mel_L1(N,D)`.
- [x] 3.4 Independently recompute Stage 02 from bound PCM and verify metric configurations, source-group labels, row order, counts, hashes, and absence of waveform modification.

## 4. Shared-Geometry Frozen Wav2Lip Rendering

- [x] 4.1 Bind the frozen Wav2Lip checkpoint, Python environment, invocation flags, ffmpeg/ffprobe, face-video hashes, and GPU/runtime provenance; test any changed renderer binding is rejected.
- [x] 4.2 Compute or load exactly one face geometry/fallback decision per record, hash it, and reuse it for `A_N`, `A_W`, and `A_D`; test per-arm geometry divergence fails the record and complete run.
- [x] 4.3 Freshly render exactly `V_N`, `V_W`, and `V_D` once per record, loading the frozen model once where possible; record command/input/output hashes and validate all 150 videos without score-based retry.
- [x] 4.4 Add hash-validated resume for render cells and verify an interrupted run accepts only complete cells with unchanged audio, face geometry, model, command, and output identities.

## 5. Strict 3-by-3 Matrix

- [x] 5.1 Implement all nine `V_{N,W,D}/A_{N,W,D}` cells per record, with video stream copy and PCM s16le audio; test the matrix enumerator produces exactly 450 unique ordered cells.
- [x] 5.2 Decode every muxed audio track and require byte-for-byte equality plus exact sample count/rate/channel/sample-width agreement with the selected source PCM; verify video elementary-stream identity and reject one-byte corruption.
- [x] 5.3 Run the pinned official file-level SyncNet V2 pipeline exactly once per valid cell and record Sync-C, Sync-D, offset, commands, logs, model hash, and media hashes; an absent or invalid cell must block scientific analysis.
- [x] 5.4 Independently validate all 150 video sources and 450 matrix cells, including no duplicate/missing IDs, no cross-record audio/video, no changed face geometry, and no parent/model/tool drift.

## 6. Registered Statistics and Decisions

- [x] 6.1 Implement deterministic 10,000-draw source-group cluster bootstrap with seed `20260904`; verify grouped resampling, percentile 95% intervals, means, win counts, and repeated-run equality against synthetic fixtures.
- [x] 6.2 Compute the registered `DAC_over_W_C`, `DAC_over_W_D`, `DAC_identity_C`, `DAC_identity_D`, and `mel_improvement` endpoints with signs exactly as specified; test hand-computable fixtures.
- [x] 6.3 Implement `DAC_REDUCES_CODEC_REPLACEMENT_PENALTY` only when all three improvement lower bounds are strictly positive; test zero-touching and incomplete endpoints fail promotion.
- [x] 6.4 Implement `DAC_CODEC_IDENTITY_COMPATIBLE` only when improvement passes and both identity lower bounds are strictly greater than `-0.10`; otherwise emit exactly `DAC_CODEC_PARTIAL_IMPROVEMENT` or `DAC_CODEC_HYPOTHESIS_NOT_SUPPORTED` as specified.
- [x] 6.5 Report all diagonal, own-audio-preference, and cross-codec cells as descriptive diagnostics without allowing them to alter the registered decision.
- [x] 6.6 Write one self-hashed terminal artifact with engineering status, scientific decision, complete endpoint values/intervals, limitations, parent hashes, and `future_tts_alignment_experiment_eligible`; verify only identity compatibility sets eligibility true.

## 7. Validation and Execution Gates

- [x] 7.1 Add an independent artifact validator for model/cohort bindings, codec invariants, PCM/media identities, fidelity metrics, render geometry, complete matrix, statistics, decisions, and sealed-scope claims; test one-field corruption at each stage.
- [x] 7.2 Run focused pytest, relevant direct-resynthesis/strict-mux regression tests, Ruff, Python compilation, and `openspec validate validate-lrs3-dac16k-codec-identity --strict`; require all checks to pass before media execution.
- [x] 7.3 Run Stage 00 read-only preflight and verify the DAC checkpoint lock, exact frozen cohort, zero cohort media decode, zero score-based selection, and zero sealed-data access.
- [x] 7.4 During the registered GPU window and on an idle GPU, execute Stages 01 through 04 with hash-validated resume; preserve incomplete outputs as engineering evidence rather than changing records or configuration.
- [x] 7.5 Finalize Stage 05/06 exactly once, independently recompute all reported numbers, and keep every TTS, MFA, DTW, Soft-DTW, residual, training, and held-out path sealed regardless of outcome.
