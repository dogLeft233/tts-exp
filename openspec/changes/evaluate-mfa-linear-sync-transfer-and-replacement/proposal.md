## Why

The sealed P1 result at `runs/lrs3_mfa_linear_real_video_sync_20260903_v4/` establishes only that one exact-length MFA-linear record can be optimized against its own real video. Before treating this TTS-only waveform adapter as useful, the experiment must determine whether one shared checkpoint transfers to untouched source groups and whether any real-video synchronization gain survives the stricter downstream test: driving a frozen talking-face generator and scoring the generated motion after replacing its audio with untouched natural audio.

## What Changes

- Require the existing `prototype-mfa-linear-real-video-sync` P2 shared-four stage to run from a fresh initialization under its frozen protocol; this follow-up stops unless P2 produces a validated `FIXED_DATA_SHARED_TTS_ONLY_SYNC_CONSTRUCTED` checkpoint.
- Freeze an eight-record adapter-heldout LRS3 cohort, one record from each of eight source groups disjoint from all P2 fit groups, using a score-independent deterministic selection rule before candidate inference or downstream scoring.
- Evaluate the fixed P2 checkpoint without optimizer access, checkpoint selection, retries, or threshold tuning against each heldout record's frozen real-video coordinates, comparing candidate audio with the exact MFA-linear identity baseline through the same official-equivalent SyncNet curve path.
- Permit natural audio during evaluation only as a detached coordinate reference and as the authoritative replacement/oracle audio in the downstream audit; it remains absent from the adapter input, checkpoint construction, and all optimization graphs.
- Gate downstream execution on a complete heldout real-video transfer result rather than promoting a fixed-data P1/P2 fit.
- Add a frozen-Wav2Lip strict replacement audit with three driver arms—natural, MFA-linear baseline, and adapted candidate—and a complete 3×3 driver-video/evaluation-audio matrix on the same heldout cohort.
- Make `generated-with-candidate + evaluated-with-natural` versus `generated-with-MFA + evaluated-with-natural` the only authoritative replacement contrast. Native diagonal cells and cross-audio cells are diagnostics and cannot establish replacement gain.
- Keep decisions fail-closed and machine-recomputable, separating engineering validity, adapter-heldout real-video transfer, and frozen-Wav2Lip replacement transfer.
- Preserve a narrow claim boundary: a pass supports empirical transfer to this frozen adapter-heldout cohort and this one frozen Wav2Lip/SyncNet stack, not population generalization, perceptual quality, content/speaker preservation, or safety across TFG models.

## Capabilities

### New Capabilities

- `mfa-linear-sync-transfer-and-replacement-evaluation`: Evaluate a frozen shared TTS-only waveform adapter on source-group-disjoint records and perform a strict natural-audio-replacement audit through a frozen talking-face generator with score-independent cohorts, complete comparison matrices, and explicit scientific gates.

### Modified Capabilities

None.

## Impact

- Adds a compact follow-up evaluator under `scripts/experiments/mfa_linear_sync_transfer/` and focused tests under `tests/experiments/mfa_linear_sync_transfer/`.
- Consumes but never mutates the sealed P1 run, a separately validated P2 run produced by the existing prototype change, the existing exact-length MFA-linear/policy assets, frozen real-video tracks, frozen Wav2Lip GAN checkpoint, and frozen SyncNet V2 checkpoint.
- Reuses the validated PCM16/MFCC/window/curve, official scoring, mux verification, hashing, QC, and replacement-audit seams already present in the repository; it adds no training architecture or production API.
- Writes a new immutable run containing prerequisite bindings, a frozen cohort manifest, baseline/candidate waveforms, real-video curves, Wav2Lip renders, all nine replacement cells per record, complete provenance, and separate transfer/replacement decisions.
- Does not access a sealed dataset test split. “Heldout” in this change means adapter-heldout and source-group-disjoint from the four P2 optimization records.
