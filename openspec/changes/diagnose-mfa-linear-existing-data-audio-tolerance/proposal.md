## Why

The fresh 94-record existing-data run completed training but stopped before real-video scoring because one deterministic evaluation candidate had normalized log-mel distance `0.100560`, only `0.000560` above the fixed `0.10` engineering limit. The user authorized a narrower exploratory tolerance so all eight fixed record-heldout candidates can receive official SyncNet scores without changing the synchronization thresholds or natural-audio isolation.

## What Changes

- Add a separate exploratory continuation using the already completed fresh 94-record step-100 checkpoint and the already locked 8-record evaluation manifest.
- Raise only the diagnostic candidate normalized log-mel limit to `0.11`; retain exact length, finite values, residual peak, PCM saturation, target-offset, official SyncNet, and replacement gates.
- Materialize and officially score all eight baseline/candidate real-video pairs without selecting or dropping rows after seeing scores.
- Permit Wav2Lip strict replacement only if the complete eight-record real-video gate passes, while labeling all conclusions exploratory because the tolerance was changed after observing the candidate QC failure.
- Preserve both earlier failed immutable run roots unchanged.

## Capabilities

### New Capabilities

- `mfa-linear-existing-data-audio-tolerance`: Explore all fixed record-heldout official scores under a narrowly relaxed post-hoc audio-tolerance diagnostic.

### Modified Capabilities

None.

## Impact

- Adds a diagnostic runner and tests under `scripts/experiments/mfa_linear_existing_data_audio_tolerance/` and a new immutable run root.
- Reads, but does not modify, the locked 94-record training checkpoint and 8-record manifest from `runs/lrs3_mfa_linear_sync_existing_data_20260903_cublas_fix/`.
- The result cannot be used as a pre-registered transfer claim because the audio-tolerance limit was selected after observing one failure; it is descriptive evidence only.
