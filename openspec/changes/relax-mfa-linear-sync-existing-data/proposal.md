## Why

The preregistered 200-record/40-source-group expansion cannot run against the repository's currently prepared assets: only 102 records across 10 source groups satisfy the existing exact-length MFA-linear and real-video contracts. To obtain an empirical result without waiting for a new LRS3 asset build, this change defines a deliberately narrower record-level heldout experiment over those 102 verified records.

## What Changes

- Use exactly 94 eligible records for fresh TTS-only adapter training and 8 different eligible records for evaluation.
- Select evaluation rows deterministically before training, with broad source-group coverage where available, while explicitly allowing source-group overlap between training and evaluation.
- Keep natural-audio isolation, the exact `[1,1,61440]` MFA-linear input, frozen SyncNet, official 31-offset scoring, waveform QC, gain thresholds, and immutable artifact rules unchanged.
- Interpret the evaluation as record-heldout within the fit-only universe, not as source-group-heldout, sealed-test, or population generalization.
- Permit the same strict natural-audio Wav2Lip replacement stage only after the complete 8-record real-video gate passes; require all 72 matrix cells and the existing six-success rule.
- Preserve the blocked 200-record run and all earlier artifacts without modification.

## Capabilities

### New Capabilities

- `mfa-linear-sync-existing-data`: Run a bounded record-heldout TTS-only MFA-linear synchronization and conditional replacement experiment using the currently verified 102-record fit-only inventory.

### Modified Capabilities

None.

## Impact

- Adds a separate scale-specific runner, configuration, tests, and immutable run root under `scripts/experiments/mfa_linear_sync_existing_data/` and `runs/`.
- Consumes the current policy/MFA/video assets but does not access validation/test split records and does not mutate the 200-record blocked run.
- Reuses the existing model, differentiable SyncNet loss, official curve, signed PCM16, FFV1, Wav2Lip, and validation seams.
- Narrows the claim boundary to unseen record IDs whose source groups may have appeared in training; it does not answer unseen-speaker/source-group transfer.
