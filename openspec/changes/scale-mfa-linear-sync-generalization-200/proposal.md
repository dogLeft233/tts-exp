## Why

The previous shared adapter improved its four fit records but did not establish reliable transfer on the eight-record adapter-heldout cohort. The next experiment must test whether scaling the fit set to 200 records improves cross-source-group transfer, while preserving the same TTS-only input contract, frozen SyncNet supervision, and fail-closed evaluation rules.

The prior structural scan exposed only 102 eligible fit-only records, so this change also needs an explicit upstream asset-expansion gate. It must never reach 200 by duplicating samples, opening sealed splits, substituting failed records, or selecting records using evaluation outcomes.

## What Changes

- Add a new immutable experiment that fresh-trains one TTS-only waveform adapter on exactly 200 eligible fit-only records.
- Lock an independent evaluation cohort of 40 records from 40 source groups disjoint from every training source group before training or candidate inference.
- Require the expanded data inventory to provide all 200 training records and 40 evaluation source groups after structural, MFA-linear, geometry, and detached natural-reference checks; otherwise stop without reducing denominators.
- Keep the existing model, loss, SyncNet frontend, 31-offset curves, audio QC, `0.003` gain threshold, and `0.002` target-gap threshold unchanged.
- Evaluate the fresh 200-record model on the 40 source-group-heldout records using official real-video SyncNet curves, requiring all 40 engineering-valid and at least 30 record-level successes plus both median gains at least `0.003` for transfer promotion.
- Permit the frozen Wav2Lip strict natural-audio replacement matrix only after the expanded real-video gate passes; require all `360` matrix cells and the same proportional success rule.
- Persist score-independent manifests, source/checkpoint/video/audio hashes, curves, decisions, validation, and a bounded report in a new run root; never modify the earlier four-record or eight-record runs.

## Capabilities

### New Capabilities

- `mfa-linear-sync-generalization-scale-200`: Fresh-train and evaluate a TTS-only MFA-linear waveform adapter with 200 fit records and a source-group-disjoint 40-record transfer cohort, with optional gated frozen Wav2Lip replacement.

### Modified Capabilities

None.

## Impact

- Adds a new scale-specific experiment package and focused tests without changing the frozen eight-record transfer implementation or its artifacts.
- Extends the locked fit-only asset inventory and MFA-linear preparation path so at least 240 eligible records can be evaluated without sealed validation/test access.
- Reuses the current exact-length waveform, frozen SyncNet, signed PCM16, official curve, FFV1, Wav2Lip, hashing, and validation seams where their contracts remain identical.
- Adds a fresh 200-record trainer/orchestrator, score-independent train/evaluation manifests, 40-record real-video aggregation, conditional 360-cell replacement accounting, and immutable terminal artifacts.
- Claims remain limited to the preregistered source-group-heldout cohort and the one frozen Wav2Lip/SyncNet stack; the result is not a population, sealed-test, perceptual, content, speaker, multi-generator, or production claim.
