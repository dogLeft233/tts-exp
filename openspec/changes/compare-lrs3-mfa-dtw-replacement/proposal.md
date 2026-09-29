## Why

The existing same-phone MFA-constrained hard-DTW audit was limited to four AISHELL-1 records and feature/waveform reachability; it did not establish whether DTW improves frozen TFG behavior on LRS3 or whether any improvement survives untouched-natural-audio replacement. A controlled comparison against the completed 133-record LRS3 MFA-linear run is needed before treating phone-local DTW as an alignment upgrade.

## What Changes

- Add a provenance-locked LRS3 experiment that reuses the exact 133-record MFA-linear face-ready cohort, MFA3 alignments, frozen model assets, and historical MFA-linear evaluation cells.
- Add deterministic same-phone hard-DTW mapping over paired natural/TTS WavLM-L6 features, with a fixed normalized path band and no cross-phone matching.
- Generate DTW candidates through the same WavLM interpolation, HiFi-GAN, exact-length, and PCM16 contracts used by MFA-linear so alignment policy is the only intended candidate change.
- Add a staged frozen-Wav2Lip evaluation: first test DTW versus MFA-linear on candidate-driven diagonal cells; promote only after a pre-registered paired dual-metric gate.
- If and only if the diagonal gate passes, evaluate the already-rendered DTW videos after strict replacement with untouched natural PCM and compare DTW, MFA-linear, and the natural baseline.
- Record complete provenance, path diagnostics, engineering gates, source-group cluster-bootstrap intervals, and an explicit terminal decision without accessing sealed validation or test data.

## Capabilities

### New Capabilities

- `lrs3-mfa-dtw-comparison`: Defines deterministic phone-local DTW candidate construction, same-cohort frozen-TFG comparison, gated strict replacement, and auditable decision semantics.

### Modified Capabilities

None.

## Impact

- Adds a new experiment package under `scripts/experiments/` and focused tests under `tests/experiments/`.
- Reads existing immutable artifacts under `runs/lrs3_mfa_linear_replacement_mfa3_exploratory_20260825/` and writes to a distinct run root.
- Reuses pinned WavLM-L6, HiFi-GAN, Wav2Lip, SyncNet, ffmpeg, and ffprobe assets; no model training or dependency changes are planned.
- GPU work is limited to candidate extraction/vocoding and frozen Wav2Lip rendering; the replacement stage remains sealed unless the diagonal promotion gate passes.
