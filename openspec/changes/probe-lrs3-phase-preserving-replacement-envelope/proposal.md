## Why

Every non-natural driver tested so far has failed to show reliable natural-audio replacement compatibility, while the DAC experiment showed that better reconstruction fidelity alone is insufficient. Before training another TTS/alignment model, we need a small positive-to-negative boundary experiment that asks whether preserving natural phase while moving spectral magnitude toward an existing aligned TTS candidate can produce any nontrivial replacement-compatible audio.

## What Changes

- Add a no-training LRS3 proof-of-concept over one fixed record from each of the 23 source groups in the existing MFA-linear cohort.
- Construct four deterministic phase-preserving spectral blends between untouched natural audio and its bound MFA-linear TTS candidate at fixed strengths `0.25`, `0.50`, `0.75`, and `1.00`.
- Add a polarity-inverted natural positive control and a 200-ms delayed-natural sensitivity control.
- Freshly render every arm through one frozen Wav2Lip contract and score only the candidate's own-audio and untouched-natural replacement cells with official SyncNet V2.
- Report the largest blend strength that remains non-inferior to the natural-driven baseline and has measurable movement in the TTS mel direction.
- Issue only a fit-cohort proof-of-concept decision. Do not train a model, claim target-speaker transfer, access sealed validation/test media, or authorize deployment.

## Capabilities

### New Capabilities

- `lrs3-phase-preserving-replacement-envelope`: Defines a provenance-locked experiment for finding whether a nontrivial natural-phase/TTS-magnitude blend can remain compatible with untouched natural audio after frozen Wav2Lip rendering.

### Modified Capabilities

None.

## Impact

- Adds an experiment package under `scripts/experiments/lrs3_phase_preserving_replacement_envelope/` and focused tests under `tests/experiments/lrs3_phase_preserving_replacement_envelope/`.
- Reads immutable natural audio, face video, and MFA-linear candidate artifacts from the completed 133-record LRS3 comparison run; it deterministically selects the first ordered record from each source group.
- Writes only to `runs/lrs3_phase_preserving_replacement_envelope_20260904/` and does not alter prior runs.
- Reuses the repository's frozen Wav2Lip, official SyncNet V2, strict PCM mux verification, hashing, and grouped-bootstrap utilities. No new learned model or external checkpoint is introduced.
