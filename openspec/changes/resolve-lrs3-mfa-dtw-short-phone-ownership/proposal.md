## Why

The first frozen LRS3 MFA-linear versus same-phone hard-DTW run reached candidate generation but was engineering-blocked at 122/133 records. The fixed WavLM center-time owner assigned no TTS frame to 11 valid 10–40 ms matched phones, so the experiment could not reach the pre-registered Wav2Lip/SyncNet comparison.

## What Changes

- Add a separately identified protocol revision for nominal feature-support ownership.
- Represent each WavLM frame by the half-stride interval centered at its nominal 20 ms frame center.
- Give a phone every frame whose nominal support has positive temporal overlap with that phone; adjacent phones may share a boundary frame.
- Keep support local to the matched MFA3 token instance, keep natural WavLM out of emitted conditioning, and reject phones outside all nominal support rather than extrapolating or falling back.
- Re-run the full frozen 133-record candidate stage in a new run root before any downstream score is observed.

## Capabilities

### New Capabilities

- `lrs3-mfa-dtw-short-phone-support`: Defines the pre-registered frame-support ownership revision that resolves short-phone candidate construction without changing the cohort or downstream gates.

### Modified Capabilities

None.

## Impact

- Adds a protocol revision and focused tests under the existing DTW experiment package.
- Writes to `runs/lrs3_mfa_dtw_replacement_short_phone_20260904/`; the blocked run remains immutable.
- Does not change the MFA-linear controls, Wav2Lip/SyncNet assets, bootstrap gate, replacement authorization, or sealed media boundaries.
