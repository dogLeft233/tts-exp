## Why

The first phase-preserving experiment found a promising descriptive result: `MAG_075` moved materially from natural audio toward the frozen MFA-linear TTS target and passed the registered replacement sub-gates. The run could not issue a scientific claim because its global 200-ms shift control did not validate the file-level SyncNet endpoint. We need one small confirmation experiment that uses unused fit records and a local timing corruption that cannot be repaired by one global offset.

## What Changes

- Add a no-training confirmation run on the second ordered record from each available source group in the same immutable 133-record parent cohort: 22 records from 22 groups, disjoint from the discovery cohort.
- Test one discovery-selected bridge, `BRIDGE_075`: natural phase and time grid with a fixed 75% move toward the bound MFA-linear TTS log magnitude.
- Add `N_REPEAT` as an independent natural-input rerender control and `LOCAL_SWAP` as a deterministic negative control that swaps the middle two quarters of natural PCM.
- Freshly render four arms through the frozen Wav2Lip contract and score six fixed cells per record with official SyncNet V2: 88 videos and 132 cells total.
- Confirm the bridge only if rerender stability and local-timing sensitivity controls pass, the bridge moves nontrivially toward TTS, replacement remains non-inferior, and replacement SyncC improves over the natural-driven baseline.
- Keep all conclusions fit-only and narrow. A positive result may authorize a separate reference-conditioned audio-head OpenSpec, but does not authorize training by itself.

## Capabilities

### New Capabilities

- `lrs3-natural-to-tts-bridge-confirmation`: Defines a control-calibrated confirmation of a natural-timed, TTS-directed internal Wav2Lip driver on unused fit records.

### Modified Capabilities

None.

## Impact

- Adds a future experiment package under `scripts/experiments/lrs3_natural_to_tts_bridge_confirmation/` and focused tests under `tests/experiments/lrs3_natural_to_tts_bridge_confirmation/`.
- Reads immutable parent manifests and bound fit-only media; it does not generate TTS, rerun MFA/DTW, train a model, or read sealed validation/test media.
- Writes only to `runs/lrs3_natural_to_tts_bridge_confirmation_20260904/` and never modifies the discovery or parent runs.
- Reuses the existing deterministic STFT, frozen Wav2Lip, strict mux, official SyncNet V2, hashing, and grouped-bootstrap contracts.
