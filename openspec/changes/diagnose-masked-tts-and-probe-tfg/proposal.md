## Why

The exploratory masked-reconstruction run found a stable `PAIRED_TTS > NAT_ONLY` modality gain, but no stable `PAIRED_TTS > PHONE_CENTROID` token gain. Two evaluation groups (`6ORDQFh0Byw` and `6yR5OUVb2gY`) were token-negative, and the current summary does not show whether their failures come from patch error, velocity error, phone composition, duration/alignment, or seed instability.

Before spending effort on a waveform decoder, first diagnose those groups from the completed artifacts. Then run a small frozen-Wav2Lip probe that feeds reconstructed mel directly into the TFG and evaluates strict natural-audio replacement with SyncNet. Direct mel injection isolates downstream visual utility without adding a vocoder confound.

## What Changes

- Add a read-only diagnostic over `runs/lrs3_masked_tts_retention_exploratory_20260901` that decomposes token and modality gains into patch and weighted-velocity terms and compares the two token-negative groups with the six positive groups by phone, duration, alignment ratio, record, and seed.
- Reuse the existing predictions and checkpoints; do not retrain the masked reconstructor.
- Freeze one evaluation record per each of the eight source groups using mask coverage and a deterministic tie-break, independent of loss or SyncNet values.
- Build full-record mel drivers by replacing only evaluated phone cores with each arm's saved prediction. Evaluate `PAIRED_TTS`, `PHONE_CENTROID`, and `NAT_ONLY` for all three seeds, plus one unchanged `NATURAL_MEL` baseline: 80 Wav2Lip renders in total.
- Feed those mels directly through the frozen Wav2Lip inference path, replace every rendered video's audio with untouched natural audio, and score with the existing frozen official SyncNet V2.
- Report paired downstream token and modality contrasts by source group. Keep all conclusions exploratory.

## Capabilities

### New Capabilities

- `masked-tts-tfg-probe`: Diagnose the failed token contrast and test whether paired TTS-conditioned mel predictions provide downstream frozen-Wav2Lip synchronization value beyond a phone centroid.

### Modified Capabilities

None.

## Impact

- Adds a small package under `scripts/experiments/masked_tts_tfg_probe/` and focused tests under `tests/experiments/masked_tts_tfg_probe/`.
- Writes a separate run under `runs/lrs3_masked_tts_tfg_probe_20260902/` and does not modify the formal or exploratory parent runs.
- Reuses existing Wav2Lip, SyncNet, evaluation predictions, natural mels, masks, and source videos. No new model training, vocoder, cohort search, or sealed-split access is required.
