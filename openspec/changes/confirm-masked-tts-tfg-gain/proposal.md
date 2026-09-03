## Why

The completed eight-record direct-mel probe found a consistent `PAIRED_TTS > NAT_ONLY` downstream gain, but its decision is correctly limited to `EXPLORATORY_MODALITY_ONLY`. Before attempting waveform decoding, validate that result on the parent run's remaining records without introducing new training, models, or audio-generation confounds.

## What Changes

- Use all 16 parent evaluation records that were not selected for the first TFG probe: two records from each of the same eight source groups.
- Reuse the saved three-seed predictions and the existing direct-mel Wav2Lip/replacement/SyncNet pipeline.
- Render the same four arms per record: `PAIRED_TTS`, `PHONE_CENTROID`, and `NAT_ONLY` for three seeds, plus one `NATURAL_MEL` reference, for 160 cells total.
- Treat `PAIRED_TTS > NAT_ONLY` as the primary modality contrast and `PAIRED_TTS > PHONE_CENTROID` as the secondary token contrast.
- Freeze the cohort and analysis before scoring, then report one confirmatory status with the existing direct-mel claim boundary.

## Capabilities

### New Capabilities

- `masked-tts-tfg-confirmation`: Confirm or reject the exploratory downstream modality gain on records not used by the first frozen-Wav2Lip probe.

### Modified Capabilities

None.

## Impact

- Reuses the existing `masked_tts_tfg_probe` implementation, checkpoints, predictions, natural mels, source videos, and strict replacement checks.
- Writes a separate run under `runs/lrs3_masked_tts_tfg_confirmation_20260902/`.
- Requires no reconstructor training, vocoder, waveform decoder, checkpoint search, or sealed-split access.
