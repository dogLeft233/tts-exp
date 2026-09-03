## Why

The completed confirmation established that paired TTS-side conditioning improves the frozen-Wav2Lip endpoint over zero TTS conditioning, but it did not establish that the exact paired trajectory is better than a phone centroid. It also did not use unchanged natural mel as an engineering promotion baseline. Before waveform decoding, separate three questions: whether the learned path is useful when natural audio is already available, whether the frozen reconstructor uses instance-specific trajectory information, and whether one minimal hard-negative training change can create that sensitivity if it is absent.

## What Changes

- Reanalyze the existing 16-record confirmation scores against the unchanged `NATURAL_MEL` reference without rerendering or changing the prior confirmatory decision.
- Add two frozen-model controls on the same 16 records: a real trajectory from a different occurrence of the same phone, and the paired trajectory reversed within the target phone.
- Reuse the existing three checkpoints, mel patching, frozen Wav2Lip, untouched-natural-audio replacement, and official SyncNet V2 path.
- If the frozen diagnosis does not show trajectory sensitivity, define one optional retraining arm that adds a fixed same-phone hard-negative ranking term while leaving the model, data split, schedule, and reconstruction loss unchanged.
- Keep every result exploratory because all parent evaluation records have already been inspected through earlier experiments.

## Capabilities

### New Capabilities

- `masked-tts-trajectory-specificity`: Audit the natural-reference baseline, test instance-specific trajectory sensitivity with matched controls, and gate one minimal hard-negative retraining attempt.

### Modified Capabilities

None.

## Impact

- Reuses artifacts under `runs/lrs3_masked_tts_retention_exploratory_20260901/` and `runs/lrs3_masked_tts_tfg_confirmation_20260902/`.
- Adds a small experiment package and focused tests; existing parent artifacts remain read-only.
- Adds 96 frozen-model render cells for the two new controls. Conditional retraining and its downstream renders run only when the frozen diagnosis says they are needed.
- Adds no vocoder, waveform decoder, checkpoint search, architecture sweep, mask redesign, or sealed-split access.
