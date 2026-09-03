## Why

The completed masked-reconstruction prototype established a limited but stable result: paired phone-aligned TTS WavLM-L6 reduced held-out natural-mel reconstruction loss versus an identically initialized/scheduled `NAT_ONLY` model (`13.10%` median relative reduction; all four groups and all three seeds positive). It did not establish whether the model retained token-level TTS information rather than using the TTS branch mainly as a high-dimensional phone-identity cue. The original cohort was also small (six train groups, four evaluation groups).

Before adding a vocoder or running a TFG audit, run one modestly larger fit-only experiment that separates paired token-level TTS features from a train-only phone-centroid control. This remains a mel-level identification experiment; it does not generate audio or claim TTS-feature retention in waveform.

## What Changes

- Reuse the completed implementation and freeze a score-blind expanded fit-only LRS3 cohort targeting 12 train source groups and eight fresh evaluation source groups, approximately 40–50 train records/1,000–1,500 masks and 20–30 evaluation records/500–800 masks.
- Keep the existing 482,256-parameter model, masking, phone alignment, mel contract, losses, optimizer, batch size, and three seeds unchanged; increase training from 600 to exactly 1,200 steps so mask exposure remains comparable after the cohort grows.
- Train three identically initialized and scheduled arms per seed: `PAIRED_TTS`, `PHONE_CENTROID`, and `NAT_ONLY`.
- Build `PHONE_CENTROID` from train groups only by averaging standardized aligned TTS-L6 core frames per lexical phone and repeating the resulting 1024-D centroid over the target core. It exposes phone identity without the paired token trajectory.
- Evaluate optional same-checkpoint train-donor interventions (`SAME_PHONE_DONOR`, `WRONG_PHONE_DONOR`) as sensitivity diagnostics only.
- Decide separately whether the original modality gain replicates and whether paired token-level TTS features outperform phone identity alone.
- Do not add waveform decoding, vocoder training, TFG/SyncNet evaluation, attention/alignment search, model scaling, or hyperparameter search.

## Capabilities

### New Capabilities

- `masked-tts-retention-scaleup`: Test on a modestly expanded fit-only cohort whether paired token-level TTS features improve masked natural-mel reconstruction beyond both natural context and a train-only phone-centroid representation.

### Modified Capabilities

None.

## Impact

- Extends `scripts/experiments/masked_tts_reconstruction/` and its focused tests rather than creating a new framework.
- Writes one new immutable run under `runs/lrs3_masked_tts_retention_scaleup_20260901/`; existing runs and checkpoints remain untouched.
- Reads only records whose parent `protocol_split` is exactly `train`; prior prototype evaluation groups are not moved into training, and the eight scale-up evaluation groups must be fresh relative to all ten prior prototype groups.
- A positive result authorizes a controlled waveform bridge with a natural roundtrip baseline. It does not itself establish perceptible TTS retention, audio quality, visual gain, SyncNet gain, or replacement effect.
