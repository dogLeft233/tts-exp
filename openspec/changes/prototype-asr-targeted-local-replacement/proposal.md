## Why

The completed ASR–local-sync study found a weak natural-audio association, but only 13.71% of low-sync cells were ASR-error cells and correlation cannot show that editing those regions changes synchronization. A small, frozen causal prototype is needed before building a prosody model: patch TTS content only into natural-ASR error blocks, compare it with equally sized non-error patches, and judge the unchanged-natural-audio replacement endpoint.

## What Changes

- Add one fit-only prototype over the existing 24-record discovery-reuse cohort and immutable outputs from `add-asr-sync-error-correlation`; no sealed media or new cohort search is allowed.
- Select usable samples from ASR artifacts alone: a natural error block must contain reference words, no mixed insertion, and every operation in the corresponding TTS reference interval must be recognized correctly by the same frozen greedy CTC output. The strict metadata-only scan finds 13 potentially usable samples, subject to hash-bound preflight reproduction.
- Construct exact-natural-length waveform drivers by replacing eligible natural blocks with their aligned TTS blocks using one deterministic duration-normalization and boundary-crossfade operator.
- Construct two deterministic matched non-error controls per sample with the same block count and approximately matched word count and duration. These are descriptive comparators, not randomized causal controls; the natural-baseline contrast remains the conditional intervention estimand.
- Render only four required conditions per usable sample: natural identity, ASR-targeted patch, and two matched controls. Score every rendered video after strict replacement with untouched natural PCM using frozen Wav2Lip and official SyncNet V2.
- Report global replacement Sync-C/Sync-D gains, fixed-coordinate edited-region diagnostics from full SyncNet distance matrices, whole-source-group bootstrap intervals, and an explicit `PROTOTYPE_SUPPORT`, `NO_PROTOTYPE_SUPPORT`, `INSUFFICIENT`, or `NOT_EVALUATED` decision.
- Keep the change deliberately small: no model training, no mel-seam modification, no prosody encoder, no hyperparameter search, no full-TTS arm, no new ASR inference, no service/framework, and no sealed confirmation split.

## Capabilities

### New Capabilities

- `asr-targeted-local-replacement-prototype`: Reproducibly test whether sparse TTS waveform patches at natural-ASR error blocks improve the strict untouched-natural-audio replacement endpoint more than matched non-error patches.

### Modified Capabilities

None.

## Impact

- Adds a narrow package under `scripts/experiments/asr_targeted_local_replacement/` and focused tests under `tests/experiments/asr_targeted_local_replacement/`.
- Reads existing fit-only manifests, locked ASR records, natural/TTS PCM, Wav2Lip, SyncNet V2, and strict-mux utilities without changing them.
- Uses the existing local model/checkpoint assets; network access may install a missing pinned package but must not introduce a mutable model dependency.
- Writes one new immutable prototype run directory with a lock, candidate WAVs, renders, replacement muxes, SyncNet matrices, metrics, minimal plots, and decisions.
