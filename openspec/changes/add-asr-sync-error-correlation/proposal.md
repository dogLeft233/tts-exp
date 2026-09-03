## Why

The repository can score whole-clip lip sync and already has paired natural/TTS LRS3 assets, but it cannot test whether ASR recognition errors occur at the same times as locally poor audio-visual synchronization. A frozen, reproducible exploratory protocol is needed now so this hypothesis can be evaluated without misusing clip-level Sync-C, leaking sealed splits, or introducing avoidable model and alignment ambiguity.

## What Changes

- Add a fit-only, paired LRS3 experiment with one natural-audio arm and one raw-TTS arm per frozen sample.
- Add an offline Wav2Vec2-CTC inference contract that uses greedy decoding only, exports frame/word timestamps, and records model/runtime provenance.
- Add a local SyncNet V2 scoring contract derived from the repository's actual distance matrix and global offset calculation, with parity checks against the upstream whole-clip evaluator.
- Add deterministic GT/ASR edit alignment, explicit time assignment for substitutions, deletions, and insertions, and a shared valid timeline for ASR-error and low-sync masks.
- Add per-record and arm-level overlap/correlation reports, a paired natural-versus-TTS comparison, minimal timeline plots, and explicit engineering/scientific decision outputs.
- Add preflight, model download/cache, resume, failure-ledger, and output-schema requirements so the experiment can be run locally on GPU or CPU without a service deployment.
- Keep the change intentionally narrow: reuse existing manifests, strict muxing, SyncNet V2, and statistics helpers; do not train models, tune on sealed data, add an API/service, or create a general experiment framework.

## Capabilities

### New Capabilities

- `asr-sync-error-correlation-experiment`: Reproducibly prepare, execute, validate, and report the paired LRS3 natural/TTS ASR-error versus local-sync correlation experiment.

### Modified Capabilities

None.

## Impact

- Adds a small experiment package under `scripts/experiments/` plus focused tests under `tests/experiments/`.
- Reads existing canonical LRS3/TTS manifests and the existing fit-only 24-record cohort; sealed internal-dev, validation, and test media remain untouched.
- Reuses `third_party/syncnet_python`, its `syncnet_v2.model`, FFmpeg/FFprobe, and the repository's strict PCM-preserving mux behavior.
- Introduces Hugging Face Transformers Wav2Vec2-CTC runtime dependencies and a pinned local model revision/cache entry; no cloud credentials or online inference are required after model acquisition.
- Writes a new immutable run directory containing manifests, stage outputs, logs, plots, summary, and decision files.
