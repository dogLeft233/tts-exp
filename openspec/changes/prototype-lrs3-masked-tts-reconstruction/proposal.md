## Why

Prior LRS3 replacement experiments did not establish that a learned head actually uses paired TTS information. The strongest diagnosis is structural: natural and TTS WavLM-L6 features are clearly distinguishable (mean CKA `0.3828`, held-out frame probe accuracy `0.9411`), while the old frame-level cross-attention was nearly uniform (mean row entropy `0.9978`) and paired-TTS, natural-KV, and shuffled-TTS ablations did not identify a paired-TTS contribution. A weak downstream Sync loss asked one adapter to discover cross-rhythm alignment, use TTS, render audio, and improve replacement simultaneously, so ignoring TTS was an easy solution.

A small, dense-supervision prototype is needed before another replacement head. It shall ask only whether phone-aligned TTS features improve reconstruction of a masked natural log-mel phone on unseen fit-only source groups beyond an identically initialized and scheduled natural-context-only model. Same-checkpoint zero/wrong-phone inputs diagnose sensitivity but do not determine feasibility. This is an injectability and learning-task feasibility test, not a TFG or replacement experiment.

## What Changes

- Add one deterministic CPU-only fit-only LRS3 prototype that reuses the local 30-record/10-source-group subset whose parent `protocol_split` is exactly `train`, cached WavLM-L6 natural/TTS features, and existing MFA3 phone timings from the Exp31 worktree assets.
- Freeze a new source-group-disjoint split before feature loading: 19 records from six groups for training and 11 records from four groups for evaluation. No parent `validation` or `test` media/features may be opened.
- Deterministically match equal-labeled lexical phones between natural and TTS MFA sequences, reject silence/`spn`/unsupported phones, and map each TTS phone trajectory onto the natural phone grid with one explicit normalized-phase feature interpolation. MFA supplies phone blocks; no waveform is stretched or generated.
- Mask each natural target phone plus a fixed guard inside a fixed 96-frame normalized natural log-mel window, expose binary mask/target channels, and predict only the clean natural phone core. The natural target region is hidden before the trainable context encoder; no natural WavLM feature, phone label, transcript token, SyncNet value, or visual score enters the model.
- Train two tiny models from an identical initial state for three deterministic CPU seeds: `FULL_CORRECT` receives aligned paired-TTS WavLM-L6 features, while `NAT_ONLY` receives an all-zero TTS tensor. The FULL checkpoint is additionally evaluated with zero and deterministic train-pool wrong-phone TTS inputs as sensitivity diagnostics, not inferential controls.
- Decide feasibility from paired held-out `FULL_CORRECT` versus trained `NAT_ONLY` reconstruction losses aggregated mask → record → source group, with whole-source-group bootstrap, a 5% practical-effect gate, all-group sign agreement, and three-seed stability. Null/shuffled results diagnose branch sensitivity but cannot promote or rescue the result.
- Keep the prototype narrow: no raw-waveform target, vocoder, Wav2Lip, SyncNet, candidate audio, visual metric, checkpoint search, learned attention/alignment, hyperparameter sweep, sealed split, or claim that TTS clarity has already produced replacement-safe visual gain.

## Capabilities

### New Capabilities

- `masked-tts-reconstruction-prototype`: Reproducibly test whether deterministic phone-aligned paired-TTS features improve masked natural-phone log-mel reconstruction on held-out fit-only source groups over an identically initialized, separately trained natural-context-only model, with same-checkpoint zero/wrong-phone sensitivity diagnostics.

### Modified Capabilities

None.

## Impact

- Adds a narrow package under `scripts/experiments/masked_tts_reconstruction/` and focused tests under `tests/experiments/masked_tts_reconstruction/`.
- Reads only hash-validated local parents under `.claude/worktrees/lrs3-wavlm-resynthesis-50/` plus their referenced `protocol_split=train` LRS3 natural/TTS assets. It does not modify or copy old runs, checkpoints, datasets, or model caches.
- Reuses cached WavLM-L6 arrays and MFA3 phone JSON; computes natural log-mel targets locally and trains only small reconstruction networks.
- Writes one new immutable prototype run containing input locks, a score-blind split, mask/alignment manifests, train-only normalization, fixed-seed checkpoints, held-out predictions, paired metrics, bootstrap output, and separate engineering/scientific decisions.
- A feasibility pass authorizes design of a subsequent factorized content-versus-natural-anchor experiment. It does not authorize a waveform head, TFG rendering, strict replacement claim, or sealed evaluation.
