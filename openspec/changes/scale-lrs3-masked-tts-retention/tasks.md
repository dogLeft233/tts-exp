## 1. Lock the expanded fit-only cohort

- [ ] 1.1 Extend metadata-only discovery to deterministically freeze 12 train groups and eight fresh evaluation groups using `design.md`; reject non-`train` parents, prior evaluation-group reuse, group overlap, hash drift, score-based fields, substitutions after lock, and insufficient denominators.
- [ ] 1.2 Reuse the existing phone alignment/mask pipeline, apply the train-only phone-support filter, and write the shared immutable mask manifest and leakage audit.

## 2. Add the phone-centroid control

- [ ] 2.1 Build/hash train-only per-phone 1024-D centroids with support provenance; verify no evaluation contribution, exact repeat-over-core/zero-outside-core tensors, and identical primary masks across arms.
- [ ] 2.2 Extend tests for deterministic group/record selection, centroid support, train-only statistics, unsupported-phone exclusion, and donor assignment.

## 3. Run the fixed scale-up

- [ ] 3.1 Generate one shared `1,200×16` schedule per seed and train nine fixed checkpoints (`PAIRED_TTS`, `PHONE_CENTROID`, `NAT_ONLY` × three seeds) from byte-identical per-seed initial states; preserve all existing model/loss/optimizer constants.
- [ ] 3.2 Evaluate all three required conditions on every frozen evaluation mask; add optional train-only same-phone and wrong-phone donor diagnostics without changing the primary denominator.

## 4. Analyze and validate

- [ ] 4.1 Compute `modality_gain` and `token_gain`, mask→record→group→seed aggregation, eight-group bootstrap intervals, relative reductions, per-group/per-seed signs, diagnostic coverage, and the exact decision statuses in `design.md`.
- [ ] 4.2 Run focused tests, compile checks, artifact validation, and `openspec validate scale-lrs3-masked-tts-retention --strict`; report `sealed_splits_accessed=false`, preserve interrupted/failed runs, and do not start waveform or TFG work from anything below `TOKEN_LEVEL_TTS_SIGNAL_SUPPORTED`.
