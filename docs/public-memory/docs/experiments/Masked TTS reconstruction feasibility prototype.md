---
title: Masked TTS reconstruction feasibility prototype
type: report
permalink: tts-exp/docs/experiments/masked-tts-reconstruction-feasibility-prototype
experiment_id: masked-tts-reconstruction-prototype
openspec_change: prototype-lrs3-masked-tts-reconstruction
status: complete
thread_id: fcd4c077-21ff-4445-a06c-07696af24afb
tags:
- masked-reconstruction
- tts
- lrs3
- feasibility
- fit-only
---

# Masked TTS reconstruction feasibility prototype

## Context

A small fit-only LRS3 prototype tested whether deterministic phone-aligned paired-TTS WavLM-L6 features are usable by a reconstruction model before attempting another waveform or visual head. The task masks a natural phone plus a four-frame guard and predicts the clean natural log-mel core. It is an injectability/learning-task feasibility test, not an enhancement or replacement test.

## Frozen implementation

- OpenSpec: `openspec/changes/prototype-lrs3-masked-tts-reconstruction/`
- Package: `scripts/experiments/masked_tts_reconstruction/`
- Focused tests: `tests/experiments/masked_tts_reconstruction/`
- Final artifact: `runs/lrs3_masked_tts_reconstruction_prototype_20260901_v2/`
- Source data: only 30 records whose parent `protocol_split` is exactly `train`; 10 source groups; 19 train records/6 groups and 11 evaluation records/4 groups.
- Primary masks: 495 train and 291 evaluation masks; readiness passed. Natural target mel uses the frozen Wav2Lip contract. TTS WavLM-L6 is mapped with `phone_phase_linear_v1`; aligned TTS input is zero outside the target core.
- Natural target-plus-guard is hidden before the trainable context branch. Leakage audit passed. The model has 482,256 trainable parameters and no attention, recurrence, waveform decoder, evaluator, or trainable pretrained module.
- FULL_CORRECT and NAT_ONLY were trained for seeds 20260901, 20260902, and 20260903 from identical initial states and byte-identical 600-step PCG64 schedules. FULL_ZERO and FULL_SHUFFLED were same-checkpoint sensitivity diagnostics only.

## Result

Engineering was `GO`; `sealed_splits_accessed=false`. All six checkpoints, 2,619 required evaluation cells, 873 available shuffled diagnostic cells, hierarchy, hashes, and artifact validation completed.

Primary result:

- Final evaluation-group `nat_gain = L_total(NAT_ONLY) - L_total(FULL_CORRECT)` was positive for all four groups: 0.07326, 0.18038, 0.11011, and 0.08302.
- Whole-source-group bootstrap median gain was 0.09656 with 95% CI `[0.07326, 0.18038]`.
- Median relative `L_total` reduction versus NAT_ONLY was 13.10%.
- All three seed-level overall medians were positive.
- Science decision: `ALGORITHM_FEASIBLE`.
- Shuffled diagnostic median `shuf_gap` was 0.22820; it remains descriptive sensitivity evidence and did not enter the feasibility gate.

## Interpretation boundary

`ALGORITHM_FEASIBLE` means only that paired phone-aligned TTS features improved masked natural log-mel reconstruction over an identically initialized and scheduled natural-context-only model under this frozen fit-only task. It does not establish phonetic-content causality, semantic clarity, TTS clarity retention, prosody disentanglement, waveform reachability, Wav2Lip/SyncNet gain, replacement effect, cross-TFG transfer, or population generalization. A later factorized natural-anchor/TTS-content head is authorized for discussion, but the next experiment must preserve these claim boundaries.

## Verification

- `PYTHONPATH=. pytest -q tests/experiments/masked_tts_reconstruction` → 7 passed
- `python -m compileall -q scripts/experiments/masked_tts_reconstruction` → passed
- `openspec validate prototype-lrs3-masked-tts-reconstruction --strict` → valid

## Observations

- [decision] Primary feasibility depends only on FULL_CORRECT versus separately trained NAT_ONLY; zero and shuffled conditions cannot promote or rescue the result.
- [insight] Dense masked natural reconstruction provided a clear learning task where clean natural target content is supervision rather than a model shortcut.
- [solution] Explicit phone-phase alignment and target masking made paired TTS feature use measurable without waveform generation, Wav2Lip, SyncNet, or sealed data.
- [constraint] Natural audio remains a high-bandwidth anchor in any downstream enhancement design; this result does not justify reducing it to duration/F0/energy/pause attributes.

## Relations

- follows [[31 - Replacement Salvage: Phone-Aligned Cross-Attention]]
- constrained_by [[Raw TTS strict replacement NO-GO]]
- informs [[Natural reference audio is available at inference]]
