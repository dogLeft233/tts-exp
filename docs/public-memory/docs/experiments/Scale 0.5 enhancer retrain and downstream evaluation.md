---
title: Scale 0.5 enhancer retrain and downstream evaluation
type: note
permalink: tts-exp/docs/experiments/scale-0.5-enhancer-retrain-and-downstream-evaluation
thread_id: ses_00fa9dc33ffeZQDZBzLeen28Dd
tags:
- enhancer
- syncnet
- wav2lip
- valid-evaluation
- two-stage
---

# Scale 0.5 enhancer retrain and downstream evaluation

## Context

The two-stage HuBERT feature-targeted waveform enhancer never reproduced raw-TTS downstream lip-sync quality. The recorded full train/valid diagnostic (residual scale 0.5, lr 1e-3, seed 29, oracle target) showed feature-level gains but no checkpoint was ever saved — `stage2_warmup_recovery_audit.py` is a diagnostic-only script. This thread retrained that exact configuration with checkpoint saving, exported the 50 valid enhanced WAVs, and ran the fixed Wav2Lip + SyncNet V2 downstream evaluation to close the open question.

## What was found before this thread

- Old enhanced (v1_seed29, residual scale 0.05, 9 epochs, stage1_goal target): valid realize_cosine 0.0159 — not comparable to oracle-target metrics because the target is the stage1 goal, not raw-TTS features
- The recorded scale 0.5 diagnostic (oracle target): valid realize_combined 0.30424 → 0.26754, content 0.248 → 0.099, energy 0.077 → 0.0033, valid clipping 0
- No scale 0.5 checkpoint existed; old enhanced WAVs must not be presented as the scale 0.5 result

## Retrain (reproduced the diagnostic)

- Script: `/tmp/reproduce_scale05_train.py` — mirrors the audit loop exactly (warmup 100 passes realize-only + recovery 100 passes full weights, lr 1e-3, seed 29, target_mode aligned_raw_tts_oracle), plus checkpoint saving
- Manifest: `/tmp/tts_two_stage_preflight/renderer_manifests/localized_stage_manifest.json` (290 train / 50 valid), stage1: `/tmp/tts_two_stage_preflight/stage1_full_1ep/best.pt`
- Checkpoint: `runs/two_stage_hubert_aishell1_20260810/stage2_scale05_full_20260812/last.pt` (epoch 1, residual_scale 0.5)
- Reproduction matched: warmup valid realize_combined 0.30366 (recorded 0.30424); recovery valid 0.26274 (recorded 0.26754)
- Two script bugs fixed along the way: `residual_peak`/`clipped_sample_count` are computed in the audit `_forward` (not in `compute_stage2_loss` diagnostics), and `CHECKPOINT_SELECTION_METRIC` is `"valid_realize_combined"` — a run_epoch key, not the evaluate dict key (`"realize_combined"`)

## Export and downstream evaluation

- Exported 50 valid enhanced WAVs via `export_feature_targeted_waveform.py` (0 clipping, residual peak ≤ 0.4528 < 0.5 bound)
- Fixed protocol: same natural face videos, Wav2Lip GAN checkpoint, SyncNet V2, min_track=50; natural/tts results reused from the cached evaluation (`/tmp/wav2lip_scale05_full/`), only enhanced ran fresh

| condition | mean Sync-C | mean Sync-D |
|---|---:|---:|
| natural | 6.002 | 7.285 |
| raw TTS | 7.154 | 6.687 |
| old enhanced (0.05) | 6.040 | 7.294 |
| new enhanced (0.5) | 5.898 | 7.381 |

- New enhanced vs natural: ΔSync-C −0.104 (20/50 positive), worse than old enhanced +0.038 (29/50)
- New vs old enhanced paired: mean diff −0.142, t=−2.241, p≈0.025 — significantly worse
- Conclusion: scale 0.5 is a downstream-verified negative result; stronger 0.5 residual hurt relative to 0.05

## Where the thread landed

The feature-targeted waveform enhancer family does not reproduce raw-TTS lip-sync quality regardless of residual scale (0.05 or 0.5). Feature-layer proximity gains (realize/content/energy) do not transfer to SyncNet. No heldout S0770 run is warranted for this negative line. The experiment report `tts-enhancer-feasibility-diagnosis` was updated with the scale 0.5 downstream table, checkpoint provenance, and the resolved blocker.

## Observations

- [decision] Retrain the recorded scale 0.5 diagnostic configuration with checkpoint saving and run the fixed 50-valid Wav2Lip/SyncNet protocol #validation
- [insight] The audit script never saved checkpoints; the recorded "scale 0.5 model" existed only as diagnostic json — `stage2_checkpoint: None` #two-stage
- [insight] Scale 0.5 reproduce check: warmup valid realize_combined 0.30366 vs recorded 0.30424; recovery 0.26274 vs 0.26754 — the diagnostic is reproducible #reproducibility
- [result] Scale 0.5 downstream: Sync-C 5.898 (−0.104 vs natural), significantly worse than old 0.05 enhanced (p≈0.025); negative result #downstream
- [learning] Old v1_seed29 realize_cosine 0.0159 is NOT comparable to oracle-target realize values — different supervision targets #metric
- [tradeoff] Script bugs: residual_peak/clipped_sample_count live in audit `_forward`, and the checkpoint selection metric key differs from the evaluate dict key #debugging
- [decision] No heldout S0770 blind confirmation for this negative line #validation

## Relations

- relates_to [[tts-enhancer-feasibility-diagnosis]]
- relates_to [[TTS Feature Alignment and Duration Model Research]]
- relates_to [[Wav2Lip Local Deployment]]
- relates_to [[SyncNet Local Deployment]]