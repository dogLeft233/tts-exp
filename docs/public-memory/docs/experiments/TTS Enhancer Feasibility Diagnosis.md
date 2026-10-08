---
title: TTS Enhancer Feasibility Diagnosis
type: report
permalink: tts-exp/docs/experiments/tts-enhancer-feasibility-diagnosis
tags:
- tts
- enhancer
- syncnet
- valid-evaluation
- alignment
---

# TTS Enhancer Feasibility Diagnosis

## Context

This thread investigates why the current two-stage TTS feature-targeted waveform enhancer does not reproduce raw-TTS downstream lip-sync quality. The evaluation policy is valid-only during development: train speakers are S0901, S0906, S0912, S0913, S0914, S0915; valid speaker is S0765; heldout S0770 must not be used for checkpoint, seed, loss-weight, or strategy selection.

## Current technical findings

## Scale 0.5 model record
## Scale 0.5 downstream evaluation (completed 2026-08-12)

The scale 0.5 candidate was retrained with the recorded configuration (realization-only warmup 100 passes + recovery 100 passes, lr 1e-3, seed 29, oracle target, manifest `renderer_manifests/localized_stage_manifest.json`), exported to 50 valid enhanced WAVs, and evaluated with the identical fixed Wav2Lip + SyncNet V2 protocol. Checkpoint: `runs/two_stage_hubert_aishell1_20260810/stage2_scale05_full_20260812/last.pt` (epoch 1, residual_scale 0.5, zero-clipping valid export, residual peak ≤ 0.4528 < 0.5 bound). Warmup reproduction matched the recorded diagnostic (valid realize_combined 0.3037 vs recorded 0.30424).

| condition | mean Sync-C (higher better) | mean Sync-D (lower better) |
|---|---:|---:|
| natural | 6.002 | 7.285 |
| raw TTS | 7.154 | 6.687 |
| old enhanced (v1_seed29) | 6.040 | 7.294 |
| **new enhanced (scale 0.5)** | **5.898** | **7.381** |

Paired against natural, the new scale 0.5 enhanced output changed Sync-C by **−0.104** (20/50 positive samples), worse than old enhanced (+0.038, 29/50). Paired t-test of new vs old enhanced deltas: mean difference −0.142, t=−2.241, p≈0.025. The scale 0.5 feature improvements (realization cosine 0.30424→0.26754, content cosine 0.248→0.099, energy 0.077→0.0033) did **not** transfer to downstream lip-sync; the stronger 0.5-scale residual slightly *hurt* relative to the 0.05-scale old output. Conclusion: residual scale 0.5 is now downstream-verified as a **negative result**; the feature-targeted waveform enhancer family does not reproduce raw-TTS lip-sync quality regardless of scale.

The current Stage 2 candidate uses residual scale `0.5`, realization-only warmup followed by recovery with the preservation/content/energy/residual objectives, and the recorded full train/valid diagnostic configuration. This record is a feature/waveform validation result, not a claim of downstream TFG success.

On valid S0765, the recorded metrics changed as follows:

| metric | before | after |
|---|---:|---:|
| realization cosine | 0.30424 | 0.26754 |
| content cosine | 0.248 | 0.099 |
| energy difference | 0.077 | 0.0033 |
| clipping samples | 0 | 0 |

On the train split, realization cosine changed from `0.22278` to `0.20501`; clipping decreased from 132 samples to 8 samples but did not reach zero. The valid output had exact waveform length, finite values, and no clipping in the recorded diagnostic.

The model was selected as a candidate from train/valid diagnostics only. Its enhanced WAVs have not yet been exported and evaluated through the fixed 50-sample valid Wav2Lip/SyncNet pipeline. The existing downstream result belongs to the older `stage2_feature_targeted_20260810_v1_seed29` output and must not be attributed to scale 0.5.

The HuBERT L6 target is learnable on small subsets. MFA-linear source-grid alignment is retained as the production diagnostic baseline: phone pooling is consistently worse, hard-DTW is approximately tied, and directly transferring TTS phone durations to the exact natural waveform length worsens the HuBERT gap. A feature residual predictor and direct waveform upper-bound optimization both reduce the target gap, so the target is not completely unreachable.

Stage 2 is also structurally trainable. A realization-only warmup followed by recovery of preservation, content, energy, and residual losses performs better than applying the full loss from the first step. Learning rate 1e-3 is more effective than the earlier 1e-4 diagnostic setting. Residual scale 0.5 is the current candidate because it improved valid feature metrics without clipping in the recorded full train/valid run.

The latest scale=0.5 full train/valid diagnostic reported:

- valid realization cosine: 0.30424 to 0.26754;
- valid content cosine: 0.248 to 0.099;
- valid energy difference: 0.077 to 0.0033;
- valid clipping: 0;
- train realization cosine: 0.22278 to 0.20501;
- train clipping: reduced from 132 to 8, not yet zero.

These are feature/waveform diagnostics only; they do not establish downstream improvement.

## Valid downstream evaluation

A fixed local Wav2Lip plus SyncNet V2 evaluation was completed on all 50 valid S0765 samples, with no failures. The evaluated enhanced audio was explicitly the older checkpoint output `runs/two_stage_hubert_aishell1_20260810/stage2_feature_targeted_20260810_v1_seed29/valid_enhanced_wav`, not the latest scale=0.5 model.

Using the same natural face videos, Wav2Lip checkpoint, SyncNet V2 model, and `min_track=50` protocol:

| condition | mean Sync-C (higher better) | mean Sync-D (lower better) |
|---|---:|---:|
| natural | 6.002 | 7.285 |
| raw TTS | 7.154 | 6.687 |
| old enhanced | 6.040 | 7.294 |

Paired against natural, raw TTS improved Sync-C by 1.153 and Sync-D by -0.597. The old enhanced output changed Sync-C by only +0.038 and Sync-D by +0.010, effectively remaining at natural quality. All conditions had nearly identical AV offsets, so the result is not explained by a gross audio-video offset mismatch.

## Decision and next action

Do not claim that residual scale 0.5 improves TFG/SyncNet until its latest checkpoint is exported to 50 valid enhanced WAV files and evaluated with the identical Wav2Lip/SyncNet protocol. The old enhanced result must not be presented as the scale=0.5 result. After the fixed valid evaluation, heldout S0770 may be evaluated once as a blind confirmation, but must not be used to tune the method.

## Observations

- [insight] Feature-layer improvement and downstream lip-sync improvement are not equivalent; the old enhanced waveform stayed at natural SyncNet quality despite the broader feature-targeted direction. #downstream
- [decision] MFA-linear alignment remains the default target geometry; phone pooling and duration transfer are rejected for the current exact-length waveform contract. #alignment
- [decision] Residual scale 0.5 is a candidate configuration, not yet a downstream-validated model. #validation
- [problem] The latest scale=0.5 enhanced WAV export and its 50-sample valid Wav2Lip/SyncNet evaluation are still missing. #blocker

## Relations

- relates_to [[TTS Feature Alignment and Duration Model Research]]
- relates_to [[Wav2Lip Local Deployment]]
- relates_to [[SyncNet Local Deployment]]

## Checkpoint provenance

The earlier note did not record a verified checkpoint path for the scale 0.5 diagnostic. The repository currently contains these Stage 2 checkpoints:

- `runs/two_stage_hubert_aishell1_20260810/stage2_feature_targeted_20260810_v1_seed29/best.pt`
- `runs/two_stage_hubert_aishell1_20260810/stage2_feature_targeted_20260810_v2_seed13/best.pt`

The 50-sample valid downstream result used the first path's exported audio: `stage2_feature_targeted_20260810_v1_seed29/valid_enhanced_wav`. The checkpoint metadata records warmup/recovery and seed information, but does not expose a `residual_scale` field that conclusively identifies either checkpoint as the scale 0.5 model. Therefore neither path is currently labeled as the verified scale 0.5 checkpoint; the scale 0.5 diagnostic should not be attributed to either one until the training/export provenance is matched explicitly.
## Rhythm factor audit — CPU phase 1

The first CPU-only rhythm audit was run at `runs/rhythm_timing/20260812_factor_audit/` using `scripts/42_extract_rhythm_factors.py`. It consumed the strict two-stage MFA manifest and the existing `/tmp/wav2lip_valid_full/summary.json`; only train and valid were included, with heldout S0770 excluded before extraction and association. The manifest contains complete natural/TTS MFA timelines including `sil` intervals, so pause statistics are available without reconstructing them from waveform energy.

The audit produced 342 usable paired records: 292 train and 50 valid, with zero rejected pairs. All 50 valid S0765 pairs joined the existing natural/raw-TTS SyncNet scores. On valid, raw TTS had mean Sync-C gain +1.15262 and mean Sync-D improvement +0.59734 over natural, with 47/50 and 44/50 samples improving respectively.

Descriptive valid rhythm differences: mean log total-duration ratio was -0.08935, mean log non-silence duration ratio -0.09992, mean pause-fraction delta +0.00256, mean internal-pause-fraction delta -0.00630, mean absolute matched-phone log-duration ratio 0.21432, mean phone-duration allocation JSD 0.00950, and mean normalized matched-boundary displacement 0.04759. These are descriptive and do not imply that a factor is causal.

The initial unadjusted associations with paired Sync-C gain are only prioritization evidence. The largest absolute Pearson association was total-duration ratio (r=-0.365), while boundary displacement (r=-0.354) and pause-fraction delta (r=-0.256) were weaker; phone-duration allocation JSD had near-zero-to-weak association (r=-0.112, Spearman +0.101). Pause-position/duration metrics had only 16 complete pairs because the simple audit currently pairs pauses by order and does not yet solve pause-label alignment. Therefore the next step is not model modification: add pause matching and stronger non-target acoustic controls, then build exact-N no-op/shuffle and single-factor counterfactuals on a small valid subset.

The audit is correlational only. It must not be used to select heldout behavior, and full TTS duration transfer remains a negative control because the previous duration-alpha experiment worsened the HuBERT L6 gap as alpha increased.
## Rhythm counterfactual CPU phase

A valid-only counterfactual builder was added as `scripts/43_build_rhythm_counterfactuals.py`. It first applies the existing 24 kHz-to-16 kHz resampling contract to TTS audio, then creates six diagnostic conditions: `natural_noop`, `tts_noop`, `tts_global_natural_length`, and `duration_alpha_{0,0.5,1}`. These conditions are diagnostic controls only, not training targets; they preserve the natural exact-N contract for all duration conditions and never read heldout S0770.

The 10-pair smoke initially exposed non-finite output from librosa phase-vocoder interpolation on a very short phone segment. `scripts/common/pnp_feature_controls.py` now detects non-finite phase-vocoder output and falls back explicitly to the exact-length source segment with `fallback_reason=phase_vocoder_nonfinite`; the corresponding regression test passes. Re-running the smoke and expanding to all 50 valid S0765 pairs produced 300 audio files with zero non-finite outputs, zero sample-count violations, zero clipped samples, and zero phase-vocoder fallbacks. The complete artifact is `runs/rhythm_timing/20260812_counterfactual_valid50/summary.json` plus per-condition WAV/JSON sidecars.

The next causal step is fixed-protocol valid-only Wav2Lip/SyncNet on a 10-pair smoke subset, followed by the full 50 pairs only if the smoke has no processing artifact. No heldout result may influence that choice.
- [result] Residual scale 0.5 downstream evaluation: Sync-C −0.104 vs natural (20/50), worse than old enhanced; feature gains did not transfer to lip-sync. Negative result. #downstream
- [problem] RESOLVED: scale 0.5 checkpoint exported and 50-sample Wav2Lip/SyncNet evaluation completed. The scale 0.5 candidate is downstream-verified negative. #blocker
## Rhythm SyncNet experiment started

Because the scale-0.5 enhancer did not show convincing downstream gains, the fixed valid-only rhythm counterfactual experiment was started. `scripts/44_eval_rhythm_tfg_matrix.py` evaluates the six pre-generated rhythm conditions through the local Wav2Lip and SyncNet V2 installations, with per-stage logs, audio/video/model hashes, valid speaker checks, and heldout exclusion. The first run exposed only a path-interface error: the counterfactual manifest stored repository-relative audio paths while Wav2Lip executes from `third_party/Wav2Lip`; the evaluator now resolves those paths against the repository root before launching inference. The restarted 10-pair × 6-condition smoke is running under `runs/rhythm_timing/20260812_syncnet_smoke10/`.
## Rhythm SyncNet smoke result and calibration

The 10-valid-pair rhythm smoke completed 60/60 cells with zero failures. Every AV offset was -2. The processed conditions were: `duration_alpha_0`, `duration_alpha_0.5`, `duration_alpha_1`, `natural_noop`, `tts_global_natural_length`, and `tts_noop`.

A critical calibration issue was found before expansion. `natural_noop` reproduced the earlier natural baseline almost exactly, but the 16 kHz written `tts_noop` scored about 0.984 Sync-C below the earlier raw 24 kHz TTS baseline on the first 10 samples. Re-running the original raw TTS reproduced the old score exactly for sample 1 (Sync-C 8.134, Sync-D 6.369), ruling out evaluator randomness. PCM16 versus float32 was not the cause: loading the raw 24 kHz TTS through the Wav2Lip environment and writing a float32 16 kHz identity file produced a waveform that Wav2Lip loaded sample-for-sample identically, yet its final SyncNet score remained low. The remaining difference is the downstream MP4/ffmpeg/SyncNet audio path: raw 24 kHz and written 16 kHz inputs trigger different resampling/encoding behavior even though Wav2Lip itself sees the same 16 kHz waveform.

Therefore raw TTS remains a positive-control ceiling, but it cannot be the direct baseline for processed rhythm interventions. All processed conditions must be compared to the same-chain `tts_noop`; duration alpha conditions should also be compared pairwise (`alpha=1` versus `alpha=0`) to isolate relative phone allocation. Under this corrected interpretation, the 10-pair smoke showed `tts_global_natural_length` versus processed identity mean ΔSync-C -1.1531 and mean ΔSync-D +0.1045. Duration alpha did not show a clean dose response: mean Sync-C was 5.2885 at alpha 0, 5.2533 at alpha 0.5, and 5.1147 at alpha 1. This supports expansion for estimation, not a causal conclusion. The valid-50 matrix has now been started with the first 10 pairs cached.
## Valid-50 rhythm matrix result

The full matrix completed 300/300 valid S0765 cells with zero failures. Aggregate scores were: processed `tts_noop` C=6.07604/D=7.28454; `natural_noop` C=6.00704/D=7.28708; `tts_global_natural_length` C=4.91076/D=7.50704; duration alpha 0 C=5.37072/D=7.37222; alpha 0.5 C=5.22802/D=7.51916; alpha 1 C=5.22682/D=7.41400. Mean AV offsets stayed approximately -2 in every condition.

Using the preregistered same-chain baselines and 10,000-draw paired bootstrap: global TTS-to-natural length conversion versus processed TTS identity changed Sync-C by -1.16528 (95% CI [-1.33156,-1.00674]) and Sync-D by +0.22250 ([+0.10236,+0.33948]), improving Sync-C in only 1/50. The duration alpha intervention also carried a large general processing penalty: alpha 0 versus processed identity ΔC=-0.70532 ([-0.88576,-0.52490]). Relative phone-duration transfer was much smaller and inconclusive for Sync-C: alpha 0.5 versus 0 ΔC=-0.14270 ([-0.28980,+0.01014]); alpha 1 versus 0 ΔC=-0.14390 ([-0.30166,+0.01164]). Alpha 1 versus 0.5 was essentially zero in Sync-C (-0.00120, [-0.15130,+0.13986]). Only 12/50 samples had monotonically decreasing Sync-C from alpha 0→0.5→1 and 7/50 had monotonically increasing Sync-C, so there is no stable duration dose-response.

A more fundamental calibration result changes the interpretation of all earlier downstream evidence. When both sources were placed on the same 16 kHz writeback chain, processed TTS identity and natural identity became statistically indistinguishable: natural minus processed TTS identity ΔC=-0.06900 (95% CI [-0.25562,+0.11558]) and ΔD=+0.00254 ([-0.12232,+0.12872]). For sample 1, original 24 kHz raw TTS reproduced the old result exactly (8.134/6.369), while its 16 kHz identity waveform scored 6.221/7.914 even though Wav2Lip loaded the two waveforms sample-for-sample identically. The resulting Wav2Lip video-frame streams were also byte-identical by frame MD5. Replacing the raw-TTS MP4 video stream with the identity MP4 audio track produced exactly the identity score, proving that the discrepancy resides in the MP4/AAC/audio resampling path rather than generated lip frames.

Therefore the earlier claim that raw TTS clearly improves lip synchronization is confounded by source audio sample rate/container encoding (raw TTS 24 kHz AAC versus natural/enhanced 16 kHz AAC). It cannot be used as evidence that TTS rhythm or TTS acoustic content improved TFG. The rhythm matrix still supports a negative conclusion: naive global length conversion strongly damages the standardized pipeline, and relative TTS phone-duration transfer provides no stable positive dose response. No timing-conditioned model or duration predictor should be selected from these results. Before any new downstream comparison, all source conditions must be standardized to the same input sample rate and mux/audio encoding contract, and the processed-identity control must be retained.
## Full video-stream sensitivity audit

A 50-sample frame-MD5 audit adds an important limitation to the sample-1 mux control. Comparing the old raw-source Wav2Lip run to the PCM16 16 kHz writeback run, natural video frames were exactly identical for 15/50 samples and TTS video frames for 0/50. Thus tiny waveform preprocessing/quantization changes can also perturb Wav2Lip output frames, even when mel shape and duration are unchanged. The stronger sample-1 float32 identity control remains valid but must not be generalized: for that sample, Wav2Lip loaded raw and identity waveforms sample-for-sample identically, produced byte-identical video frames, and the mux experiment isolated the score difference to the audio track. Across the full PCM16 matrix, both Wav2Lip-frame sensitivity and AAC/sample-rate sensitivity may contribute. This strengthens the protocol requirement: processed identity controls, sample-rate/mux standardization, and video-stream identity checks are all required before attributing SyncNet differences to rhythm or enhancement.
## Pause-allocation experiment started

The completed scale-0.5 result was recovered from the project note rather than rerun: checkpoint `stage2_scale05_full_20260812/last.pt`, valid-50 Sync-C 5.898, ΔC -0.104 versus natural, a downstream negative.

A stricter pause-only intervention has now been implemented. `reallocate_pause_samples()` preserves every TTS speech sample, every speech-span duration, every existing gap sample, total gap samples, and exact output length; it only changes how the existing gap-sample pool is allocated among phone-boundary slots. It passes 14/14 related tests. `scripts/46_build_pause_counterfactuals.py` generates `pause_alpha_0` (strict in-memory identity), `pause_alpha_0.5`, `pause_alpha_1` (natural-normalized gap allocation at fixed TTS total silence), `pause_internal_to_edges`, and `pause_reverse`. The 10-valid-pair CPU smoke produced 50/50 finite exact-length, zero-clipping controls with full sample-multiset conservation and no rejected pairs. A 10×5 Wav2Lip/SyncNet smoke has started, using `pause_alpha_0` as the same-chain baseline.
## Pause smoke invalidated by Wav2Lip cwd collision

The first pause Wav2Lip/SyncNet smoke must not be used. Another local evaluation ran concurrently from the same `third_party/Wav2Lip` working directory. Wav2Lip hard-codes `temp/result.avi`, so concurrent processes overwrote one another's intermediate video; pause sample 7 identity collapsed to C=1.003/D=12.143 and exposed the collision. The pause run was stopped and all its downstream outputs are considered contaminated. `scripts/44_eval_rhythm_tfg_matrix.py` now creates a per-cell working directory with an isolated `temp/` and symlinked Wav2Lip code/assets. An isolated sample-1 identity check restored the expected C=6.252/D=7.798. The CPU pause counterfactual WAVs remain valid; downstream evaluation will be deleted and rerun from scratch after GPU contention clears.
## Pause-allocation smoke result and valid-50 expansion

After fixing Wav2Lip per-cell cwd isolation, the clean 10×5 pause smoke completed 50/50 with zero failures. Identity calibration passed: all 10 `pause_alpha_0` WAV hashes equal the earlier processed `tts_noop` WAV hashes; repeated Sync-C differed by only -0.0034 on average and AV offsets were identical. On the 10-pair smoke, moving pause allocation toward natural did not show a positive aggregate trend: alpha 0.5 versus identity ΔC=-0.1022 (95% bootstrap CI [-0.2978,+0.0916]); alpha 1 ΔC=-0.0636 ([-0.2815,+0.1949]). Response directions were heterogeneous. Moving all internal pause samples to utterance edges reduced Sync-C in the 3 active samples by -0.2853, but n=3 is too small.

The experiment has been expanded to all 50 valid pairs. CPU construction produced 250/250 pause controls with zero rejections, finite exact-N audio, zero clipping, and full source-sample multiset conservation. Alpha 0.5/1 changed gap allocation in all 50 samples; edge/reverse controls were active in 19 samples with internal pauses. The isolated valid-50 Wav2Lip/SyncNet matrix has started under `runs/rhythm_timing/20260812_pause_syncnet_valid50/`.
## Final valid-50 pause-allocation result

The isolated pause matrix completed 250/250 valid S0765 cells with zero failures. Calibration passed: all 50 `pause_alpha_0` WAV hashes matched the earlier processed `tts_noop` identities; repeated identity scores differed by only +0.00346 Sync-C and +0.00176 Sync-D on average, with all 50 AV offsets identical.

Condition means were: identity C=6.07950/D=7.28630; alpha 0.5 C=6.05776/D=7.37512; alpha 1 C=6.01568/D=7.41412; internal-to-edges C=6.05880/D=7.32138; reverse C=6.07792/D=7.29400. Paired valid-50 bootstrap results show no Sync-C improvement from moving pause allocation toward natural: alpha 0.5 ΔC=-0.02174 (95% CI [-0.13092,+0.08572], 21/50 better); alpha 1 ΔC=-0.06382 ([-0.19672,+0.06782], 23/50 better). Sync-D worsened consistently: alpha 0.5 ΔD=+0.08882 ([+0.00980,+0.17250]); alpha 1 ΔD=+0.12782 ([+0.03432,+0.22192]). Only 4/50 samples showed monotonically improving Sync-D as alpha increased, while 13/50 showed monotonic worsening. Sync-C monotonic increase/decrease counts were 8/50 and 13/50.

Among the 19 samples with actual internal pauses, moving all internal pause samples to the utterance edges gave ΔC=-0.04868 (CI [-0.21268,+0.11474]) and ΔD=+0.09021 (CI crosses zero); reversing pause allocation gave ΔC=+0.00211 and ΔD=+0.02353, both CIs crossing zero. Intervention magnitude had no stable association with alpha benefit. Therefore pause placement/duration reallocation is not a positive TFG mechanism under the standardized valid-only pipeline. Combined with the global-length and phone-duration results, no tested rhythm transfer warrants adding a duration predictor, length regulator, or timing side-channel to the enhancer. Local phone/viseme boundary acoustics remain conceptually separate and were not isolated by this pause experiment.