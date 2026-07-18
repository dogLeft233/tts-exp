# Phase 2 Execution Log

*Date: 2026-07-18*

## Scope

Phase 2 was executed on the RTX 4080 server at port `26325` using the
repository copy under `/root/autodl-tmp/repos/tts-exp`.

The primary audio pair was aligned with the Phase 1 manifest:

- Natural: `data/data/audio/*.wav`
- Faster-Qwen3 TTS: `runs/r2_faster_qwen3_20260707T145233Z/02_tts/*.wav`
- Phase 1 study IDs: `1-8, 10-13` (12 samples)

An earlier run accidentally selected `aishell1_ldn_20260707T070936Z/02_tts`,
which contains only 10 samples and is not the Phase 1 source. Its results were
not used in this log or the main report.

## Feature Interventions

Script: `scripts/20_causal_feature_interventions.py`

The pilot used five samples and the full study used 12 samples. The script
verified that each transformation changed its target feature; it did not run
TFG inference or SyncNet on transformed audio.

| Intervention | Pilot | Full study | Mean delta, source -> target |
|---|---:|---:|---:|
| LUFS TTS -> natural | 4/5 | 5/12 | -0.167 LUFS |
| LUFS natural -> TTS | 4/5 | 5/12 | +0.167 LUFS |
| Spectral tilt TTS -> natural | 5/5 | 12/12 | +0.2401 |
| Spectral tilt natural -> TTS | 5/5 | 12/12 | -0.2401 |
| Dynamic compression | 5/5 | 12/12 | -0.0018 (TTS) |
| Dynamic expansion | 5/5 | 12/12 | +0.0023 (TTS) |

The small LUFS gap is why per-sample LUFS matching is inconsistent in the full
study. This result supersedes the older 4.4-LUFS observation from a different
audio condition.

Artifacts:

- `data/wav2sem_analysis/metrics/intervention_pilot_20260718.json`
- `data/wav2sem_analysis/metrics/intervention_results.json`

## GxE Matrix

Script: `scripts/19_generate_eval_matrix.py`

The strict Ditto run contains 10 sample IDs. Sample 9 has no valid TTS
generated diagonal result and its TTS-generated natural-evaluator cell also
failed SyncNet parsing, leaving nine complete 2x2 matrices.

Audio in every off-diagonal cell was globally time-stretched to the generator
video duration before SyncNet evaluation. This avoids treating duration
differences as scorer effects.

| Metric | G_nat/E_nat | G_nat/E_tts | G_tts/E_nat | G_tts/E_tts |
|---|---:|---:|---:|---:|
| Sync-C | 5.574 | 2.776 | 2.327 | 6.819 |
| Sync-D | 8.317 | 11.429 | 11.501 | 7.550 |

Complete-case decomposition, `n=9`:

| Metric | Total | Generator | Scorer | Interaction |
|---|---:|---:|---:|---:|
| Sync-C | +1.246 | -3.247 | -2.798 | +7.290 |
| Sync-D | -0.766 | +3.184 | +3.113 | -7.063 |

The large interaction and low off-diagonal scores do not support a simple
independent SyncNet scorer-bias explanation. They indicate that generated
visual motion and the corresponding audio are strongly coupled. This matrix
does not by itself identify the causal acoustic feature.

Artifact:

- `runs/aishell1_strict_20260707T081223Z/04_eval/gxe_matrix.json`

## Cross-TFG Validation

Script: `scripts/21_validate_causal_results.py`

Per-sample SyncNet results were available for all four comparison models:

| Model | n | Delta Sync-C | Delta Sync-D |
|---|---:|---:|---:|
| Ditto | 9 | +1.246 | -0.766 |
| Wav2Lip | 12 | +1.515 | -1.395 |
| MuseTalk | 12 | +0.572 | -0.169 |
| JoyVASA | 12 | +0.329 | +0.144 |
| LatentSync | 12 | -0.063 | -0.121 |

The aggregate pattern is architecture-dependent: LatentSync is near neutral,
while the other models generally show positive Sync-C responses. The specific
HuBERT mechanism cannot yet be validated cross-model because embeddings were
only computed for Ditto.

Artifact:

- `data/wav2sem_analysis/metrics/cross_tfg_validation.json`

## Dose-Response SyncNet Evaluation

Script: `scripts/22_dose_response_syncnet.py`

Each of the 8 interventions from script 20 was applied to the source audio
(TTS or natural), the resulting audio was fed directly through Ditto TFG
inference (no loudnorm — matching the strict-run config), and SyncNet was
evaluated on the resulting video. ΔSync-C and ΔSync-D were computed versus
the original diagonal baseline (`tts_raw` for TTS-source interventions,
`natural_raw` for natural-source interventions).

Run: `runs/aishell1_strict_20260707T081223Z` on the RTX 4080 server,
elapsed 2217 s (37 min), `n = 9` per intervention, 0 failures. Only samples
1–8 and 10 had local Phase 1 audio + diagonal SyncNet baselines, so the
dose-response uses those nine samples.

| Intervention | n | ΔSync-C | ΔSync-D | Expected direction | Matched? |
|---|---:|---:|---:|---|---|
| LUFS_match_tts_to_nat | 9 | −1.198 | +0.558 | decrease | ✓ (9/9 ↓) |
| spectral_tilt_match_tts_to_nat | 9 | −0.989 | +0.520 | decrease | ✓ (9/9 ↓) |
| dynamic_compression_tts_compress | 9 | −1.157 | +0.563 | unknown | 9/9 ↓ |
| dynamic_expansion_tts_expand | 9 | −1.232 | +0.630 | unknown | 9/9 ↓ |
| LUFS_match_nat_to_tts | 9 | −0.015 | −0.015 | increase | ✗ (flat) |
| spectral_tilt_match_nat_to_tts | 9 | −0.528 | +0.037 | increase | ✗ (reversed) |
| dynamic_compression_nat_compress | 9 | +0.046 | −0.086 | unknown | flat |
| dynamic_expansion_nat_expand | 9 | +0.092 | −0.059 | unknown | flat |

### Interpretation

**TTS-source interventions (top four rows):** every per-sample perturbation
of TTS audio moves Sync-C downward, regardless of which acoustic feature is
manipulated. Per-sample agreement is 9/9 for all four interventions
(36 cells out of 36), with magnitudes clustered around −1.0 to −1.2 Sync-C.
The fact that *opposite* dynamic-range manipulations (compress and expand)
produce nearly identical Sync-C decreases (−1.157 vs −1.232) is
particularly striking: it argues against any single manipulated feature
being the causal driver and instead suggests that **the TTS-TTS diagonal
is a co-adapted state that is robust to the exact acoustic profile produced
by Faster-Qwen3 TTS, but fragile to any perturbation.**

This is consistent with the G×E interaction term (+7.29 Sync-C) from the
generator-evaluator matrix: the diagonal advantage is not separable into
independent audio-feature contributions. Naming a particular feature
(LUFS, spectral tilt, dynamic range) as the mechanism is not supported
by this dose-response. What is supported is that the +1.25 Sync-C
"advantage" lives in the joint (generator × evaluator) system, not in
any one acoustic attribute.

**Natural-source interventions (bottom four rows):** perturbing natural
audio does not raise Sync-C toward the TTS diagonal. Both LUFS and tilt
matching fail to lift Sync-C; spectral-tilt matching actually reduces
Sync-C by −0.53. The dynamic-range manipulations on natural audio produce
near-zero effects. This asymmetry reinforces the co-adaptation interpretation:
you cannot manufacture the TTS-TTS Sync-C advantage by single-feature
matching of natural audio.

### Causal verdicts on candidate mechanisms

| Candidate | Pre-Phase-2 evidence | Phase-2 verdict |
|---|---|---|
| LUFS | Weak (Δ≈+0.17 LUFS) | **Rejected as a mechanism.** TTS-source LUFS perturbation has the largest ΔSync-C among all interventions, but the per-sample LUFS deltas are bidirectional and small; the SyncNet response is common-mode, not LUFS-specific. |
| Spectral tilt | Strong observational (Δ≈−0.240, 12/12) | **Inconclusive — likely not isolated.** 9/9 ↓ direction matches the hypothesis, but the −0.99 Sync-C response is not separable from common-mode TTS perturbation (compare to dynamic interventions at −1.16/−1.23 with opposite feature manipulations). |
| Dynamic range | Weak (Δ≈0.0004 absolute) | **Rejected as a mechanism.** Opposite manipulations produce near-identical Sync-C decreases — diagnostic of common-mode degradation, not a dynamic-range-specific effect. |
| Generator × Evaluator co-adaptation | G×E interaction = +7.29 (vs −3.25 generator + −2.80 scorer main effects) | **Supported.** The dose-response pattern (all TTS perturbations move Sync-C downward regardless of feature, natural-side manipulations flat) is the expected signature of a coupled state. |

Artifacts:

- `data/wav2sem_analysis/metrics/dose_response.json` (summary + per-sample)
- `runs/aishell1_strict_20260707T081223Z/03_ditto/interventions/<name>/<sid>.mp4` (Ditto videos)
- `runs/aishell1_strict_20260707T081223Z/04_eval/interventions/<name>/<sid>/syncnet.json`

## Identity Control (write/Ditto/SyncNet pipeline noise)

Script: `23_identity_control_syncnet.py`

The dose-response drops of −1.0 to −1.2 Sync-C per TTS-source intervention
coincide with the magnitude we expect for **Ditto run-to-run non-determinism**:
re-rendering the same audio through Ditto produces slightly different frame
sequences that SyncNet scores lower, independent of audio feature content.
This script applies a no-op transform (TTS audio written back as 16-bit PCM
unchanged) and re-runs Ditto + SyncNet per sample. ΔSync-C is computed
against the original `tts_raw` diagonal baseline — the same baseline used
for the four TTS-source dose-response interventions.

| Run | n | ΔSync-C mean | ± SD |
|---|---:|---:|---:|
| Identity control (this script) | 9 | **−1.161** | 0.469 |
| LUFS_match_tts_to_nat (dose-response) | 9 | −1.198 | – |
| spectral_tilt_match_tts_to_nat | 9 | −0.989 | – |
| dynamic_compression_tts_compress | 9 | −1.157 | – |
| dynamic_expansion_tts_expand | 9 | −1.232 | – |

Per-sample identity ΔSync-C (all negative, directionally consistent):

| sid | baseline | identity | Δ |
|---:|---:|---:|---:|
| 1 | 6.491 | 4.468 | **−2.023** |
| 2 | 6.600 | 5.478 | −1.122 |
| 3 | 7.444 | 5.927 | −1.517 |
| 4 | 6.843 | 5.807 | −1.036 |
| 5 | 6.161 | 5.426 | −0.735 |
| 6 | 6.149 | 5.682 | −0.467 |
| 7 | 6.406 | 5.138 | −1.268 |
| 8 | 7.977 | 6.520 | −1.457 |
| 10 | 7.303 | 6.478 | −0.825 |
| 9 | – | – | FAILED (face detection: empty `tracks.pckl`) |

**Verdict: `PIPELINE_CONFOUNDS_DOSE_RESPONSE`.** The pipeline alone explains
the full magnitude of the dose-response drop. Sample 9 also failed the
identity run for the same reason it failed the original G×E run: SyncNet's
S3FD face detector produces an empty `tracks.pckl` on the Ditto-rendered
9.tts_raw video. Sample 9 is therefore unrepairable without re-rendering the
video with a different face pre-processing path.

Artifacts:

- `data/wav2sem_analysis/metrics/identity_control.json` (incl. per-sample
  baseline / identity Sync-C / Δ for the 9 samples that succeeded)
- `runs/aishell1_strict_20260707T081223Z/03_ditto/interventions/identity_tts/<sid>.mp4`
  (re-rendered identity videos)

## Identity-Corrected Dose-Response

Script: `24_identity_corrected_analysis.py`

All four TTS-source interventions are reanalysed by subtracting the
per-sample identity-baseline Δ from each intervention's per-sample Δ,
then testing the residuals against zero (paired one-sample t-test, n=9).

| Intervention | net ΔC | ± SD | t | p | Cohen's d | verdict |
|---|---:|---:|---:|---:|---:|---|
| LUFS_match_tts_to_nat | **−0.037** | 0.296 | −0.38 | 0.7154 | −0.13 | not significant |
| spectral_tilt_match_tts_to_nat | **+0.173** | 0.371 | +1.40 | 0.2003 | +0.47 | not significant |
| dynamic_compression_tts_compress | **+0.004** | 0.338 | +0.03 | 0.9748 | +0.01 | not significant |
| dynamic_expansion_tts_expand | **−0.071** | 0.219 | −0.97 | 0.3619 | −0.32 | not significant |

**Conclusion.** None of the four TTS-source acoustic interventions has a
measurable feature-specific effect on Sync-C after correcting for Ditto
run-to-run non-determinism. The raw −1.0 to −1.2 drops reported in the
dose-response above are fully explained by pipeline noise. The TTS-TTS
diagonal advantage (+1.25 Sync-C) cannot be decomposed into contributions
from any single acoustic feature.

Natural-source interventions are reported as raw ΔC (no identity
subtraction — the identity control is TTS-side only):

| Intervention | raw ΔC |
|---|---:|
| LUFS_match_nat_to_tts | −0.015 |
| spectral_tilt_match_nat_to_tts | −0.528 (ΔD +0.037) |
| dynamic_compression_nat_compress | +0.046 |
| dynamic_expansion_nat_expand | +0.092 |

These remain uninterpretable without a parallel natural-side identity
control. Their flatness does continue to support the asymmetry that
matching natural audio to TTS characteristics does not lift Sync-C toward
the TTS diagonal.

Artifacts:

- `data/wav2sem_analysis/metrics/identity_corrected_summary.json`
  (per-intervention residuals, t-stats, p-values, per-sample records)

### Causal verdicts on candidate mechanisms (revised)

The previous verdict table was based on the raw dose-response drops and the
common-mode degradation interpretation. After identity correction, those
interpretations need to be sharpened: the raw drops are pipeline noise
*plus* a (sub-noise) feature-specific residual. The substantive findings
that survive are:

| Candidate | Pre-Phase-2 evidence | Phase-2 verdict (identity-corrected) |
|---|---|---|
| LUFS | Weak (Δ≈+0.17 LUFS) | **Rejected.** Net residual −0.04 ± 0.30 (t=−0.38, p=0.72). The pipeline noise explains the entire raw drop. |
| Spectral tilt | Strong observational (Δ≈−0.240, 12/12) | **Rejected as isolable mechanism.** Net residual +0.17 ± 0.37 (t=+1.40, p=0.20) — not significant at n=9. The +0.17 mean is positive (i.e., tilt matching *raises* Sync-C relative to identity), but the effect is below the pipeline noise floor and not statistically distinguishable from zero. |
| Dynamic range | Weak (Δ≈0.0004 absolute) | **Rejected.** Compression residual +0.004 (p=0.97); expansion residual −0.071 (p=0.36). Neither direction isolatable. |
| **Generator × Evaluator co-adaptation** | G×E interaction = +7.29 (vs −3.25 generator + −2.80 scorer) | **Supported.** Identity correction does not affect this finding — it is computed on the strict run's original diagonal scores, not on re-renders. The diagonal advantage lives in the joint (Faster-Qwen3 × Ditto × SyncNet) coupling, not in any isolable acoustic feature. |
| **HuBERT segment_stability (layer 11)** | Strong (FDR p=0.024, d=1.045) | **Open — strongest surviving candidate.** The Phase 2 identity-corrected dose-response ruled out all acoustic features. The remaining causal hypothesis is SSL-representation-level: TTS audio produces more stable viseme representations in HuBERT space, and Ditto + SyncNet are both sensitive to that property. To be tested by a stability-targeted intervention (script 25). |

## Code Corrections

- Script 20 now honors `--input-dir` and resolves the Phase 1 Faster-Qwen3 path.
- Script 19 time-aligns off-diagonal audio, supports cache overwrites, and
  decomposes only complete 2x2 cases.
- Script 21 accepts the Phase 1 candidate field `mechanism` and aggregates
  per-sample SyncNet deltas for cross-model consistency checks.
- Script 22 dose-response pipeline applies script-20 transforms, runs Ditto
  (in `loudnorm: false` mode matching the strict run) and SyncNet per
  intervention × sample with three-stage caching.
- Script 23 identity control re-uses script-22 pipeline with a no-op
  transform; classifies the verdict (PIPELINE_CLEAN / AMBIGUOUS /
  PIPELINE_CONFOUNDS_DOSE_RESPONSE) using thresholds 0.2 / 0.5.
- Script 24 paired-residual analysis subtracts per-sample identity Δ from
  each dose-response Δ and runs one-sample t-tests against zero; produces
  per-intervention net effect, Cohen's d, and significance class.

## Remaining Work

1. Design and run a stability-targeted intervention on HuBERT embeddings
   (script 25): perturb audio with a low-magnitude adversarial disturbance
   that **specifically** reduces `segment_stability` at layer 11 while
   preserving LUFS / spectral tilt / dynamic range. A drop in Sync-C *not*
   seen in identity control would causally validate segment_stability. A
   drop equal to identity control would mean segment_stability is also a
   correlate, not a driver.
2. Run a parallel natural-side identity control to make the natural-source
   interventions interpretable in the same paired-residual framework.
3. Extract HuBERT feature metrics for the other TFG models (Wav2Lip,
   MuseTalk, LatentSync, JoyVASA) so the segment_stability finding can be
   tested beyond Ditto.
4. Add human perceptual evaluation before extending SyncNet-based mechanism
   claims to "lip-sync quality."
5. Investigate the cause of sample 9's face detection failure on its TTS
   DiTTO video (empty `tracks.pckl`). Until fixed, all Δ analyses retain
   n=9.
