# TTS Enhancement of TFG Lip Sync: Mechanism Analysis

*Last updated: 2026-07-18 (script 25 stability intervention results added)*

## Summary

This report characterises why text-to-speech (TTS) synthesis produces higher SyncNet
lip-sync scores than natural speech when processing Talking Face Generation (TFG)
videos. We set out to identify the acoustic mechanisms driving this TTS advantage and
determine whether the effect reflects genuine visual-motor quality improvement or a
SyncNet scoring artifact.

**Findings**: TTS audio produces a robust diagonal SyncNet advantage for Ditto
(ΔC = +1.246, ΔD = −0.766; n=9 paired samples). The Phase 2 dose-response
experiment — eight audio interventions applied to TTS and natural sources and
fed back through Ditto + SyncNet — initially suggested that all TTS-source
perturbations drop Sync-C by ≈ −1.0 to −1.2, hinting at common-mode
degradation. **An identity control experiment then showed that this entire
magnitude is produced by Ditto run-to-run non-determinism alone**: re-running
the same TTS audio through the pipeline *without any modification* produces
ΔSync-C = −1.161 ± 0.469 (n=9). After per-sample subtraction of the identity
baseline, none of the four TTS-source acoustic interventions (LUFS, spectral
tilt, dynamic compression, dynamic expansion) shows a feature-specific
residual significantly different from zero (paired t-test: t=−0.38/+1.40/+0.03/−0.97,
all p > 0.20, n=9). The +1.25 Sync-C diagonal advantage cannot be decomposed
into any single isolable acoustic contribution.

Two causal candidates survive Phase 2:

- **Generator × Evaluator co-adaptation** (G×E interaction = +7.29, vs
  −3.25 generator + −2.80 scorer main effects): the diagonal advantage
  lives in the joint coupling of Faster-Qwen3 audio + Ditto generator +
  SyncNet evaluator, not in any single acoustic attribute. This finding
  does not depend on re-rendered audio and is not affected by the identity
  control.
- **HuBERT `segment_stability` at layer 11**: Phase 1 observational
  (FDR p=0.024, d=1.045); Phase 2 stability intervention (script 25,
  n=12, PGD under ε=0.005 budget) produced an **inconclusive causal
  verdict**. In the TTS direction the intervention is strongly
  feature-specific (paired residual −1.18 ± 0.35 vs same-ε random-noise
  control, t=−11.79, d=−3.40): destabilizing HuBERT-L11 stability on
  TTS drops Sync-C by ~1.18 beyond pure noise. But the natural-side
  counterpart is directionally wrong — stabilizing natural toward
  TTS *drops* Sync-C by −4.21 ± 0.83 instead of rising toward the
  TTS diagonal. Stability is therefore a **statistically significant
  causal interceptor in the TTS direction but the diagonal advantage
  remains primarily a non-decomposable G×E co-adapted state**.

LatentSync remains a near-zero cross-model control.

---

## Evidence Layers

### 1. Observational

TTS audio yields higher Sync-C (confidence) and lower Sync-D (distance) compared
to natural speech across all evaluated conditions.

| Condition | Sync-C (mean) | Sync-D (mean) |
|-----------|-------------|-------------|
| natural_raw | 5.66 | 7.96 |
| tts_raw | 6.91 | 7.19 |
| Δ (TTS − natural) | **+1.25** | **−0.77** |

Data: 9 paired AISHELL-1 samples from the strict Ditto run, SyncNet v2
evaluation. The broader Phase 1 audio/embedding set contains 12 samples;
sample 9 has no valid paired Ditto TTS score in the strict run.

### 2. Acoustic

Acoustic feature analysis reveals systematic differences between TTS and natural
speech:

- **LUFS**: Faster-Qwen3 TTS averages −39.15 LUFS versus −39.31 LUFS for
  natural speech (Δ ≈ +0.17 LUFS), which is not a meaningful loudness gap
- **Spectral tilt**: TTS averages −1.22 versus −0.98 for natural speech
- **Dynamic range**: TTS energy-envelope standard deviation is 0.00619 versus
  0.00578 for natural speech

These features remain intervention targets, but the primary data does not
justify treating LUFS as the established driver.

### 3. Encoding (Wav2Sem-style) — COMPLETED

HuBERT/XLS-R embedding analysis examines whether TTS affects speech encoder
representations differently than natural speech. Phase 1 pipeline completed on
RTX 4080 (26325 server):

**Pipeline executed**:
- `14_prepare_mandarin_alignment.py`: 72 manifest entries (12 samples × 3 conditions × 2 variants), uniform-segmented pinyin tokens as MFA fallback
- `15_extract_ssl_embeddings.py`: 144 NPY+JSON files (HuBERT + XLS-R), GPU-accelerated (~7s total)
- `16_feature_separability.py`: 18 per-condition results + 24 comparisons (HuBERT, viseme level only; XLS-R skipped due to time constraints)
- `17_link_features_to_tfg.py`: 120 univariate tests across 5 TFG models
- `18_render_wav2sem_report.py`: 203-line report + 5 figures

**Key finding**: TTS significantly improves **segment stability** at HuBERT
layer 11 (FDR p=0.024, Cohen's d=-1.045). TTS produces more temporally
consistent viseme embeddings within each phoneme segment. This effect is
most pronounced at deep layers (11 vs 0/6), suggesting it affects
higher-level phonetic representations, not just raw acoustics.

**SyncNet correlation**: Per-sample feature deltas could not be correlated
with per-sample SyncNet deltas (all Spearman rho = NaN). The separability
metrics are population-level statistics computed across all tokens for a
given condition. Per-sample variance within identical viseme label sets is
too small to produce meaningful rank correlations with SyncNet. Future work
should use per-sample embedding distances (e.g., L2 between natural/TTS
frame embeddings) rather than population-level separability metrics.

**Full report**: `data/wav2sem_analysis/report.md` (203 lines + 5 figures)

### 4. Causal

Controlled audio interventions (`scripts/20_causal_feature_interventions.py`)
were completed as feature-transformation verification on the 12-sample Phase 1
audio set:

| Intervention family | Target | Pilot verification | Full-study verification |
|---|---|---|---|
| LUFS matching (both directions) | Loudness | 4/5 | 5/12 per direction |
| Spectral-tilt matching (both directions) | Spectral balance | 5/5 | 12/12 per direction |
| Dynamic compression/expansion | Energy envelope | 5/5 | 12/12 per direction |

The transformations successfully modify their target features. LUFS matching
is inconsistent at the per-sample level because the primary TTS/natural LUFS
gap is small. These runs do **not** yet re-evaluate TFG videos with the
modified audio, so downstream causal effects on SyncNet remain pending.

### 5. G×E Separation

The Generator×Evaluator matrix (`scripts/19_generate_eval_matrix.py`) decomposes
the TTS advantage into:

- **Generator effect**: TTS audio genuinely improves TFG video quality
  (G_tts E_natural − G_natural E_natural)
- **Scorer effect**: SyncNet itself prefers TTS audio characteristics
  (G_natural E_tts − G_natural E_natural)
- **Interaction**: The synergy between TTS generation and TTS audio for scoring

**Status**: Completed on the 26325 RTX 4080 server with duration-aligned
off-diagonal audio. Nine complete 2×2 matrices are available; sample 9 lacks
both TTS-generated cells. Results are in
`runs/aishell1_strict_20260707T081223Z/04_eval/gxe_matrix.json`.

| Metric | G_nat/E_nat | G_nat/E_tts | G_tts/E_nat | G_tts/E_tts |
|---|---:|---:|---:|---:|
| Sync-C | 5.574 | 2.776 | 2.327 | 6.819 |
| Sync-D | 8.317 | 11.429 | 11.501 | 7.550 |

Complete-case decomposition (n=9):

- Sync-C: total +1.246, generator −3.247, scorer −2.798, interaction +7.290
- Sync-D: total −0.766, generator +3.184, scorer +3.113, interaction −7.063

The large interaction and low off-diagonal scores indicate that TTS and
natural audio are not interchangeable evaluator tracks even after global
duration alignment. This is evidence against a simple independent scorer-bias
explanation, not proof that SyncNet has no acoustic sensitivity.

### 6. Cross-TFG

Cross-model validation (`scripts/21_validate_causal_results.py`) tests whether
mechanisms identified in Ditto generalise to other TFG architectures:

| Model | n | Δ Sync-C | Δ Sync-D | Source |
|---|---:|---:|---:|---|
| Ditto | 9 | +1.246 | −0.766 | `cross_tfg_validation.json` |
| Wav2Lip | 12 | +1.515 | −1.395 | `tfg_wav2lip/04_eval/eval_meta.json` |
| MuseTalk 1.5 | 12 | +0.572 | −0.169 | `tfg_musetalk/04_eval/eval_meta.json` |
| JoyVASA | 12 | +0.329 | +0.144 | `tfg_joyvasa/04_eval/eval_meta.json` |
| LatentSync | 12 | −0.063 | −0.121 | `tfg_latentsync/04_eval/eval_meta.json` |

**Key observation**: three comparison models show positive Sync-C deltas, while
LatentSync remains near zero. This supports an architecture-dependent TTS
response, but does not identify which acoustic or representation mechanism
causes it.

**Status**: Per-sample SyncNet data is now available for all four comparison
models. Mechanism-specific cross-TFG validation remains incomplete because
the SSL feature metrics were only computed for Ditto.

---

## Key Findings

### Loudness is not established as the primary driver

In the Faster-Qwen3 Phase 1 dataset, TTS and natural audio differ by only about
0.17 LUFS. Bidirectional LUFS matching verified the transformation on only 5/12
study pairs. The earlier 4.4-LUFS observation came from a different audio
condition and should not be used to explain this experiment's TTS advantage.

### Representation stability

HuBERT layer 11 segment stability is the only Phase 1 metric surviving FDR
correction (q=0.024, d=−1.045). This is currently the strongest mechanism
candidate, although its downstream SyncNet mediation has not been established.

### Spectral and dynamic characteristics

Spectral-tilt and dynamic-range transformations are reproducible on all 12
audio pairs. Their effect on TFG generation and SyncNet scores remains pending.

### LatentSync as negative control

LatentSync shows near-zero aggregate ΔC (−0.063) and ΔD (−0.121). This
indicates that the TTS advantage is not a uniform cross-model effect — it
depends on specific model architecture and audio processing characteristics.
LatentSync's diffusion-based approach may apply internal normalisation
that attenuates loudness effects.

---

## Candidate Mechanisms Table

| Mechanism | Evidence Strength | Generator or Scorer | Cross-TFG | Actionable |
|---|---|---|---|---|---|
| LUFS/loudness | **Rejected (Phase 2 identity-corrected)** — raw drop −1.20 explained by pipeline noise; per-sample residual vs identity = −0.037 ± 0.296 (t=−0.38, p=0.72, n=9) | N/A | Architecture-dependent | No |
| Spectral tilt | **Rejected as isolable mechanism (Phase 2 identity-corrected)** — raw drop −0.99 explained by pipeline noise; residual vs identity = +0.173 ± 0.371 (t=+1.40, p=0.20, n=9). The +0.17 mean is intriguing but below the pipeline noise floor and not statistically distinguishable from zero at this sample size. | N/A | Architecture-dependent | Conditional — bounded tilt shifts at larger n or smaller noise floor |
| Dynamic range (compress) | **Rejected (Phase 2 identity-corrected)** — residual = +0.004 ± 0.338 (t=+0.03, p=0.97). | N/A | Untested | No |
| Dynamic range (expand) | **Rejected (Phase 2 identity-corrected)** — residual = −0.071 ± 0.219 (t=−0.97, p=0.36). | N/A | Untested | No |
| **Generator × Evaluator co-adaptation** | **Supported (Phase 2 G×E)** — G×E interaction = +7.29 (vs −3.25 generator + −2.80 scorer main effects). This finding is computed on the strict run's original diagonal scores and is not affected by Ditto non-determinism. The +1.25 Sync-C diagonal advantage lives in the joint Faster-Qwen3 × Ditto × SyncNet coupling. | **Both** (joint state) | Architecture-dependent (LatentSync neutral cross-model) | Yes — preserve generator/scorer coupling |
| **HuBERT `segment_stability` (layer 11)** | **Phase 1: Strong observational (FDR p=0.024, d=1.045).** **Phase 2 stability intervention (script 25, n=12): INCONCLUSIVE** — strongly feature-specific in the TTS direction (paired residual vs same-ε random-noise control = −1.18 ± 0.35, t=−11.79, d=−3.40) but bidirectional test fails (stabilizing natural drops Sync-C by −4.21 instead of raising). Verdict: stability is a **significant causal interceptor for the TTS direction** but is not alone sufficient to recreate the diagonal advantage; G×E co-adaptation remains dominant. | Generator (TTS direction only) | Single-model (HuBERT computed only for Ditto) | Partial — temporal smoothing of TTS audio may help; cannot boost natural alone |
| Boundary sharpness | **Moderate (Phase 1)** — n.s. at FDR (mean d=0.58). | Generator | Single-model | Conditional |
| Silhouette score | **Weak (Phase 1)** — consistent direction, n.s. | Generator | Single-model | No |
| Intra-class compactness | **Weak (Phase 1)** — no strong signal | Generator | Single-model | No |

---

## Methods

### Experimental design

- **n=12** paired natural/TTS utterances from AISHELL-1 Mandarin Chinese
- **TTS**: Faster-Qwen3 (primary), F5-TTS (validation)
- **TFG models**: Ditto (primary), Wav2Lip, MuseTalk, LatentSync, JoyVASA (cross-TFG)
- **Audio analysis**: LUFS (pyloudnorm EBU R128), spectral tilt, energy envelope
- **Embedding analysis**: HuBERT-base, XLS-R-300M, layers 0-12
- **Causal inference**: Controlled audio transformations plus duration-aligned G×E SyncNet evaluation
- **Cross-TFG**: Per-sample SyncNet deltas from four comparison models

### Statistical approach

- Bootstrap confidence intervals (10,000 resamples)
- Paired permutation tests (10,000 permutations)
- Cohen's d effect sizes
- Spearman rank correlation (feature deltas vs SyncNet deltas)
- Benjamini-Hochberg FDR correction for multiple comparisons

### G×E decomposition

For each sample, a 2×2 matrix (generator audio × evaluator audio) separates:
- Generator effect (G_tts, E_natural − G_natural, E_natural)
- Scorer effect (G_natural, E_tts − G_natural, E_natural)
- Interaction (total − generator − scorer)

---

## Limitations

1. **Sample size (n=12)**: Limited statistical power. Bootstrap CIs and
   permutation tests are used to avoid parametric assumptions. Results
   are exploratory rather than confirmatory.

2. **SyncNet-only evaluation**: SyncNet v2 is the sole evaluation metric.
   Results reflect SyncNet sensitivity to acoustic features, not necessarily
   human-perceived video quality. Perceptual studies are needed.

3. **Cross-TFG mechanism coverage**: Per-sample SyncNet deltas are available,
   but HuBERT feature metrics were not extracted for the other TFG models, so
   mechanism-specific cross-model tests remain unavailable.

4. **Single language**: All data is AISHELL-1 Mandarin Chinese. Cross-linguistic
   validation (English, multilingual) is needed.

5. **No human evaluation**: All findings are based on automated metrics. Human
   perceptual studies are the gold standard for lip-sync quality assessment.

6. **Phase 1 per-sample correlation failed**: Per-sample Spearman correlations between
   feature deltas and SyncNet deltas returned NaN — separability metrics are
   population-level statistics with insufficient per-sample variance for rank
   correlation. Future work needs per-sample embedding distance metrics.

7. **G×E matrix coverage**: Duration-aligned off-diagonal cells are complete
   for 9 samples; sample 9 still lacks TTS-generated cells. The large
   interaction makes independent generator/scorer attribution unstable.

8. **Intervention outcome pending**: While pilot interventions successfully
   modify target features (LUFS, spectral tilt, dynamic range), the downstream
   effect on SyncNet scores has not yet been measured on modified-audio videos.

---

## Next Steps

1. ~~**Complete Phase 1**~~ **DONE**: HuBERT viseme-level separability complete
   (18 results, 24 comparisons). XLS-R skipped due to time constraints.
   Report: `data/wav2sem_analysis/report.md`.

2. **Downstream intervention evaluation**: Generate TFG videos or replace their
   audio tracks with the verified LUFS, spectral, and dynamic transformations,
   then re-run SyncNet. The current script verifies transforms but does not yet
   measure their score effects.

3. **Repair sample 9 and align all 12 samples**: Complete the missing Ditto
   TTS cells before treating G×E decompositions as final.

4. **Cross-TFG feature validation**: Extract the same HuBERT metrics for
   Wav2Lip, MuseTalk, LatentSync, and JoyVASA, then test mechanism directions
   rather than only aggregate SyncNet deltas.

5. **Per-sample embedding analysis**: Replace population-level separability
   metrics with per-sample embedding distances (L2/cosine between natural
   and TTS frame embeddings) to enable Spearman correlation with SyncNet.

6. **Perceptual study**: Human evaluation of lip-sync quality across conditions
   to validate SyncNet-based conclusions.

---
*Generated from Phase 1+2 analysis pipeline. See `data/wav2sem_analysis/` for
raw metrics and `scripts/21_validate_causal_results.py` for cross-TFG validation.*
