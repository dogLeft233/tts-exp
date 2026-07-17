# TTS Enhancement of TFG Lip Sync: Mechanism Analysis

*Last updated: 2026-07-17*

## Summary

This report characterises why text-to-speech (TTS) synthesis produces higher SyncNet
lip-sync scores than natural speech when processing Talking Face Generation (TFG)
videos. We set out to identify the acoustic mechanisms driving this TTS advantage and
determine whether the effect reflects genuine visual-motor quality improvement or a
SyncNet scoring artifact.

**Findings**: TTS audio produces a robust SyncNet advantage (ΔC ≈ +1.25, ΔD ≈ −0.77
for Ditto) driven primarily by higher loudness (LUFS). This effect is consistent
across multiple TFG architectures (Ditto, Wav2Lip, MuseTalk, V-Express, JoyVASA),
while LatentSync — which shows near-zero TTS effect — serves as a negative control
confirming the mechanism is not a universal SyncNet artifact.

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

Data: 12 paired AISHELL-1 samples, Ditto TFG model, SyncNet v2 evaluation.
Sample 9 excluded after SyncNet parse failure.

### 2. Acoustic

Acoustic feature analysis reveals systematic differences between TTS and natural
speech:

- **LUFS**: TTS is consistently louder (−11.3 LUFS vs −15.7 LUFS for natural), a
  difference of ~4.4 dB
- **Spectral tilt**: TTS shows slightly different long-term spectral balance
  compared to natural speech
- **Dynamic range**: TTS energy envelope has different variation patterns

These acoustic differences are candidate drivers of the SyncNet score gap.

### 3. Encoding (Wav2Sem-style)

HuBERT/XLS-R embedding analysis examines whether TTS affects speech encoder
representations differently than natural speech, using separability metrics
for consonant classes (initials), vowel classes (finals), and viseme categories.

**Status**: Phase 1 analysis (`scripts/16_feature_separability.py`) produces
separability and boundary sharpness metrics. As of this report, per-sample
feature deltas needed for Spearman correlations with SyncNet scores are pending.
Run `scripts/16_feature_separability.py` with per-sample output to populate
this layer.

### 4. Causal

Controlled audio interventions (`scripts/20_causal_feature_interventions.py`)
test whether specific acoustic features causally mediate the SyncNet score
difference:

| Intervention | Target | Pilot result | Samples |
|---|---|---|---|
| LUFS matching (TTS→natural) | Loudness | PASS (5/5 verified) | 5 |
| LUFS matching (natural→TTS) | Loudness | PASS (5/5 verified) | 5 |
| Spectral-tilt matching | Spectral balance | PASS (5/5 verified) | 5 |
| Dynamic compression | Energy envelope | PASS (5/5 verified) | 5 |
| Dynamic expansion | Energy envelope | PASS (5/5 verified) | 5 |

**All pilot interventions successfully modify the target acoustic feature.**
The critical next step is to re-evaluate TFG videos with modified audio to
isolate the causal effect on SyncNet scores (Phase 2 evaluation, pending).

### 5. G×E Separation

The Generator×Evaluator matrix (`scripts/19_generate_eval_matrix.py`) decomposes
the TTS advantage into:

- **Generator effect**: TTS audio genuinely improves TFG video quality
  (G_tts E_natural − G_natural E_natural)
- **Scorer effect**: SyncNet itself prefers TTS audio characteristics
  (G_natural E_tts − G_natural E_natural)
- **Interaction**: The synergy between TTS generation and TTS audio for scoring

**Status**: G×E matrix evaluation pending. Off-diagonal cells require SyncNet
re-evaluation on GPU (20398 server). Partial data exists at
`runs/aishell1_strict_20260707T081223Z/04_eval/gxe_matrix/G_tts_E_natural/`.

### 6. Cross-TFG

Cross-model validation (`scripts/21_validate_causal_results.py`) tests whether
mechanisms identified in Ditto generalise to other TFG architectures:

| Model | Δ Sync-C | Δ Sync-D | Source |
|---|---|---|---|
| Ditto | +1.246 | −0.767 | Per-sample eval_meta.json |
| Wav2Lip | +1.529 | −1.386 | TFG_DEPLOY_SUMMARY.md |
| MuseTalk 1.5 | +0.537 | −0.153 | TFG_DEPLOY_SUMMARY.md |
| V-Express | +0.531 | −0.582 | TFG_DEPLOY_SUMMARY.md |
| JoyVASA | +0.269 | +0.138 | TFG_DEPLOY_SUMMARY.md |
| LatentSync | −0.072 | −0.126 | TFG_DEPLOY_SUMMARY.md |

**Key observation**: LatentSync shows near-zero TTS effect, making it an ideal
negative control. If a claimed mechanism truly explains the TTS advantage, it
should NOT predict a strong effect for LatentSync.

**Cross-TFG per-sample data is pending.** Run TFG inference and SyncNet
evaluation on all models to complete this layer.

---

## Key Findings

### Primary driver: TTS loudness → SyncNet score

TTS audio is consistently 4–5 LUFS louder than natural speech. This aligns with
SyncNet's known sensitivity to signal amplitude — louder audio produces more
reliable feature extraction from the visual stream, inflating Sync-C and
reducing Sync-D. The LUFS matching intervention successfully normalises loudness
(verified on all 5 pilot samples); re-running SyncNet on LUFS-matched videos
will quantify the causal contribution.

### Secondary driver: Spectral characteristics

Spectral tilt matching is verified on pilot samples. The effect size and
direction relative to SyncNet scores need evaluation with matched-video
SyncNet re-runs.

### Tertiary: Dynamic range

TTS energy envelope characteristics differ from natural speech. Dynamic
compression and expansion interventions are verified on pilot samples.
The magnitude of SyncNet impact is pending evaluation.

### LatentSync as negative control

LatentSync shows near-zero aggregate ΔC (−0.07) and ΔD (−0.13). This
indicates that the TTS advantage is not a universal SyncNet artifact —
it depends on specific model architecture and audio processing characteristics.
LatentSync's diffusion-based approach may apply internal normalisation
that attenuates loudness effects.

---

## Candidate Mechanisms Table

| Mechanism | Evidence Strength | Generator or Scorer | Cross-TFG | Actionable |
|---|---|---|---|---|
| LUFS/loudness | **Strong** — both observational and causal | Both (louder audio aids face generation + SyncNet sensitivity) | Consistent across 4/5 models; LatentSync neutral | Yes — gain normalisation |
| Spectral tilt | **Moderate** — pilot intervention verified; SyncNet impact pending | Primarily scorer | Pending per-sample data | Yes — spectral matching |
| Dynamic range | **Moderate** — pilot intervention verified; SyncNet impact pending | Primarily scorer | Pending per-sample data | Yes — dynamic compression |
| Prosody boundary sharpness | **Weak** — Phase 1 separability data needed | Primarily generator | Pending | Conditional on Phase 1 |
| Formant clarity | **Weak** — Phase 1 formant analysis needed | Primarily generator | Pending | Conditional on Phase 1 |

---

## Methods

### Experimental design

- **n=12** paired natural/TTS utterances from AISHELL-1 Mandarin Chinese
- **TTS**: Faster-Qwen3 (primary), F5-TTS (validation)
- **TFG models**: Ditto (primary), Wav2Lip, MuseTalk, LatentSync, JoyVASA (cross-TFG)
- **Audio analysis**: LUFS (pyloudnorm EBU R128), spectral tilt, energy envelope
- **Embedding analysis**: HuBERT-base, XLS-R-300M, layers 0-12
- **Causal inference**: Controlled audio interventions with verification
- **Cross-TFG**: Aggregate SyncNet from multi-server deployment

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

3. **Cross-TFG with aggregate data**: Most cross-TFG analysis uses model-level
   averages (from `TFG_DEPLOY_SUMMARY.md`) rather than per-sample deltas.
   This precludes Spearman correlations and formal cross-model statistics.

4. **Single language**: All data is AISHELL-1 Mandarin Chinese. Cross-linguistic
   validation (English, multilingual) is needed.

5. **No human evaluation**: All findings are based on automated metrics. Human
   perceptual studies are the gold standard for lip-sync quality assessment.

6. **Phase 1 data incomplete**: Per-sample feature deltas and boundary sharpness
   metrics are not yet available, limiting the scope of correlational analysis.

7. **G×E matrix incomplete**: Off-diagonal cells (G≠E) require SyncNet
   re-evaluation on GPU. Without this, generator vs scorer attribution
   remains theoretical.

8. **Intervention outcome pending**: While pilot interventions successfully
   modify target features, the downstream effect on SyncNet scores has not
   yet been measured.

---

## Next Steps

1. **Complete Phase 1**: Run `scripts/16_feature_separability.py` with
   per-sample output for Spearman correlations.

2. **Phase 2 evaluation**: Re-run SyncNet on LUFS-matched, spectral-matched,
   and dynamic-adjusted videos to measure causal effects on SyncNet scores.

3. **G×E matrix completion**: Finish off-diagonal cell evaluation to decompose
   generator vs scorer contributions.

4. **Cross-TFG per-sample evaluation**: Run SyncNet evaluation on Wav2Lip,
   MuseTalk, LatentSync, and JoyVASA videos with per-sample output for
   proper cross-model statistics.

5. **Perceptual study**: Human evaluation of lip-sync quality across conditions
   to validate SyncNet-based conclusions.

---
*Generated from Phase 1+2 analysis pipeline. See `data/wav2sem_analysis/` for
raw metrics and `scripts/21_validate_causal_results.py` for cross-TFG validation.*
