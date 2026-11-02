---
title: tts_tfg_mechanism_report
type: note
permalink: tts-exp/docs/experiments/tts-tfg-mechanism-report
---

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

Controlled audio interventions (`scripts/analysis/causal_feature_interventions.py`)
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

Cross-model validation (`scripts/analysis/validate_causal_results.py`) tests whether
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

## Replacement-aware bottleneck and breakthrough ranking (2026-08-22)

最新目标要求 `mux(TFG(face, enhanced_driver), natural)` 在 official file-level SyncNet 中优于 `mux(TFG(face, natural), natural)`，并最终形成跨冻结 TFG 的即插即用 waveform/audio head。当前瓶颈不是 proxy parity 或训练量，而是 TTS 优势主要存在于 diagonal generator×evaluator co-adaptation；Ditto 2×2 中 `G_tts/E_natural` 相对 `G_natural/E_natural` 的 generator effect 为 `-3.247`，说明 raw TTS motion 本身不能直接换回 natural。

进一步边界：zero-residual WavLM→HiFi-GAN 不是 waveform identity，strict 2×2 已证明 reconstruction video/audio crossed preference；Wav2Lip 使用 mel、Ditto 使用 HuBERT、MuseTalk/LatentSync 使用 Whisper，单一 HuBERT target 不具备自然的跨模型接口。Wav2Sem 提供跨 3D facial-animation baseline 的 plug-and-play representation precedent，但不输出通用 waveform，也未研究不同 replacement audio。

突破优先级：

1. 跨 TFG 2×2 replacement map，用于寻找 generator effect 为正或接近零的模型族；用户指出该类 map 已做过很多，不再作为下一执行 gate。
2. sample-exact waveform residual head：`C=N+εR(N,T)`，要求 zero action 逐采样等于 natural，不经过 full encode/decode；先用低维、严格保时的 residual/DSP 参数验证正 replacement 区间，再蒸馏。
3. 多 frontend ensemble：联合 Wav2Lip mel、Ditto HuBERT、MuseTalk/LatentSync Whisper，只保留跨 frontend 一致的 TTS-benefit direction，同时约束 natural 时间坐标。
4. natural timing teacher + TTS motion-quality teacher：从 landmarks、lip aperture、visual embeddings 或 VSR units 中分离时间兼容性与运动质量。
5. 低维 multi-TFG black-box search（CMA-ES/CEM）只用于 existence proof；objective query 昂贵，不直接搜索高维 waveform。
6. AV-HuBERT/VSR units 可作逐窗口 replacement-compatible 辅助约束，但暂无证据支持其作为完全 generator-independent target。

降级方向：继续增加当前 WavLM→HiFi-GAN cross-attention 的 samples/steps、普通 VC、phone-local DTW、单一 global-shift CEM、单一 HuBERT layer target、继续细化已通过 aggregate parity 的 Wav2Lip proxy。

外部资料：Wav2Sem https://arxiv.org/abs/2505.23290；FaceFormer https://arxiv.org/abs/2112.05329；AV-HuBERT https://arxiv.org/abs/2201.02184；Wav2Lip https://github.com/Rudrabha/Wav2Lip；Ditto https://github.com/antgroup/ditto-talkinghead；MuseTalk https://github.com/TMElyralab/MuseTalk；LatentSync https://github.com/bytedance/LatentSync；CMA-ES https://cma-es.github.io/。

## LRS3 conservative alignment official replacement audit (2026-08-22)

用户要求跳过已反复完成的跨 TFG 2×2 map，直接把现有 LRS3 对齐方法导出并通过 Wav2Lip/SyncNet 最终链路验证。使用完整 validation 69 条上已经通过 fixed-video gate 的最强诊断组合：

```text
MFA-linear exact-length audio
→ natural-derived 50 ms energy envelope，ratio [0.8,1.25]，strength 0.25
→ per-clip fixed-video CEM global residual shift，范围 ±120 ms
```

fixed-video 结果曾为：conservative envelope α=.25 在 69 条上 median reward gain `+0.01164`、positive `82.6%`；再加 timing 后 median gain `+0.32147`、positive `79.7%`，两者均为 GO。此前没有保存 corrected WAV，也没有运行 Wav2Lip。

本次新增 exporter/official evaluator，并在 balanced LRS3 validation n=15 上同次重跑三臂：natural control、envelope、envelope+timing。每个 candidate 驱动 official Wav2Lip，然后视频流不重编码、音轨精确换回 original natural PCM，再跑 official file-level SyncNet。

结果相对同次 natural-control replacement：

- envelope：mean `ΔSync-C -1.216`，median `-1.371`，`0/15` 提升；mean `ΔSync-D +1.432`，`0/15` 改善；
- envelope+timing：mean `ΔSync-C -1.107`，median `-1.130`，`0/15` 提升；mean `ΔSync-D +1.357`，`0/15` 改善；
- timing 相对 envelope 有小幅增量：mean `ΔSync-C +0.108`、mean `ΔSync-D -0.074`，但远不足以接近 natural baseline；
- driver-audio 自身视频也不稳：envelope+timing mean `ΔSync-C +0.084` 但 median `-0.192`，Sync-D mean 变差 `+0.129`；
- 45/45 replacement 的 video stream、original PCM 和 sample count integrity 全部通过，AV offset match 为 `14/15`（两种 alignment arm 均如此）。

结论：该 natural-derived energy/timing alignment 是 fixed-natural-video reward 的强 oracle，但不能迁移到 Wav2Lip-generated-video + natural replacement。它再次证明“candidate audio 对固定真实视频更对齐”与“candidate 驱动 TFG 后的视频可换回 natural”不是同一目标；不应继续扩大该 envelope/CEM 路线或蒸馏其 NO-GO policy。

Artifacts：

- smoke：`runs/lrs3_cem_fixed_video_20260820/17_envelope_timing_wav2lip_smoke/`
- official n15：`runs/lrs3_cem_fixed_video_20260820/18_envelope_timing_wav2lip_n15/`
- evaluator：`scripts/experiments/lrs3_rhythm/eval_lrs3_envelope_timing_alignment.py`

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
raw metrics and `scripts/analysis/validate_causal_results.py` for cross-TFG validation.*

## LRS3 natural-phase spectral-envelope residual audit (2026-08-22)

为避免完整 waveform VC 和 phone-local warp 的影响，新增单一 sample-exact residual：保留 natural waveform 的时间轴、STFT phase、逐样本长度和 broadband energy，只把 raw TTS 的静态短时 log-magnitude spectral envelope（16 kHz STFT 800/160/800，频率平滑 15 bins）映射到 natural，并以 strength `0.5` 加回 natural。`strength=0` 逐样本 identity；candidate 通过 official Wav2Lip，之后视频流不重编码、音轨换回 original natural PCM，再跑 official file-level SyncNet。

balanced LRS3 n=15 official audit：candidate 自身 driver 视频相对同次 natural control 的 mean `ΔSync-C +0.040`、median `+0.015`，10/15 提升；mean `ΔSync-D -0.052`，10/15 改善。可是 replacement 相对 natural control 的 mean `ΔSync-C -0.00013`、median `-0.009`，仅 7/15 提升；mean `ΔSync-D -0.0065`、median `+0.027`，6/15 改善，joint better 4/15。AV offset match 为 15/15，video stream、original PCM 和 sample count integrity 全部通过。

结论：该保时、保相位的 spectral-envelope residual 比 phone-warp 更接近 replacement neutrality，并再次复现 candidate-driver 的小幅对角增益，但没有产生稳定正向 replacement gain；这是 NO-GO，不应继续扩大 n 或把 strength=0.5 的近零结果称为成功。Artifact：`runs/lrs3_cem_fixed_video_20260820/28_spectral_envelope_tts_wav2lip_n15/`；script：`scripts/experiments/lrs3_rhythm/eval_lrs3_spectral_envelope_tts.py`。

动态频谱补充实验（同一 script `--dynamic --strength 0.5`，balanced LRS3 n=15，artifact `runs/lrs3_cem_fixed_video_20260820/30_dynamic_spectral_envelope_tts_wav2lip_n15/`）把 TTS 的逐时间帧 log-magnitude envelope 线性映射到 natural frame count，再保留 natural phase 和 sample clock。candidate-driver mean `ΔSync-C -0.262`、median `-0.278`，仅 3/15 提升；replacement mean `ΔSync-C -0.0645`、median `-0.046`，仅 3/15 提升；replacement mean `ΔSync-D +0.0223`、median `+0.057`，6/15 改善，joint 3/15。AV offset 15/15 匹配，视频流、original PCM 和 sample count integrity 全部通过。结论：逐时 spectral transfer 比静态 envelope 更差，保相位频谱残差整类没有 replacement 正增益证据，应停止继续调 strength 或扩大 cohort。

## LRS3 motion-teacher Gate 0 protocol audit (2026-08-22)

新增 Gate 0 审计脚本：`scripts/experiments/lrs3_motion_teacher/audit_assets_and_protocol.py`，并通过手工 targeted assertions/`py_compile`。成功 run：`runs/lrs3_motion_teacher_20260822/00_protocol_audit_retry1/`。

审计固定了 LRS3 online dataset hash、30 train source groups 内的 24 fit + 6 internal-dev groups、6 validation groups，并保留 7 test groups/84 clips 为 `sealed_unvisited`；工作 manifest 包含 271 fit、76 internal-dev、69 validation records，不打开 test 媒体或派生 feature。所有工作记录的 video、natural audio、paired TTS audio、visual cache SHA-256、transcript pairing、16 kHz mono 和 natural sample count 均通过；Wav2Lip/SyncNet/SFD checkpoint 与两个 venv interpreter 资产通过。

Decision：`GO`，`next_allowed_stage=render_wav2lip_teacher_cache`。可选 MediaPipe face-landmarker asset 位于当前进程无权限的历史 Ditto 路径，因此记录为 `inaccessible` 而不是误报为 core audit failure；AV-HuBERT/VSR asset 当前为 `missing`，viseme branch 后续必须单独 BLOCKED 处理。首次 run `runs/lrs3_motion_teacher_20260822/00_protocol_audit/` 因可选资产权限探测 bug 保留为失败 artifact，未覆盖重跑。

## LRS3 motion-teacher Gate 1 Wav2Lip pilot (2026-08-22)

Gate 1 teacher renderer：`scripts/experiments/lrs3_motion_teacher/render_wav2lip_teacher_cache.py`，复用 official `run_wav2lip`，新增批量 face-box fallback 工具 `derive_wav2lip_fallback_box.py`，并让 official runner 接受兼容的可选 `fallback_box` 参数。首次 24-fit-group 版本和第二次固定 box 坐标版本分别因 sample selection 不完整及 Wav2Lip `(y1,y2,x1,x2)` 坐标误用而保留为失败 artifacts；最终成功 run：`runs/lrs3_motion_teacher_20260822/01_wav2lip_teacher_pilot_retry3/`。

成功渲染 30 个 train source groups、每组 2 条，共 60 条 `V_N=TFG(face,N)` 与 `V_T=TFG(face,TTS)`，SyncNet 未用于选样。所有视频 25 fps、video metadata/hash 完整；N/T video stream MD5 无重复；TTS 时长差异保留，没有人为截断或 waveform 对齐。4 个 clip 使用自动 8-frame 批量 face detector median fallback，1 个 clip 使用 predeclared box；fallback 来源、坐标格式和 detector coverage 都写入 teacher manifest。test lock 仍为 `sealed_unvisited`。

Decision：`GO`，`next_allowed_stage=audit_landmark_teacher`。下一步只能做 landmark trajectory reliability/teacher-existence audit，不能直接训练 head，也不能读取 test 媒体。

## LRS3 motion-teacher Gate 2 landmark audit (2026-08-22)

Gate 2 脚本：`scripts/experiments/lrs3_motion_teacher/audit_landmark_teacher.py`；trajectory/normalization 模块与测试已加入：`scripts/common/landmark_trajectories.py`、`tests/common/test_landmark_trajectories.py`。对 Gate 1 成功的 60 条 Wav2Lip natural/TTS teacher pilot 执行资产检查后，结果为 `BLOCKED`：MediaPipe `face_landmarker.task` 只有一个历史 Ditto 路径可见但当前进程无权限，repo 内没有可访问副本，syncnet/wav2lip 两个运行环境也未安装 `mediapipe`。

Artifact：`runs/lrs3_motion_teacher_20260822/02_landmark_teacher_audit/`；status=`blocked_missing_landmark_asset`，`next_allowed_stage=audit_viseme_assets`。没有读取 test 媒体、没有生成 landmark trajectory、没有训练 landmark head，也没有用 Wav2Lip face box 冒充 landmark teacher。Landmark branch 必须等待明确 hash/license 的 detector asset 和 differentiable landmark model，不能通过增加训练量绕过该 BLOCKED。

## LRS3 motion-teacher Gate 2 rerun and viseme asset gate (2026-08-22)

在用户授权安装依赖后，`mediapipe==1.0.1` 安装到 `~/.venvs/syncnet`，官方 `face_landmarker.task` 下载并固定为 SHA-256 `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`。landmark extractor 改用独立 IMAGE mode，保留缺失帧为 NaN 并显式统计，不跨视频复用 VIDEO timestamp state。

60 条 Gate 1 teacher pilot 的 landmark audit artifact：`runs/lrs3_motion_teacher_20260822/02_landmark_teacher_audit_retry2/`。45/60 序列达到 valid-frame ≥95%，但 real-video geometry teacher benefit `error(V_N,V_R)-error(mapped(V_T),V_R)` median `-0.05836`，positive fraction `1/60=0.0167`；26 个 source groups 的 accepted median 全为负，非退化 TTS trajectory 仅说明 T/N 不同，不能说明有益。Decision=`NO_GO`，不训练 landmark residual head，不调 target strength。

Viseme asset gate artifact：`runs/lrs3_motion_teacher_20260822/09_viseme_asset_audit/`。English phoneme-to-viseme YAML 和 MFA parser 可用但明确只是 diagnostic labels；AV-HuBERT/VSR runtime 与 pinned checkpoint 均缺失，因此 `BLOCKED`，不能用 rule-based map 或 audio HuBERT 替代 learned visual-viseme teacher。下一条允许路线为独立的 Wav2Lip-native model-specific teacher audit。

## LRS3 motion-teacher Gate 9 native output audit (2026-08-22)

由于 landmark branch NO-GO、AV-HuBERT/VSR asset gate BLOCKED，执行独立的 Wav2Lip-specific native output teacher existence audit：不使用 SyncNet，提取 `V_N/V_T/V_R` 224×224 输出/源视频 lower-face 32×32 normalized frame-difference descriptor，按 normalized frame clock 映射 TTS descriptor 到 real video。

Artifact：`runs/lrs3_motion_teacher_20260822/10_native_teacher_audit/`。60/60 records 完成，fps 均为 25，descriptor 非退化，但 teacher benefit median `-1.91355`，positive fraction `2/60=0.0333`；28 个 accepted source-group median 为负。Decision=`NO_GO`，`next_allowed_stage=null`。该分支只能说明当前 Wav2Lip output-space 的简单 native motion descriptor 不提供可迁移的 TTS quality target，不能声称 Wav2Lip 内部所有 latent 均无效。

截至此 gate：landmark teacher 为 NO_GO，viseme teacher 为 BLOCKED_MISSING_VSR_ASSET，native output teacher 为 NO_GO；按预注册规则不进入 waveform-head training、组合 loss、cross-TFG 或 test。已有 MediaPipe 安装和 model asset 仅用于审计，不能将失败 teacher 变成训练监督。

## LRS3 Wav2Lip raw-TTS strict PCM 2×2 (2026-08-22)

Artifact：`runs/lrs3_motion_teacher_20260822/11_raw_tts_strict_2x2_retry1/`。输入为 Gate 1 已冻结的 60 clips / 30 train source groups `V_N/V_T` cache；四格全部重新 mux untouched mono 16 kHz PCM16，禁用 atempo、AAC、shortest 和任何 time transform。240/240 cells 完成，video payload + 完整 video PTS timeline、decoded PCM/sample count、A/V start timing 全部通过；120 个 cached teacher audio streams 与声明 N/T driver 的 codec-aware binding 全部通过。SyncNet 多人脸时按预注册的 maximum-frame-count→duration→filename 规则选择 primary track，不读取分数；test 未打开。

同次四格均值 `C/D`：`Y_NN=7.4274/7.2116`，`Y_NT=1.3328/13.5556`，`Y_TN=1.5677/13.2147`，`Y_TT=6.8941/8.0779`。正式 target `Y_TN−Y_NN`：mean ΔC `−5.85965`（group-bootstrap 95% CI `[-6.13335,-5.56798]`），median `−5.975`，C better `0/60`；mean ΔD `+6.00315`（95% CI `[5.7212,6.28463]`），median `+6.1075`，D better `0/60`；joint `0/60`，positive groups `0/30`。interaction mean 为 C `+11.421`、D `−11.4809`，再次证明强 diagonal co-adaptation，而不是 replacement-compatible raw-TTS motion。Decision=`NO_GO`，next=`audit_wav2lip_mel_intervention`。

该结果严格否定“直接复制 raw TTS 生成 motion 后换回 N”作为 Wav2Lip teacher；它尚不否定把 TTS mel residual phone-align 到 natural clock 后在真实 Wav2Lip mel seam 进行小幅 causal intervention。后者是本地 Wav2Lip TTS-transfer 路线最后一个 existence audit；若仍 NO-GO，则停止该主线，不训练 waveform head、不访问 validation/test。

## LRS3 Wav2Lip phone-aligned mel causal intervention (2026-08-23)

Strict raw-TTS 2×2 NO-GO 后，最后一个 Wav2Lip-native existence audit 已完整结束。Artifact：`runs/lrs3_motion_teacher_20260823/12b_wav2lip_mel_intervention_causal_retry4/`；30 条 train-only records / 30 source groups，固定 `alpha=0.25`、exact-label non-pause/non-`spn` phone mask、shared boxes、untouched natural PCM replacement 与 official file-level SyncNet，validation/test 未访问。

正式 target 相对 natural-mel identity：mean ΔC `-0.10233`（group-bootstrap 95% CI `[-0.13610,-0.06610]`），mean ΔD `+0.06883`（CI `[+0.02443,+0.11217]`），joint better `3/30`。八项预注册 GO 检查全部失败。shuffle 同样 NO-GO（mean ΔC `-0.16153`、ΔD `+0.13227`）；raw arm 灾难性失败（mean ΔC `-5.38563`、ΔD `+5.37040`）。Target 比 shuffle 少伤害，但仍系统性差于 identity。

Gate 2A 与 full run 的 identity parity、shared-box、video/PCM/sample-count/PTS/A-V timing integrity 全部通过；target/identity AV offset `30/30` 相同，MFA teacher subset provenance 完整。因此这是科学 NO-GO，不是 mel seam、codec、mux、face detection 或 alignment inventory 的基础设施失败。Phone boundaries 只提供粗时钟等价，无法消除 TTS 与 natural 的 phone 内 onset/coarticulation/能量/F0/formant/频谱实现差异；Wav2Lip 将这些差异转成与最终 natural audio 不兼容的运动。

Decision：关闭当前 Wav2Lip TTS-motion / phone-aligned TTS-mel transfer 主线；不进入 waveform reachability，不训练 audio head，不事后 sweep alpha/mask/threshold。完整分析见 [[LRS3 Wav2Lip phone-aligned mel causal NO-GO]]。
