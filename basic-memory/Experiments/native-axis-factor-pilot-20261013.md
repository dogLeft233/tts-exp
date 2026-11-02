---
title: native-axis-factor-pilot-20261013
type: note
permalink: tts-exp/experiments/native-axis-factor-pilot-20261013
status: concluded
protocol: native_axis_factor_pilot_v1
scientific_status: NATIVE_9CELL_AND_PHYSICAL6_CONCLUDED_ARTICULATORY_FIT_DIAGNOSTIC_SEPARATE
---

# native-axis-factor-pilot-20261013

## 当前科学结论：干净原生9格全部完成

三模型×SEP_ANG_ACTIVE05/STAB_ANG_FIT/radial ENV v2×两人=18因子端点，复用6baseline，ROOT实际原矩阵/嘴部轨迹和新Wav forward/原PCM核验完成。单行joint pass4/18，model-factor双人复现0/9。完整可复核汇总 `native_axis_model_factor_scope_render_20261014_attempt02/root_complete_native_coordinate_grid_summary.json` SHA9e03f9c50dee1bb5300952572a3b88f9ac84fe97b7fc6c9679fa5bcb6449cc4f；Muse新增2行实际summary85c4a6d5…，剩4行实际summary0bf45888…。

| factor/model | ΔC S0044/S0052 | ΔD0 S0044/S0052 | 本格双人joint |
|---|---|---|---|
| SEP/Muse新增 | −0.012/−0.036 | +0.073/+0.065 | fail |
| STAB/Ditto新增 | +0.029/−0.064 | −0.123/+0.119 | fail |
| radial ENV/Wav新增 | −0.043/+0.034 | −0.171/+0.070 | fail |

新6行独立嘴部开合中位数/velocity RMS：MuseSEP44 −0.000478/−0.004410，52 −0.000687/−0.009495；DittoSTAB44 +0.001689/+0.002314，52 +0.005369/−0.010375；WavENV44 −0.002141/−0.012535，52 +0.002722/−0.084511（IPD、IPD/s）。全部tracking完整；运动有变化但无跨样本统一质量方向。Sync-C增益不能代替原时钟D0，更不能直接称嘴型准确。

当前固定剂量的三种干净native坐标均不构成稳定增强方法；不再扩dose/speaker或重开旧矩阵。原生latent干预未保证waveform可实现，物理谱形保留为独立下一问题；这不是“全部声学机制无效”的结论。旧canonical operator22/24与其能量collateral只保留为历史实验。以下旧阶段文字不作为新执行授权，以当前Task/HANDOFF为准。

## 物理phone谱形：六行筛查已完成

已执行 `physical_phone_spectral_shape_fit_20261014_attempt04/source.py` SHA89c644de…；protocolf951d47a…，receipt2d1e685b…，WAVreadbackf3e443e8…，ROOT实际4WAV QAd6713acf…合格。9FIT/18对/36WAV，83phones中57有≥3人支持，半signed median/±3dB。两targetN59135/65855、16kmonoFLOAT32，全局RMS误差≤3.71e-10，零剂量实际差≤1.32e-17；OLA后edited phase约0.01969/0.01575rad、boundary约0.01301/0.01114rad，edited局部power median abs约0.01597/0.01619dB，不能称严格相位/局部能量不变。这些是CPU控制资格；下游效果见本节六行结果。

6candidate CPU native encode（2波形×3TFG）已完成，source9486dac7…、receipt2464541e…、ROOT实际6NPY QA0f991e86…，位于attempt04/native_reencode_cpu_attempt02/。0baseline/roundtrip重encode、0API/download/deps/MFA/training/tests。实际margin Δ：Wav −0.000704/−0.000232、Muse +0.000202/−0.000471、Ditto −0.000154/−0.000035，dispersion响应也混合；geometry receipt1dc5711e…，仅diagnostic，不调dose/选样本。physical共享方向经过不同encoder仍不等于统一几何响应。

效果前冻结的6新短render/score/mouth screen（protocol6ca540bf…）已完成，复用no-op，原natural PCM/25fps/脸/seed/profile不变。ROOT独立实际6行矩阵/mouth及输入/clock检查通过，汇总 `physical_phone_spectral_shape_fit_20261014_attempt04/root_physical_six_endpoint_summary.json` SHA46fc88944e09af05ef0b73b733fda0e09c79efa1db34034b68e8edc3d0cadf12。

| model | ΔC S0044/S0052 | ΔDmin S0044/S0052 | ΔD0 S0044/S0052 | 双人joint |
|---|---:|---:|---:|---|
| Wav2Lip | −0.016/+0.101 | −0.091/−0.131 | −0.125/+0.011 | fail |
| MuseTalk | −0.032/−0.052 | +0.064/+0.018 | +0.000455/−0.115 | fail |
| Ditto | +0.043/−0.040 | +0.043/+0.147 | +0.043/+0.147 | fail |

0/6行joint通过，0/3模型双人通过，等模型/人meanΔC+0.001。冻结≥2模型/双人joint/meanΔC≥0.050扩展门失败，此candidate不做更多GPU/dose/cohort。此结论限定本FIT phone均值谱形/剂量/复用2speaker，不概括全部物理局部声学机制。

两处C上升都D0变差；Ditto44是背景B增加+.086超过Dmin恶化+.043，Wav52最佳lag不变且Dmin改善但D0+.011。Wav44、Muse52则D0改善而C下降。因此共同提升C不能直接解释为原时钟更准。独立aperture/velocity Δ：Wav44−.006248/−.023528、52−.002804/−.071681；Muse44−.002176/+.021950、52−.001097/−.003985；Ditto44+.000488/−.007623、52+.001183/−.016357。完整tracking合格，仅描述运动，无真人准确度标签。

实际ROOT summaries Muse6e5c7529…、Ditto12b05d74…、Wavc3d6aabe…，baseline metrics复用已验缓存。Wav actual forward/native/CPU candidate逐值相等，full natural PCM及primary原PCM前缀相等；Ditto多1context frame，score支持=min(video帧,floor(natural N/640))−5，与no-op同为87/97。ROOT初稿query_count假设错误保存并另source修复，无实验重跑。源预检/score-wrapper早期schema失败均0render或0score，保存原件；OS controller rc缺失记录null，不为元数据重跑。未运行实现测试。

独立a/i/u开合contrast（12旧tracks CPU，receipt8063808f…）补充描述性运动证据：S0044唯一a中央区间无frame，UNMEASURABLE；S0052仅1个a中央frame。MuseSEP contrast+0.006714而C−0.036，DittoSTAB−0.000925/C−0.064，WavENV−0.011890而C+0.034/D0变差。保留支持计数，不能按contrast增大称准确或追溯修改gate。

## 独立构音proxy CPU诊断已收束

限定结果与完整协议见 [[构音属性与口部功能坐标诊断]]。9FIT已有native数组的natural-only LOSO/common-basephone/类与人等权诊断已完成；双唇仅Ditto、高度仅Wav/Muse、圆唇均未过门，0共同三模型属性。ROOT从原native数组/TextGrid恢复1896token、162anchors/1972margins和speaker统计一致，QA95c21c92…。此FIT观察不改报告39的因果筛查结论，不事后翻转符号/改mapping，0新GPU预算。下一功能坐标由active Task另冻。

## Summary
A bounded native-feature-space pilot for the three shared directions identified in the project research map. The full protocol is kept in the repository Basic Memory Experiments folder under the dated Chinese title; this ASCII note keeps the experiment indexed in Basic Memory.

Protocol: [[跨TFG原生输入三因素因果pilot Spec 2026-10-13]]. Current synthesis: [[研究交接总览]]. Current work: [[tts-exp/tasks/native-axis-factor-pilot-20261013]].

## Frozen design
Two unseen speakers, three TFGs, and four arms per context: RMS identity, phoneme separation (SEP), segment stability (STAB), and local energy/spectral envelope (ENV), at most 24 generated units after CPU/native-front-end qualification. Preserve 16 kHz mono, target RMS 0.0376838172675746, the natural sample clock, transcript, MFA boundaries, and untouched natural PCM for final scoring and mux.

## Endpoints
Measure each model in its own native input space: Wav2Lip mel, MuseTalk pinned audio features, and Ditto HuBERT/ONNX features. Record SyncNet C, B, D, fixed-lag distance, normalized mouth aperture, and aperture velocity. Without human or real-video labels, motion is reported as motion-consistent or not, never as proof of mouth-shape accuracy.

## Stop rule
Do not expand unless at least two of three models and one factor agree in direction on both speakers without a purity failure or motion-quality regression. Do not reopen the retired four-center or Confirm branches.

## State
AVAILABLE_OPERATOR_SCREEN_COMPLETE_ANGULAR_FOLLOWUP. All available canonical arms are now scored: 22/24 planned cells, including six no-op baselines and 16 factor endpoints; Wav2Lip ENV is unavailable in the remaining two cells. None of these operators passes both speakers in any TFG. The old round-mask packet is engineering-only; old Wav2Lip STAB is pre-calibration; SEP_RD is a separate closed variant. The source-native angular/radial analysis below supplies a more isolated coordinate and a new, separately budgeted four-row follow-up. The enhancement-head goal remains active.

## Current execution and interpretation boundary

The admitted corrected render roots on wjj@192.168.50.2 are `.codex/scratch/native_axis_pilot_ditto_canonical_render_20261014_attempt01/` and `.codex/scratch/native_axis_pilot_musetalk_canonical_render_20261014_attempt02/` under `/home/wjj/tts-audio/tts-exp`; matched no-op videos are reused from the earlier model render roots. Generation uses the same fixed-RMS driver within each contrast. Corrected StageA scoring has already remuxed raw originals from `data/fresh_aishell_disjoint_ref_target_20260929/audio/`: S0044 raw PCM SHA `7ddcdeb90ef2de65d68d1df9a41cf9fc6464255585087dec1cfdae3c3716e976`, S0052 raw PCM SHA `451711ed1799a75086a30fe91b04cbfab188ea6fe74d662341b9ba29c05bd5fc`.

The original CPU operator qualification is `.codex/scratch/native_axis_pilot_operator_qualification_20261014_attempt01/`; the admitted canonical float32 arrays are `.codex/scratch/native_axis_pilot_mask_audit_20261014_attempt02/`. SEP targets a 0.5 FIT SD directional projection rather than guaranteeing an equal nearest-rival distance increase. STAB retains 0.75 residual amplitude and preserves token centroids. Both have native-norm collateral, which is reported separately; fixed waveform RMS does not make native feature energy invariant.

Mouth aperture, range and velocity are independent motion descriptors. Greater or smaller velocity alone has no universal quality ordering. Earlier boolean motion gates retain their original arithmetic as historical screening fields; they do not establish more accurate articulation or justify rejecting a result solely because velocity decreases. Without a real-video or human target, the question that can be answered is whether scores change with measurable mouth motion, where on the phoneme clock it changes, and whether the score benefit survives the raw-original track.

The canonical operator endpoints are complete and these operators will not be expanded. ROOT owns interpretation and BM; GPT-6 Luna/max agents handle execution and CPU analysis. ROOT's old scratch endpoint-input attempt01 used an extra transcode and remains engineering-only; it and the misplaced remote scratch worker are excluded execution sources. Scientific source videos are unchanged.

## Corrected canonical factor screen (Ditto and MuseTalk, 2026-10-14)

The CPU repair produced create-only float32 arrays under `.codex/scratch/native_axis_pilot_mask_audit_20261014_attempt02/`, with calibration metrics, candidate, TextGrid hashes and a fixed supported-phone mask bound. Ditto corrected score packet: `.codex/scratch/native_axis_pilot_ditto_canonical_score_20261014_attempt01/` (manifest SHA `83b784e574b5893e4e2697dc3065e18fe924d0f6eed41b60b92c5392fe3b30a5`). MuseTalk corrected score packet: `.codex/scratch/native_axis_pilot_musetalk_canonical_score_20261014_attempt01/` (manifest SHA `7bac2949f257fa71fcad86074079de0e827a4ce80fc91ee90b95e69743588b72`). Each has two speakers × three corrected factors, matched to existing no-op baselines, raw-original PCM and source clock. All rows have complete MediaPipe tracking.

| model | speaker | SEP ΔC/ΔDmin/ΔD0 | STAB ΔC/ΔDmin/ΔD0 | ENV ΔC/ΔDmin/ΔD0 |
|---|---|---|---|---|
| Ditto | S0044 | `+0.056/-0.022/-0.022` pass | `-0.124/+0.054/+0.054` fail | `+0.131/-0.120/-0.120` pass |
| Ditto | S0052 | `-0.137/+0.102/+0.102` fail | `-0.157/+0.140/+0.140` fail | `-0.058/+0.133/+0.133` fail |
| MuseTalk | S0044 | `+0.028/+0.064/+0.001` fail | `-0.169/+0.272/+0.098` fail | `+0.083/-0.085/+0.060` fail |
| MuseTalk | S0052 | `+0.048/-0.008/-0.078` pass | `-0.145/+0.252/+0.070` fail | `-0.065/+0.038/-0.003` fail |

No factor passes on both speakers in Ditto or MuseTalk. Positive SEP/ENV rows are speaker-specific; STAB fails every corrected row. Lag is unchanged within every pair. Velocity and aperture deltas remain descriptive and mixed, not mouth-accuracy labels. This closes the corrected two-model factor screen without authorizing more MuseTalk/Ditto GPU rows.

## Remaining bounded work and angular/radial coordinate proposal

Canonical Wav2Lip SEP/STAB CPU arrays and all four endpoints are complete. CPU receipt `.codex/scratch/native_axis_wav2lip_canonical_cpu_20261014_attempt01/materialized/receipt.json` has SHA `0e310ba7bdcb444a4023de9c7aa93a152831b49432f0e9aae74ded7a9b4ef753`. Supported chunks are 59/89 and 58/100; unsupported chunks are identity. ENV is explicitly `UNAVAILABLE` (FIT sign agreement `0.530017 < 0.60`, no arrays created).

Wav2Lip final manifest is `.codex/scratch/native_axis_wav2lip_canonical_cpu_20261014_attempt01/local_readback_attempt01/native_axis_wav2lip_final_manifest.json`, SHA `e90224a6f6d9547c474722281489ec6091061cc18c9dae52675b79afa5029303`. Observed forward chunks equal the prescribed factor arrays; raw natural PCM, 25 fps, zero-lag column 15 and complete tracking are retained. S0044 SEP worker rc1 was a JSON serialization failure after valid render/score; create-only recovery reused those arrays/video and added visual/receipts without rerender or rescoring. Other three workers rc0. Earlier adapter failures are engineering-only.

| Wav2Lip speaker | factor | ΔCbest | ΔDmin | ΔD0 | Δaperture median | Δvelocity RMS | response gate |
|---|---|---:|---:|---:|---:|---:|---|
| S0044 | SEP | +0.035 | -0.194 | -0.213 | -0.003325 | -0.041438 | pass |
| S0044 | STAB | +0.108 | -0.281 | -0.612 | -0.016727 | -0.072768 | pass |
| S0052 | SEP | -0.004 | +0.026 | -0.047 | -0.002387 | -0.035939 | fail |
| S0052 | STAB | -0.095 | -0.016 | -0.230 | -0.001036 | -0.044695 | fail |

No Wav2Lip factor passes both speakers either. Compact full-cohort coverage is `.codex/scratch/native_axis_factor_pilot_coverage_manifest_20261014_attempt04.json`, SHA `a1a4b082c999648e4b84fe4a9c827e52a139ee777f793cc3e75297deb20b5ec3`. Five of 16 factor endpoints pass the screen, but none replicates within a model across both speakers. This is a result for these operators, not a clean rejection of their intended factors.

The current corrected screen evaluates these specific operators. SEP/STAB have native-energy collateral, so its negative replication result does not isolate separation/stability as independent causal quantities. A CPU-only follow-up will evaluate a more reproducible coordinate system before any additional GPU budget. Write a flattened native frame/chunk as `x_t = r_t u_t`, with unit direction `u_t`, and define the phone direction as `p = normalize(mean_t u_t)` using an unweighted mean. Feature norm is not physical acoustic energy, especially in normalized Wav2Lip mel.

This coupling is algebraic: `mean(||x||^2) = ||mean(x)||^2 + mean(||x-mean(x)||^2)`. Shrinking Euclidean within-token residuals while holding the raw centroid fixed necessarily lowers native feature energy. The angular/radial proposal preserves unit-centroid direction and permits its magnitude to change, rather than imposing these incompatible Euclidean invariants.

- SEP candidate: one orthogonal rotation for every frame within a phone token moves `p` toward a FIT own-versus-rival angular margin. It preserves every `r_t` and the token's pairwise angular Gram matrix, so radial envelope and within-token angular stability remain fixed. Boundary jumps are measured separately.
- STAB candidate: slerp unit vectors toward the original `p`, then apply one common rotation to restore the resulting phone direction to the original `p`. It preserves every `r_t` and the unit-centroid prototype margin while reducing angular dispersion if the CPU gate passes. Centroid magnitude is allowed to change and is reported.
- ENV candidate: change only the radial trajectory while preserving `u_t` and token total squared norm. Its direction must come from FIT pairs; no unavailable model coordinate is invented.

Before materializing even a small GPU comparison, estimate natural/TTS angular margin, angular dispersion and radial trajectory differences on the existing nine FIT speakers. The hypothesis is tested rather than assumed: a lack of consistent FIT direction is a reason to keep this route CPU-only. The first feasibility dose is capped at a small angle and is not selected from holdout scores. This work reuses existing features and tracks, and does not restart four-center/Dev/Confirm or expand the cohort. These are native-input mechanism probes. Independent edits of overlapping 80×16 Wav2Lip chunks need not correspond to one continuous mel grid or waveform; eventual waveform-head transfer requires a separate realizability/projection check.

## CPU angular/radial result and next micro-pilot

FIT receipt `.codex/scratch/native_axis_angular_radial_cpu_20261014_attempt01/receipt.json` (SHA `dbb147a03213dcec3cf2794851cb9b198512e150d8147562786ea7cc1a9cb6be`) uses nine speakers and 18 pairs. SEP uses LOSO natural-only, speaker-equal phone anchors with at least three other speakers, shared by N/T evaluation. Fixed-FIT-rival angular margin TTS-minus-natural means are Wav2Lip `+0.000521`, MuseTalk `+0.002957`, Ditto `+0.001750`; each has 7/9 positive speakers, but only 3/9 speakers are positive in all three. This is a common average direction with heterogeneity, not a universal individual effect. Mean angular dispersion falls in Wav2Lip/MuseTalk and increases in Ditto. Centered radial shape trends early-positive/late-negative on average with mixed speaker support; constant log-norm shift is reported separately from shape.

The first norm-preserving `SEP_ANG05` improved its fixed-rival margin but reduced actual nearest-any-rival margin in four tokens. It is not launched. The revised `SEP_ANG_ACTIVE05` chooses the current nearest competing natural-FIT prototype per token, then applies one common orthogonal rotation. Fixed candidate angles are `[0.05,0.025,0.0125,0.00625,0.003125,0.0015625]` rad; the first float32 candidate whose actual own-minus-max-other margin increases by more than `1e-7` is used, otherwise identity. This geometric selection never reads SyncNet. It keeps each frame norm and within-token Gram fixed; cross-phone boundary changes remain declared collateral.

Executed source is `.../sep_ang_active05/source.py` (SHA `2b6bce34a26830ece5d4d304bfe8667e04483d2c6bb2ccf0a8360389a401bbf0`), receipt `.../sep_ang_active05/outputs/receipt.json` (SHA `a860afb9ca8c0125f1f487ada40f1f36d4caad93d0310a6bda951f68f58db2a6`). All 68 supported Wav2Lip/Ditto tokens selected 0.05 rad and increased the actual margin. Four float32 candidates match baseline source hashes, keep unselected frames bitwise identical, and have max frame-norm error `2.03e-7`, unit-Gram error `1.50e-9`, relative squared-norm error `2.24e-10`. ROOT read the source, receipts and four candidate arrays; `root_array_qa.json` passes shape/hash/source/finite/unselected/norm checks. No endpoint benefit is inferred from this CPU result.

**Frozen next budget, before new endpoint scores:** Wav2Lip and Ditto × S0044/S0052 × `SEP_ANG_ACTIVE05`, four new renders total, matched canonical no-op baselines, same natural PCM/time/RMS and evaluator. Preserve the response gate `ΔCbest>0 && ΔDmin<=0 && ΔD0<=0`; report local mouth motion separately. This is exploratory reuse of the two speakers, not fresh confirmation. Only if all four rows pass can MuseTalk receive two additional rows; otherwise stop expansion of this candidate. No training, old four-center, Dev or Confirm is opened.

Interpretation is limited to this angle cap and this phone-label margin. The induced margins can exceed the small FIT TTS-minus-natural mean; a failed 0.05-rad candidate would close this candidate's expansion, not every smaller dose or every separation definition. Before opening its endpoint numbers, ROOT also counted current competitors after stripping only IPA tone marks: 11/68 token rivals share the same base-phone string (Ditto 6/18 and 1/16; Wav2Lip 3/18 and 1/16). These are not automatically distinct visual classes. This supplies a specific semantic boundary for future work; it is not an explanation assigned to a result and does not change the four-row gate or current candidates.

### Completed angular micro-pilot

All four prescribed rows are complete, with untouched original score PCM, 25 fps, matching no-op clocks, unchanged lag and valid tracking. Wav2Lip forward captures equal the candidate chunks; ROOT also read the actual Ditto dynamic runner snapshot (SHA `cf84ac09b2a8bc3c9442867c47a12abeb6eb4c3d9fa75d4dd2f68bb9854e304e`) and its direct `audio2motion_queue.put(feats)` injection. Ditto OS return codes and the second renderer PID were not persisted: retain them as null, do not infer rc0. Complete output/source/PCM/clock/tracking receipts and the score manifest support the endpoint screen; no repeat render was commissioned for that job-control gap.

| model | speaker | ΔCbest | ΔDmin | ΔD0 | Δaperture median | Δvelocity RMS | response gate |
|---|---|---:|---:|---:|---:|---:|---|
| Wav2Lip | S0044 | +0.006 | -0.159 | -0.142 | -0.011124 | -0.014491 | pass |
| Wav2Lip | S0052 | -0.170 | +0.083 | -0.094 | -0.004405 | -0.027660 | fail |
| Ditto | S0044 | +0.006 | -0.015 | -0.015 | +0.000680 | +0.006023 | pass |
| Ditto | S0052 | -0.080 | +0.161 | +0.161 | -0.002632 | -0.013442 | fail |

The four-row expansion gate fails. The same two speakers have concordant response signs across these two models, but neither model replicates improvement across speakers. Raising actual nearest-rival margin while fixing native norms and within-token geometry is therefore not a general enhancer at this dose. MuseTalk expansion is closed. Motion changes have different signs in the two S0044 passes and no universal size/velocity interpretation.

ROOT summary is `.../sep_ang_active05/root_four_row_summary.json` (SHA `357b987e1ab83380aaf0a556d4b4ecba2be475b98f9d38e78bdcc0c448bb7235`). Wav2Lip compact manifest is `.../sep_ang_active05/readback_wav2lip_attempt01/compact_readback_manifest.json` (SHA `d20309a6e8d1d8831fb045ae609eb22cb98beaace48f246ba59088d6cc25b36d`). Ditto compact root is `.codex/scratch/native_axis_sep_ang_active05_ditto_readback_20261014_attempt01/`, manifest SHA `dd5858fbafd28c5fca47d1c7927d514abbc9417ee7cef09c5b92a8bbdddaee05`. GPU was idle with no compute processes at UTC 12:58:17; no subsequent GPU task is admitted.

### Actual nearest-rival FIT refinement

The same-support CPU supplement now computes both fixed-rival and actual-nearest-other margins under the same LOSO natural anchors. Receipt `.codex/scratch/native_axis_actual_nearest_loso_20261014_attempt02/receipt.json` (SHA `c889116baa257d6f7779838c7e625f2599596ef6d7bab58b613ef2a3566b34d0`), executed source SHA `96db19e36be63d30f6e11d47876bb918ff7034918dbe62fa7208882a9ed2c664`, supersedes attempt01's incorrect token-count display. Paired eligible token-fold counts are 293/294/294.

| model | actual nearest-rival ΔTTS−natural | speaker SD | positive speakers | fixed-rival Δ |
|---|---:|---:|---:|---:|
| Wav2Lip | -0.000249 | 0.001278 | 4/9 | +0.000521 |
| MuseTalk | +0.003932 | 0.005981 | 6/9 | +0.002957 |
| Ditto | +0.003140 | 0.002189 | 9/9 | +0.001750 |

The fixed-prototype average direction is common; actual nearest-rival margin is not. N/T actual competitor labels change on 53.5%/46.6%/51.3% of paired tokens. This pinpoints a coordinate-definition boundary rather than erasing the earlier fixed-direction observation. Next work is CPU-only: strip tone marks to define a base-phone label, recompute actual nearest-other margins with equal-speaker LOSO support, and compare on common paired tokens. This is a segmental-label probe, not a claimed viseme mapping or a new GPU enhancement candidate.

## Existing motion localization

The CPU-only readback `.codex/scratch/native_axis_motion_localization_20261014_attempt02/receipt.json` (SHA `ed96c4c05b6fe56a76b4fa78eaa1ecbe2308d1f61b8cc1c7249d14a7d3fff019`) analyzes 12 Ditto/MuseTalk factor rows. It compares absolute aperture/velocity deltas on changed frames, a 40 ms adjacent shell and outside, and joins lag-0 SyncNet queries to their five-frame visual windows. Changed frames map to known phones and paired timestamps align. Positive-gain windows are often target-overlapping (63.3–90.9%), but target support covers 47–62% of frames and the windows overlap broadly, so this is not target specificity. Same-window Spearman gain versus aperture/velocity change is weak and variable (about `-0.06…+0.33` / `-0.01…+0.28`). No p/b/m token is present. Read `summary_rho_corrected.md`; the older summary has a display-format bug. This shows measurable motion responses without a stable local score-motion relationship or an articulation-accuracy label.

## Superseded round-mask GPU engineering packet (2026-10-14)

The clean score packet is `.codex/scratch/native_axis_pilot_score_20261014_attempt02/` with source-script SHA `ea73b4f3841bf6b026c5d834d8854b441022cf6d8447f5e9d17549d2385dc9a2`, 16 rows, raw-original PCM, 224x224 crops, official SyncNet matrices, and independent MediaPipe tracks. Local readback is under `.../local_readback_attempt01/`; compact row data is `compact_summary.json`. Raw-original PCM SHA is `7ddcdeb90ef2...` for S0044 and `451711ed1799...` for S0052. All visual rows have 100% valid tracking; frame counts are Ditto 93/103 and MuseTalk 92/102, with 87/97 SyncNet queries.

For each factor below, deltas are factor minus same-model/speaker no-op. The response gate is `Delta Cbest > 0`, `Delta Dmin <= 0`, and `Delta D0 <= 0`; this is a screening gate, not a statistical claim.

| model | speaker | factor | ΔCbest | ΔDmin | ΔD0 | lag Δ | Δvelocity RMS | Δaperture median | response gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Ditto | S0044 | SEP | -0.113 | -0.028 | -0.028 | 0 | +0.0063 | +0.0023 | fail |
| Ditto | S0044 | STAB | -0.130 | +0.118 | +0.118 | 0 | -0.0069 | -0.0023 | fail |
| Ditto | S0044 | ENV | +0.031 | -0.056 | -0.056 | 0 | -0.0176 | +0.0026 | pass |
| Ditto | S0052 | SEP | +0.003 | +0.078 | +0.078 | 0 | +0.0108 | -0.0029 | fail |
| Ditto | S0052 | STAB | -0.338 | +0.360 | +0.360 | 0 | -0.0298 | +0.0006 | fail |
| Ditto | S0052 | ENV | -0.023 | +0.094 | +0.094 | 0 | -0.0228 | +0.0028 | fail |
| MuseTalk | S0044 | SEP | +0.009 | +0.031 | +0.030 | 0 | +0.0072 | -0.0003 | fail |
| MuseTalk | S0044 | STAB | -0.245 | +0.224 | +0.010 | 0 | -0.0120 | -0.0001 | fail |
| MuseTalk | S0044 | ENV | +0.036 | +0.016 | -0.000 | 0 | -0.0054 | -0.0001 | fail |
| MuseTalk | S0052 | SEP | +0.041 | +0.011 | +0.081 | 0 | +0.0023 | -0.0005 | fail |
| MuseTalk | S0052 | STAB | -0.194 | +0.306 | +0.009 | 0 | -0.0049 | -0.0005 | fail |
| MuseTalk | S0052 | ENV | +0.015 | +0.151 | -0.090 | 0 | -0.0093 | +0.0002 | fail |

This packet is retained only as historical engineering evidence. Its round-mask rows are superseded by the corrected canonical screen above and must not be counted as scientific factor cells. The old apparent Ditto S0044 ENV pass, and the old MuseTalk C/D pattern, therefore do not support a factor claim. Aperture and velocity deltas remain descriptive and do not label articulation accuracy.

The design review records two operator boundaries: SEP's intended projection is algebraically centered with within-phone residual ratio 1, but empirical nearest-rival movement can be nonpositive and feature energy changes by roughly 0.974–1.022; STAB preserves centroids and has residual variance ratio 0.5625, while target energy is about 0.865–0.979. These are factor operators with declared collateral, not perfectly isolated physical coordinates. The corrected canonical rerender binds the actual float32 tensors and interval-ownership masks.

The CPU audit quantified the former mismatch (same TextGrid for MuseTalk and Ditto): S0044 SEP rounded/canonical target frames `57/57` with XOR 4; S0044 STAB `57/54` with XOR 5; S0052 SEP `61/57` with XOR 4; S0052 STAB `64/61` with XOR 7. Phone assignment can differ even where union counts match. The canonical rematerialization and rerender are now complete; the old round-mask cells stay engineering-only.

## Executed STAB contrast

- S0044: Cbest `6.996809` vs `6.958532` (`+0.038277`), Canchor `+0.245365`, D0 `-0.470227`, Dmin `-0.263139`, B `-0.224862`; lag/offset stayed `+3/-3`. Mouth velocity RMS changed `-0.009740` from the no-op baseline.
- S0052: Cbest `6.332500` vs `6.329837` (`+0.002663`), Canchor `+0.072287`, D0 `-0.038471`, Dmin `+0.031153`, B `+0.033815`; lag/offset stayed `+3/-3`. Mouth velocity RMS changed `-0.017112` from baseline.
- Algebraic purity: known-phone mel columns only, excluded intervals unchanged, retained residual fraction `0.75`, centroid shifts `2.97e-6` and `2.16e-6`; exact source PCM lineage restored for S0052. The S0052 WAV container header differs from its historical baseline header, but decoded float32 PCM matches the authoritative hash and is recorded separately.
- Canonical remote run manifest: `/dev/shm/cross_tfg_native_axis_injection_pilot_20261013_attempt01/stab_w2l_operator_manifest.json`, updated SHA `db4d69e6…`; CPU attempt04 binds the exact render adapter SHA `3bdd7aeb…` and preserves attempt03 as stale-source provenance. Paired local readback is under `.codex/scratch/native_axis_injection_pilot_20261013_attempt01/local_readback_attempt01/`.

## FIT calibration boundary

The existing FIT-only archives did not contain a frozen STAB standard-deviation receipt. The Wav2Lip archive `runs/cross_tfg_audio_head_20261003/native_encoder_wm_archive_readback_attempt01/restored_attempt01/metadata/archive_manifest.json` (SHA `0d73826e…`) contains encoder-feature NPZs but no replayable 80 Hz mel grids; the full-context cache `runs/cross_tfg_audio_head_20261003/fit_v4_native_fullcontext_readback_attempt02/artifacts/baseline_cache/baseline_cache_manifest.json` (SHA `a65821c0…`) has frame indices/RMS curves only. A new CPU MFA alignment regenerated 40 TextGrids for the S0052–S0061 WAV/`.lab` corpus under `.codex/scratch/native_axis_fit_mfa_20261014_attempt01/output/` from the pinned `mandarin_china_mfa`/`mandarin_mfa` environment. This is a new calibration source; it does not retroactively qualify the earlier STAB rows, and S0052 is excluded from calibration to avoid pilot leakage.

## CPU FIT calibration result

The initial retry `native_axis_fit_mfa_20261014_attempt02_retry01` is preserved as an exploratory CPU-only calibration because its flat MFA corpus logged one speaker. A corrected speaker-partitioned run `native_axis_fit_mfa_20261014_attempt02_retry02` is PASS: ten independent speaker partitions (S0052–S0061), 40 TextGrids, all ten MFA commands return 0, MFA v2.2.17, `mandarin_mfa` acoustic asset SHA `c4c91c49…`, `mandarin_china_mfa` dictionary asset SHA `0881d4e3…`, and exact per-speaker commands recorded in `mfa_receipt.json`. The corresponding calibration `native_axis_fit_mfa_20261014_attempt02_retry02_calibration` is CPU-only and PASS on S0053–S0061: 18 utterance pairs, 36 condition rows, 631 natural plus 631 TTS temporal segments, 6 short known segments retained in the segment map but excluded only from temporal metrics, endpoint error 0 samples, and known-phone coverage 1.0. It uses the current Wav2Lip source pins (`audio.py` `c282d0bd…`, `hparams.py` `4a4c31dc…`, `inference.py` `42cfc8d3…`) and speaker-equal SD with `ddof=1`. The corrected calibration receipt is `.codex/scratch/native_axis_fit_calibration_20261014_attempt02_retry02_calibration/calibration_receipt.json` (SHA `00b16c4a…`); metric SHA is `2d947138…`, native mel array SHA is `7db9a927…`, segment-map SHA is `33ffa6cd…`, and centroid SHA is `22ce8f59…`. The original short-segment failure remains preserved separately.

The calibration supports an exploratory Wav2Lip coordinate table: 58 of 84 phones have at least three independent FIT speakers for SEP prototype/rival estimates; the median rival cosine distance is `0.007187` and median half-dose within-phone residual SD is `0.5010`. The eventual pilot labels are only partially covered: S0044 has 20/22 unique phones supported and S0052 has 17/20; unsupported labels remain identity. Speaker-equal natural reference means are centroid norm `17.3696` (SD `1.0910`), adjacent-cosine cost `0.013237` (SD `0.002413`), frame-norm CV `0.094706` (SD `0.009709`), and RMS residual `0.385002` (SD `0.032042`). TTS values are descriptive only: `17.9958`, `0.012612`, `0.093926`, and `0.389918`. ENV is `UNAVAILABLE` because the conservative local-envelope direction is inconsistent (583 grids, sign agreement `0.5283 < 0.60`); no ENV direction is fabricated.

The corrected calibration closes the Wav2Lip FIT source/provenance gate for an exploratory intervention: 58 of 84 phones have at least three independent FIT speakers for SEP prototype/rival estimates; the median rival cosine distance is `0.007298` and median half-dose within-phone residual SD is `0.513903`. The eventual pilot labels are only partially covered: S0044 has 20/22 unique phones supported and S0052 has 17/20; unsupported labels remain identity. Speaker-equal natural reference means are centroid norm `17.3822` (SD `1.079927`), adjacent-cosine cost `0.013233` (SD `0.002413`), frame-norm CV `0.095795` (SD `0.010204`), and RMS residual `0.386635` (SD `0.032038`). TTS values are descriptive only: centroid norm `17.9945`, adjacent-cosine cost `0.012625`, frame-norm CV `0.094123`, and RMS residual `0.390109`. ENV remains `UNAVAILABLE` because the conservative local-envelope direction is inconsistent (583 grids, sign agreement `0.5300 < 0.60`); no ENV direction is fabricated. This is still a calibration/coordinate result, not a SyncNet or replacement result, and it does not qualify MuseTalk/Ditto coordinates. The create-only binding receipt `.codex/scratch/native_axis_fit_calibration_20261014_attempt02_retry02/retry02_binding_receipt.json` binds the calibration to the MFA receipt/provenance hashes and exact commands without modifying arrays.

The earlier STAB result remains `NO_COORDINATE`/`SCORE_RESPONSE` evidence rather than a retroactively qualified factor. Before any GPU row, complete the same low-cost CPU calibration for MuseTalk and Ditto using the corrected TextGrid bundle. If those coordinate gates are closed, the next GPU budget is one explicitly bounded Wav2Lip SEP arm on the existing two-speaker cohort with a fresh baseline/receipt; no second-TFG STAB or ENV render is authorized until its own coordinate calibration is closed.

## Cross-TFG CPU calibration result

The bounded CPU-only run `native_axis_musetalk_ditto_fit_calibration_20261014_attempt03` is PASS for both native frontends: 18 utterance pairs, 36 condition rows, 9 FIT speakers (S0053–S0061), corrected retry02 MFA dependency PASS, no renderer/SyncNet/tests, and GPU utilization remained 0%. MuseTalk uses its pinned Whisper/audio sequence (`[T,50,384]`, 25 fps floor clock); Ditto uses the pinned HuBERT/ONNX sequence (`[T,1024]`, source `ceil` frame rule and explicit CPU provider). The first Muse import guard failure and the first Ditto frame-rule guard failure remain archived as engineering receipts; the final run corrected both without weakening dependency guards.

MuseTalk has 53/79 known phones eligible for SEP prototypes. Its speaker-equal natural reference means are centroid norm `118.4600`, adjacent-cosine cost `0.177417`, frame-norm CV `0.003579`, and RMS residual `0.326769`; descriptive TTS values are `119.8225`, `0.178629`, `0.004061`, and `0.322959`. Ditto has the same 53/79 SEP support; natural reference means are centroid norm `9.3290`, adjacent-cosine cost `0.173334`, frame-norm CV `0.018306`, and RMS residual `0.107871`; descriptive TTS values are `9.3420`, `0.185897`, `0.016744`, and `0.108699`. MuseTalk and Ditto ENV are both `UNAVAILABLE` under the current diagnostic envelope gate (sign agreement `0.5566` and `0.5118` respectively); these values do not establish a causal ENV coordinate. The native directions are not numerically interchangeable: Wav2Lip shows a small TTS centroid increase with slightly lower adjacent cost, MuseTalk a larger centroid increase with slightly higher adjacent cost, and Ditto near-zero centroid change with higher adjacent cost. This is evidence for model-native coordinate differences, not a shared factor result.

The CPU calibration closes the model-entry qualification boundary. It does not include SEP/STAB/ENV interventions, SyncNet scores, or mouth-motion measurements. The SEP_RD CPU packet now passes, so the next bounded action is at most two Wav2Lip SEP_RD renders; MuseTalk/Ditto renders remain stopped until the two-row response and motion gate is read.

## MuseTalk/Ditto ENV coordinate candidate

A separate CPU-only candidate builder applied the frozen TTS-minus-natural local normalized log-norm trajectory to the native feature sequences. MuseTalk and Ditto each have 49 available phone coordinates and 28 unavailable phones because paired support was below the required speaker threshold. The corrected attempt03 packet marks the available coordinates `purity_status=PURE`; unit-direction maximum error is at most `3.33e-16`, total feature-energy relative error at most `3.80e-16`, and excluded-frame change is zero. The candidate therefore provides a valid model-native ENV coordinate packet for future intervention, while preserving the old same-sign slope diagnostic as a separate descriptive field. It is not a rendered factor result; no GPU, SyncNet, or mouth-motion run was performed. Protocol and summary are `.codex/scratch/native_axis_env_coordinate_candidate_20261014_attempt03/env_coordinate_protocol.json` (SHA `4e89e1cfe1e4cd1b…`) and `env_coordinate_summary.json` (SHA `72fcebfe56bdecb2…`); attempt02 remains preserved.

## Wav2Lip SEP CPU qualification

The frozen prototype-to-rival directional operator was evaluated CPU-only on S0044 and S0052. Native mel shapes remained `[80,296]` and `[80,330]`, float32/finite/source/RMS/clock contracts passed, excluded and unsupported labels stayed unchanged, and off-target mean deltas stayed below two FIT SDs. The directional projection moved by `0.500000 SD` on both rows within float tolerance, but the actual centroid-to-nearest-rival distance moved only `+0.104493 SD` for S0044 and `+0.237511 SD` for S0052; that older packet is retained as a failed coordinate attempt. The separately named distance-directed SEP_RD attempt03 then passed CPU-only: actual rival-distance movement is `0.499999997/0.500000004 SD` for S0044/S0052, off-target mean STAB/frame-CV deltas remain within the two-SD limits, excluded/unsupported masks are unchanged, and exact renderer chunk counts are 89/100. The receipt SHA values are `10669b15…` and `aa0c0c6d…`; script SHA is `5a89756d…`. This qualifies only a native coordinate packet. No GPU render, SyncNet score, or mouth-motion measurement has yet been run.

## CPU native-frontend rebind

The current source bundle was rebound on the remote host for the same S0044/S0052 inputs. Wav2Lip CPU mel/chunk extraction passed for both rows in `.codex/scratch/native_axis_cpu_wav2lip_receipt_20261014.json` (S0044 `[80,296]` mel and `[89,80,16]` chunks; S0052 `[80,330]` and `[100,80,16]`, finite, fixed RMS, known-phone coverage 95.357143%/84.905660%). MuseTalk CPU extraction passed in `.codex/scratch/native_axis_cpu_frontends_probe_20261014_attempt02/` using the recovered S0052 PCM hash `17a6809b…`: S0044 conditioning `[92,50,384]`, SHA `c35630ea…`; S0052 `[102,50,384]`, SHA `c6e96b39…`; both finite and CPU-only. Attempt01's S0052 `d936…` PCM is engineering-only and excluded from scientific comparisons.

Ditto attempt04 exposed the native provider binding: the pinned config forces `cuda/ori` even with an empty CUDA mask. A scratch-only adapter copied the parsed nested config, set `device=cpu` and `force_ori_type=false`, and preserved the exact config/ONNX/source hashes; it did not modify tracked Ditto source or scientific inputs. Authoritative attempt06 then passed Ditto CPU ONNX extraction for both rows: S0044 `[93,1024]`, conditioning SHA `49141d2f…`; S0052 `[103,1024]`, SHA `7f35a40b…`; finite, CPU, `model_type=onnx`, and explicit `CPUExecutionProvider` route. Attempt04 remains preserved as the original provider-binding failure. No factor intervention, render, SyncNet, or visual measurement was run in this structural stage.

## Wav2Lip SEP_RD two-row GPU result

The bounded GPU follow-up used only S0044 and S0052, the final CPU SEP_RD packet, the original natural PCM for scoring/mux, and the exact 25 fps native clock. Both renders passed source, RMS, mask, frame-count, mux, SyncNet execution, and MediaPipe tracking quality gates (89/89 and 100/100 valid frames). The scientific response gate failed: S0044 had `Delta Cbest=-0.080363`, `Delta Dmin=+0.019251`; S0052 had `Delta Cbest=+0.074784`, `Delta Dmin=-0.092860`; best lag and official AV offset stayed `+3/-3` on both. Mouth velocity RMS deltas were negative on both rows (`-0.00364849`, `-0.00821861` IPD/s), so the no-regression motion endpoint also failed. The final packet is `.codex/scratch/native_axis_injection_pilot_20261013_attempt01/sep_rd_gpu_readback_attempt01/sep_rd_two_row_gate_attempt01.json` (latest SHA `972967be…`). This is a clean negative/heterogeneous result for Wav2Lip SEP_RD, not evidence that the coordinate or SyncNet pipeline is broken; no further Wav2Lip SEP_RD rows are authorized.

## MuseTalk/Ditto ENV one-speaker preflight blocker

The first one-speaker ENV preflight was stopped at dependency resolution. The pinned autoavsr Python runtime has torch but no transformers; the pinned mfa runtime has transformers but no torch; mixing their site-packages produces a NumPy C-extension incompatibility. The receipt `.codex/scratch/native_axis_one_speaker_env_preflight_20261014_attempt01_blocked_receipt.json` is `DEPENDENCY_BLOCKED` (SHA `512be1cc…`), with `operator_executed=false`, renderer/GPU/SyncNet/MediaPipe/tests all false. Existing MuseTalk/Ditto native frontend pins remain valid; a NumPy-only replay over already extracted arrays is the final low-cost check before closing the ENV branch.

The NumPy-only replay `native_axis_numpy_only_env_preflight_20261014_attempt01` closes that dependency gap for feature-only purity. MuseTalk and Ditto natural/TTS rows are finite; RMS absolute error is `6.58e-12/9.95e-12`, 25 fps boundary/overlap and relative-position checks pass, excluded masks are unchanged, and the maximum unit-direction/full-energy errors are `3.83e-9/8.04e-11` for MuseTalk and `1.10e-8/2.79e-10` for Ditto. Receipt SHA is `0df85b1b…`, summary SHA `d475346e…`. No renderer, SyncNet, MediaPipe, or test ran in this preflight.

## MuseTalk/Ditto ENV two-model causal result

One common supported utterance was rendered with ENV and no-op conditions for MuseTalk and Ditto (four renders total). All renders used the same natural PCM for generation and scoring, exact 25 fps timing, and complete MediaPipe tracking. The response gate failed in both models. MuseTalk ENV-minus-no-op was `Delta Cbest=-0.252169`, `Delta Dmin=+0.277637`, `Delta D0=+0.127375`; Ditto was `-0.106188`, `+0.063300`, `+0.063300`. Mouth tracking quality was complete, but motion-consistency gates failed: aperture velocity RMS changed by `+0.028210` for MuseTalk and `+0.004887` for Ditto under a no-regression criterion, and the summary reports `motion_consistent=false` for both. Final compact summary is `.codex/scratch/native_axis_env_gpu_causal_rows_20261014_attempt01/causal_summary.json`; run receipt SHA is `00937b88…`. The packet recommends no further rows.

## Base-phone result and current stability-isolation work

Base-phone LOSO supplement is complete: .codex/scratch/native_axis_base_phone_loso_20261014_attempt01/receipt.json SHA 0389cc386b2b1b582944eb4df488421a8160584e7859b906aa375a19423d5336. Removing only tone marks gives actual-margin means Wav2Lip -0.000275 (4/9 positive), MuseTalk +0.001784 (5/9), Ditto +0.002915 (8/9), on 334/335/335 eligible paired fold-tokens. Tone-aware comparisons use a common 293/294/294-token subset, with base common-support results separate in the receipt. Pooling tones changes the own prototype as well as competitors; it is not merely deleting tone rivals or defining visemes. It supplies no common three-model positive direction and admits no further SEP GPU candidate.

Current CPU-only next step closes the stability-isolation gap: STAB_ANG_FIT in Wav2Lip/MuseTalk, whose FIT angular dispersion fell. Rounded preset mean-angle reductions are 0.004/0.008 rad, from FIT scale, not endpoint selection. Slerp unit vectors toward the original token direction, then orthogonally recenter their unit centroid; reconstruct original per-frame norms and preserve actual unit-centroid nearest-rival margin. Float32 direction/margin drift must be <=1e-6, unsupported/short/unowned frames identity, with boundary collateral disclosed. Ditto's observed stability direction is opposite, so it is excluded. Four CPU candidates are complete and ROOT-reviewed against source hashes and actual float32 arrays; source SHA `42a6e69b49fdf8f8609324e84afb65ba042436709bff2389e05168d44a75cd06`, receipt `13f0e8dc6b49ef98412ebbda09289da73743da3526fa9bfe2609adb3e46fd2cf`, ROOT QA `55d732257e3dc25abbc67b2576881c98f7c19d222993e012f9284a60429227aa`. All 18/16 supported tokens apply in each model; maximum native norm relative drift 9.67e-9, centroid direction drift 1.96e-8, nearest-rival margin drift 7.86e-10, unsupported frames bitwise identity. Boundary collateral max 0.292 degrees is reported separately. Frozen admission `root_four_row_admission.json` SHA `f02c77c1fde62dab6109249583a3093e31fbd5f21d6998e737ec6a9921618b91` admits only four renders: Wav2Lip two rows first, then MuseTalk two rows after baseline-source equality and GPU idle. Baselines are reused; source/actual injection, fixed-RMS driver, raw natural PCM/25fps, Cbest/Dmin/D0 and MediaPipe endpoints remain bound. Both speakers must pass per model, both models for a transferable candidate; no n=2 inferential claim. No dose sweep, new cohort, training, Dev or Confirm.
### STAB_ANG_FIT current endpoints

Wav2Lip compact manifest `local_readback_wav2lip_attempt01/compact_readback_manifest.json` SHA `e1a1fd636dd80657b9fbed2b5fe526fec7a78babe88d9c52ef418fad482f3978`, ROOT raw-array/motion summary `root_wav_pair_summary.json` SHA `47217783f3e59520635c7def9f87cfd36e285203cef311c70fbe26c768595a1c`, both beneath the STAB CPU root. Both worker/render rc0; actual forward arrays match candidates, original-natural final PCM and 25fps retained, full tracking89/89 and100/100. ROOT recomputed Cbest/Dmin/D0 from raw query×lag matrices (lag0 column15) and aperture/velocity from track NPZs; max score readback difference3.34e-6.

| model | speaker | dCbest | dDmin | dD0 | d aperture median (IPD) | d velocity RMS (IPD/s) | response |
|---|---|---:|---:|---:|---:|---:|---|
| Wav2Lip | S0044 | -0.038 | -0.081 | -0.238 | -0.004311 | -0.014628 | fail |
| Wav2Lip | S0052 | +0.028 | -0.071 | +0.046 | -0.002649 | -0.082540 | fail |

Both fail the frozen joint gate. Lower aperture/velocity describes changed motion without an accuracy ordering. S0052 Cbest improves while original-clock D0 worsens; a score increase is not by itself improved timing. MuseTalk S0044/S0052 renders are complete (PIDs2478690/2479386, startticks46934378/46943806); nohup OS rc remains null, with completed prompt/output/clock receipts. Final StageA scoring must remux raw original PCM (S0044 `7ddcdeb9…`, S0052 `451711ed…`); renderer driver hashes `b59ec…`/`17a680…` are not final score PCM. Those two endpoint rows are now complete; final results follow. No expansion of the four-row budget.
MuseTalk compact artifact `readback_compact_attempt02/compact_readback.json` SHA `58759f95a17e1481ab8909310447fe46fb735433f57303460e5cfa991929e49e` includes candidate and no-op raw matrices/tracks, source/prompt/PCM/clock receipts. Candidate score manifest SHA `eca600f505b3c3b7982b2f6671ee61250232054f0b971556acf493e3b48bb135`. Scorer PID2480139 exited rc0, startticks unavailable; renderer OS rc remains null. UTC13:38:23 GPU0 util0/5MiB/no compute apps; no more GPU task is admitted.

| MuseTalk speaker | dCbest | dDmin | dD0 | d aperture median (IPD) | d velocity RMS (IPD/s) | best lag no-op→candidate | response |
|---|---:|---:|---:|---:|---:|---|---|
| S0044 | -0.046 | +0.071 | -0.067 | +0.000527 | +0.025070 | -2→-1 | fail |
| S0052 | -0.079 | +0.091 | -0.001 | +0.001875 | -0.010691 | -2→-2 | fail |

ROOT checked the compact science payload and independently recomputed both arms' query×lag and mouth arrays. `root_four_row_stability_summary.json` SHA `e56f5aa1106457716ae1bc79508005a65dd2fef8cb057dc36e0e71bdf4c42f2c` closes the four rows with zero joint passes. Raw float32 mean reductions differ from receipt scalar reductions by only a few e-6; displayed three-decimal values and gates agree. Stable unit features change physical mouth trajectories without a repeatable C/D benefit in these two TFGs at these fixed doses. The coordinate defines angular dispersion and radial norm; it does not isolate arbitrary physical spectral envelope, prove content preservation beyond its unit-centroid guard, or imply waveform realizability.

### Next research question, not a new GPU admission

Before inventing another low-dimensional direction, establish whether structured TTS-native angular information can transfer under natural time and natural native radii. The closest historical LRS3 Wav2Lip phone-aligned mel residual (30 source groups, alpha0.25) was negative and did not impose this norm control, so a new proposal would first use MuseTalk, where actual FIT margin/stability directions and prior explicit11 replacement evidence align. Candidate design: two current samples, structured phone-phase-mapped TTS unit directions vs within-phone phase-shuffled sham, restoring each natural frame norm and leaving unmatched/short phones identity; reuse natural no-ops and score raw natural PCM. Potential maximum is four new renders, not authorized yet. Existing two samples lack located TTS donor audio; Luna is only locating locked transcripts, natural references, existing Qwen/MFA/native runtimes and a private runtime-key boolean. The proposal itself admits no renderer or tensor construction. A separately frozen `next_asset_phase_admission.json` now admits only two Qwen TTS requests and at most two optional ASR checks/two voice enrollments, after locked spoken-text review, without new model download or training. Existing MFA/Muse CPU runtime may be prepared; further execution/support is frozen separately. The source script and private no-echo PTY are prepared by Luna; ROOT injects the previously authorized key at runtime, never into files/argv/BM/logs. S0052 official “三零胜六负” is specifically checked against existing spoken-text records before synthesis; this is a potential notation discrepancy, not an established error. User permission to use Qwen TTS already exists; missing assets or runtime key are operational inputs, not a request for new user permission. ROOT will decide the smallest worthwhile positive-control budget from that preparation.
Content check before donor creation is complete. `.codex/scratch/qwen_s0052_asr_check_20261014_attempt01/receipt.json` SHA `ad918f21017b099de30caca9c84a4ea08d77b5d55b48d534aae73c0943190e36` records exactly one Qwen-ASR Flash POST, retry0, HTTP200, enable_itn=false and the frozen S0052 original WAV hash. Raw response body SHA `24066d71877ee4f76dfaf82e369b033590361b23f29a40e95d3885fb4348239c` returns “此前的战绩是三零胜六负。”; only the terminal period is removed to lock the donor target. This agrees with official/MFA text; there is no observed digit-reading conflict and no claim of manual phonetic validation. Agent's waiting PTY was cancelled before any request because ROOT cannot access agent-scoped execution sessions. ROOT ran the same reviewed frozen ASR script using `F:/Apps/MiniConda/python.exe`, supplied credentials only at its no-echo prompt, and observed no credential echo. The earlier bundled-Python launch failed at requests import before prompting/calling; no request was spent. Luna now prepares the two donor request assets/provider/runtime only, including short-reference eligibility; no additional ASR or renderer is admitted.
### Coverage clarification and CPU-only missing-model follow-up

The canonical ENV4 rows consume the v2 PURE coordinate hashes directly (Muse a658b45a… and Ditto d9e4b990…). ROOT read all4 actual canonical arrays/source arrays and verified u-direction errors<=1.45e-8 and total-feature-energy error<=2.87e-10, with57/56 changed frames exactly matching recorded target counts. Receipt `native_axis_pilot_mask_audit_20261014_attempt02/root_env_radial_readback_qa.json` prevents interpreting the earlier CPU protocol renderer=false field as absence of later render endpoints. Frame-radius changes are intended ENV treatment. This remains radial/local-power geometry, not physical spectral envelope.

Remaining clean native model-factor cells are Muse SEP, Ditto STAB and Wav ENV. ROOT admits CPU-only construction of6float32 arrays, no GPU: reproduce SEP_ANG_ACTIVE05 for Muse; Ditto STAB0.008rad contraction with original radii/unit-centroid margin (counter-TTS-direction scientific control); separately named Wav ENV v2 from9speaker native chunks/paired11point phone log-norm trajectories, natural-source binding and total-energy compensation. The old Wav sign0.530 unavailable row remains historical; a different explicit coordinate is not retroactive qualification. Source and array gates precede any new renderer budget, maximum6rows if later admitted. This fills user-requested model coverage rather than changing the old four-row stop gate. All3 physical spectral-envelope cells remain missing and are not replaced by radial tests or the joint donor question.

Two-donor source inputs now exist at `.codex/scratch/qwen_independent_donors_20261014_attempt01/input_manifest.json` SHA1668c859d03febd4296ae1671815a2ad7dfcccc9b5731298f38e76229e6d94f3: independent same-speaker references W0161/W0495 at6.971/6.312s. Worker SHA906dabee… wasROOT-read and run once through hidden stdin. S0044 enrollment200, synthesis404 with InvalidParameter/Model not exist; no audio downloaded, S0052 never reached. The worker is terminalrc1, no secret echo; raw responses/metadata remain create-only. Do not rerun cloud_attempt01 or infer a scientific failure. Agent now researches a documented available Qwen clone interface/model and the minimal dependency/call cost beforeROOT admits an explicit version correction. No newcloud call/renderer is yet authorized for that repair.
## 缺格CPU候选实际准入与预算边界

ROOT已直接核验6个候选，不运行实现测试或GPU：24个manifest member hash/bytes、source/candidate file/raw SHA、float32/finite/shape、canonical TextGrid ownership、未选帧identity；独立复算Muse SEP actual-rival margin/Gram、Ditto单位中心/角离散度减少、Wav径向方向/总r²。create-only QA `native_axis_model_factor_scope_cpu_20261014_attempt01/root_actual_six_array_qa.json` SHA `254aafd5a4efaeb55701a790031bda4de511c7aa46ac79d40217773e7be137e8`。CPU receipt `0738b04b…`，manifest `b41c2b8c…`。Muse最小margin增量+0.013658/+0.008112；Ditto角离散度减少0.008rad误差≤1.87e-9，单位中心漂移≤2.09e-8；Wav单位方向误差≤3.84e-9，总r²相对误差≤3.07e-10。每行18/16token、48/52帧。

Wav径向v2是新预先定义的FIT坐标（9人/18句/409paired occurrences/49phone），48phone有任意grid处正负speaker混合；不是一致TTS方向证明，不改旧0.530017 sign门。Ditto STAB增加与observed TTS方向相反，只测稳定性增加的因果反应。新增缺格不进入旧四行stop门复算。

后续GPU上限6短render，仅Muse SEP×2→Ditto STAB×2→Wav radial ENV×2串行，复用baseline/frontend/MFA、原自然PCM/25fps。当前只有CPU科学资格；wrapper实际来源/source-to-noop和GPU owner确认后才ROOT launch准入。每对Cbest>0 && Dmin<=0 && D0<=0同时两人通过才有小独立复现资格；口部开合/速度独立描述，无人工质量标签、不自动扩矩阵。物理谱包络仍独立未测。

ADR037冻结旧HTTP404之后官方realtime最小修正：qwen3-tts-vc-realtime-2026-01-15，2新voice enrollment+2WS synthesis，无retry，同既有独立refs和锁定target text，raw PCM/WAV不改。仅隔离SDK/worker准备，无新cloud调用或donor renderer结果。

## Donor realtime实际终局与时间成本收敛

ADR037官方SDK worker SHA8065d47a…/manifest f7e8c9c7…已由ROOT读源码后在私有TTY执行，凭据getpass无回显/不入文件argv日志。实际rc1：S0044 enrollment HTTP200、1 WS session，只有session.created=1，无session.updated/response.audio/终局事件，等满240s后realtime_terminal_timeout，PCM0B；S0052未请求。该工程失败不证明模型不可用，也不作科学阴性。source/runtime/原raw response/receipt保留，未重试、未改运行中源码。SDK基类on_close为空，纯close/error不能唤醒当前worker Event；未自动重连。

ROOT将接口支线停放，不继续耗费预算修cloud。完整终局见 `.codex/scratch/qwen3_vc_realtime_donor_attempt01/outputs/root_terminal_readback.json`，run receipt SHA ce536307aa96023fe0043bbb7b3372a749ca00f5d18f17e0b751ffdbf8866e61，sample receipt SHA d3b9bc3bfe167a65cff05f8d865612559cd08a497cfee38981b869c9eb681bb4。现有9FIT配对WAV可供应phone条件物理谱形估计；目前Luna只确认36 WAV覆盖，ROOT尚未冻结该算子/GPU预算。此替代避免将target donor可用性当主研究依赖。

## 物理谱形CPU模板协议（未执行下游）

历史38局部donor/flux窗修补、Sep8 utterance平均ENV及WORLD SP均有物理控制，但不能直接供应当前两句候选或把其未过门当所有物理坐标的阴性。旧38仅local RMS且本地run缺失；Sep8仅result留存；WORLD bridge亦未等效。Luna有界查询已保留这些边界。现有9FIT说话人18对36 WAV均本地可读/16k monoPCM16，逐文件hash/bytes与MFA manifest f4c2b762…相等，无下载/重MFA。

ROOT冻结 `physical_phone_spectral_shape_fit_20261014_attempt01/protocol.json` SHA f951d47a3d9161945a846493f997a54278eb5b6af379be7c79e825496698a6ec：16k/STFT1024-hop256 periodicHann；log幅谱real cepstrum保留|q|≤32sample(2ms)平滑，再消除frame总power；token→utterance→speaker equal，paired TTS-N后speaker median，半signed median并clip±3dB，≥3独立FIT speaker/phone。不用target donor/score拟合，不强制方向。支持帧乘template并补偿STFT framepower，原complex phase/iSTFT原N，FLOAT32 WAV全局RMS固定，不peak scale；actual重分析phase和局部power通过OLA会变，须量化，不作严格原相位/局部能量保持主张。

CPU预算仅36已有FIT WAV、2candidate+2零剂量roundtrip，0native frontend/API/GPU/renderer/training/MFA/新依赖/实现测试。roundtrip及原长度/RMS/clip门实际通过才讨论物理renderer/sham预算；现在无此GPU准入。该协议为机制候选，不是可部署增强头成果。

Muse SEP pair wrapper02已经ROOT读diff/source-pin通过，admission `root_musetalk_sep_pair_admission.json` SHA11abd849…，只2render+原PCM score/mouth，旧baseline复用。agent收到单次launch授权，实际job信息另读，不凭授权声称已经运行。
