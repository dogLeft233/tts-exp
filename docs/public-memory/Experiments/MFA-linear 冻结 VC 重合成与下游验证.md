---
title: MFA-linear 冻结 VC 重合成与下游验证
type: experiment
permalink: tts-exp/experiments/mfa-linear-冻结-vc-重合成与下游验证
thread_id: 596d4aad-dd9d-48b1-8585-6ca75a768641
status: concluded
date: '2026-08-12'
tags:
- vc
- mfa-linear
- wavlm
- knn-vc
- wav2lip
- syncnet
- valid-only
- experiment
---

# MFA-linear 冻结 VC 重合成与下游验证

## 背景与目标

目标是在不先训练项目专用 VC 模型的前提下，生成一类新音频：保留配对 TTS 的连续音素/声学轨迹，同时把节奏、音素时长和停顿放到自然语音的时间网格上。为避免重复 phone-local waveform warp 的失败，本轮不切分、拉伸和拼接波形，而是在特征空间映射时间，再用冻结声码器重合成。

本轮只使用 valid speaker `S0765`；heldout speaker `S0770` 未被加载、选择、调参或评估。当前结论均为 valid-only 探索性证据。

## 方法

固定使用 `bshall/knn-vc` revision `c616845c4e309e24d5927f15adbdf277a3d65358`：

- WavLM-Large layer 6，1024 维，约 20 ms 帧移；
- 配套 frozen prematched HiFi-GAN；
- MFA 提供自然/TTS 两侧 phone、word、silence 区间；
- `MFA-linear` 在每个匹配 phone 内保留 TTS 连续特征轨迹，以 phone 内相对位置插值到自然 phone 持续时间；
- 输出为 16 kHz，严格裁剪或右补零到自然音频样本数，不做 loudness normalization。

MFA 只是映射时间网格的来源，不代表输出波形的真实 phone 边界已被验证；若要声明边界正确，仍需重新 forced alignment。

## 冻结 kNN-VC 初筛（n=10）

WavLM 合成空间中，以 natural→aligned-TTS 距离 `0.206188` 为基线（越低越好）：

| 条件 | 输出→aligned-TTS | 相对变化 | 改善数 |
|---|---:|---:|---:|
| natural direct resynthesis | 0.283867 | -37.67% | 0/10 |
| unconstrained kNN | 0.202080 | +1.99% | 6/10 |
| same-phone kNN | 0.231101 | -12.08% | 1/10 |
| same-phone + position | 0.229974 | -11.54% | 2/10 |
| **MFA-linear** | **0.143052** | **+30.62%** | **10/10** |
| wrong-phone control | 0.311138 | -50.90% | 0/10 |

独立 HuBERT L6（768 维）审计也支持 MFA-linear：combined distance 从 natural→TTS 的 `0.1885608` 降到输出→TTS 的 `0.1655008`，约改善 `11.9%`，10/10 改善。

判断：same-phone kNN 是 No-Go；离散检索破坏 phone 内连续轨迹和上下文。MFA-linear 是唯一在两个表示空间都稳定改善的条件。phone-local waveform warp 仍不是可靠 oracle，因为会引入 phase、transient、边界拼接及 contextual SSL geometry 问题。

## Wav2Lip + SyncNet 下游验证（n=15）

固定前 15 个 valid 样本，每个样本三臂：

1. `natural_raw`：已有 `natural_noop` 16 kHz 评估链音频；
2. `raw_tts`：已有 `tts_noop` 16 kHz 评估链音频；
3. `mfa_linear`：新生成的自然时钟/TTS 特征重合成音频。

三臂使用相同 sample、paired key、reference face、Wav2Lip checkpoint、SyncNet V2 model 和 `min_track=50`。30 个 natural/TTS baseline 分数经 audio/face/model/path/hash 逐项核验后复用；15 个 MFA-linear 视频与分数新生成。最终 45/45 finite scores、0 failures。

| 条件 | Sync-C ↑ | Sync-D ↓ | AV offset 均值 |
|---|---:|---:|---:|
| natural | 5.758133 | 7.269000 | -2 |
| raw TTS | 6.049133 | 7.192667 | -2 |
| **MFA-linear** | **6.722400** | **6.915333** | **-2** |

### MFA-linear 对 natural

- ΔSync-C `+0.964267`，bootstrap 95% CI `[+0.650133, +1.265867]`，13/15 改善；paired t `p=4.025e-05`，Wilcoxon `p=0.000305`。
- ΔSync-D `-0.353667`，95% CI `[-0.574000, -0.123000]`，12/15 改善；paired t `p=0.01057`，Wilcoxon `p=0.01458`。
- 两指标联合改善 12/15。

### MFA-linear 对 raw TTS

- ΔSync-C `+0.673267`，95% CI `[+0.318800, +1.024333]`，12/15 改善；paired t `p=0.00315`，Wilcoxon `p=0.00427`。
- ΔSync-D `-0.277333`，95% CI `[-0.510400, -0.061333]`，11/15 改善；paired t `p=0.03304`，Wilcoxon `p=0.03813`。
- 两指标联合改善 10/15。

当前是 **下游 Go**：在这套冻结、固定、valid-only 的 Wav2Lip/SyncNet 设置中，MFA-linear 在特征空间与下游指标上均成功。但不能外推为 heldout/跨数据集泛化、精确输出 phone 边界、可懂度/音质/主观偏好改善，或所有 talking-face 模型都会受益。`raw_tts` 是评估链的 `tts_noop` 16 kHz 音频，不是原始 24 kHz TTS。

## 既有稳定证据及其边界

这些完整、固定 cohort 能独立支持“中文 raw TTS 往往比自然音频更利于唇形同步”，但**不能独立证明 MFA-linear**：

### AISHELL-1 Ditto，n=100

`basic-memory/docs/experiments/15-tfg-link.md` 记录固定参考帧、同文本配对的 100 natural + 100 TTS：

- natural Sync-C `5.067`，TTS `6.089`，Δ `+1.022`；
- `p<0.0001`，Cohen's `d=1.24`，bootstrap CI `[0.86, 1.18]`；
- TTS 的 Sync-C 在 92/100 样本更高；
- Sync-D 仅 Δ `-0.069`，`p=.37`，不显著；两指标联合改善为 51%。

逐样本数据：`results/aishell100_phoneme/tfg_link/syncnet_100.json`。这是当前最强的大样本一般性 TTS 优势证据，但模型是 Ditto，且没有 MFA-linear 臂。

### AISHELL-1 跨 talking-face 模型，固定 n=13

`basic-memory/docs/results/SYNC-C-SCORES.md` 与 `basic-memory/docs/experiments/02-tfg-cross-model.md` 记录同一固定 cohort：

- Wav2Lip：`7.393 → 8.922`，ΔSync-C `+1.53`；
- IMTalker：`6.235 → 7.369`，Δ `+1.13`；
- Ditto：`5.670 → 6.819`，Δ `+1.15`；
- V-Express `+0.53`，MuseTalk1.5 `+0.54`，JoyVASA `+0.66`，EchoMimicV2 `+0.19`；
- Hallo2 与 LatentSync 均 `-0.07`；整体 7/9 模型为正。

该表证明现象跨多个架构出现，但 n=13，未记录各模型的样本级正向数、p 值或效应量；部分模型评测协议也有额外 caveat。因此它是一般性支持，不是 MFA-linear 的独立复现。

### 其他对照

- `basic-memory/docs/experiments/01-tts-tfg-baseline.md`：旧 Ditto n≈9–10 严格基线中 raw TTS ΔSync-C `+1.25`，`p=.0005`。
- `basic-memory/docs/experiments/16-phone-local-warp-validation.md`：固定 n=10 中 raw TTS 相对 natural `+0.4175`、7/10 为正，但 phone-local waveform warp 相对 natural `-0.4877`、仅 2/10 为正。这同时支持 raw-TTS 优势存在和 waveform warp 路线失败。

## 关键工程陷阱

`third_party/Wav2Lip/inference.py` 固定使用相对 `temp/temp.wav`、`temp/result.avi`、`temp/faulty_frame.jpg`。多样本或并发共享 cwd 会造成 temp 污染，即使返回码为 0 也可能生成错误来源的视频。

首轮受影响结果已全部丢弃。最终有效结果从零重跑：每个样本独立使用 `wav2lip_work/mfa_linear/<sample_id>/temp/`，并用绝对路径调用脚本和输出视频。今后必须保持每个 cell 独立 cwd/temp。

## 产物位置

### MFA-linear 音频

- `runs/knn_vc_poc_valid15_mfa_linear_20260812/audio/*__paired_tts_mfa_linear.wav`
- 汇总：`runs/knn_vc_poc_valid15_mfa_linear_20260812/summary.json`

### natural / raw-TTS 对照音频

- `runs/rhythm_timing/20260812_counterfactual_valid50/audio/*_natural_noop.wav`
- `runs/rhythm_timing/20260812_counterfactual_valid50/audio/*_tts_noop.wav`
- 汇总：`runs/rhythm_timing/20260812_counterfactual_valid50/summary.json`

### 原始源音频

- natural：`results/rhythm_style_500/aishell1_test_400/natural/`
- TTS：`results/rhythm_style_500/aishell1_test_400/tts/`

### 最终有效视频与结果

- 视频：`runs/rhythm_timing/20260812_syncnet_valid15_mfa/wav2lip/mfa_linear/{1..15}.mp4`
- manifest：`runs/rhythm_timing/20260812_syncnet_valid15_mfa/manifest.json`
- 逐项分数：`runs/rhythm_timing/20260812_syncnet_valid15_mfa/summary.json`
- 配对统计：`runs/rhythm_timing/20260812_syncnet_valid15_mfa/analysis.json`
- 独立工作目录：`runs/rhythm_timing/20260812_syncnet_valid15_mfa/wav2lip_work/mfa_linear/{1..15}/`

## 下一步证据计划

1. 不按历史得分挑样本；把 MFA-linear 扩展到预先定义的完整 50 个 valid `S0765` 样本。natural/TTS 50 条 baseline 已存在，只需补 35 个 MFA-linear 音频、视频和 SyncNet cell。
2. 方法、checkpoint、插值、长度和统计冻结后，再测试 heldout speaker 或另一完整固定 cohort。
3. 增加输出重新 MFA 对齐、ASR/PER、音质指标和主观听测。
4. 在第二个 talking-face 模型上做完整配对复验；历史 raw-TTS 证据不能替代 MFA-linear 的独立复制。

## AISHELL-1 预定义多说话人复核（n=25）

为检验单 speaker n=15 结果是否可跨 speaker，冻结了 5 个 speaker × 每 speaker 5 条 pair：`S0765`（valid）以及 `S0901/S0906/S0912/S0913`（train）。`S0770`、`S0914`、`S0915` 未进入 cohort；选择按固定 speaker 列表和 paired_key 前缀排序，不读取 SyncNet 分数。

严格有效链路产物位于 `runs/aishell1_mfa_linear_n25_20260813/`：

- cohort：`data_boundary/aishell1_n25_speaker_balanced_cohort.json`
- TTS：`tts_strict/tts_meta.json`（严格源 TTS canonical 到 16 kHz）
- tokens：`mfa/tokens.json`（冻结 strict MFA source tokens）
- MFA-linear：`mfa_linear_strict/summary.json`
- 下游：`eval_strict/summary.json`、`eval_strict/analysis.json`

本地 MFA 2.2.17 对新 staging 音频的 phones tier 全部输出整段 `spn`，speech-phone gate 正确拒绝；因此没有使用这些无效 TextGrid。最终使用严格源 manifest 中已通过独立 MFA 门控的 natural/TTS tokens，并显式记录本地重跑失败。WavLM 最后 frame 可能落在 MFA 末端后约一个 hop 的尾部，生成器对最后 token 做了显式、可审计的 feature-tail extension。

严格版本 75/75 fresh Wav2Lip+SyncNet cells 完成、0 failures；所有音频/视频 SHA、固定脸 SHA、模型 SHA、`min_track=50` 通过校验。固定单参考脸协议仅是统一视觉输入的 exploratory audio-condition 对照，不是 speaker/video 泛化。

| arm | Sync-C | Sync-D | AV offset |
|---|---:|---:|---:|
| natural_raw | 5.98816 | 7.26468 | -2.00 |
| raw_tts | 6.05640 | 7.23668 | -2.00 |
| mfa_linear | 5.91896 | 7.33340 | -2.08 |

MFA-linear vs natural：Sync-C Δ=`-0.0692`，5-speaker cluster bootstrap 95% CI `[-0.6521,+0.5137]`，11/25 positive；Sync-D Δ=`+0.0687`，CI `[-0.1891,+0.3266]`，9/25 lower，joint 8/25。MFA-linear vs raw TTS：Sync-C Δ=`-0.1374`，CI `[-0.5542,+0.1976]`，12/25 positive；Sync-D Δ=`+0.0967`，CI `[-0.1605,+0.4095]`，12/25 lower，joint 9/25。两项 Sync-C CI 均跨 0，不能支持 n=25 多 speaker 下的 MFA-linear 正向收益。

首轮 `eval/` 结果已判定无效：它把严格 tokens 与新生成 TTS 混用，SHA 核验发现二者不一致；不得引用。旧单 speaker n=15 结果仍保留，不能与 n=25 当作同一 cohort 重复测量。

[decision] 在当前预定义多 speaker AISHELL-1 cohort、固定脸和冻结链路下，MFA-linear 不显示可重复的 SyncNet 正向收益；单 speaker S0765 n=15 阳性不能外推。 #mfa-linear #aishell1 #wav2lip #syncnet
[boundary] 该结果不构成 heldout S0770、真实多 speaker video、跨数据集或跨 talking-face 模型泛化结论。 #scope

## Observations
- [status] concluded

- [decision] same-phone kNN 为 No-Go；保留连续 TTS trajectory 的 MFA-linear 进入下一阶段。 #vc #wavlm
- [result] MFA-linear 在 WavLM 与独立 HuBERT 审计中均 10/10 向 TTS 靠近。 #feature-audit
- [result] valid-only n=15 中，MFA-linear 对 natural：ΔSync-C +0.964267、ΔSync-D -0.353667，联合改善 12/15。 #wav2lip #syncnet
- [result] MFA-linear 对 raw TTS：ΔSync-C +0.673267、ΔSync-D -0.277333，联合改善 10/15。 #wav2lip #syncnet
- [evidence] Ditto n=100 中 raw TTS ΔSync-C +1.022、92/100 更高；这是一般 TTS 优势而非 MFA-linear 复制。 #ditto
- [evidence] 固定 n=13 跨模型中 7/9 的 raw TTS Sync-C 更高，Wav2Lip +1.53、IMTalker +1.13。 #cross-model
- [insight] 连续 phone 内 TTS trajectory 比离散 same-phone kNN 更关键；waveform warp 不是可靠 oracle。 #alignment
- [problem] Wav2Lip 的共享相对 `temp/` 会造成跨样本污染。 #reproducibility
- [solution] 每个评测 cell 使用独立 cwd/temp，丢弃污染结果并从零重跑。 #provenance
- [boundary] 当前 MFA-linear 证据仅为 `S0765` valid-only，不构成 heldout、主观音质或精确边界声明。 #scope

## 独立 MFA-linear rerun（2026-08-13）

为响应重新生成要求，在不覆盖既有 `mfa_linear_strict` 的情况下，新建了两个独立目录：

- `runs/aishell1_mfa_linear_n25_20260813/mfa_linear_rerun_20260813/`：首次独立重算，25/25 成功，但发现 summary 的 `cohort_manifest` 字段沿用了源池 manifest 路径，未准确记录实际冻结 cohort；该目录保留作审计对照，不作为最终 provenance 目录。
- `runs/aishell1_mfa_linear_n25_20260813/mfa_linear_rerun_fixed_20260813/`：修复 provenance 字段后重新独立生成，作为最终 rerun 产物。

最终 rerun 使用同一冻结输入：cohort `data_boundary/aishell1_n25_speaker_balanced_cohort.json`、strict TTS `tts_strict/tts_meta.json`、frozen tokens `mfa/tokens.json`。生成脚本现在记录实际传入 cohort 路径及 `cohort_manifest_sha256`，并保留 frozen WavLM-Large L6 / prematched HiFi-GAN model hashes。

最终检查结果：25/25 samples、0 failures、25/25 output audio hash 有效、25/25 exact natural length、全部 finite；speaker 分布为 `S0765/S0901/S0906/S0912/S0913` 各 5 条；cohort 路径/SHA 和 tokens SHA 均匹配。旧 `mfa_linear_strict/summary.json` 与第一份 rerun summary 均保持不变，新旧最终音频 SHA 不同，证明是独立重算。

代码修复：`scripts/pilot_generate_mfa_linear.py` 的 `generate()`/CLI 现在接收实际 manifest path，并将实际 cohort path/SHA 写入 summary；语法检查及相关回归测试通过。针对性测试最终为 `24 passed, 3 warnings`；修复后聚焦测试为 `15 passed, 3 warnings`。首次直接调用 `.venv/bin/python -m pytest` 因环境未安装 pytest 失败，随后使用 `uv run --with pytest --python .venv/bin/python` 成功，未修改环境。

- [result] provenance-corrected independent MFA-linear rerun completed 25/25 with 0 failures. #mfa-linear #reproducibility
- [solution] Existing strict output was preserved; rerun uses a separate directory and records the actual cohort manifest path/SHA. #provenance
- [boundary] This rerun regenerates MFA-linear audio only; no new Wav2Lip/SyncNet evaluation was performed. #scope

## 原 n=15 严格 bridge rerun（2026-08-13）

为区分 n=25 新协议差异，按原 n=15 协议重新生成并评测：固定 `S0765` valid speaker、原 sorted paired_key 前 15 条、原始 24 kHz TTS source、strict MFA token metadata、原逐样本 face `natural_raw/{1..15}.mp4`、Wav2Lip/SyncNet checkpoint 和 `min_track=50` 均保持不变。旧目录未覆盖。

### 新生成音频

新音频目录：`runs/knn_vc_poc_valid15_mfa_linear_bridge_20260813/`。使用原 `scripts/run_knn_vc_poc.py --count 15 --condition paired_tts_mfa_linear` 入口重新生成，不使用当前 n=25 的 16 kHz canonical TTS 流程。

音频 gate：15/15 成功、0 failures；selection hash 与原实验相同；model revision/checkpoint hash、映射策略相同；输出均 16 kHz、exact natural length、finite、无 clipping。样本 3/7/10 的 known TTS-only fallback 保持记录。新生成音频与旧 15 条音频 SHA 全部不同，确认是独立重算。

### 下游 bridge

新评测目录：`runs/rhythm_timing/20260813_syncnet_valid15_mfa_bridge/`。MFA-linear 15 条全部 fresh Wav2Lip→SyncNet；natural/raw-TTS 各 15 条沿用原协议已核验的 smoke10 cache，并在 scores 中保留 `cache_reused=true`。每个 MFA cell 使用独立 `wav2lip_work/mfa_linear/<sample_id>/temp/`。

完整性：45/45 score、0 failures、45/45 video hash 有效、所有分数 finite、15 个逐样本 face hash、模型 hash 和 `min_track=50` 一致。旧 `runs/knn_vc_poc_valid15_mfa_linear_20260812/summary.json`、旧 `runs/rhythm_timing/20260812_syncnet_valid15_mfa/{manifest,summary,analysis}.json` 哈希均未变化。

### 结果

| 比较 | bridge ΔSync-C | 原报告 ΔSync-C | bridge ΔSync-D | 原报告 ΔSync-D | C+ | D− | joint |
|---|---:|---:|---:|---:|---:|---:|---:|
| MFA-linear − natural | `+0.941600` | `+0.964267` | `-0.359533` | `-0.353667` | 13/15 | 12/15 | 12/15 |
| MFA-linear − raw TTS | `+0.650600` | `+0.673267` | `-0.283200` | `-0.277333` | 12/15 | 12/15 | 10/15 |

Bridge 的 bootstrap CI：对 natural 的 ΔC `[+0.6182,+1.2458]`、ΔD `[-0.5851,-0.1200]`；对 raw TTS 的 ΔC `[+0.2943,+1.0068]`、ΔD `[-0.5117,-0.0743]`。与原报告数值接近，方向和 positive/joint counts 基本一致，支持原 n=15 正向结果不是由一次偶然的生成或 Wav2Lip 并发污染造成的。

限制：natural/raw-TTS 在 bridge 中仍是原协议 cache，不是三臂全部 fresh；bridge 是 S0765 valid-only n=15 原协议复现，不是多 speaker 或 heldout 泛化。相关回归测试 `24 passed, 3 warnings`。

- [result] 原协议 bridge 复现了 MFA-linear 的下游正向效果。 #bridge #mfa-linear #syncnet
- [insight] n=15 bridge 与原报告的差异很小，说明旧结果对重新生成和独立 fresh MFA-linear 评测稳定。 #reproducibility
- [boundary] 结论适用于原 S0765 n=15 协议，不自动外推到 n=25 多 speaker。 #scope

## n=25 前15条音频 × n=15 face slots 对照（2026-08-13）

应用户要求，固定使用 n=25 cohort 的前 15 条 MFA-linear 音频：`1–5`（S0765）、`101–105`（S0901）、`151–155`（S0906）；face 改用原 n=15 的逐槽位 `natural_raw/{1..15}.mp4`。natural/raw-TTS 仍沿用原 n=15 summary 中的 15 个 face-slot cache，Wav2Lip/SyncNet checkpoint、batch 参数、`--nosmooth` 和 `min_track=50` 保持不变。

由于 n=25 音频和 n=15 face slot 不是同一 utterance，专用脚本 `scripts/eval_n25_audio_n15_face_wav2lip_syncnet.py` 将 `face_slot` 与 `audio_source_sample_id/paired_key/speaker_id` 分开记录，并明确 `same_utterance_pairing=false`，避免把跨 utterance 对照误标成配对实验。新产物：`runs/rhythm_timing/20260813_n25_audio_n15_face_slots/`。

完整性：45/45、0 failures；natural/raw-TTS 各 15 条 `cache_reused=true`；MFA-linear 15 条 fresh Wav2Lip→SyncNet；所有音频、face、视频 hash 有效，所有分数 finite，15 个独立 `wav2lip_work/mfa_linear/face_slot_*` 工作目录。

结果（按 face slot 比较，不是同文本配对）：

- MFA-linear vs natural：ΔSync-C `+0.3654`，95% bootstrap CI `[-0.1673,+0.8519]`；ΔSync-D `-0.0223`，CI `[-0.3385,+0.2995]`；C positive 9/15，D improved 8/15，joint 7/15。
- MFA-linear vs raw TTS：ΔSync-C `+0.0744`，CI `[-0.4403,+0.4965]`；ΔSync-D `+0.0541`，CI `[-0.1737,+0.2847]`；C positive 10/15，D improved 6/15，joint 5/15。

解释：n=25 前 15 条音频放到 n=15 face slots 后，相对 natural 仍有小幅平均 Sync-C 正向，但 CI 跨 0；相对 raw TTS 基本无差异。这个结果不是自然/TTS/MFA-linear 同 utterance 的严格 paired effect，也不是 speaker 泛化结论，只能作为跨音频/face-slot 的探索性对照。相关回归测试 `24 passed, 3 warnings`。

- [result] n=25 audio × n=15 face slots fresh evaluation complete 45/45. #cross-protocol #syncnet
- [boundary] Face-slot comparison is cross-utterance and must not be interpreted as a paired same-text result. #scope

## n=15 两组统一 fixed-face 三臂实验（2026-08-13）

为隔离 face 条件影响，原 n=15 的 15 条 S0765 natural/raw-TTS/MFA-linear 音频分别做了两次全 fresh 评测；每次 15 个音频全部使用同一张 face，未复用旧逐样本 face cache。

新增 evaluator：`scripts/eval_valid15_same_face_wav2lip_syncnet.py`。每组输出三臂各 15 条、45 个 fresh Wav2Lip→SyncNet cell，每个 cell 独立 cwd/temp。

### face 1（n=25 原 fixed-face）

目录：`runs/rhythm_timing/20260813_syncnet_valid15_same_face1/`

face：`ditto_videos/natural_raw/1.mp4`，SHA `abf9ae5b14333f793adcd72e16d26780dfbb2a3ab8753f6c0c1682cf4c4f4f2b`。

- 45/45，0 failures，所有输入/输出/face/video SHA 有效，三臂全部 `cache_reused=false`。
- MFA-linear vs natural：ΔSync-C `+0.8590`，95% CI `[+0.5423,+1.1640]`；ΔSync-D `-0.2998`，CI `[-0.4780,-0.1022]`；C+ 13/15，D− 12/15，joint 12/15。
- MFA-linear vs raw TTS：ΔSync-C `+0.3352`，CI `[-0.0284,+0.7141]`；ΔSync-D `+0.1217`，CI `[-0.0850,+0.3373]`；C+ 12/15，D− 6/15，joint 6/15。

### face 15

目录：`runs/rhythm_timing/20260813_syncnet_valid15_same_face15/`

face：`ditto_videos/natural_raw/15.mp4`，SHA `6161b3331e34794674c56ff95874fd068b0d144fc79c8b426cb76a42e7980f93`。

- 45/45，0 failures，所有输入/输出/face/video SHA 有效，三臂全部 `cache_reused=false`。
- MFA-linear vs natural：ΔSync-C `+0.8054`，CI `[+0.5163,+1.1046]`；ΔSync-D `-0.2870`，CI `[-0.4841,-0.0788]`；C+ 13/15，D− 10/15，joint 10/15。
- MFA-linear vs raw TTS：ΔSync-C `+0.3907`，CI `[+0.0315,+0.7759]`；ΔSync-D `+0.0159`，CI `[-0.1755,+0.2115]`；C+ 11/15，D− 7/15，joint 4/15。

### 结论

两张不同 face 下，MFA-linear 相对 natural 的 Sync-C 和 Sync-D 均保持正向，且 CI 不跨 0；因此原 n=15 的增强效果不依赖某一张逐样本 face，也不只是 n=25 `face1` 的偶然条件。face1/face15 对 MFA-linear vs natural 的结果接近（ΔC +0.859/+0.805，ΔD -0.300/-0.287）。相对 raw TTS 的收益较弱，face1 的 Sync-D 甚至变差，face15 的 Sync-C 较小正向；这仍说明主要增强是相对 natural，而非稳定超过 raw TTS。

这两组仍是 S0765 valid-only n=15，不能推断多 speaker 泛化；但它们强力削弱“单张 face 导致原 n=15 正向结果”的解释。相关回归测试 `24 passed, 3 warnings`。

- [result] Two fixed-face fresh n15 evaluations both preserved MFA-linear vs natural improvement. #fixed-face #mfa-linear #syncnet
- [insight] Positive effect is stable across face1 and face15, so it is not dependent on one particular face. #reproducibility
- [boundary] Raw-TTS comparison is weaker and fixed-face experiments remain S0765 valid-only. #scope

## n=15 原数据上的采样率条件隔离（2026-08-13）

为验证 n=25 的采样率处理是否破坏 MFA-linear 增强条件，固定原 n=15 的全部条件：S0765 valid 前 15 条、原 natural/TTS source、strict MFA token 时间戳/标签、原逐样本 face、冻结 WavLM/kNN-VC/HiFi-GAN 和 Wav2Lip/SyncNet 参数；只替换 24k→16k TTS waveform 的处理方式。

新增：

- `scripts/prepare_aishell1_n15_sampling_rate_isolation.py`：从 strict source manifest 选原 sample 1..15，保留 token arrays 和 source SHA；按 n=25 合同用 `ffmpeg -ar 16000 -ac 1 -c:a pcm_s16le` 生成 canonical TTS。
- `scripts/eval_valid15_sampling_rate_isolation.py`：四臂全部 fresh、逐样本 face、60 个 Wav2Lip→SyncNet cell。

新音频产物：

- canonical TTS：`runs/knn_vc_poc_valid15_mfa_linear_sr16k_20260813/tts_canonical/`，15/15、全部 16 kHz mono；
- new MFA-linear：`runs/knn_vc_poc_valid15_mfa_linear_sr16k_20260813/mfa_linear/`，15/15、0 failures、exact natural length、finite。

评测：`runs/rhythm_timing/20260813_syncnet_valid15_sampling_rate_isolation/`。四臂为 `natural_raw`、`raw_tts`、原内存 `scipy resample_poly` 的 `mfa_linear_old_resample_poly`、n=25 风格 ffmpeg canonical 的 `mfa_linear_n25_ffmpeg_16k`。60/60、0 failures、所有 audio/video/face SHA 有效、没有 cache reuse、同 utterance paired、每样本 face 保持原协议。

结果：

- old resample_poly vs natural：ΔSync-C `+0.9666`，CI `[+0.6465,+1.2741]`；ΔSync-D `-0.3707`，CI `[-0.5975,-0.1361]`。
- new ffmpeg 16k vs natural：ΔSync-C `+0.9343`，CI `[+0.6251,+1.2311]`；ΔSync-D `-0.3409`，CI `[-0.5682,-0.1189]`。
- new ffmpeg 16k vs old resample_poly：ΔSync-C `-0.0323`，CI `[-0.1379,+0.0683]`；ΔSync-D `+0.0298`，CI `[-0.0652,+0.1181]`；C+ 6/15，D− 6/15，joint 4/15。

结论：在原 n=15 数据、strict tokens、逐样本 face 和 fresh 四臂评测下，n=25 风格的 16 kHz ffmpeg canonicalization 没有破坏 MFA-linear 增强；new/old 两种采样率处理的直接差异很小且 CI 跨 0。n=25 实验未复现增强，主要原因不太可能是单纯 24k→16k canonicalization；应继续关注 speaker/cohort、MFA token/输入链其他差异或 n=25 评测协议。

相关回归测试：`24 passed, 3 warnings`。

- [result] Sampling-rate isolation preserved positive MFA-linear gain under n15 protocol. #sampling-rate #mfa-linear
- [insight] ffmpeg 16k canonicalization vs in-memory scipy resample_poly changed neither direction nor meaningful magnitude. #reproducibility
- [boundary] This is S0765 valid-only n15 and does not establish n25 multi-speaker generalization. #scope

## 完整 n=25 使用 n=15 采样率条件（2026-08-13）

按用户确认，保留完整 n=25 cohort（S0765/S0901/S0906/S0912/S0913，各 5 条）和 n=25 原协议统一 face1，只把 TTS/MFA-linear 的 waveform 处理切换成 n=15 条件：原始 24 kHz TTS 在内存中用 `scipy.signal.resample_poly(up=2, down=3)` 变为 16 kHz。natural、strict MFA tokens、speaker/cohort、frozen WavLM/kNN-VC/HiFi-GAN、Wav2Lip/SyncNet 参数和 `min_track=50` 保持不变；不重新 MFA。

新增：

- `scripts/prepare_aishell1_n25_resample_poly_tts_meta.py`：校验 n25 原始 TTS source SHA，用 float32 `resample_poly(2,3)` 生成 metadata/audio。
- `scripts/eval_aishell1_n25_resample_poly_face1.py`：统一 face1、三臂全 fresh 的 75-cell evaluator。

产物：

- resample-poly TTS：`runs/aishell1_mfa_linear_n25_resample_poly_20260813/tts_resample_poly_16k/`，25/25；
- resample-poly MFA-linear：`runs/aishell1_mfa_linear_n25_resample_poly_20260813/mfa_linear/summary.json`，25/25、0 failures、exact natural length、finite；
- fresh 评测：`runs/rhythm_timing/20260813_syncnet_aishell1_n25_resample_poly_face1/`。

评测完整性：75/75、0 failures、natural/raw-TTS/MFA-linear 各 25，统一 face1 hash，所有 audio/video/face SHA 有效，`cache_reused=false`。这是 single fixed-face audio-condition 对照，不是 speaker-matched video 泛化。

结果：

- resample_poly MFA-linear vs natural：ΔSync-C `-0.07092`，95% CI `[-0.52104,+0.34988]`；ΔSync-D `+0.06944`，CI `[-0.15644,+0.28912]`；C+ 13/25，D− 9/25，joint 9/25。
- resample_poly MFA-linear vs raw TTS：ΔSync-C `-0.1634`，CI `[-0.43324,+0.09132]`；ΔSync-D `+0.10964`，CI `[-0.10388,+0.33676]`；C+ 14/25，D− 13/25，joint 10/25。

结论：把 n=25 的 TTS/MFA-linear 采样率处理改成 n=15 的 in-memory `resample_poly` 没有恢复增强；相对 natural 仍略负，和 n=25 原 ffmpeg-16k fixed-face 结果（ΔSync-C -0.0431）一致地不支持正向收益。因此 n=25 失败主要不是 24k→16k canonicalization 方式造成的。限制仍是完整 n25 的非 S0765 speaker 没有匹配 face，统一 face1 不代表真实多 speaker video 泛化。

回归测试：`24 passed, 3 warnings`。

- [result] n25 under n15 sampling condition completed 75/75 without restoring MFA-linear gain. #n25 #sampling-rate #mfa-linear
- [insight] Same n25 cohort/face with resample_poly remains near-zero or negative, ruling out simple resampling implementation as the main cause. #ablation
- [boundary] Fixed-face single visual protocol limits speaker-level interpretation. #scope

## n=25 speaker-level 汇总与 cohort 影响

基于 `runs/rhythm_timing/20260813_syncnet_aishell1_n25_resample_poly_face1/summary.json`，按 speaker 汇总 MFA-linear 相对 natural/raw-TTS 的结果：

- S0765：vs natural ΔC `+0.7268`、ΔD `-0.3440`，C+ 4/5、D− 4/5、joint 4；vs raw TTS ΔC `+0.2762`、ΔD `-0.0388`，joint 3。
- S0901：vs natural ΔC `-0.9146`、ΔD `+0.3466`，C+ 1/5、D− 0/5、joint 0；vs raw TTS ΔC `-0.3196`、ΔD `+0.1920`，joint 1。
- S0906：vs natural ΔC `+0.2516`、ΔD `+0.1056`，C+ 4/5、D− 1/5、joint 1；vs raw TTS ΔC `+0.0772`、ΔD `-0.1378`，joint 3。
- S0912：vs natural ΔC `-0.8692`、ΔD `+0.3578`，C+ 1/5、D− 1/5、joint 1；vs raw TTS ΔC `-1.0994`、ΔD `+0.7074`，C+ 0/5、D− 0/5、joint 0。
- S0913：vs natural ΔC `+0.4508`、ΔD `-0.1188`，C+ 3/5、D− 3/5、joint 3；vs raw TTS ΔC `+0.2486`、ΔD `-0.1746`，joint 3。

初步判断：speaker composition 明显有影响。S0765/S0913 总体偏正，S0901/S0912 明显偏负，S0906 居中；因此 n=25 总体接近零/略负，很可能被 speaker-dependent 差异抵消。该证据仍受统一 S0765 face 的限制，不能区分真实声学/韵律差异与 speaker-face mismatch。

统计说明：当前 n=25 analysis 使用的是 5-speaker cluster bootstrap，即每次把 speaker 作为 block 重采样、保留该 speaker 的 5 条 utterance；这不是对 speaker 做聚类算法，也没有做 k-means、层次聚类或任何无监督聚类。speaker 只是预先固定的 cohort 分组和统计单位。

- [insight] n=25 speaker-level 方向差异很大，cohort 组成可能抵消 S0765 的正向效果。 #speaker #cohort
- [clarification] 没有执行聚类算法；cluster bootstrap 只是按 speaker block 重采样以处理同 speaker 样本相关性。 #statistics

## 负增益 speaker 试听与特征分析方向（2026-08-13）

当前 n25 resample_poly/face1 结果中，明显负向 speaker 为 S0901、S0912；正向对照主要为 S0765、S0913，S0906 居中。试听时应同时听 natural、raw TTS、MFA-linear，而不是只听 MFA-linear。

### 试听路径

S0901：
- natural：`results/rhythm_style_500/aishell1_test_400/natural/0101.wav` … `0105.wav`
- raw TTS：`runs/aishell1_mfa_linear_n25_20260813/tts_strict/tts/101.wav` … `105.wav`
- MFA-linear：`runs/aishell1_mfa_linear_n25_resample_poly_20260813/mfa_linear/101.wav` … `105.wav`
- 最明显负向样本：103（ΔSync-C -3.250，ΔSync-D +0.904）。

S0912：
- natural：`results/rhythm_style_500/aishell1_test_400/natural/0201.wav` … `0205.wav`
- raw TTS：`runs/aishell1_mfa_linear_n25_20260813/tts_strict/tts/201.wav` … `205.wav`
- MFA-linear：`runs/aishell1_mfa_linear_n25_resample_poly_20260813/mfa_linear/201.wav` … `205.wav`
- 最明显负向样本：203（ΔSync-C -1.880，ΔSync-D +0.995）；201 也明显负向。

### 声学/韵律分析优先级

1. natural/raw-TTS/MFA-linear 的时长、静音比例、voiced ratio、F0 均值/方差/范围、能量和语速；
2. strict MFA phone 数、phone duration、phone coverage、unmatched/fallback frames、最后 token tail extension；
3. WavLM/HuBERT feature-space 中 natural→TTS 与 MFA-linear→TTS 的 cosine/MSE，按 phone 和 speaker 汇总；
4. raw TTS 相对 natural 的 SyncNet 变化，用于区分 speaker 声学问题与 speaker-face mismatch；
5. ASR/PER、音质和听检，特别关注 S0901/103、S0912/203 的音素清晰度、停顿边界和局部轨迹。

当前已有线索：S0901 的 raw TTS 相对 natural 已负向（平均 ΔC -0.595、ΔD +0.155），更像 speaker/audio 与固定 S0765 face 的匹配或声学问题；S0912 的 raw TTS 相对 natural 反而平均正向（ΔC +0.230、ΔD -0.350），但 MFA-linear 变负，提示 S0912 更应优先检查 MFA token/phone 映射和连续特征轨迹。结论仍不能仅凭 SyncNet 归因于声学特征。

没有执行聚类算法；之前的 cluster bootstrap 只是按 speaker block 重采样。后续 speaker 分组必须预先固定，避免按分数事后挑选。

- [insight] S0901 的 raw TTS 已相对 natural 变差，S0912 的 raw TTS 尚可但 MFA-linear 变差，两者可能不是同一故障机制。 #acoustic #speaker
- [decision] 先做试听、声学/韵律、MFA coverage 和 SSL feature audit，再判断 speaker cohort 影响。 #analysis

## Relations

- relates_to [[TTS Feature Alignment and Duration Model Research]]
- relates_to [[Ralph Step 1 - Controlled Baseline]]
- relates_to [[Ralph Step 2 - Phone Target Geometry Screening]]
- relates_to [[增强头下游评估 (Ditto + SyncNet)]]

## n=25 audio-only artifact audit（2026-08-14）

用户试听后报告 S0901 MFA-linear 有类似电流干扰和重复的听感。为避免仅凭 SyncNet 推断，新增 `scripts/audit_aishell1_n25_audio_artifacts.py`，对冻结 n25 的 natural、raw TTS、resample-poly MFA-linear 三臂做了 75 条 audio-only 审计，并生成：

- `runs/aishell1_mfa_linear_n25_resample_poly_20260813/audio_artifact_audit_20260813/per_file.jsonl`
- `per_event.jsonl`、`paired_summary.json`、`speaker_summary.json`、`correlations.json`
- `plots/sample_*.png`（25 个三臂 waveform/spectrogram 对照）
- `report.md`

输入 gate 校验了 25 个 paired utterances、5 speakers × 5、三臂 audio SHA、16 kHz mono finite、tokens/MFA/SyncNet provenance；MFA-linear 明确使用 resample-poly summary，未混入原 ffmpeg 条件结果。测试共 `14 passed`。

### 结果

- 没有发现高置信度的非平凡重复片段；普通持续元音的周期性已在检测器中排除。
- MFA-linear 相对 natural 的 artifact-event count 平均差为 `+0.20`，相对 raw TTS 为 `+0.84`；boundary-event count 分别为 `+0.36`、`+0.52`。这些是候选瞬态/边界事件，不等价于确认的电流声。
- S0901 的 MFA-linear 相对 natural 平均 peak 约 `+0.081`、crest factor `+3.04`、artifact-event duration `+0.231 s`；相对 raw TTS 分别约 `+0.129`、`+4.29`、`+0.337 s`。S0912 相对 natural peak `+0.095`、crest `+0.95`、artifact-event duration `+0.100 s`；相对 raw TTS peak `+0.128`、crest `+1.37`、duration `+0.318 s`。
- MFA-linear 没有显示稳定的高频能量升高或 50/60 Hz 独有线谱：S0901 MFA high-band ratio 约 `0.0061`，低于 natural `0.0111` 和 raw TTS `0.0212`；S0912 MFA 约 `0.0085`，低于 raw TTS `0.0121`，接近 natural `0.0070`。因此当前证据更支持“局部瞬态/重合成边界伪影或峰值异常”，不支持直接诊断为电流干扰。
- S0901/103 的 MFA 候选集中在约 `1.50–1.56 s`、`1.90–1.99 s`、`2.68–2.82 s`、`2.87–2.93 s`；S0912/203 集中在约 `0.57–0.75 s`、`0.83–1.07 s`、`1.11–1.25 s`、`1.37–1.56 s` 等区间。部分自然/raw TTS 也有同类瞬态，因此需结合听感和更细粒度边界/重合成诊断。

### 限制与判断

MFA summary 只有 aggregate fallback/coverage，没有 fallback frame indices；所有 right-zero-pad 尾部均单独标为 padding。SyncNet 使用固定 S0765 face，只能作为探索性相关，不可解释为 speaker acoustic causality。当前最合理的下一步是对候选时间段做 raw waveform 局部对齐、phone boundary 对照和重新 MFA/ASR 检查，而不是继续扩大 SyncNet 评测。

## Natural-anchor MFA residual alpha blend（2026-08-14）

针对 S0765、S0901、S0912 各 5 条 paired utterance，进行了 audio-only 五档 feature blend：

```text
z_alpha = z_natural + alpha * (z_tts_warp - z_natural)
alpha = 0 / 0.25 / 0.5 / 0.75 / 1
```

### 输入与输出

- cohort：`runs/aishell1_mfa_linear_n25_20260813/data_boundary/aishell1_n25_speaker_balanced_cohort.json`
- resample-poly TTS metadata：`runs/aishell1_mfa_linear_n25_resample_poly_20260813/tts_meta.json`
- TTS WAV：`runs/aishell1_mfa_linear_n25_resample_poly_20260813/tts_resample_poly_16k/`
- frozen tokens：`runs/aishell1_mfa_linear_n25_20260813/mfa/tokens.json`
- alpha=1 reference：`runs/aishell1_mfa_linear_n25_resample_poly_20260813/mfa_linear/summary.json`
- 新输出：`runs/aishell1_mfa_linear_n25_resample_poly_20260813/natural_anchor_alpha_20260814/`
- 代码：`scripts/run_natural_anchor_alpha_blend.py`
- 测试：`tests/test_natural_anchor_alpha_blend.py`

使用冻结 `bshall/knn-vc` revision `c616845c4e309e24d5927f15adbdf277a3d65358`，WavLM-Large L6 1024-d、prematched frozen HiFi-GAN、CUDA；没有运行 Wav2Lip/SyncNet。75/75 输出成功，全部 exact natural length、finite、0 clipping。alpha=1 与已有 MFA-linear 逐条数值等价：15/15 最大绝对差 ≤1e-5，worst max diff `5.94e-6`、worst RMSE `8.23e-8`；SHA 不同仅是 CUDA/浮点重算。

### 音频结果

S0901 的 speaker mean peak/crest 随 alpha 并非单调改善：

```text
alpha 0    0.3240 / 10.977
alpha 0.25 0.2944 / 11.009
alpha 0.5  0.2994 / 12.189
alpha 0.75 0.3199 / 12.801
alpha 1    0.3313 / 12.449
```

S0912 则呈现较明显的下降趋势：

```text
alpha 0    0.5724 / 9.834
alpha 0.25 0.5463 / 9.726
alpha 0.5  0.5165 / 9.375
alpha 0.75 0.4927 / 8.762
alpha 1    0.4991 / 8.647
```

重点样本 S0901/103 在 alpha 0.5/0.75 的 crest factor 反而升至 `12.487/13.503`；S0912/203 的 peak 则从 alpha 0 的 `0.7543` 降至 alpha 1 的 `0.6226`。因此“减小 alpha 就能修复变质”不成立；alpha=0 的 natural direct resynthesis 本身也可能有较高峰值，说明问题不只是 TTS residual 强度。

### 判断

该实验验证了 direct natural-anchor feature blend 的可行性和 alpha=1 参考一致性，但尚未解决 S0901/S0912 的局部变质。下一步应优先加入 residual norm/velocity constraint、phone-boundary smoothing、silence anchor 和 natural F0/energy conditioning，仍先做 audio-only QC，再决定下游视频评测。
- [inputs] AISHELL-1 S0765 valid 前 15 条 + n=25 多 speaker cohort; natural results/rhythm_style_500/aishell1_test_400/natural/, raw-TTS 24 kHz 源; MFA tokens runs/aishell1_mfa_linear_n25_20260813/mfa/tokens.json
- [outputs] MFA-linear 音频 runs/knn_vc_poc_valid15_mfa_linear_20260812/ 与 runs/aishell1_mfa_linear_n25_20260813/mfa_linear*/; Wav2Lip 视频与 SyncNet 分数 runs/rhythm_timing/2026081[23]_*/(见正文各节)
- [model] bshall/knn-vc revision c616845c4e309e24d5927f15adbdf277a3d65358 (WavLM-Large L6 1024-d + frozen prematched HiFi-GAN); Wav2Lip; SyncNet V2 min_track=50
- [code_paths] scripts/run_knn_vc_poc.py, scripts/pilot_generate_mfa_linear.py, scripts/eval_mfa_linear_wav2lip_syncnet.py, scripts/eval_valid15_same_face_wav2lip_syncnet.py, scripts/eval_aishell1_n25_wav2lip_syncnet.py, scripts/run_natural_anchor_alpha_blend.py, scripts/audit_aishell1_n25_audio_artifacts.py

## Residual norm/velocity constraint sweep（2026-08-14）

在 alpha blend 后，固定 alpha=1，仅对 residual trajectory `d = z_tts_warp - z_natural` 加约束，测试 S0765/S0901/S0912 各 5 条 paired utterance。输入、模型和代码路径：

- cohort：`runs/aishell1_mfa_linear_n25_20260813/data_boundary/aishell1_n25_speaker_balanced_cohort.json`
- resample-poly TTS：`runs/aishell1_mfa_linear_n25_resample_poly_20260813/tts_meta.json`
- tokens：`runs/aishell1_mfa_linear_n25_20260813/mfa/tokens.json`
- MFA reference：`runs/aishell1_mfa_linear_n25_resample_poly_20260813/mfa_linear/summary.json`
- generator：`scripts/run_natural_anchor_constraints.py`
- tests：`tests/test_natural_anchor_constraints.py`
- output：`runs/aishell1_mfa_linear_n25_resample_poly_20260814/natural_anchor_constraints_20260814/`
- model：frozen `bshall/knn-vc` revision `c616845c4e309e24d5927f15adbdf277a3d65358`，WavLM-Large L6 + prematched frozen HiFi-GAN，CUDA。

条件为：`unconstrained`、`norm_q90`、`norm_q75`、`velocity_q90`、`norm_q90_velocity_q90`、`norm_q75_velocity_q90`。q threshold 按每条 utterance 的 unconstrained residual trajectory 计算；没有使用 SyncNet 分数选参。90/90 音频成功、exact natural length、finite、0 clipping；相关回归测试共 19 passed。

### 结果

S0901 mean peak/crest：

```text
unconstrained       0.3313 / 12.449
norm_q90            0.3313 / 12.598
norm_q75            0.3317 / 13.180
velocity_q90        0.3337 / 12.545
norm_q90+velocity   0.3337 / 12.679
norm_q75+velocity   0.3313 / 13.159
```

S0912 mean peak/crest：

```text
unconstrained       0.4991 / 8.647
norm_q90            0.4805 / 8.383
norm_q75            0.4758 / 8.421
velocity_q90        0.4976 / 8.621
norm_q90+velocity   0.4807 / 8.379
norm_q75+velocity   0.4748 / 8.400
```

S0901/103 的 q75+velocity crest 从 `11.529` 增至 `12.012`，而 S0912/203 从 `10.758` 降至 `10.656`。说明简单 residual magnitude/velocity clipping 具有明显 speaker/sample heterogeneity，不能稳定修复 S0901，也不应直接进入下游视频评测。

[decision] 下一步若继续，优先采用 phone-boundary/context-aware constraint、silence anchor 或 natural F0/energy 条件；保留 speaker-specific audio gate，不再直接扩大 quantile clipping 网格。 #constraint #audio #mfa-linear

## 0813 MFA 音频驱动 Wav2Lip 与严格原音反事实（2026-08-14）
由于当前本地 checkout 没有可用 Ditto/TFG 代码与 checkpoint，且远端 TFG GPU 不可用，本次按用户确认改用本地 Wav2Lip。该实验**不是 TFG/Ditto 重新生成结果**，而是验证 0813 MFA-linear 音频驱动的 Wav2Lip 口型，以及在完全相同视频帧上替换原始自然音频后的反事实变化；原有视频没有删除或覆盖。

### 固定 cohort、输入与模型

- cohort：AISHELL-1 valid split，单说话人 `S0765`，预先固定 paired-key 排序前 15 条；heldout `S0770` 未加载、选择或评估。
- 冻结清单：`runs/rhythm_timing/20260814_wav2lip_0813_mfa_bridge/manifest.json`。
- 0813 MFA-linear 输入音频：`runs/knn_vc_poc_valid15_mfa_linear_bridge_20260813/audio/*__paired_tts_mfa_linear.wav`；源汇总：`runs/knn_vc_poc_valid15_mfa_linear_bridge_20260813/summary.json`。
- 自然输入音频：`results/rhythm_style_500/aishell1_test_400/natural/0001.wav` … `0015.wav`。
- Wav2Lip face 输入：`runs/two_stage_hubert_aishell1_20260810/ditto_videos/natural_raw/1.mp4` … `15.mp4`；仅作为视觉输入，不表示本轮重新运行 Ditto。
- Wav2Lip Python：`[redacted-local-path]`；入口：`third_party/Wav2Lip/inference.py`；checkpoint：`third_party/Wav2Lip/checkpoints/wav2lip_gan.pth`；SHA256 `ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8`。
- Wav2Lip 参数：`--face_det_batch_size 4 --wav2lip_batch_size 4 --nosmooth`。
- SyncNet Python：`[redacted-local-path]`；代码目录：`third_party/syncnet_python/`；模型：`third_party/syncnet_python/data/syncnet_v2.model`；SHA256 `961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442`；`min_track=50`。

### 相关代码与运行协议

- Wav2Lip/SyncNet 评测协议参考 `scripts/eval_valid15_same_face_wav2lip_syncnet.py`、`scripts/eval_mfa_linear_wav2lip_syncnet.py`，复用逐样本独立工作目录、模型哈希和 SyncNet 解析方式。
- 每个样本使用独立 `wav2lip_work/` cwd/temp，避免 Wav2Lip 相对路径 `temp/temp.wav`、`temp/result.avi` 造成跨样本污染。
- 条件 A 的实际 Wav2Lip 命令记录在 `generation.json`，核心形式为：

```bash
[redacted-local-path] third_party/Wav2Lip/inference.py \
  --checkpoint_path third_party/Wav2Lip/checkpoints/wav2lip_gan.pth \
  --face <natural_raw/{id}.mp4> \
  --audio <0813__paired_tts_mfa_linear.wav> \
  --outfile <new_output.mp4> \
  --face_det_batch_size 4 --wav2lip_batch_size 4 --nosmooth
```

- 条件 B 仅替换条件 A 视频的音频流，使用严格无损 mux：

```bash
ffmpeg -n -i <condition_a.mp4> -i <natural.wav> \
  -map 0:v:0 -map 1:a:0 -c:v copy -c:a copy <output.mkv>
```

使用 Matroska + `pcm_s16le`，不做 `atempo`、裁剪、`-shortest` 或静音填充；逐条验证视频流 MD5 与条件 A 相同，解码 PCM MD5 与自然 WAV 相同，`audio_transform=none`、`video_stream_preserved=true`、`audio_samples_preserved=true`。

### 三种条件与结果

| 条件 | Sync-C ↑ | Sync-D ↓ | AV offset 均值 |
|---|---:|---:|---:|
| 自然音频生成 Wav2Lip + 自然音频 | 5.7581 | 7.2690 | -2.0000 |
| 0813 MFA-linear 生成 Wav2Lip + MFA 音频 | **6.7226** | **6.9063** | -2.0000 |
| 同一 MFA 视频帧 + 严格替换自然音频 | 5.0791 | 8.2683 | -2.0667 |

0813 MFA-linear 生成视频相对自然音频生成视频：ΔSync-C `+0.9645`、ΔSync-D `-0.3627`；Sync-C 改善 13/15、Sync-D 改善 12/15、联合改善 12/15。自然生成臂来自原协议中逐项核验的 cache，而非本轮全 fresh 重跑，因此该比较适合作为同协议基线，不是同一执行批次的 bit-identical 对照。

在完全相同的 MFA 视频帧上严格替换自然音频，相对原 MFA 音轨：ΔSync-C `-1.6435`、ΔSync-D `+1.3621`；Sync-C、Sync-D、联合改善均为 0/15。与自然音频直接生成的视频相比，替换条件还额外降低 Sync-C `-0.6790`、增加 Sync-D `+0.9993`。

### 为什么听感接近但替换后 SyncNet 明显下降

“人耳听感接近”和“逐帧唇形驱动线索相同”不是同一条件。MFA-linear 使用自然 phone 时间网格并强制总样本数与自然音频相同，但不保证逐采样、逐帧或 phone 内声学事件重合：

- MFA-linear 在自然 phone 区间内保留并重映射 TTS 的连续特征轨迹；phone 的粗粒度起止可以接近，但 phone 内元音过渡、协同发音、辅音闭塞/释放位置仍可能不同。
- 自然语音包含弱化、吞音、气声、连读和不规则能量变化；MFA/TTS 轨迹通常更清晰、稳定，可能更容易被 Wav2Lip 和 SyncNet读取。
- 视频为 25 fps，一帧约 40 ms；人耳可能不敏感的 20–80 ms 局部差异，足以改变短窗口中的音素—视素对应。
- MFA forced-alignment 边界是估计值，不是精确生理唇部运动边界；“总长度完全一致”也不等于“所有局部事件完全对齐”。
- 三种条件的平均 AV offset 都约为 `-2`，不支持主要由整体时间平移造成；更合理的解释是局部轨迹不一致与 MFA 音频本身更适合该 Wav2Lip/SyncNet 链路共同作用。

总下降可描述性地拆为两部分：MFA 音频生成视频相对自然生成视频本来就有约 `+0.9645` Sync-C 的生成/可读性优势；保留 MFA 口型再换成自然音频，又相对自然音频自行生成的视频产生约 `-0.6790` Sync-C 的额外不匹配。该拆分不是因果估计，因为自然生成臂为经核验 cache。

### 尚需补充的 processed-identity control

当前严格替换同时改变了音频内容和封装合同：条件 A 是 Wav2Lip 生成的 MP4/AAC 音轨，条件 B 是 MKV/PCM 原始自然音轨。SyncNet 会解码两者，编解码差异通常不应单独解释如此大的变化，但严格归因前仍需增加：

> 保留同一条件 A 视频流，把原始 MFA WAV 也用完全相同的 MKV/PCM 无损方式重新封装并重新评分。

若 `MFA-video + MFA-PCM-MKV` 仍接近 `6.7226/6.9063`，才可更有力地排除容器、编码与 mux 路径，确认下降主要来自自然音频与 MFA 驱动口型的局部不匹配。在完成该 identity control 前，不应把全部 `-1.6435` Sync-C 都解释为嘴型内容差异。

### 产物路径

- run root：`runs/rhythm_timing/20260814_wav2lip_0813_mfa_bridge/`
- 生成元数据：`runs/rhythm_timing/20260814_wav2lip_0813_mfa_bridge/generation.json`
- 条件 A 视频：`runs/rhythm_timing/20260814_wav2lip_0813_mfa_bridge/wav2lip/mfa_linear/{1..15}.mp4`
- 条件 B 严格自然音频替换：`runs/rhythm_timing/20260814_wav2lip_0813_mfa_bridge/natural_audio_replace_strict/{1..15}.mkv`
- SyncNet 汇总：`runs/rhythm_timing/20260814_wav2lip_0813_mfa_bridge/syncnet_eval/summary.json`
- 配对分析：`runs/rhythm_timing/20260814_wav2lip_0813_mfa_bridge/syncnet_eval/analysis.json`
- 逐样本分数：`runs/rhythm_timing/20260814_wav2lip_0813_mfa_bridge/syncnet_eval/scores/mfa_audio/` 与 `.../scores/natural_audio_replace_strict/`
- 日志和 SyncNet 中间数据：`runs/rhythm_timing/20260814_wav2lip_0813_mfa_bridge/logs/`、`runs/rhythm_timing/20260814_wav2lip_0813_mfa_bridge/syncnet_eval/syncnet_data/`。

- [result] 0813 MFA-linear Wav2Lip 为 Sync-C `6.7226`、Sync-D `6.9063`；自然生成基线为 `5.7581/7.2690`；同帧严格替换自然音频为 `5.0791/8.2683`。 #0813 #wav2lip #counterfactual
- [decision] 原音替换采用 Matroska + PCM_S16LE，并验证视频流与解码 PCM 的逐条 MD5；不改变速度、波形、样本数，不裁剪、不补静音。 #audio-integrity
- [insight] 相同总时长和近似听感不保证 phone 内声学事件逐帧一致；25 fps 下几十毫秒局部差异即可影响 SyncNet。 #alignment #syncnet
- [insight] 当前下降包含 MFA 音频的生成/可读性优势与自然音频替换后的局部口型不匹配两部分，不能仅凭总长度解释。 #counterfactual
- [risk] AAC/MP4 与 PCM/MKV 合同不同仍是次要混杂；需运行 MFA-waveform 的同视频流 PCM/MKV processed-identity control。 #provenance
- [boundary] 本轮是本地 Wav2Lip 音频反事实，不是 TFG/Ditto 重新生成；自然生成臂使用核验 cache，结论不能外推到 heldout speaker 或跨模型。 #scope

### 固定 cohort、输入与模型

- cohort：AISHELL-1 valid split，单说话人 `S0765`，预先固定 paired-key 排序前 15 条；heldout `S0770` 未加载、选择或评估。
- 冻结清单：`runs/rhythm_timing/20260814_wav2lip_0813_mfa_bridge/manifest.json`。
- 0813 MFA-linear 输入音频：`runs/knn_vc_poc_valid15_mfa_linear_bridge_20260813/audio/*__paired_tts_mfa_linear.wav`；源汇总：`runs/knn_vc_poc_valid15_mfa_linear_bridge_20260813/summary.json`。
- 自然输入音频：`results/rhythm_style_500/aishell1_test_400/natural/0001.wav` … `0015.wav`。
- Wav2Lip face 输入：`runs/two_stage_hubert_aishell1_20260810/ditto_videos/natural_raw/1.mp4` … `15.mp4`；仅作为统一视觉输入，不表示本轮重新运行 Ditto。
- Wav2Lip Python：`[redacted-local-path]`；入口：`third_party/Wav2Lip/inference.py`；checkpoint：`third_party/Wav2Lip/checkpoints/wav2lip_gan.pth`；SHA256 `ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8`。
- Wav2Lip 参数：`--face_det_batch_size 4 --wav2lip_batch_size 4 --nosmooth`。
- SyncNet Python：`[redacted-local-path]`；代码目录：`third_party/syncnet_python/`；模型：`third_party/syncnet_python/data/syncnet_v2.model`；SHA256 `961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442`；`min_track=50`。

### 相关代码与运行协议

- Wav2Lip/SyncNet 评测协议参考并复用了 `scripts/eval_valid15_same_face_wav2lip_syncnet.py`、`scripts/eval_mfa_linear_wav2lip_syncnet.py` 的逐样本独立工作目录、模型哈希和 SyncNet 解析方式。
- 每个样本使用独立 `wav2lip_work/` cwd/temp，避免 Wav2Lip 相对路径 `temp/temp.wav`、`temp/result.avi` 造成跨样本污染。
- 条件 A 的实际 Wav2Lip 命令记录在 `generation.json`，核心形式为：

```bash
[redacted-local-path] third_party/Wav2Lip/inference.py \
  --checkpoint_path third_party/Wav2Lip/checkpoints/wav2lip_gan.pth \
  --face <natural_raw/{id}.mp4> \
  --audio <0813__paired_tts_mfa_linear.wav> \
  --outfile <new_output.mp4> \
  --face_det_batch_size 4 --wav2lip_batch_size 4 --nosmooth
```

- 条件 B 仅替换条件 A 视频的音频流，使用严格无损 mux：

```bash
ffmpeg -n -i <condition_a.mp4> -i <natural.wav> \
  -map 0:v:0 -map 1:a:0 -c:v copy -c:a copy <output.mkv>
```

使用 Matroska + `pcm_s16le`，不做 `atempo`、裁剪、`-shortest` 或静音填充；逐条验证视频流 MD5 与条件 A 相同，解码 PCM MD5 与自然 WAV 相同，`audio_transform=none`、`video_stream_preserved=true`、`audio_samples_preserved=true`。

### 产物路径

- run root：`runs/rhythm_timing/20260814_wav2lip_0813_mfa_bridge/`
- 生成元数据：`runs/rhythm_timing/20260814_wav2lip_0813_mfa_bridge/generation.json`
- 条件 A 视频：`runs/rhythm_timing/20260814_wav2lip_0813_mfa_bridge/wav2lip/mfa_linear/{1..15}.mp4`
- 条件 B 严格自然音频替换：`runs/rhythm_timing/20260814_wav2lip_0813_mfa_bridge/natural_audio_replace_strict/{1..15}.mkv`
- SyncNet 汇总：`runs/rhythm_timing/20260814_wav2lip_0813_mfa_bridge/syncnet_eval/summary.json`
- 配对分析：`runs/rhythm_timing/20260814_wav2lip_0813_mfa_bridge/syncnet_eval/analysis.json`
- 逐样本分数：`runs/rhythm_timing/20260814_wav2lip_0813_mfa_bridge/syncnet_eval/scores/mfa_audio/` 与 `.../scores/natural_audio_replace_strict/`
- 日志和 SyncNet 中间数据：`runs/rhythm_timing/20260814_wav2lip_0813_mfa_bridge/logs/`、`runs/rhythm_timing/20260814_wav2lip_0813_mfa_bridge/syncnet_eval/syncnet_data/`。

### 结果（n=15，所有 30 个 condition cells 完成，0 failures）

| 条件 | Sync-C ↑ | SD | Sync-D ↓ | SD | AV offset 均值 |
|---|---:|---:|---:|---:|---:|
| `wav2lip_0813_mfa_audio` | 6.7226 | 0.6302 | 6.9063 | 0.3745 | -2.0000 |
| `wav2lip_0813_video_natural_audio_strict` | 5.0791 | 0.9809 | 8.2683 | 0.8491 | -2.0667 |

严格自然音频替换相对 MFA 音频驱动视频：ΔSync-C `-1.6435`、ΔSync-D `+1.3621`；Sync-C 改善 `0/15`，Sync-D 改善 `0/15`，联合改善 `0/15`。因此在该 Wav2Lip counterfactual 中，口型运动明显依赖 0813 MFA-linear 音频；把音频替换为原始自然语音而保持视频帧不变，会显著降低同步分数。该结论只适用于 `S0765` valid-only、固定 natural face 输入和 Wav2Lip/SyncNet 协议，不能外推为 TFG/Ditto、heldout speaker、跨模型泛化或自然音频质量结论。

与既有 `runs/rhythm_timing/20260813_syncnet_valid15_mfa_bridge/summary.json` 的 Wav2Lip MFA-linear 参考结果接近（既有 mean Sync-C 约 `6.6997`、Sync-D 约 `6.9095`），但本轮独立重生成不要求 bit-identical；差异可来自独立 Wav2Lip 编码/人脸检测。

- [result] 0813 MFA-linear 音频驱动的 Wav2Lip 条件 mean Sync-C `6.7226`、Sync-D `6.9063`；严格替换自然音频后变为 `5.0791`、`8.2683`。 #0813 #wav2lip #counterfactual
- [decision] 原音替换采用 Matroska + PCM_S16LE，并验证视频流与解码 PCM 的逐条 MD5；不改变速度、波形、样本数，不裁剪、不补静音。 #audio-integrity
- [insight] 在相同生成视频帧上，替换自然音频使 Sync-C/Sync-D 全部 15 对变差，支持口型运动由 MFA 音频条件驱动。 #syncnet
- [boundary] 本轮是本地 Wav2Lip 音频反事实，不是 TFG/Ditto 重新生成；不能用于宣称 TFG 的自然音频替换效果。 #scope

## Natural silence anchor / phone-boundary smoothing（2026-08-14）

固定 alpha=1，在 S0765/S0901/S0912 各 5 条 paired utterance 上测试 natural silence anchor 和局部 boundary smoothing。输入与模型保持不变：cohort `runs/aishell1_mfa_linear_n25_20260813/data_boundary/aishell1_n25_speaker_balanced_cohort.json`、resample-poly TTS `runs/aishell1_mfa_linear_n25_resample_poly_20260813/tts_meta.json`、tokens `runs/aishell1_mfa_linear_n25_20260813/mfa/tokens.json`、reference `runs/aishell1_mfa_linear_n25_resample_poly_20260813/mfa_linear/summary.json`、frozen knn-vc revision `c616845c3e309e24d5927f15adbdf277a3d65358`。

代码：`scripts/run_natural_anchor_boundary_silence.py`；测试：`tests/test_natural_anchor_boundary_silence.py`；输出：`runs/aishell1_mfa_linear_n25_resample_poly_20260814/natural_anchor_boundary_20260814/`。条件为 unconstrained、silence anchor、boundary smoothing ±2/±3 frames 及两种组合；90/90 成功、exact-N、finite、0 clipping；相关测试总计 22 passed。

### 结果

S0901 mean peak/crest：unconstrained `0.3313/12.449`；silence anchor `0.3380/12.649`；boundary r2 `0.3315/12.272`；boundary r3 `0.3585/13.257`；combined r2 `0.3357/12.314`；combined r3 `0.3618/13.275`。没有稳定改善，r3 变差。S0912 mean peak/crest：unconstrained `0.4991/8.647`；silence anchor `0.4810/8.463`；boundary r2 `0.4932/8.545`；combined r2 `0.4854/8.576`，存在小幅改善但不普适。

Silence anchor 提高 output→natural feature cosine（S0901 `0.6983→0.8094`，S0912 `0.7441→0.7952`），但这再次说明 feature proximity 不等价于音质/可懂度/SyncNet。S0901/103 silence anchor peak 从 `0.2312` 升至 `0.2618`，不能修复重点样本。

### 评测协议限制

已有 mux 实验显示：将生成视频的音频轨替换成原始音频后，SyncNet 分数会下降。因此 SyncNet 对音频轨 sample rate、编码和 mux 路径敏感；未来必须统一 natural/raw/generated 的解码音频轨合同，并保留 processed-identity control，不能把 audio-track replacement 的分数差异直接解释为嘴型质量。

[decision] 当前 silence/boundary frozen feature 修改没有稳定修复，暂不进入下游；下一步若继续，优先做 phone-context-aware transition 或 natural F0/energy conditioning，并在统一音频轨协议下评估。 #mfa-linear #audio #syncnet

## Phone-context residual crossfade（2026-08-14）

在 alpha、residual norm/velocity、silence anchor 和 moving-average boundary smoothing 均不能稳定修复 S0901 后，新增 `scripts/run_natural_anchor_context_crossfade.py` 与 `tests/test_natural_anchor_context_crossfade.py`。固定 alpha=1，在 natural MFA phone boundary 两侧用 context residual 做 ±2/±3 WavLM frame 线性过渡；测试 unconstrained、r2/r3、silence-anchor+r2/r3。

输入沿用冻结 n25 cohort、resample-poly TTS、strict tokens 和 `bshall/knn-vc` revision `c616845c4e309e24d5927f15adbdf277a3d65358`（WavLM-Large L6 + prematched HiFi-GAN）。输出为 `runs/aishell1_mfa_linear_n25_resample_poly_20260814/natural_anchor_context_20260814/`。75/75 成功、exact natural length、finite、0 clipping；unconstrained 与已有 MFA-linear 15/15 数值等价（max abs <= 1e-5）；最终相关回归 `24 passed`。

S0901 peak/crest 从 unconstrained `0.3313/12.449` 变为 r2 `0.3369/11.955`、r3 `0.3618/11.917`；S0912 从 `0.4991/8.647` 变为 r2 `0.4970/8.429`、silence+r2 `0.4847/8.346`、r3 `0.5875/10.053`。重点样本 S0912/203 的 r3 peak 从 `0.6226` 增至 `0.6970`。r2 只有轻微且 speaker-dependent 的变化，r3 明显不稳定；更高 output→natural cosine 仍不等价于音质或同步改善。

[decision] phone-context crossfade 未通过 audio gate，不运行 Wav2Lip/SyncNet；停止扩大手工 residual smoothing 网格，下一阶段考虑可训练的 phone-level confidence/gating 或显式 content/prosody/speaker 分离。 #mfa-linear #training #speaker

## Vocoder-domain split（2026-08-14）

为区分 speaker-dependent 伪影来自 MFA-linear trajectory 还是 HiFi-GAN 输入域，新增 `scripts/run_aishell1_vocoder_domain_split.py` 与 `tests/test_aishell1_vocoder_domain_split.py`。固定 S0765/S0901/S0912 各 5 条，测试 natural direct、raw-TTS direct、MFA-linear 三种 feature 条件，经 regular `g_02500000.pt` 和 prematched `prematch_g_02500000.pt` 两种 frozen kNN-VC HiFi-GAN；共 90 个 audio-only cells。

输出：`runs/aishell1_mfa_linear_n25_resample_poly_20260814/vocoder_domain_split_20260814/`。90/90 成功、0 failures、MFA-linear exact natural length、0 clipping；MFA-linear prematched 与已有 reference 15/15 数值完全一致（max abs diff 0）；相关测试 `6 passed`。

Prematched→regular 的 MFA-linear mean peak：S0765 `0.0782→0.0820`，S0901 `0.3313→0.2533`，S0912 `0.4991→0.3886`。S0901/S0912 的 raw-TTS direct 也同步降峰，说明 prematched HiFi-GAN 是 speaker-dependent amplitude/resynthesis confound。可是 regular vocoder 下 MFA-linear output→conditioning feature distance 相对 raw-TTS direct 仍上升：S0901 `0.0567→0.1406`，S0912 `0.0573→0.0924`；trajectory mismatch 没有消失。

[decision] 不再把 prematched HiFi-GAN 当作唯一 baseline；先扩展 regular/prematched 分叉到完整 n25 audio gate，再决定训练 regular-vocoder adapter 还是 learned phone-trajectory mapper。SyncNet 未用于模型选择。 #vocoder #speaker #mfa-linear

## Learned vocoder input calibration（2026-08-14）

完整 n25 domain split 后，训练了一个 2048 参数 identity-initialized bounded feature affine adapter：四个 train speaker（S0901/S0906/S0912/S0913）的 native natural/raw-TTS WavLM-L6 features 作为输入，冻结 regular HiFi-GAN 作为 teacher，冻结 prematched HiFi-GAN 作为 student；S0765 五条 speaker-disjoint valid 只用于验证。代码 `scripts/train_vocoder_input_adapter.py`，测试 `tests/test_train_vocoder_input_adapter.py`，输出 `runs/aishell1_mfa_linear_n25_resample_poly_20260814/vocoder_input_adapter_20260814/`。

第一次运行因 inference-mode WavLM tensor 不能参与 adapter backward 失败；在缓存处加入 `.detach().clone()` 后修复，相关测试 `4 passed`，250 steps 成功。训练 spectral loss `0.02910→0.01809`，waveform loss `0.02458→0.01310`。

S0765 validation 的 MFA-linear：prematched peak/crest `0.0782/8.858`，adapted `0.0699/8.394`，regular `0.0820/7.812`；adapted raw-TTS direct peak/crest `0.0747/8.708`。说明小 adapter 能部分降低 prematched 的 amplitude/crest confound，且 spectral flux 没有明显变差；但 MFA-linear output→conditioning distance 从 prematched `0.1297` 变为 adapted `0.1312`，没有提升 trajectory fidelity。

[decision] learned input calibration 是安全的局部 vocoder 修复，但不是 MFA-linear speaker/trajectory 问题的解决方案；暂不运行 SyncNet，下一步转向 learned phone-level trajectory model。 #training #vocoder #speaker

## Learned phone trajectory denoiser（2026-08-14）

A 659,584-parameter bounded temporal adapter was trained on native raw-TTS WavLM-L6 trajectories with artificial boundary/frame interpolation corruption, then applied to MFA-linear natural-clock features and frozen regular HiFi-GAN. Four speakers trained; S0765 was speaker-disjoint validation. The first run had an argument-order bug; the corrected crop-local run is in `runs/aishell1_mfa_linear_n25_resample_poly_20260814/phone_trajectory_adapter_cropfix_20260814/`, with tests `4 passed`.

The corrected training loss moved (feature `0.02469→0.02289`, velocity `0.05017→0.04644`), proving the denoiser received a real signal. However, MFA-linear output did not materially change: S0765 peak `0.0820→0.0818`, distance `0.0821→0.0821`; S0901 `0.2533→0.2536`, distance `0.1406→0.1407`; S0912 `0.3886→0.3887`, distance `0.0924→0.0924`.

[decision] Native raw-TTS corruption denoising does not transfer to the cross-phone-duration MFA-linear mismatch. This route is stopped; a useful mapper needs explicit duration-conditioned phone rendering or a pretrained TTS/codec decoder with separate content, prosody/timing and target-speaker controls. #trajectory #negative #training


## TTS–natural phone duration discrepancy audit（2026-08-14）

为检验负增益 speaker 是否因为 TTS 与 natural 的音素时长差距更大，新增 `scripts/analyze_aishell1_phone_duration_gap.py` 与 `tests/test_aishell1_phone_duration_gap.py`。使用冻结 n=25 的 paired strict-token `runs/aishell1_mfa_linear_n25_20260813/mfa/tokens.json`，并连接当前 resample-poly fixed-face1 评测 `runs/rhythm_timing/20260813_syncnet_aishell1_n25_resample_poly_face1/summary.json`。speaker 分组在计算 duration 前固定：good=`S0765/S0913`、poor=`S0901/S0912`、intermediate=`S0906`，没有使用 duration 值选组，也没有使用 S0770。

输出：`runs/aishell1_mfa_linear_n25_resample_poly_20260814/phone_duration_gap_audit_20260814/`，包含 `analysis.json`、`per_sample.jsonl`、`speaker_summary.json`、`group_summary.json` 和 `report.md`；3 个针对性测试通过。

### 结果

逐 matched speech phone 的 speaker 均值：

```text
speaker   group          phone MAE     relative abs gap   speech-duration TTS-natural
S0765     good           0.0139 s      0.1435             -0.176 s
S0901     poor           0.0308 s      0.2635             -0.660 s
S0906     intermediate   0.0230 s      0.2166             -0.424 s
S0912     poor           0.0222 s      0.1732             -0.464 s
S0913     good           0.0391 s      0.3119             -0.742 s
```

good group（S0765/S0913）与 poor group（S0901/S0912）比较：

```text
phone MAE:          good 0.0265 s, poor 0.0265 s, poor-good ≈ 0.0000 s
relative abs gap:   good 0.2277,   poor 0.2184,   poor-good -0.0093
speech total gap:   good -0.459 s,  poor -0.562 s,  poor-good -0.103 s
```

### 判断

当前证据不支持“坏 speaker 的 MFA-linear 失败主要由更大的逐音素 TTS–natural 时长误差造成”。S0901 确实同时有较大 phone MAE 和明显整句时长缩短，但 S0912 的 phone MAE 低于 S0906，S0913 作为效果较好的 speaker 反而拥有全组最大的 phone MAE。因此单纯扩大 duration-conditioned mapper 不能由这个分析直接得到支持。

poor group 的 TTS speech 总时长平均多短约 0.103 s，但这更像整体语速/停顿或长度偏差，不能替代逐 phone mismatch 证据。5 个 speaker 的 speaker-level duration–SyncNet 相关只作描述性关联，不能做统计或因果结论；固定 S0765 face 也混入 speaker–face mismatch。

[boundary] 这仍使用历史 strict tokens；完整 MFA 3.4.1 clean n=25 tokens 尚未完成。clean n25 后必须重算一次，尤其要复核 S0901/S0912 的结论。 #duration #speaker #mfa-linear
[insight] good/poor speaker 的逐 phone duration MAE 几乎相同，duration gap 不是当前 speaker-dependent MFA-linear 失败的充分解释。 #negative #speaker


## Phone duration 与音频伪影联合检查（2026-08-14）

进一步把 phone-duration 分析与已有 `audio_artifact_audit_20260813/per_file.jsonl` 按 sample identity 连接；不重新检测、不修改旧 audio audit。修正版输出为 `runs/aishell1_mfa_linear_n25_resample_poly_20260814/phone_duration_gap_audit_fix_20260814/`，包含 audio audit SHA 与 duration/artifact speaker-level correlations。

MFA-linear 每 speaker 的 audio-artifact-event mean：

```text
S0765: artifact 0.4, boundary 1.2, repetition 0.0
S0901: artifact 4.0, boundary 1.4, repetition 0.0
S0906: artifact 0.6, boundary 0.2, repetition 0.0
S0912: artifact 7.4, boundary 1.2, repetition 0.0
S0913: artifact 0.2, boundary 0.0, repetition 0.0
```

phone MAE 与 artifact-event count 的 speaker-level Pearson `r=-0.088`，与 artifact-event duration `r=-0.063`；与 boundary-event count 为 `r=-0.491`，方向也不是“duration 越大伪影越多”。S0912 是 artifact 候选最多的 speaker，但 phone MAE=`0.0222 s`；S0913 是 phone MAE 最大的 good speaker（`0.0391 s`），但 artifact 候选最少。因此当前听感异常更像 speaker/vocoder/trajectory/boundary 相关的组合问题，而不是单一 TTS-natural phone duration gap。

[insight] phone duration mismatch 既不能解释 good/poor MFA-linear 分组，也不能解释 audio-artifact detector 的 speaker 排序；clean MFA-3 n25 后需重算，但不应仅凭 duration gap 训练 mapper。 #duration #artifact #speaker


## Sample 101 cloud Qwen/MiniMax TTS comparison（2026-08-14）

样本 `101` 属于 `S0901`，paired key=`aishell1_test_400__BAC009S0901W0122`，文本为“其市管国管住房公积金政策也均进行调整”。以 sample 101 的 natural 音频作为 reference，尝试了 Qwen DashScope 和 MiniMax voice-clone；没有覆盖既有结果，也没有把 API key 写入文件或 metadata。

### Qwen DashScope

Qwen voice registration 最终成功；本地适配器原先自动生成 `prefix_101`，下划线导致 `preferred_name` 被拒绝，改用合法名称后复用注册 voice 完成 synthesis。

- model：`qwen3-tts-vc-2026-01-22`
- output：`runs/aishell1_mfa_linear_n25_resample_poly_20260814/cloud_tts_sample101_qwen_retry2_20260814/qwen/101.wav`
- canonical：16 kHz mono PCM、`93440` samples、`5.840 s`、peak=`0.2968`、finite
- output SHA：`c367130b6575f4a711b9b329a1f6f5beca02372381701a073140425c925ec081`

相对 sample 101 natural（`8.333 s`）和原本地 TTS（`4.640 s`），Qwen 时长为 `5.840 s`，整体更接近 natural。单样本 acoustic proxy 也更接近 natural：spectral centroid `1484` vs natural `1528` vs local `1881 Hz`；spectral flatness `0.0121` vs `0.0155` vs `0.0824`；ZCR `0.0898` vs `0.0922` vs `0.1399`。但 Qwen F0 mean=`131.0 Hz`，仍高于 natural `96.3 Hz`，且时长仍短 `2.493 s`。

### 同环境 clean MFA-3 三路对齐

使用 MFA `3.4.1`、`mandarin_china_mfa`、`mandarin_mfa`、同一 character-segmented transcript 和同一单 speaker corpus 对齐 natural/local TTS/Qwen：

```text
                 speech phones  total duration  silence duration  phone MAE vs natural
natural           50             8.333 s         2.813 s           —
local TTS         50             4.640 s         1.180 s           43.6 ms
Qwen cloud        50             5.840 s         0.770 s           23.0 ms
```

Qwen 将 matched phone duration MAE 从本地 TTS 的 `43.6 ms` 降至 `23.0 ms`，signed per-phone delta 也从 `-41.2 ms` 改善到 `-9.0 ms`。但 Qwen 的 pause/silence duration 只有 `0.770 s`，比 natural 少 `2.043 s`；local TTS 少 `1.633 s`。因此 Qwen 更接近 natural 的 phone duration 和频谱形态，却没有保留 natural 的停顿节奏，不能称为完整 style preservation。

### MiniMax

使用旧 `/v1/files/upload` 流程在 `api.minimax.chat` 和当前 `api.minimaxi.com` 均返回 `status_code=1004 login fail`，没有生成音频。该 key 可能是 MiniMax Token Plan/其他服务凭证，或没有 Speech API 权限；没有继续重复请求。

### 产物

- consolidated summary：`runs/aishell1_mfa_linear_n25_resample_poly_20260814/cloud_tts_sample101_qwen_retry2_20260814/comparison_summary.json`
- style proxy：`style_qc.json`
- same-environment MFA：`mfa3_alignment_threeway/threeway_summary.json`
- failed/diagnostic attempts are kept in separate `cloud_tts_sample101_20260814/` and `mfa3_alignment*/` directories.

[insight] 更换 Qwen 云端 voice-clone 确实改善了 S0901/101 的 phone-duration 与部分频谱 proxy，但主要缺陷从 phone duration 转移为停顿缺失；voice cloning 不能自动复制 natural prosody。 #qwen #speaker #style #mfa
[boundary] 仅有单条样本，尚未做 Wav2Lip/SyncNet；不能据此宣称 speaker 风格或下游泛化已解决。 #scope


## Sample 101 Fish Audio 尝试（2026-08-14）

使用 sample 101 / `S0901` natural `0101.wav` 作为 reference，调用仓库现有 `FishAudioProvider`、model=`s2-pro`、private voice model。Fish Audio 的 voice model 创建阶段成功，但 `/v1/tts` 合成阶段返回：

```text
HTTP 402: Insufficient API credit. API credit is managed independently from platform credit.
```

因此没有生成音频；这不是 API key 格式、模型名或本地音频请求格式错误。失败尝试目录：`runs/aishell1_mfa_linear_n25_resample_poly_20260814/cloud_tts_sample101_fish_20260814/`。provider registry 保留了已创建的 Fish voice model 信息，后续补充 API credit 后可复用，不必重新创建 voice model。

[boundary] 当前没有 Fish Audio 音频可试听，也不应把这次失败当作模型质量负结果；需要在 Fish developer 页面补充独立 API credit 后再重试。 #fish-audio #api #scope


## Fish Audio retry success（2026-08-14）

使用新 Fish API key 复用已创建的 sample 101 voice model，`s2-pro` synthesis 成功。canonical 音频：`runs/aishell1_mfa_linear_n25_resample_poly_20260814/cloud_tts_sample101_fish_retry_20260814/fish/101.wav`；16 kHz mono、`90604` samples、`5.66275 s`、peak=`0.9555`、SHA=`1ae86dd6767ebc90220779983a245846227db810c18ec6445637249235f0e019`。尚未做 MFA/Wav2Lip/SyncNet，peak 接近 1 需试听时注意潜在高电平。

[result] Fish Audio voice-clone 已生成可试听音频；前一次失败是旧 key 的 API credit 不足，新 key 成功。 #fish-audio #speaker


## Sample 101 Qwen/Fish MFA-linear listening outputs（2026-08-14）

使用同一四路 MFA-3 corpus（natural/local/Qwen/Fish）、`mandarin_china_mfa`、`mandarin_mfa`、共同 natural time grid，以及 frozen WavLM-Large L6 + prematched HiFi-GAN 生成两条 MFA-linear 音频。两条均 exact natural length=`133328` samples、finite。

- Qwen MFA-linear：`runs/aishell1_mfa_linear_n25_resample_poly_20260814/cloud_tts_mfa_linear_sample101_retry_20260814/audio/qwen/101.wav`；peak=`0.4144`；MFA coverage=`0.9952`、fallback=`2`。
- Fish MFA-linear：`runs/aishell1_mfa_linear_n25_resample_poly_20260814/cloud_tts_mfa_linear_sample101_retry_20260814/audio/fish/101.wav`；peak=`0.9237`；MFA coverage=`0.8125`、fallback=`78`。

[boundary] Fish MFA phone matching coverage 明显低于 Qwen，试听时需把 Fish MFA-linear 当作低覆盖 exploratory output，不能与 Qwen 进行等价质量结论。 #mfa-linear #fish-audio #qwen


## Corrected LRS3 target-context RhythmWarp retraining and multimodal evaluation（2026-08-18）

为排查 RhythmWarp 生成音频及其自然音轨替换视频效果较差的问题，重新检查了 LRS3 validation 数据准备、训练循环和下游 Wav2Lip/SyncNet 评测。实验保留旧 cache、旧 checkpoint 和旧结果不变，所有 corrected 产物写入独立目录。

### 已修复的问题

`scripts/prepare_lrs3_rhythm_dataset.py` 原来使用 paired TTS token 的 `start_s/end_s`（TTS source 时间轴）截取自然视频 visual embedding；正确的 visual context 应使用 `target_start_s/target_end_s`（自然语音时间轴）。修复后新增 `phone_context_from_visual()`，并添加 target-timeline regression test。

`scripts/train_lrs3_rhythm_warp.py` 的 validation 函数会调用 `model.eval()`，但返回训练循环后没有恢复 `model.train()`，导致 validation 后下一次 cuDNN GRU backward 报错。现在会保存并恢复调用前的 training mode，并添加 regression test。

相关测试最终为 `15 passed`。

### Corrected 数据集

- 目录：`runs/lrs3_qwen_cloud_n500_20260818/03_rhythm_data_mfa_target_context/`
- 500 条尝试，498 条有效，2 条显式失败；43 个 source groups；validation 69 条；
- corrected visual-context cosine mean：`0.99999997`；
- 旧 cache 的 context cosine mean：`0.1516`；phone 中点平均错位约 `1.2184 s`，`83.98%` 的 phone visual window 完全无重叠；
- source manifest、TTS hash 和 provenance 校验通过。

本地 LRS3 metadata 没有可靠 speaker ID，因此该 validation 不能宣称 speaker-disjoint 泛化。

### Corrected 训练

- checkpoint：`runs/lrs3_qwen_cloud_n500_20260818/04_rhythm_train_mfa_target_context_retry/checkpoint.pt`
- train/validation：345/69；full objective；validation path loss 每 100 steps 评估；patience=5；
- best step：`300`；实际在 `800` steps early stop；
- validation path loss：`0.932157 → 0.931140`；
- boundary MAE：global-resample baseline `0.424066 s`，corrected RhythmWarp `0.415243 s`，改善约 `2.08%`；
- checkpoint dataset SHA 与 corrected dataset SHA 匹配。

### Corrected LRS3 validation 下游评测

输出：`runs/rhythm_timing/20260818_lrs3_validation_repaired_wav2lip_syncnet/`

音频 gate：69/69 生成成功，69/69 exact natural length，全部 finite，0 clipping，0 failures。Wav2Lip/SyncNet 共 `207/207` score cells，0 failures；8 条 face detector 漏检样本使用同一视频的固定 box fallback，natural 和 RhythmWarp arms 保持一致。

| 条件 | Sync-C（越高越好） | Sync-D（越低越好） |
|---|---:|---:|
| natural audio → Wav2Lip | 6.9000 | 7.5345 |
| corrected RhythmWarp audio → Wav2Lip | 6.1575 | 8.5082 |
| corrected RhythmWarp video + natural audio | 2.0720 | 12.4749 |

corrected RhythmWarp 相对 natural：ΔC=`-0.7425`、ΔD=`+0.9737`；C 改善 `14/69`，D 改善 `6/69`，joint improvement `4/69`。相对修复前 RhythmWarp 的 `6.1549/8.5077`，修复后为 `6.1575/8.5082`，下游几乎没有变化。

音轨替换完整性通过：69/69 视频流保持，69/69 解码 PCM 保持。此前对 natural-audio Wav2Lip 视频做相同自然音频的 re-mux control，Sync-C/D 从 `6.988/7.016` 变为 `7.028/6.924`，因此 replacement 的大幅下降不是 Matroska/PCM/ffmpeg mux bug。

replacement arm 必须解释为“由 RhythmWarp 音频驱动生成的 video frames 与自然音频的跨条件 mismatch”测试，不能当作自然音频质量的独立评分。它的低分说明生成视频帧依赖 RhythmWarp 音频条件，不代表自然音频本身一定差。

### Corrected 与旧版本的 Sight 定性对比

固定 validation sample，而不是按分数挑选，进行了旧/修复版匹配视频的全模态检查：

- `lrs3_6VWPHKABRQA_00003`：旧版约 4/5，修复版约 3/5；修复版出现轻微滞后、plosive closure 不清和局部 micro-jitter；
- `lrs3_6XS8TA4RBog_00002`：旧版约 3/5，修复版约 2/5；修复版有约 60–120 ms 拖尾描述、口型模糊和快速片段 warping；
- `lrs3_6YKYo00mFAg_00003`：旧版约 3/5，修复版约 2/5；两者均有 blur/弱 plosive，修复版有更多 micro-stutter 描述；
- `lrs3_6qTfX4U6CS0_00002`：旧版约 2/5，修复版约 3/5；修复版连续性略好，但仍有 blur 和软化。

Sight 结果与 SyncNet 不完全一致，没有显示 corrected audio/video 一致优于旧版的主观趋势；Sight 只作定性辅助证据，不替代完整 69 条 paired SyncNet 矩阵。

### 当前结论

- target-timeline visual-context 错位已被真实修复；
- validation mode/cudNN 训练 bug 已修复；
- corrected boundary MAE 有小幅改善，但没有转化为下游 Wav2Lip/SyncNet 改善；
- corrected RhythmWarp 在 69 条 LRS3 validation 上仍是 downstream No-Go；
- 不应继续单纯增加同一模型训练步数。

更值得重查的是 `pseudo_sync_boundaries()` 的监督可靠性、full objective 中 offline Sync path loss 在非 sync-only 模式被 detach 的设计，以及 Wav2Lip/HiFi-GAN 对重合成 WavLM feature 的声学域偏差。当前结果仍不能声称 speaker-disjoint 或跨模型泛化。

- [problem] target visual context 原来用 TTS source timeline，导致 validation 中大多数 phone context 与自然视频窗口错位。 #lrs3 #visual-context
- [solution] 改用 `target_start_s/target_end_s` 并加入 regression test；corrected cache context cosine 达到 `0.99999997`。 #alignment #provenance
- [solution] validation 后恢复 RhythmWarp training mode，消除 cuDNN GRU backward failure。 #training
- [result] corrected boundary MAE 仅从 `0.424066 s` 改善到 `0.415243 s`，下游 SyncNet 没有实质改善。 #negative #syncnet
- [result] corrected LRS3 evaluation completed `207/207` cells with 0 failures and preserved replacement video/PCM integrity. #wav2lip #syncnet
- [insight] replacement 的极低分是 generated-video/natural-audio cross-condition mismatch，不能作为自然音频质量独立结论。 #counterfactual #audio-integrity
- [boundary] local LRS3 metadata 不支持 speaker-disjoint 泛化声明；Sight 结果仅为定性辅助。 #scope
- [decision] 下一步优先重审 pseudo Sync boundary 与 training objective，不重复同一 corrected RhythmWarp 训练。 #future-work
