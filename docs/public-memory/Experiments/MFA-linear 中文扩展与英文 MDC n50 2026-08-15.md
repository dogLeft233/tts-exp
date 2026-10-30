---
title: MFA-linear 中文扩展与英文 MDC n50 2026-08-15
type: experiment
permalink: tts-exp/experiments/mfa-linear-中文扩展与英文-mdc-n50-2026-08-15
date: '2026-08-15'
status: done
s0770_used: false
tags:
- mfa-linear
- chinese
- english
- mdc
- ramc
- alimeeting
- syncnet
---

# MFA-linear 中文扩展与英文 MDC n50 2026-08-15

## Context

本线程核对了 MFA-linear 中文数据扩展，并建立了英文对照实验。记忆中的两个中文自然语音数据集是 MagicData-RAMC 与 AliMeeting；它们此前已经完成各 25 条、各 25 个说话人的独立试点，因此没有重复下载或覆盖旧结果。本轮在此基础上完成了独立的 MDC English n=50 MFA-linear run。

## 中文数据集状态

- MagicData-RAMC：电话双人自由对话，25 条、25 speakers，带金标逐轮转录和时间戳；许可 CC BY-NC-ND 4.0。
- AliMeeting Eval：真实会议近场声道，25 条、25 speakers，带 TextGrid 转录；许可 CC BY-SA 4.0。
- 两者的中文 MFA-linear 三臂试点、MFA、冻结 WavLM-L6/prematched HiFi-GAN、Wav2Lip/SyncNet 产物均已存在。
- 原始多样域试点：AliMeeting MFA-linear 相对 natural ΔSync-C `-0.899`（p=`0.040`）；RAMC `-1.223`（p=`0.013`）。这与 AISHELL-1 朗读域的正向结果相反，说明自发/对话域不能直接视为 AISHELL-1 的扩大版。
- 后续 RAMC cloud-Qwen 重查把 ΔSync-C 改善到 `+0.40088`，但 Sync-D `+0.02884`（变差方向），仍未通过严格双指标 gate；不能把它写成路线成功。

## Cloud-Qwen 中文重跑（2026-08-16）

为区分“本地 faster_qwen3 轨迹较弱”和自发语音时钟本身的问题，按用户要求对同一 RAMC/AliMeeting cohort 重新调用云端 Qwen VC；没有覆盖旧本地或旧 RAMC cloud-Qwen 结果。

- 模型：`qwen3-tts-vc-2026-01-22`；每条 natural 音频作为该条 reference；RAMC/AliMeeting 各 25 条、各 25 speakers；未使用 S0770。
- TTS：RAMC 25/25、AliMeeting 25/25；metadata 保存 reference/canonical/provider/token 所需 hash，API key 未写入文件。
- MFA：独立 MFA 3.4.1 runtime `[redacted-local-path]`，`mandarin_china_mfa` dictionary + `mandarin_mfa` acoustic model；natural/Qwen 均 50/50 TextGrid。
- 对齐结果保留 unknown speech：RAMC 6 个（样本 06/12/15 的 natural/Qwen 成对 token），AliMeeting 8 个（样本 01/02/10/20 的 natural/Qwen 成对 token）；没有把 `spn` 转为 silence。
- MFA-linear：冻结 WavLM-Large L6 + pinned prematched HiFi-GAN；RAMC/AliMeeting 均 25/25，finite mono 16 kHz，exact natural length，无 0.999 clipping。
- Audio QC：RAMC mean coverage `0.9282`、min `0.8480`、fallback `769`；AliMeeting mean coverage `0.9232`、min `0.8169`、fallback `1327`。coverage 只作诊断，不作退化的因果解释。

### Cloud-Qwen fixed-face SyncNet 结果

固定 `data/data/image/1.png`，`min_track=50`，每个 cell 独立 Wav2Lip/SyncNet cwd/temp；两域均 75/75 分数完整。Sync-C 越高越好，Sync-D 越低越好；结果仍是 fixed-face exploratory，不代表 speaker-matched 泛化或嘴型因果。

| 域/arm | mean Sync-C | mean Sync-D |
|---|---:|---:|
| RAMC natural raw | 6.32384 | 7.13368 |
| RAMC cloud-Qwen raw TTS | 6.98580 | 6.96964 |
| RAMC cloud-Qwen MFA-linear | 6.68076 | 7.12952 |
| AliMeeting natural raw | 7.08004 | 6.79412 |
| AliMeeting cloud-Qwen raw TTS | 7.66576 | 6.71100 |
| AliMeeting cloud-Qwen MFA-linear | 7.28272 | 6.79204 |
| pooled natural raw (n=50) | 6.70194 | 6.96390 |
| pooled cloud-Qwen raw TTS (n=50) | 7.32578 | 6.84032 |
| pooled cloud-Qwen MFA-linear (n=50) | 6.98174 | 6.96078 |

配对差异：

- RAMC raw TTS − natural：ΔC `+0.66196` (p=`0.000155`)，ΔD `-0.16404` (p=`0.1697`)；MFA-linear − natural：ΔC `+0.35692` (p=`0.01982`)，ΔD `-0.00416` (p=`0.9669`)；MFA-linear − raw TTS：ΔC `-0.30504` (p=`0.01018`)，ΔD `+0.15988` (p=`0.1015`)。
- AliMeeting raw TTS − natural：ΔC `+0.58572` (p=`4.82e-06`)，ΔD `-0.08312` (p=`0.2231`)；MFA-linear − natural：ΔC `+0.20268` (p=`0.01839`)，ΔD `-0.00208` (p=`0.9717`)；MFA-linear − raw TTS：ΔC `-0.38304` (p=`3.2e-05`)，ΔD `+0.08104` (p=`0.1813`)。
- pooled n=50：MFA-linear − natural 的 ΔC `+0.27980` (p=`0.001277`)，ΔD `-0.00312` (p=`0.9564`)；MFA-linear − raw TTS 的 ΔC `-0.34404` (p=`4e-06`)，ΔD `+0.12046` (p=`0.03369`)。因此 cloud-Qwen MFA-linear 相对 natural 只在 Sync-C 上有稳定正向，且两个域都明显落后于 cloud-Qwen raw TTS；严格双指标 gate 未通过。

### 新重跑产物

- RAMC：`runs/ramc_alimeeting_pilot50/ramc25_qwen_cloud_rerun_20260815/`
- AliMeeting：`runs/ramc_alimeeting_pilot50/alimeeting25_qwen_cloud_rerun_20260815/`
- 汇总分析：`runs/ramc_alimeeting_pilot50/chinese_qwen_cloud_rerun_20260816_analysis_extended.json`

这次结果强化了“云端 TTS 能恢复 raw-TTS 的 Sync-C 红利，但把 cloud-Qwen trajectory 硬映射到自发语音 natural clock 后仍损失相当部分收益”的判断；下一步不应继续重复同一 50 条或把 MFA-linear 宣称为优于 raw TTS。

## 英文数据与协议

主 cohort 选择现有 **Effect AI Scripted Speech 1.0 - English (MDC)** 50 条：逐条英文转录、原始音频已下载、CC0-1.0，且不是 LibriSpeech encoder pretraining 测试集。现有 LibriSpeech English 只有 13 条并使用 native `.phn`，不作为本轮 MFA 扩展主 cohort。

- Run: `runs/mdc_en_mfa_linear_50_20260815/`
- TTS：本地 `faster_qwen3` 0.6B self-clone，50/50；未调用云端 API。
- MFA：3.4.1，独立 `MFA_ROOT_DIR=[redacted-local-path]`，`english_mfa` dictionary/acoustic model；natural/TTS 100/100 TextGrid entries 完成。
- `en_015` natural 初次批处理未导出 TextGrid；隔离单条重跑成功后合并，未静默丢弃。
- MFA-linear：冻结 WavLM-Large L6 + pinned kNN-VC/prematched HiFi-GAN，50/50 finite exact-natural-length waveforms。
- Wav2Lip/SyncNet：固定 `data/data/image/1.png`，`min_track=50`，三臂 150 个目标 cell 中 149 个有分数；完整三臂 paired n=49。`en_011/raw_tts` 在 `min_track=50` 下没有可解析 Confidence，重复 cell 仍失败；保留为缺失，不降低阈值、不插补。

## English downstream result (fixed-face exploratory)

配对 n=49，Sync-C 越高越好、Sync-D 越低越好。

| comparison | ΔSync-C | 95% CI | ΔSync-D | 95% CI | joint |
|---|---:|---|---:|---|---:|
| raw TTS − natural | `+0.5037` | `[+0.0783,+0.8869]` | `-0.4099` | `[-0.6384,-0.1778]` | 31/49 |
| MFA-linear − natural | `+0.2952` | `[+0.0669,+0.5287]` | `-0.4150` | `[-0.6002,-0.2199]` | 25/49 |
| MFA-linear − raw TTS | `-0.2085` | `[-0.6052,+0.2160]` | `-0.0051` | `[-0.1973,+0.1662]` | 9/49 |

Paired t p-values：

- raw TTS vs natural：Sync-C `0.0205`，Sync-D `0.00135`。
- MFA-linear vs natural：Sync-C `0.0171`，Sync-D `0.000121`。
- MFA-linear vs raw TTS：Sync-C `0.3207`，Sync-D `0.9573`。

英文 n=49 fixed-face cohort 中，MFA-linear 相对 natural 两个指标均为正向，但没有超过 raw TTS；它与 raw TTS 的差异不显著。结果比既有 LibriSpeech n=13 更有信息量，但仍不能说明英文 MFA-linear 优于 raw TTS，也不能说明 speaker-matched 泛化。

## Provenance/schema fixes

- `scripts/generate_mdc_english_tts.py`：补齐 `sample_id`、`reference_sha256`、`canonical_16k_audio`、`source_audio_sha256`、`canonical_audio_sha256`。
- `scripts/prepare_mdc_english_pairs.py`：补齐 `sample_id`、`transcript` 别名。
- `scripts/eval_pilot_wav2lip_syncnet.py`：n=1 smoke 的 SD 返回 0，配对 t-test p 值不可估计时写 `null` 而不是 NaN。
- 相关回归测试 28 passed；三份修改脚本 `py_compile` 通过。

## Artifacts

- Pair manifest: `runs/mdc_en_mfa_linear_50_20260815/00_pairs/pair_manifest.json`
- TTS metadata: `runs/mdc_en_mfa_linear_50_20260815/02_tts/tts_meta.json`
- MFA alignment/tokens: `runs/mdc_en_mfa_linear_50_20260815/03_alignment/`
- MFA-linear: `runs/mdc_en_mfa_linear_50_20260815/04_mfa_linear/`
- Fixed-face downstream: `runs/mdc_en_mfa_linear_50_20260815/05_wav2lip_syncnet_full/`
- Main analysis: `runs/mdc_en_mfa_linear_50_20260815/05_wav2lip_syncnet_full/analysis.json`
- Batch logs and raw outputs: `runs/mdc_en_mfa_linear_50_20260815/05_wav2lip_syncnet_full_batches/`

## Decision
英文扩展可行，且已完成一个有用的 n=50 MFA-linear run；英文结果支持“raw TTS/MFA-linear 相对自然音频在固定脸 SyncNet 上可能有优势”，但不支持 MFA-linear 超过 raw TTS。2026-08-16 的 cloud-Qwen 中文重跑也已完成：RAMC 与 AliMeeting 各 25 条的 MFA-linear 相对 natural 只稳定改善 Sync-C，Sync-D 基本不变；两个域中 cloud-Qwen raw TTS 都显著优于 MFA-linear（pooled ΔC `-0.34404`、ΔD `+0.12046`）。因此中文自发域的当前决策仍是：不要继续重复同一 50 条或把 MFA-linear 宣称为优于 raw TTS；若继续 TFG，优先改进显式 pause/自发 speech clock renderer，或扩大受控朗读域，而不是继续调用云端 TTS 做相同映射。所有 fixed-face 结果都不支持 speaker-matched 泛化或嘴型因果结论。
## Observations
- [result] 2026-08-16 cloud-Qwen RAMC/AliMeeting rerun completed TTS 50/50, MFA 100/100, MFA-linear 50/50, and fixed-face SyncNet 150/150. #cloud-qwen #mfa-linear
- [result] Cloud-Qwen MFA-linear vs natural pooled n=50: ΔSync-C `+0.27980` (p=`0.001277`), ΔSync-D `-0.00312` (p=`0.9564`); vs cloud-Qwen raw TTS: ΔC `-0.34404` (p=`4e-06`), ΔD `+0.12046` (p=`0.03369`). #syncnet
- [learning] Correct Chinese MFA 3.4.1 runtime requires both `mandarin_china_mfa` dictionary and `mandarin_mfa` acoustic model; the previously prepared English-only runtime and MFA 2.2 retry were retained as failed/diagnostic artifacts, not silently substituted. #mfa
- [boundary] Unknown speech remained explicit (`spn`: RAMC 6, AliMeeting 8); no score threshold was lowered and no SyncNet cell was interpolated. #provenance

- [result] RAMC + AliMeeting 中文扩展已经完成 50 条并显示强烈域依赖。
- [result] MDC English MFA 100/100、MFA-linear 50/50、fixed-face paired SyncNet 49/50/50 coverage 完成。
- [result] 英文 MFA-linear 相对 natural 为 ΔC `+0.2952`、ΔD `-0.4150`，但相对 raw TTS 无显著优势。
- [boundary] fixed-face 结果不代表 speaker-matched 泛化；`en_011/raw_tts` 缺失分数被显式保留。

## Relations

- relates_to [[多样域中文试点：MFA-linear 在自发对话域失败]]
- relates_to [[RAMC n25 Qwen cloud MFA-linear 2026-08-15]]
- relates_to [[AISHELL-1 n25 Qwen cloud MFA-linear 2026-08-14]]
