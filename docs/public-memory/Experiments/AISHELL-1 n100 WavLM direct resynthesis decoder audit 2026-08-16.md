---
title: AISHELL-1 n100 WavLM direct resynthesis decoder audit 2026-08-16
type: experiment
permalink: tts-exp/experiments/aishell-1-n100-wav-lm-direct-resynthesis-decoder-audit-2026-08-16
tags:
- aishell1
- wavlm
- hifigan
- decoder-audit
- syncnet
- experiment
---

# AISHELL-1 n100 WavLM direct resynthesis decoder audit 2026-08-16

## Context

验证自然音频直接经过冻结 WavLM 编码，再经过当前配套声码器解码，是否能恢复原始波形，以及这种 direct-resynthesis 是否会影响 Wav2Lip/SyncNet。该实验不使用 Qwen、MFA 或 conditional adapter。

## Encoder and decoder

- Encoder：冻结 WavLM-Large 第 6 层，1024 维，16 kHz 输入，320-sample frame stride。
- Decoder：固定 `bshall/knn-vc` revision `c616845c4e309e24d5927f15adbdf277a3d65358` 配套的 frozen prematched HiFi-GAN，checkpoint interface 为 `prematch_g_02500000.pt`。
- 该链路是声学特征重合成，不是无损 codec；WavLM 表示和 HiFi-GAN 生成均可能丢失原始相位、瞬态和细节。

## Selection and reconstruction

- 从冻结 AISHELL-1 n=100 cohort 中按 speaker 分层选取 S0914、S0915、S0916 各 5 条，共 15 条；选择按 sample ID 排序，与 SyncNet 分数无关。
- 15/15 生成成功，全部 finite、严格 natural length、未削波。
- 0/15 波形与原始音频完全相同。
- 平均 waveform SNR：`-2.6827 dB`。
- 平均波形相关系数：`0.0123`。
- direct 输出重新经过 WavLM 后，与原始 WavLM 特征的平均 frame cosine：`0.8834`。
- 因此解码器保留了一部分 WavLM 表征结构，但不是原始波形的近似复制；波形差异很大。

## Wav2Lip and SyncNet

固定同一 face、Wav2Lip checkpoint、SyncNet checkpoint、`min_track=50`，natural raw 与 direct-resynthesis 共 30/30 cell 完成。

| arm | mean Sync-C | mean Sync-D |
|---|---:|---:|
| natural raw | 5.6630 | 7.2003 |
| direct WavLM→HiFi-GAN | 5.9657 | 7.0696 |

Direct − natural：

- Sync-C `+0.3027`，paired p=`0.1662`，cluster bootstrap 95% CI `[-0.3992, +0.8518]`，9/15 改善。
- Sync-D `-0.1307`，paired p=`0.2965`，cluster bootstrap 95% CI `[-0.3290, +0.1232]`，8/15 改善。
- 双指标同时改善 `7/15`。

Speaker 分组：

- S0914：Sync-C `+0.8518`、Sync-D `-0.3290`，joint `4/5`。
- S0915：Sync-C `-0.3992`、Sync-D `+0.1232`，joint `1/5`。
- S0916：Sync-C `+0.4554`、Sync-D `-0.1862`，joint `2/5`。

## Counterfactual audio-track replacement

为区分“direct 视频画面本身”与“direct 音轨”的影响，保留 15 个 direct 视频的 video stream，只用对应 natural raw 音频替换音轨；使用相同 `min_track=50` 的 SyncNet 协议重新评分。视频流 hash 在 15/15 个样本中保持一致。

| 音轨 | mean Sync-C | mean Sync-D |
|---|---:|---:|
| direct 原音轨 | 5.9657 | 7.0696 |
| 同一画面 + natural 原音轨 | 5.6878 | 7.3703 |

natural 音轨 − direct 音轨：Sync-C `-0.2779`，cluster bootstrap 95% CI `[-0.6258, +0.1696]`，6/15 改善；Sync-D `+0.3007`，CI `[+0.0762, +0.4292]`，5/15 改善，paired p=`0.0332`；双指标同时改善 `5/15`。S0914 的 Sync-C 变化 `-0.6258`，S0915 `+0.1696`，S0916 `-0.3774`，存在 speaker 差异。

这个反事实结果表明：对于由 direct 音轨驱动生成的 Wav2Lip 画面，换回 natural 音轨没有提升，整体反而降低 Sync-C、恶化 Sync-D；画面更匹配生成它时使用的 direct 音轨。该结果不表示 natural 音频本身较差，而是说明两种音轨对应的唇动/声学时间结构不同。

## Interpretation and limits

当前解码器确实可以把 WavLM 特征重新合成为可用音频，但输出不是原始音频本身。SyncNet 结果在 15 条小样本、3 个 speaker 上呈现平均正向，但 speaker 间不一致、置信区间跨 0，不能据此宣称 direct-resynthesis 优于 natural raw，也不能把 fixed-face 结果解释为真实 speaker 泛化。

## Artifacts

- Generator: `scripts/generate_aishell1_n100_wavlm_direct_resynthesis.py`
- Evaluator: `scripts/eval_aishell1_direct_wavlm_resynthesis_wav2lip_syncnet.py`
- Audio/QC: `runs/aishell1_qwen_mfa_linear_n100_20260816/13_direct_wavlm_hifigan_resynthesis/`
- Counterfactual audio replacement/evaluation: `scripts/eval_aishell1_direct_video_original_audio_syncnet.py`
- Replaced-video output: `runs/aishell1_qwen_mfa_linear_n100_20260816/15_direct_video_original_audio_syncnet/`

## Observations

- [insight] WavLM-L6 + prematched HiFi-GAN 是可解码的声学重合成链，但不是无损编码器/解码器。
- [result] direct waveform 与 natural 波形明显不同，重新编码后的特征仍保持较高但非完美 cosine 相似度。
- [boundary] direct-resynthesis 的 SyncNet 平均改善不显著且 speaker-dependent，不能替代独立听感和跨 speaker 验证。

## Relations

- relates_to [[MFA-linear 冻结 VC 重合成与下游验证]]
- relates_to [[AISHELL-1 n100 conditional MFA adapter 2026-08-16]]