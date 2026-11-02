---
title: LRS3 DAC 16 kHz 编解码器身份兼容性实验结果
type: report
permalink: tts-exp/experiments/lrs3-dac-16-k-hz-编解码器身份兼容性实验结果
tags:
- codec
- DAC
- replacement
- negative
- lrs3
status: concluded
---

# LRS3 DAC 16 kHz 编解码器身份兼容性实验结果

## Context

本实验用于验证 WavLM-L6 → HiFi-GAN 音频路径的保真度是否是冻结 Wav2Lip replacement 失败的主要原因之一。实验在固定的 LRS3 fit-only cohort 上，将历史 WavLM/HiFi-GAN 重建与 Descript DAC 原生 16 kHz 重建进行对比，并通过冻结 Wav2Lip 和完整 3×3 音频/视频矩阵测试 codec fidelity 与 replacement identity compatibility。

## Protocol and execution

- Cohort：50 条记录、38 个 source group；严格绑定 parent summary、source manifest 和 ordered sample-ID hashes。
- DAC：release `0.0.5`，source commit `408235a9dcd2983684c87615a1bc2a8954f6eb47`，`16khz/8kbps`，checkpoint SHA-256 `95ab7176b67137d4d4c6c54b8d6ef3cea797faec228cb03ad084badcad570b4d`。
- 使用 DAC checkpoint 的全部 12 个 quantizers，输入为精确 16 kHz mono PCM16；仅允许输入端 hop padding 和输出端裁剪到原始长度。
- 完成 50/50 DAC 重建、50/50 fidelity panel、150 个共享几何 Wav2Lip 视频和 450 个严格 mux/SyncNet matrix cell。
- 所有独立 stage validator 均通过；没有访问 sealed validation/test media，没有使用 TTS、MFA、DTW、Soft-DTW、训练或 score-based selection。
- Bootstrap：10,000 draws，seed `20260904`，以 source group 为 cluster resampling 单位。

## Results

DAC 相对于历史 WavLM/HiFi-GAN control 的 Wav2Lip mel fidelity 改善为：

- mean `mel_improvement = 0.194773916963`
- 95% CI `[0.183910382809, 0.206323284891]`
- 50/50 条记录为正改善

但 replacement 相关 endpoint 没有通过：

- `DAC_over_W_C`：mean `-0.021300`，95% CI `[-0.050756, 0.006500]`
- `DAC_over_W_D`：mean `-0.020820`，95% CI `[-0.043980, 0.000661]`
- `DAC_identity_C`：mean `0.012880`，95% CI `[-0.011086, 0.036850]`
- `DAC_identity_D`：mean `0.009660`，95% CI `[-0.014864, 0.034801]`

注册判定：

- `DAC_REDUCES_CODEC_REPLACEMENT_PENALTY = false`
- `DAC_IDENTITY_COMPATIBLE = false`
- `scientific_decision = DAC_CODEC_HYPOTHESIS_NOT_SUPPORTED`
- `engineering_decision = GO`
- `future_tts_alignment_experiment_eligible = false`

## Interpretation

DAC 确实改善了当前 Wav2Lip mel 保真度，但该改善没有转化为冻结 Wav2Lip replacement 优势，也没有建立 identity compatibility。因而，当前 replacement 失败不能主要归因于 WavLM-L6 → HiFi-GAN 的编码器/解码器保真度不足；剩余问题更可能涉及音视频条件、视觉相位、声学身份与同步目标之间的结构性不匹配。该结论限定在本实验注册 endpoint 和 50 条 fit-only cohort 内，不是对整个 LRS3 人群的总体性结论。

## How to apply

- 不应仅凭更好的 mel 或 waveform fidelity 继续推进 TTS/MFA/alignment replacement 路线。
- 在现有冻结 TFG 证据下，保持后续 TTS alignment experiment sealed，除非重新注册并获得新的科学依据。
- 后续研究应优先分析 reference-conditioned 的音视频相位/身份分离问题，而不是继续进行单纯 codec bitrate 或 quantizer 搜索。

## Observations
- [status] concluded

- [result] DAC 在 50 条记录上稳定改善 Wav2Lip mel fidelity，但没有改善两个 codec replacement penalty endpoint。 #codec #fidelity
- [decision] DAC codec identity compatibility 判定失败，后续 TTS/alignment experiment 不具备资格。 #replacement #negative
- [insight] 更高的声学重建保真度并不等价于 frozen Wav2Lip replacement 兼容性。 #audio-visual #representation
- [constraint] 结论只适用于固定 50-record/38-source-group fit-only cohort 和预注册 endpoints。 #scope

## Relations

- relates_to [[tts-exp|Wav2Lip replacement is primary objective]]
- relates_to [[Full-frame Wav2Lip training audit]]
- relates_to [[Raw TTS strict replacement NO-GO]]
- relates_to [[tts-exp|Natural reference audio available at inference]]

- extends [[LRS3 WavLM-HiFi-GAN direct resynthesis replacement]]
