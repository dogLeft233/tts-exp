---
title: LRS3 WavLM-HiFi-GAN direct resynthesis replacement
type: experiment
permalink: tts-exp/experiments/lrs3-wav-lm-hi-fi-gan-direct-resynthesis-replacement
tags:
- lrs3
- wavlm
- hifigan
- wav2lip
- syncnet
- replacement
- direct-resynthesis
---

# LRS3 WavLM-HiFi-GAN direct resynthesis replacement experiment

## Context

在 LRS3 上验证冻结 WavLM 编码与 prematched HiFi-GAN 解码的 direct waveform resynthesis，判断这种 audio codec/interface 变化是否能在冻结 Wav2Lip 中产生视觉 SyncNet 增益，以及该增益能否在换回 untouched natural audio 后保留。该实验不是训练过的 autoencoder：WavLM-Large layer 6 与 HiFi-GAN 均冻结，不使用 TTS，也不微调 encoder。

## Protocol

- Cohort：LRS3 manifest 顺序前 50 条，38 个 source groups。
- Audio：natural waveform → frozen WavLM-Large layer 6（1024-d）→ prematched HiFi-GAN；输出保持 16 kHz、mono、PCM_16，并调整到自然音频的精确 sample count。
- Video：冻结 `wav2lip_gan.pth`，natural audio 与 direct-resynthesized audio 分别驱动同一原始 face video。
- Cells：`natural_video_natural_audio`、`direct_video_direct_audio`、`direct_video_natural_audio`。
- Replacement：视频 stream copy，音频 PCM s16le；对 mux 前后解码 PCM 做 byte-level equality 检查。
- Statistics：Sync-C 越高越好，Sync-D 越低越好；报告 direct-vs-natural 与 strict replacement 的 source-group cluster-bootstrap 95% CI。

## Results

正式 artifact 位于 `runs/lrs3_wavlm_hifigan_direct_20260826/summary.json`，状态为 complete：

| Cell | Sync-C mean | Sync-D mean |
|---|---:|---:|
| natural video + natural audio | 7.228 | 7.359 |
| direct video + direct audio | 7.128 | 7.420 |
| direct video + natural audio | 6.997 | 7.590 |

Direct driver 相对 natural driver：

- Sync-C delta = -0.100，95% CI [-0.221, 0.023]。
- Sync-D benefit（baseline D - candidate D）= -0.061，95% CI [-0.173, 0.060]。
- C/D/joint wins = 20/22/17 out of 50。

Strict natural-audio replacement（direct video + natural audio 相对 natural video + natural audio）：

- Sync-C delta = -0.231，95% CI [-0.294, -0.170]。
- Sync-D benefit = -0.232，95% CI [-0.293, -0.173]。
- C/D/joint wins = 6/7/4 out of 50。

所有 150 个 mux 的 decoded PCM 都与对应输入音频完全一致；所有 AV offsets 均为 -2。重建音频平均 WavLM re-encoding cosine distance 为 0.126，无 clipping。默认逐帧人脸检测失败的 5 个片段使用了首帧静态框 fallback，两种 audio arm 共 10 个 render；这些结果应视为带有该 renderer caveat 的 exploratory result。

## Interpretation

WavLM→HiFi-GAN direct resynthesis 没有产生可重复的 SyncNet visual gain。更重要的是，在 strict natural-audio replacement 中，direct video 的 Sync-C 与 Sync-D 均显著变差，因此该 codec/interface 路径不能解释或解决 replacement-safe 增益问题。该结果支持“TFG 的视觉收益可能依赖于生成音频的可见声学细节或 audio-video co-adaptation；仅保持高层 WavLM 内容表示不足以让收益脱离 waveform 音频而保留”的判断，但不证明所有冻结 codec 都无效。

## Reproducibility

- Code：`scripts/experiments/lrs3_direct_audio/run_wavlm_hifigan_lrs3.py`
- Protocol tests：`tests/experiments/lrs3_direct_audio/test_protocol.py`，5 passed。
- Code commit：`98e177f` on branch `worktree-lrs3-wavlm-resynthesis-50`。

## Observations

- [decision] 使用冻结 WavLM layer 6 与 prematched HiFi-GAN，不进行 encoder 微调或 audio-head training。 #protocol
- [insight] direct-resynthesized audio 自身未带来 SyncNet 增益；natural audio replacement 后没有保留增益。 #replacement
- [result] 50-record LRS3 evaluation 完成 150/150 score cells，strict PCM equality 为 150/150。 #experiment
- [learning] WavLM feature-level similarity（平均 cosine distance 0.126）不能作为 replacement-safe waveform reachability 的充分条件。 #audio
- [tradeoff] 对漏检片段使用首帧静态框 fallback 保证 cohort 完整，但降低了 renderer protocol 的统一性，因此结果保留 exploratory caveat。 #wav2lip

## Relations

- relates_to [[TTS 视觉教师到 replacement-safe 音频控制链路]]
- follows [[LRS3 MFA-linear TFG native/replacement 2026-09-13|LRS3 MFA-linear replacement NO-GO]]
- constrained_by [[tts-exp|Wav2Lip replacement is primary objective]]
- depends_on [[tts-exp|Natural reference audio available at inference]]
