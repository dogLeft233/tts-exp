---
title: Fixed-video teacher-map distillation
type: task
permalink: tts-exp/tasks/fixed-video-teacher-map-distillation
status: done
priority: high
current_step: 5
completed: '2026-08-21'
tags:
- task
- syncnet
- fixed-video
- teacher-map
- audio-replacement
---

# Fixed-video teacher-map distillation

## Context

本轮尝试在完全不把 Wav2Lip、Ditto 或其他 TFG 放入训练图的前提下，让固定自然视频监督更接近最终的 original-audio replacement 目标。方法是把原来的单一 fixed-coordinate SyncNet loss 改成 natural-audio teacher alignment-map distillation：original natural audio 的 SyncNet audio embedding 生成局部对齐分布，candidate audio 在同一固定视频上的分布通过 KL loss 复现；SyncNet、WavLM 和 HiFi-GAN 均冻结，仅训练 TTS cross-attention residual adapter。

## Implementation

- `scripts/experiments/syncnet_supervision/online_syncnet_objective.py` 新增保留 absolute visual/audio indices 的局部 alignment-map KL objective，并支持局部 radius 与 teacher confidence threshold。
- `scripts/experiments/aishell1_direct_audio/train_tts_cross_attention_fixed_video.py` 新增显式 `teacher_alignment_map` supervision 选项，旧 `fixed_coordinate` 仍是默认模式。
- `tests/experiments/syncnet_supervision/test_online_syncnet_objective.py` targeted tests：13 passed；真实 WavLM/HiFi-GAN/SyncNet smoke 通过，candidate waveform finite 且 exact-length。

## Evidence

无阈值 teacher-map 的 60-step pilot：proxy reward 从 `-0.8896` 提升到 `-0.7566`。官方 fixed natural video audit 的 Sync-C mean delta 为 `+0.0257`、median `+0.094`，Sync-D mean delta 为 `-0.0808`；15/15 的视频流、PCM 和 sample-count integrity 均通过。

随后进行唯一一次下游 Wav2Lip blind audit。candidate audio 自身生成视频相对 natural baseline 的 Sync-C mean delta 为 `+0.2874`、Sync-D 为 `-0.1125`；但生成视频换回 original natural audio 后，Sync-C mean delta 仅 `+0.0515`、median `-0.044`，Sync-D mean delta 变差 `+0.1638`。这不能称为稳定 replacement improvement。

`min_teacher_confidence=0.5` 的高置信 variant 虽然 proxy reward 从 `-1.0267` 提升到 `-0.6399`，但官方 fixed-video Sync-C mean delta 只有 `+0.0060`、median `-0.010`，因此未继续做 Wav2Lip 审计。

## Decision

本路线标记为 **NO-GO**：teacher-map 能产生有效梯度并改善部分 fixed-video 指标，但没有解决 candidate → TFG → generated video → original-audio replacement 的跨样本迁移。停止继续调 radius、confidence 或叠加新的 SyncNet loss；若继续研究，需要新的可验证中间监督，例如真实 timestamp/viseme supervision，而不是继续强化当前 fixed-video proxy。

## Artifacts

- Pilot: `runs/aishell1_qwen_mfa_linear_n100_20260820/32_tts_cross_attention_teacher_map_pilot/`
- Official fixed-video audit: `runs/aishell1_qwen_mfa_linear_n100_20260820/33_tts_cross_attention_teacher_map_fixed_video_official/`
- Wav2Lip audit: `runs/aishell1_qwen_mfa_linear_n100_20260820/34_tts_cross_attention_teacher_map_wav2lip_three_conditions/`
- Confidence pilot/audit: `runs/aishell1_qwen_mfa_linear_n100_20260820/36_tts_cross_attention_teacher_map_confidence_pilot/`, `37_tts_cross_attention_teacher_map_confidence_fixed_video_official/`

## Observations

- [decision] TFG 退出训练图，但保留一次最终下游 blind audit；不能只凭 fixed-video proxy 宣称 replacement 成功。
- [status] done
- [insight] 固定视频 alignment-map 比 scalar fixed-coordinate loss 更接近 natural reference 的局部时序，但仍不足以约束未知 TFG 的响应。
- [problem] candidate 与自身生成视频的同步提升没有稳定迁移到生成视频与 original natural audio 的配对。
- [decision] 本轮不再增加 confidence、radius、局部 warp 或额外 critic；路线 NO-GO。

## Relations

- relates_to [[固定视频 SyncNet 监督 Gate B 结果]]
- relates_to [[自然参考音频可在推理时获得]]
- relates_to [[Wav2Lip replacement-aware cross-attention supervision]]