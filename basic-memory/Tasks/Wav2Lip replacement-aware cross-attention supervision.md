---
title: Wav2Lip replacement-aware cross-attention supervision
type: task
permalink: tts-exp/tasks/wav2-lip-replacement-aware-cross-attention-supervision
status: done
priority: high
current_step: 5
started: '2026-08-21'
tags:
- task
- wav2lip
- syncnet
- cross-attention
- audio-replacement
completed: '2026-08-21'
---

# Wav2Lip replacement-aware cross-attention supervision

## Observations

- [description] 验证目标：`natural audio → WavLM/TTS cross-attention → candidate audio → frozen Wav2Lip(face, candidate audio) → generated video + original natural audio → frozen SyncNet loss`。
- [status] done
- [assigned_to] claude
- [current_step] 5

## Steps

1. [x] 用已修复的可微 Wav2Lip/SyncNet 链运行受控 pilot；SyncNet audio target 使用 original natural audio，不使用 candidate audio self-consistency。
2. [x] 记录 replacement proxy loss、adapter gradient、finite/exact-length 和 checkpoint provenance。
3. [x] 选择 validation checkpoint，生成 15 条 candidate audio。
4. [x] 将 candidate audio 送入真实 Wav2Lip 文件生成视频，再将音轨替换为 original natural audio，运行官方 SyncNet 三条件评测。
5. [x] 比较 candidate+generated audio、generated video+original audio 与 original audio baseline；记录 proxy/真实路径差异。

## Results

- Explicit 30-step pilot 的 best validation 是 step 0，replacement proxy 没有改善；训练链路有梯度但 cross-attention 未学到稳定 replacement 方向。
- 官方三条件结果：candidate audio 自身生成视频相对 natural baseline 的 Sync-C 平均 `+0.2868`、Sync-D 平均 `-0.1212`；但 generated video 换回 original audio 后，Sync-C 平均 `+0.0651`、中位数 `+0.112`，Sync-D 平均变差 `+0.1642`、中位数变差 `+0.122`。
- 换轨视频流、original PCM 和 sample count 完整性均通过 `15/15`。
- Artifact: `runs/aishell1_qwen_mfa_linear_n100_20260820/29_wav2lip_replacement_target_explicit_pilot/`
- Official artifact: `runs/aishell1_qwen_mfa_linear_n100_20260820/30_wav2lip_replacement_target_explicit_three_conditions/`

## Context

- Repo: `/home/wjj/tts-audio/tts-exp`
- Implementation: `scripts/experiments/aishell1_direct_audio/train_tts_cross_attention_wav2lip_replacement.py`
- WavLM, prematched HiFi-GAN, Wav2Lip and SyncNet were frozen; only cross-attention was trainable.
- Training used original natural audio as SyncNet audio target. The Wav2Lip training path used differentiable mel preprocessing and a 96-to-224 visual approximation; file-level evaluation remained authoritative.
- Conclusion: the target chain is implemented and numerically trainable, but current proxy did not learn a stable post-replacement improvement. Do not fine-tune decoder or add self-consistency loss as a workaround; treat the route as blocked at the current differentiable Wav2Lip/proxy alignment.
