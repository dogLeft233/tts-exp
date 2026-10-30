---
title: AISHELL-1 n25 audio-track replacement SyncNet 2026-08-15
type: experiment
permalink: tts-exp/experiments/aishell-1-n25-audio-track-replacement-sync-net-2026-08-15
status: done
date: '2026-08-15'
s0770_used: false
tags:
- aishell-1
- syncnet
- audio-track-replacement
- mfa-linear
- phase2-renderer
---

# AISHELL-1 n25 audio-track replacement SyncNet 2026-08-15

## Context

在同一批 cloud-Qwen AISHELL-1 n=25 生成视频上，将 paired natural 原始音频替换为视频原有音轨，分别比较 MFA-linear 与 full phone-trajectory phase2 renderer。目标是测量固定视频帧下的 audio/video consistency 变化；这不是唇形运动的因果证明。

## Frozen protocol

- Cohort: 25 utterances，5 speakers × 5：S0765、S0901、S0906、S0912、S0913。
- S0770 未用于选择、调参或评估。
- MFA-linear source videos: `runs/aishell1_mfa_linear_n25_resample_poly_20260815/qwen_cloud_mfa_linear_n25_wav2lip_syncnet_qwenprov_20260815/wav2lip/mfa_linear/`。
- Phase2 source videos: `runs/aishell1_mfa_linear_n25_resample_poly_20260815/phone_trajectory_prosody_phase2_20260815/qwen_cloud_phase2_wav2lip_syncnet_20260815/wav2lip/phone_duration_lr/`。
- Paired natural audio: `results/rhythm_style_500/aishell1_test_400/natural/`。
- Replacement: `ffmpeg` maps source video stream and paired natural audio stream, `-c:v copy`, `-c:a pcm_s16le`，MKV container；无 tempo scaling、crop、silence padding、`-shortest`。
- SyncNet V2，`min_track=50`；每个 arm/speaker/sample 使用独立目录。
- Wav2Lip and SyncNet checkpoint SHA-256 recorded in the run metadata；natural/video/canonical hashes recorded per sample。

## Completion and stream verification

- 50/50 replacement videos generated。
- 50/50 SyncNet scores finite；0 failures。
- 50/50 video elementary-stream MD5 matched its source MP4，确认视频帧未被重新编码；只有音频轨道被替换。

## Replacement-arm comparison

Sync-C higher is better；Sync-D lower is better。

| Arm | n | Sync-C | Sync-D |
|---|---:|---:|---:|
| MFA-linear + natural PCM replacement | 25 | 5.5034 | 8.1166 |
| Phase2 renderer + natural PCM replacement | 25 | 5.4395 | 8.1192 |
| Phase2 − MFA-linear | — | −0.0639 | +0.0026 |

- Phase2 Sync-C beat MFA-linear on 10/25；Sync-D beat it on 13/25；both metrics jointly improved on 10/25。
- Paired t-test: Sync-C `p=0.3108`；Sync-D `p=0.9733`。
- Speaker-cluster bootstrap 95% CI: ΔSync-C `[-0.1838, +0.0511]`；ΔSync-D `[-0.1268, +0.1532]`。

### Speaker-level phase2 − MFA-linear

| Speaker | n | ΔSync-C | ΔSync-D |
|---|---:|---:|---:|
| S0765 | 5 | −0.0272 | −0.1458 |
| S0901 | 5 | +0.0036 | −0.0642 |
| S0906 | 5 | +0.1378 | −0.1390 |
| S0912 | 5 | −0.1618 | +0.0906 |
| S0913 | 5 | −0.2720 | +0.2714 |

## Replacement versus original video audio track

| Arm | Original C | Replacement C | ΔC | Original D | Replacement D | ΔD |
|---|---:|---:|---:|---:|---:|---:|
| MFA-linear | 6.5691 | 5.5034 | −1.0657 | 7.1016 | 8.1166 | +1.0150 |
| Phase2 renderer | 6.5628 | 5.4395 | −1.1233 | 7.1048 | 8.1192 | +1.0144 |

- MFA-linear replacement improved Sync-C on 3/25 and Sync-D on 0/25。
- Phase2 replacement improved Sync-C on 2/25 and Sync-D on 2/25。
- Paired t-tests for replacement vs original were strongly negative for Sync-C and positive for Sync-D in both arms；speaker-cluster bootstrap 95% CIs excluded zero for both metrics in both arms。
- The decrease is nearly the same for both arms；it does not support a meaningful phase2 advantage under this replacement protocol。

## Artifacts

- Replacement output, per-sample scores, logs, and manifest: `runs/aishell1_mfa_linear_n25_resample_poly_20260815/phone_trajectory_prosody_phase2_20260815/qwen_cloud_audio_track_replace_syncnet_20260815/`。
- Main analysis: `.../qwen_cloud_audio_track_replace_syncnet_20260815/analysis.json`。
- Summary: `.../qwen_cloud_audio_track_replace_syncnet_20260815/summary.json`。

## Interpretation and limits

该结果表示：在固定 generated video frames、只替换 audio stream 的 counterfactual protocol 下，natural PCM audio 与这些视频帧的 SyncNet consistency 明显低于原始 Wav2Lip audio track；MFA-linear 与 phase2 renderer 的 replacement 结果基本相同。不能把差异直接解释为嘴型运动的因果效应，也不能作为 speaker-matched 泛化证据，因为本批次使用 fixed S0765 face，且 replacement 同时改变了音频内容与封装/编码合同。若需要严格分离内容与封装影响，应增加同一 audio 内容的 PCM-MKV identity control。

## Observations

- [result] 50/50 strict PCM/no-tempo/no-crop/no-padding replacements and SyncNet scores completed。
- [result] Video elementary streams preserved for all 50 outputs。
- [result] Phase2 vs MFA-linear after replacement: no statistically significant difference。
- [insight] Natural-audio replacement lowers both arms similarly relative to their original audio tracks；the old observation is replicated at n=25 but should remain a fixed-face exploratory consistency result。
- [boundary] Do not claim lip-motion causality or speaker-independent generalization。

## Relations

- relates_to [[TTS 视觉教师到 replacement-safe 音频控制链路]]
