---
title: Direct WavLM 重合成 2×2 音轨 Identity Control
type: experiment
permalink: tts-exp/experiments/direct-wav-lm-重合成-2x2-音轨-identity-control
status: concluded
date: '2026-08-20'
tags:
- wavlm
- resynthesis
- audio-integrity
- syncnet
- wav2lip
- counterfactual
---

# Direct WavLM 重合成 2×2 音轨 Identity Control

## Context

为判断“原始 natural 音频经过 WavLM 编码与 prematched HiFi-GAN 解码后，驱动 Wav2Lip/TFG 生成的视频，再换回原始音频时 SyncNet 下降”是否只是音轨封装或编解码伪影，固定已有 direct-resynthesis 视频帧，构造 natural/reconstructed audio 的严格 2×2 矩阵。

## Protocol

- Cohort：AISHELL-1 n=15，S0914/S0915/S0916 各 5 条；使用既有 direct-resynthesis 视频，不重新运行 Wav2Lip。
- Audio N：原始 natural WAV。
- Audio R：natural → frozen WavLM-Large L6 → frozen prematched HiFi-GAN 的 direct reconstruction。
- Video N：由 natural audio 驱动生成的视频帧。
- Video R：由 reconstructed audio 驱动生成的视频帧。
- 四个 cell：`V_N/A_N`、`V_N/A_R`、`V_R/A_N`、`V_R/A_R`。
- 所有替换统一使用 16 kHz mono `pcm_s16le` + Matroska；视频流 `-c:v copy`，无 AAC、无 `-shortest`、无 tempo/crop/padding。
- 每个 cell 都验证视频 elementary stream MD5、解码 PCM SHA-256、样本数和模型 hash；SyncNet V2 使用 `min_track=50`。

代码：`scripts/experiments/aishell1_direct_audio/eval_aishell1_direct_wavlm_2x2_identity_syncnet.py`

产物：`runs/aishell1_qwen_mfa_linear_n100_20260816/16_direct_wavlm_2x2_identity_control_syncnet/`

## Results

| Video frames | Audio track | Sync-C ↑ | Sync-D ↓ |
|---|---|---:|---:|
| `V_N` | `A_N` | 5.6638 | 7.2095 |
| `V_N` | `A_R` | 5.4419 | 7.4875 |
| `V_R` | `A_N` | 5.7236 | 7.3597 |
| `V_R` | `A_R` | 5.9517 | 7.1179 |

60/60 cells completed with zero failures. Video stream, decoded PCM and sample count were preserved in 60/60 cases.

Identity replacement deltas relative to the original source videos were near zero:

- `V_N/A_N` replacement: ΔSync-C `+0.0008`, ΔSync-D `+0.0092`;
- `V_R/A_R` replacement: ΔSync-C `-0.0139`, ΔSync-D `+0.0483`.

On fixed natural video, replacing natural audio with reconstructed audio changed Sync-C by `-0.2219` and Sync-D by `+0.2781`. On fixed direct-resynthesis video, replacing reconstructed audio with natural audio changed Sync-C by `-0.2281` and Sync-D by `+0.2419`.

## Interpretation

The identity cells show that the earlier score drop is not explained by AAC/MP4 versus PCM/MKV muxing, video re-encoding, audio sample loss, or `-shortest` truncation. The crossed preference is consistent: natural-driven video prefers natural audio, while reconstructed-audio-driven video prefers reconstructed audio. The audio used to generate the video frames remains part of the downstream compatibility condition.

The direct-resynthesis summary had mean WavLM feature cosine `0.8834` relative to the original natural features, but waveform correlation only `0.0123` and mean SNR `-2.68 dB`. Therefore SSL feature proximity is insufficient to make the reconstructed waveform substitutable for natural audio in the talking-face model. Remaining relevant differences include phase, transient/onset structure, energy envelope, F0/formant details, coarticulation and vocoder-domain artifacts.

## Limits

This is an exploratory n=15, three-speaker, single-fixed-face control. The fixed-video audio-swap directions are consistent but not a large-sample causal estimate; the direct-video swap has paired p-values `0.165` for Sync-C and `0.059` for Sync-D. It establishes audio/video condition mismatch under this Wav2Lip/SyncNet protocol, not naturalness or cross-model generalization.

## Observations
- [status] concluded

- [decision] Treat PCM/MKV identity replacement as a passed control; do not attribute the cross-condition SyncNet drop to mux alone. #audio-integrity
- [result] `V_N` prefers `A_N` and `V_R` prefers `A_R` in the strict 2×2 matrix. #counterfactual #syncnet
- [insight] WavLM feature similarity does not imply waveform substitutability for a talking-face generator. #representation #wav2lip
- [boundary] Evidence is limited to 15 utterances, 3 speakers and one fixed face. #scope

## Relations

- relates_to [[MFA-linear 冻结 VC 重合成与下游验证]]
- relates_to [[AISHELL-1 n25 audio-track replacement SyncNet 2026-08-15]]
- relates_to [[TTS Feature Alignment and Duration Model Research]]
