---
title: Wav2Lip proxy-to-official parity gate
type: experiment
permalink: tts-exp/experiments/wav2lip/wav2-lip-proxy-to-official-parity-gate
status: complete
date: '2026-08-21'
gate: GO_with_local_window_warning
tags:
- wav2lip
- parity
- syncnet
- full-frame
- proxy
- replacement
---

# Wav2Lip proxy-to-official parity gate

## Context

项目目标不是 candidate audio 与其生成视频的自洽分数，而是：candidate audio 驱动 Wav2Lip 生成视频后，把音轨替换回 original natural audio，official SyncNet 仍然改善。为验证训练 proxy 是否接近文件级测试链路，增加了 fixed-box full-frame compositor、official per-frame box 导出和 raw/file-level parity audit。

## Implementation

- `scripts/experiments/aishell1_direct_audio/wav2lip_full_frame_proxy.py`
  - 固定 boxes 的 224/96 crop；
  - lower-half masking；
  - differentiable 96×96 patch paste 回完整 frame；
  - 支持 waveform-to-video 梯度。
- `third_party/Wav2Lip/inference.py`
  - 增加可选 `--boxes_output`，默认官方推理行为不变；
  - 导出实际使用的 per-frame boxes，并复现 mel chunks 多于 source frames 时的 frame cycling。
- AISHELL-1 parity audit：比较 official raw AVI、proxy full-frame AVI、mel、frame count、fixed-box visual embedding、official Sync-C/D/AV offset。
- LRS3 parity/direct audit：在原始 224×224 face-cropped LRS3 视频上使用 official per-frame boxes，另外生成 zero-residual direct audio 作为训练前基线。

## AISHELL-1 results

Three-speaker、5 variants、15 rows 的 aggregate parity gate 为 GO：mel max error `1.79e-5`，frame MAE `0.245`，fixed-box visual cosine mean `0.9940`，Sync-C/D mean absolute delta `0.074/0.072`，AV offset 100% 一致。full n15 natural/enhanced 30 rows 也无 execution failure；局部最低 cosine `0.8311` 只作为 warning。

随后 full-frame cross-attention n15 pilot 完成。official replacement audit 的 video stream 和 original PCM integrity 均通过，但 replaced-vs-natural 只有 Sync-C mean `+0.040`，Sync-D mean `+0.153`（变差），不能称为 robust replacement improvement。

## LRS3 n300 training and official audit

LRS3 online dataset 有 500 条、43 个 source groups，split 为 347 train / 69 validation / 84 test。n300 training 使用 300 条 train clips、30 个 train groups，validation 使用独立 6 个 groups 的 69 条样本。使用 static full-frame boxes on 224×224 LRS3 face crops、official mel frame cycling、frozen Wav2Lip/SyncNet、zero-initialized TTS cross-attention residual。

Proxy validation reward 从 step 0 的 `-7.2133` 改善到 step 300 的 `-6.8677`，369 条输出全部 finite 且 exact-length。但 official balanced validation n15 audit 中，trained candidate video 相对 natural baseline 的 Sync-C mean delta 为 `-2.473`，15/15 都下降；换回 natural audio 后 Sync-C 仍为 `-1.325`，Sync-D 变差 `+0.546`。LRS3 n300 route 为 NO-GO。

Artifacts:

- LRS3 n300 training: `runs/lrs3_qwen_cloud_n500_20260818/07_wav2lip_replacement_lrs3_train300/`
- LRS3 trained official audit: `runs/lrs3_qwen_cloud_n500_20260818/09_wav2lip_replacement_lrs3_official_audit_n15_retry/`

## LRS3 proxy parity and zero-residual direct baseline

在均衡 LRS3 validation n15 上，zero-residual direct protocol 为：natural WavLM features → zero-residual `TTSCrossAttention` → frozen HiFi-GAN。proxy 使用 official per-frame boxes 导出的 full-frame raw path。

Parity 结果：frame count agreement `100%`，frame MAE mean `0.861`，mel max error `5.01e-5`，fixed-box visual cosine mean `0.9914`、minimum `0.8350`，official/proxy Sync-C/D absolute delta mean `0.134/0.127`，AV-offset agreement `100%`。因此 LRS3 proxy-to-official parity 通过。

Zero-residual direct official candidate video 相对 natural baseline：Sync-C mean delta `+0.087`，Sync-D mean delta `-0.121`，两个指标各有 8/15 改善。换回 original natural audio 后：Sync-C mean delta `-0.114`，Sync-D mean delta `+0.149`，只有 4/15 与 5/15 分别改善。所有 video stream、audio PCM 和 sample count integrity 均通过。

Artifact:

- LRS3 parity/direct audit: `runs/lrs3_qwen_cloud_n500_20260818/12_lrs3_proxy_direct_audit_n15/`

## Decision

parity gate 已证明当前 proxy 的 mel、完整帧合成、official boxes/frame cycling 和 official SyncNet 结果在聚合层面足够接近；proxy implementation 不是主要失败原因。zero-residual direct audio 的 candidate-video 分数略有改善，但 replacement 分数仍低于 natural baseline；n300 训练反而显著恶化 candidate video。不要继续增加 Wav2Lip proxy 复杂度、样本量或 steps，除非先改变 replacement supervision、candidate audio renderer 或引入更直接的真实视频运动/viseme/timestamp 中间监督。

## Observations

- [solution] 96×96 patch 直接上采样到 224×224 不是足够的 proxy；固定 boxes 的 full-frame paste 后再 crop 才能与官方路径比较。
- [insight] official boxes、mel 和 frame cycling 对 raw parity 是关键；LRS3 official per-frame box audit 复现成功。
- [insight] proxy validation 改善不保证 official replacement 改善；LRS3 n300 是明确反例。
- [insight] zero-residual direct candidate 略优于 natural，但换回 original audio 后仍退化，说明 decisive bottleneck 是 generated video motion 与 original natural audio 的关系。
- [decision] parity 通过只是必要 gate，不是 replacement objective 成功证明；最终 checkpoint 仍必须通过 official file-level replacement audit。

## Relations

- relates_to [[Wav2Lip replacement-aware cross-attention supervision]]
- relates_to [[Fixed-video teacher-map distillation]]
- relates_to [[LRS3 SyncNet RhythmWarp 训练]]