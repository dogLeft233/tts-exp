---
title: Encoder-only SyncNet 微调与固定视频迁移性
type: experiment
permalink: tts-exp/experiments/encoder-only-sync-net-微调与固定视频迁移性
status: concluded
date: '2026-08-21'
tags:
- syncnet
- wav2lip
- fixed-video
- audio-replacement
- wavlm
- encoder-finetuning
- cross-attention
- multi-tfg
---

# Encoder-only SyncNet 微调与 TTS cross-attention 阶段性验证

## Context

项目最终目标是：增强音频驱动 TFG 生成视频后，将音轨替换回原始自然音频，SyncNet 分数仍然提升。

当前 TTS cross-attention 使用：

```text
natural WavLM features → Query
paired TTS WavLM features → Key / Value
→ residual cross-attention
→ frozen prematched HiFi-GAN
→ candidate audio
```

先用 Wav2Lip-in-the-loop 训练尝试过，随后按阶段性决策移除 Wav2Lip 训练链，改用固定自然视频监督，隔离验证 candidate audio 是否真的改善。

## Wav2Lip-in-the-loop 结果

Wav2Lip、HiFi-GAN、WavLM 和 SyncNet 均冻结，只有 cross-attention 更新。100-step、learning rate `1e-5` 的 pilot 中：

- replacement proxy validation reward 只从 `-4.14746` 到 `-4.14612`，改善约 `0.00134`；
- 增强音频与其自身 Wav2Lip 生成视频的 Sync-C 平均相对自然音频 baseline 为 `+0.311`；
- 生成视频换回原始音频后，Sync-C 平均仅 `+0.0019`，中位数 `-0.026`；
- 换回原始音频后的 Sync-D 平均变差 `+0.1784`。

这说明 Wav2Lip-in-the-loop 可以反向传播，但主要学习到让视频适应 candidate audio，未形成稳定的 post-replacement 增益。训练图中的 `96→224` 视觉上采样也只是完整文件级合成/重裁剪的近似。

artifact：`runs/aishell1_qwen_mfa_linear_n100_20260820/24_tts_cross_attention_long_wav2lip_three_conditions/summary.json`

## 固定自然视频监督结果

下一阶段完全不调用 Wav2Lip 训练，只使用 fixed natural video visual embeddings 和 frozen SyncNet，训练 60 steps，WavLM、HiFi-GAN 和 SyncNet 均冻结。

训练 proxy validation：

```text
step 0 mean reward:  -4.63597
best step 30:       -4.21340
```

官方 fixed-video PCM mux audit（15 条）：

- Sync-C 平均 `+0.0499`，中位数 `+0.084`，`10/15` 提升；
- Sync-D 平均 `-0.0945`，中位数 `-0.156`，`10/15` 改善；
- 视频流、音频 PCM 和 sample count 完整性均 `15/15`。

artifact：`runs/aishell1_qwen_mfa_linear_n100_20260820/27_tts_cross_attention_fixed_video_official/summary.json`

这证明 TTS cross-attention 在**音频层面相对 direct-resynthesis baseline 有真实 fixed-natural-video 增益**。

## Wav2Lip 独立下游结果

同一批 candidate audio 不参与 Wav2Lip 训练，只在训练后进行官方三条件评测：

1. 原始自然音频 → Wav2Lip → SyncNet；
2. candidate audio → Wav2Lip → SyncNet；
3. candidate audio → Wav2Lip 视频 → 换回原始自然音频 → SyncNet。

相对原始音频 Wav2Lip baseline：

- candidate audio 自身生成视频：Sync-C 平均 `+0.2925`，Sync-D 平均 `-0.1288`；
- candidate Wav2Lip 视频换回原始音频：Sync-C 平均 `+0.0153`，中位数 `+0.078`，`8/15` 提升；
- 但换轨后的 Sync-D 平均变差 `+0.1699`，中位数变差 `+0.191`，仅 `6/15` 改善；
- 视频流和音频完整性均 `15/15`。

speaker 分组显示迁移不稳定：S0914 的换轨 Sync-C 平均 `+0.3842`，S0915 为 `-0.2450`，S0916 为 `-0.0934`。

artifact：`runs/aishell1_qwen_mfa_linear_n100_20260820/28_tts_cross_attention_fixed_video_wav2lip_three_conditions/summary.json`

## 当前结论与决策

- [decision] 固定自然视频监督可以有效改善 candidate audio 本身；该监督阶段不使用 Wav2Lip #training
- [insight] candidate audio 的 fixed-natural-video 增益没有完整迁移到 Wav2Lip 生成视频换回原始音频的目标；换轨 Sync-C 只剩弱平均增益，Sync-D 变差 #evaluation
- [decision] 暂不微调 HiFi-GAN decoder；先保留 frozen decoder 和 cross-attention 的音频层面结果 #decoder
- [decision] 后续可把该 candidate audio 作为通用音频增强输出，分别在多个 TFG（Wav2Lip、Ditto、LeapTalk 等）独立验证，而不是继续优化单一 Wav2Lip 训练代理 #generalization
- [problem] 当前最大瓶颈是不同 TFG 对 candidate audio 的视频适应机制不一致，导致 audio-level gain 与 post-replacement gain 不等价 #audio-replacement

## Relations

- relates_to [[Direct WavLM 重合成 2×2 音轨 Identity Control]]
- relates_to [[固定视频 SyncNet 监督 Gate B 结果]]
- relates_to [[MFA-linear 冻结 VC 重合成与下游验证]]
