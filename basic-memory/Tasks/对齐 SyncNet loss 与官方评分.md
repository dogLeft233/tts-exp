---
title: 对齐 SyncNet loss 与官方评分
type: task
permalink: tts-exp/tasks/对齐-sync-net-loss-与官方评分
assigned_to: claude
context: LRS3 WavLM layer 6 + HiFi-GAN SyncNet-only finetuning reduced the current
  fixed normalized-cosine proxy but degraded official SyncNet on official face crops.
  The next implementation must align crop/audio coordinates, use raw Euclidean distance,
  then add local offset ranking and only later confidence/quality regularization.
current_step: 10
parent_task: 建立可学习音频对齐模型
priority: high
started: '2026-08-27'
status: in-progress
steps:
- 建立官方 25fps face-tracked crop 与同步音频坐标 parity
- 建立固定样本/固定窗口的 pre/post diagnostic set
- 实现 raw Euclidean fixed-target SyncNet loss
- 验证 fixed distance 与 official Sync-D 同向
- 加入 local offset ranking/stability，防止 target shift 漂移
- 加入小权重 Sync-C confidence margin
- 必要时加入 audio anchor/mel/STFT quality regularization
- 按固定 split 重训并运行 Wav2Lip 官方 SyncNet 评估
- 完成 candidate-driver、fixed natural-video 和 untouched natural-audio replacement 决策门
tags:
- task
- syncnet
- wav2lip
- lrs3
- loss
---

# 对齐 SyncNet loss 与官方评分

## Context

在 LRS3 上验证 WavLM layer 6 + HiFi-GAN 的 SyncNet-only 微调是否能产生独立于候选音频的视觉增益。最终判据是：candidate audio 驱动冻结 Wav2Lip 得到的视频，换回 untouched natural audio 后，仍优于 natural video + natural audio；candidate audio/candidate video 的分数提升不能单独算作视觉收益。

当前训练只更新 WavLM layer 6 和 HiFi-GAN，SyncNet 参数冻结且 visual embedding detach。训练输入已对齐到官方 25 fps、224x224 face-tracked crop 和同步 crop audio，目标使用官方 zero-padded raw Euclidean distance、每条 natural baseline 的 offset 以及 local offset ranking。

## 已完成方案

- 官方 crop preparation、crop/audio provenance 和 SHA256 manifest。
- 可微 MFCC 与 `python_speech_features` parity；官方未归一化 Euclidean fixed loss 与 `vshift=15` offset curve parity。
- per-sample natural baseline target offset；不再使用统一 `+2`。
- local offset ranking，negative distance detached，避免通过拉远错误 offset 伪造 Sync-C。
- weak direct-resynthesis waveform anchor，`anchor_weight=0.1`。
- 50 条 crop manifest 和 33 条 train split endpoint；focused suite `14 passed`，相关模块 `py_compile` 通过。

## 实验结果

统一 `+2` raw fixed target 会让 fixed loss 下降但官方 Sync-D/Sync-C 恶化并出现 offset drift。Per-sample target + ranking（无 anchor）在固定诊断上改善，但 train endpoint replacement 仍为 NO_GO：candidate-driver Sync-C delta `-0.3075`、Sync-D benefit `-0.0063`；replacement Sync-C delta `-0.3997`、Sync-D benefit `-0.3145`，joint wins `0/33`。

Anchor checkpoint：

```text
/home/wjj/tts-audio/tts-exp/runs/lrs3_wavlm_hifigan_syncnet_per_sample_rank_anchor_steps20_v2_20260827/checkpoint.pt
```

训练固定诊断 step 20：fixed raw `7.9431`、Sync-D `7.9431`、Sync-C `5.0802`、best shift `+0.50`、anchor L1 `0.02748`。

Anchor train endpoint：

```text
/home/wjj/tts-audio/tts-exp/runs/lrs3_wavlm_hifigan_syncnet_per_sample_rank_anchor_steps20_v2_eval_train_v2_20260827/
```

相对无 anchor 版本，candidate-driver 的 Sync-C 改善 `+0.2575`（95% CI `[+0.1594,+0.3458]`），Sync-D benefit 改善 `+0.0760`（CI `[+0.0092,+0.1407]`）；但相对 natural 的最终 candidate-driver 结果仍不显著：C delta `-0.0509`（CI `[-0.2281,+0.1291]`），D benefit `+0.0694`（CI `[-0.0939,+0.2665]`）。

Untouched natural-audio replacement 仍明确失败：C delta `-0.3595`（CI `[-0.4383,-0.2886]`），D benefit `-0.3708`（CI `[-0.4484,-0.2976]`），joint wins `0/33`。

同一 natural-driver video 的固定 crop 对照显示 anchor 确实比 unfinetuned direct-resynthesis 更兼容：finetuned 相对 unfinetuned 的 C delta `+0.0309`（CI 跨 0），D benefit `+0.0687`（CI `[+0.0208,+0.1162]`），joint wins `14/33`；但 finetuned 仍低于 natural（C `-0.2260`、D benefit `-0.1392`）。

## Domain mismatch diagnosis

33 条 endpoint 样本的 natural/candidate 音频样本数完全一致；natural 和 finetuned 视频均为 25 fps，官方 pipeline 的 track 起点均为 0，track 长度一致，因此没有发现粗粒度 audio/video coordinate bug。

官方 Wav2Lip 使用 `mel_step_size=16`、`80/fps` 的 mel chunk 索引，并按 mel chunk 数与视频帧数取共同长度。同一 natural-driver 上，anchor finetuned 音频相对 natural 的 Wav2Lip mel L1 为 `0.3936`，unfinetuned 为 `0.3669`，只有 `4/33` 条 anchor 更接近 natural。与此同时，同一 finetuned video 换 candidate 音频相对换 natural 音频的 Sync-D benefit 为 `+0.4402`，说明生成视频与 candidate 音频高度共适应。

因此当前证据支持：主要失败接口是 Wav2Lip mel-conditioned audio-to-video domain mismatch 和音频驱动共适应，而不是训练 crop 的粗粒度时序错位。Waveform anchor 只修复了部分 audio-side compatibility，没有实现 replacement-safe visual gain。

## Decision and next step

当前 anchor 版本的 replacement scientific decision 为 `NO_GO`。不运行 held-out 作为成功证明，也不继续盲目叠加 confidence、STFT 或复合正则。若继续，只设计一个明确针对 Wav2Lip mel 条件域的单一窄实验，并保留 candidate-driver、fixed natural-driver 和 untouched natural-audio replacement 三个 gate；在 replacement 通过前不扩大数据或模型规模。

## Observations

- [decision] candidate-driver 提升不能解释为独立视觉增益；untouched natural-audio replacement 是硬门槛 #syncnet #wav2lip
- [status] in-progress
- [decision] raw Euclidean、per-sample offset、local ranking 和 weak waveform anchor 的组合仍不足以产生 replacement-safe gain #lrs3
- [insight] anchor 改善了相对 unfinetuned 的固定视频兼容性，尤其是 Sync-D，但 candidate 音频仍未恢复到 natural 的 Wav2Lip mel 域 #audio
- [insight] endpoint 的视频与 candidate 音频存在明显共适应；换回 natural audio 后性能显著下降 #replacement
- [problem] 训练用 SyncNet MFCC crop 域与 Wav2Lip mel-conditioned video 域之间存在明确 domain gap #domain-mismatch
- [next] 只验证一个 Wav2Lip-mel-domain 窄改动，继续使用 replacement gate；暂不加入更多复合 loss

## Relations

- parent_task [[建立可学习音频对齐模型]]
- relates_to [[raw-tts-strict-replacement-no-go]]
- relates_to [[direct-waveform-upper-bound]]
- relates_to [[natural-reference-audio-is-available-at-inference]]