---
title: 优质静态图 B075 MFA-linear native replacement 2026-09-13
type: experiment
permalink: tts-exp/experiments/优质静态图-b075-mfa-linear-native-replacement-2026-09-13
status: concluded
date: 2026-09-13
tags:
- static-image
- b075
- mfa-linear
- replacement
- syncnet
- wav2lip
---

# 优质静态图 B075 MFA-linear native replacement 2026-09-13

[status] concluded

## 目的与口径

复现 B025/B050 的静态人脸条件，检查 B075 和纯 MFA-linear 在相同静态输入上的：

- native：候选音频驱动 Wav2Lip，仍用候选音频评分；
- natural replacement：同一个候选生成视频，最终评分音频换回自然音频 N。

本实验明确不使用 `LOCAL_SWAP`。这里的 replacement 是把自然音频作为生成视频的最终音轨/评分音频，不是局部音频交换。

## 固定输入

- 固定图片：`data/data/image/{3,6,9}.png`，与 B025/B050 完全相同。
- 样本：复用 B025/B050 实验最终纳入的前 11 条 LRS3 音频样本，共 11 个 source group。
- 每个条件：3 张图片 × 11 条音频 = 33 个 image×audio 组合。
- 固定生成 ROI、评分 crop、25 fps、Wav2Lip checkpoint 和 SyncNet V2。
- B025/B050 的既有视频与矩阵作为同口径参考；本次新增 B075/MFA-linear 静态视频并评分。
- 每个 image×audio 单元在 N、B025、B050、B075、MFA-linear 的 native/replacement cells 上取共同有效窗口 W；三图结果先按音频聚合，再对 11 个 source group 做 bootstrap，10,000 次，seed=20260913。

## 结果

Sync-C 的 Δ 是候选减自然基线，越大越好；D gain 是自然基线 D 减候选 D，越大越好。区间为 source-group paired bootstrap 95% CI。B025/B050 在本表中使用包含全部条件的统一共同支持 W 重新计算，因此与原低强度笔记中的数字可能有小差异。

| 条件 | native ΔC | replacement ΔC | native D gain | replacement D gain | native C+ | replacement C+ |
|---|---:|---:|---:|---:|---:|---:|
| B025 | -0.081 [-0.141,-0.031] | -0.103 [-0.158,-0.044] | -0.101 [-0.178,-0.025] | -0.109 [-0.164,-0.055] | 7/33 | 7/33 |
| B050 | -0.182 [-0.293,-0.075] | -0.343 [-0.444,-0.242] | -0.092 [-0.229,+0.040] | -0.365 [-0.465,-0.268] | 7/33 | 1/33 |
| B075 | -0.323 [-0.527,-0.135] | -0.661 [-0.871,-0.483] | -0.074 [-0.243,+0.082] | -0.674 [-0.856,-0.522] | 7/33 | 0/33 |
| MFA-linear | -0.169 [-0.436,+0.057] | -0.967 [-1.299,-0.695] | +0.013 [-0.175,+0.198] | -0.985 [-1.270,-0.735] | 13/33 | 0/33 |

## 结论

[result] 在与 B025/B050 完全一致的静态图片条件下，B075 native 已低于自然基线；把同一 B075 生成视频的评分/最终音频换回自然音频后，Sync-C 进一步下降，ΔC=-0.661，CI 完全为负。MFA-linear native 的 C 区间跨 0，但 natural replacement 明显为负，ΔC=-0.967，CI 完全为负。

[conclusion] 该结果没有发现 static-image replacement 增益，反而显示候选音频驱动的生成视频与候选音频存在较强共适应：换回自然音频不能把视频提升到自然基线，可能使 SyncNet 分数进一步下降。它支持“native 分数不能作为独立视觉收益，replacement 是更严格的视觉收益门槛”，但不能把 replacement 下降解释为自然音频本身有害。

[boundary] 这是 11 条、3 张固定图片的 seen-fit 探索性实验，未做多重比较校正，也不证明跨身份/跨图片泛化；不授权训练 audio head 或生成头。

## 完整性核验

- 静态视频：66/66，所有 Wav2Lip source frame index 均为 0。
- 新 SyncNet score cells：132/132 完成。
- 可播放媒体：132/132 完成。
- native/replacement 的视频源相同；封装后音频与目标 WAV 的 PCM 逐样本一致。
- GPU 任务串行执行，结束后显存恢复到约 322 MiB。
- 本实验未读取或生成任何 LOCAL_SWAP 音频。

## 产物

- 运行根目录：`runs/static_image_b075_mfa_20260913/`
- 报告：`runs/static_image_b075_mfa_20260913/report.md`
- 输入协议：`runs/static_image_b075_mfa_20260913/inputs.json`
- 视频清单：`runs/static_image_b075_mfa_20260913/video_manifest.json`
- 评分清单：`runs/static_image_b075_mfa_20260913/score_manifest.json`
- 分析：`runs/static_image_b075_mfa_20260913/analysis.json`
- 可播放媒体：`runs/static_image_b075_mfa_20260913/media/`
- 实现脚本：`scripts/experiments/static_image_b075_mfa.py`

## Relations

- extends [[优质静态图低强度 Bridge replacement 2026-09-13]]
- relates_to [[LRS3 MFA-linear TFG native/replacement 2026-09-13]]
- relates_to [[Bridge 自然视频 SyncNet 强度扫描 2026-09-13]]

## Changelog

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 完成与 B025/B050 相同静态图片、前11样本的 B075/MFA-linear native/replacement 实验；排除 LOCAL_SWAP | September 13, 2026 | user request / agent execution |