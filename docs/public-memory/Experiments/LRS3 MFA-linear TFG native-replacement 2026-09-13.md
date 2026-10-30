---
title: LRS3 MFA-linear TFG native/replacement 2026-09-13
type: experiment
permalink: tts-exp/experiments/lrs3-mfa-linear-tfg-native-replacement-2026-09-13
status: concluded
date: 2026-09-13
tags:
- mfa-linear
- wav2lip
- syncnet
- replacement
- lrs3
---

# LRS3 MFA-linear TFG native/replacement 2026-09-13

[status] concluded

## 目的

在同一批 LRS3 样本和同一套 TFG/Wav2Lip 生成流程上，区分两件事：

1. TFG 使用 MFA-linear 音频生成视频后，保留 MFA-linear 作为最终音轨（native）。
2. 同一批 MFA-linear 生成的视频帧不变，只把最终封装音轨换成自然音频（natural replacement）。

本实验不包含 `LOCAL_SWAP`。这里的 replacement 指最终 mux 阶段把自然音频放到已经生成的视频上，不是局部替换视频驱动音频，也不是重新生成视频。

## 实验协议

- 固定 22 条 LRS3 natural-to-TTS bridge confirmation cohort 样本。
- 固定样本对应的人脸视频和自然音频；MFA-linear 音频来自既有 cohort。
- 对每条样本只用 MFA-linear 音频生成一次 Wav2Lip 视频，随后产生两个评估单元：
  - `V_MFA_LINEAR / A_MFA_LINEAR`：MFA-linear native。
  - `V_MFA_LINEAR / A_N`：同一视频帧，最终音轨换成 natural。
- 采用严格 mux 校验：替换条件的视频 elementary stream 与 native 相同，音频 PCM 与目标音频相同。
- SyncNet V2：Sync-C 越高越好，Sync-D 越低越好；paired record 统计，bootstrap 10,000 次，seed=20260913。
- B075 两个结果直接复用 2026-09-04 已确认的同一 cohort 结果，作为对照：
  - B075 native：`V_B075/A_B075`
  - B075 natural replacement：`V_B075/A_N`

## 结果

相对各自 natural baseline 的配对变化如下：

| 条件 | ΔSync-C（95% bootstrap CI） | Sync-C 提升条数 | ΔSync-D（95% bootstrap CI） | Sync-D 改善条数 | offset 改善条数 |
|---|---:|---:|---:|---:|---:|
| B075 native（既有确认结果） | -0.648 [-0.810, -0.476] | 1/22 | -0.497 [-0.610, -0.367] | 2/22 | 1/22 |
| B075 natural replacement（既有确认结果） | +0.031 [-0.034, +0.103] | 9/22 | +0.017 [-0.034, +0.068] | 11/22 | 0/22 |
| MFA-linear native（本次） | -0.893 [-1.079, -0.712] | 1/22 | -0.862 [-0.996, -0.731] | 0/22 | 0/22 |
| MFA-linear natural replacement（本次） | +0.022 [-0.066, +0.117] | 11/22 | +0.009 [-0.059, +0.075] | 12/22 | 1/22 |

## 结论

[result] 在本实验的 TFG-generated-video endpoint 上，B075 和 MFA-linear native 都明显低于 natural baseline；仅仅把最终音轨换回 natural 后，Sync-C 基本恢复到 baseline 附近，但没有得到统计上可靠的正向增益。B075 replacement 的 ΔSync-C=+0.031，MFA-linear replacement 的 ΔSync-C=+0.022，两个 CI 都跨过 0。

[conclusion] 这说明 replacement 的主要作用是恢复音频与已经生成的视频之间的输入/评估兼容性，不能证明 MFA-linear 或 bridge 改善了视频中的唇形运动。native 与 replacement 的差异支持“生成视频质量/驱动条件”和“最终评估音轨”是两个不同因素；在本 cohort 上，生成视频的变化没有被 replacement 转化为可复现的 Sync-C 提升。

[boundary] 因此不能把自然音频 replacement 的近 baseline 结果解释成 LOCAL_SWAP 成功，也不能据此授权 audio head 训练。LOCAL_SWAP 是另一个独立实验，本次明确未使用。

## 复现与产物

- 运行根目录：`runs/lrs3_mfa_linear_tfg_replacement_20260913/`
- 报告：`runs/lrs3_mfa_linear_tfg_replacement_20260913/report.md`
- 视频生成清单：`runs/lrs3_mfa_linear_tfg_replacement_20260913/01_videos/manifest.json`
- SyncNet 评分清单：`runs/lrs3_mfa_linear_tfg_replacement_20260913/02_scores/manifest.json`
- 分析 JSON：`runs/lrs3_mfa_linear_tfg_replacement_20260913/03_analysis.json`
- MFA native 视频：`runs/lrs3_mfa_linear_tfg_replacement_20260913/02_scores/mux/MFA_LINEAR/MFA_LINEAR/`
- MFA natural replacement 视频：`runs/lrs3_mfa_linear_tfg_replacement_20260913/02_scores/mux/MFA_LINEAR/N/`

## 可复现性边界

[boundary] 本次 MFA-linear render 使用当前 `[redacted-local-path]`，SHA256 为 `e50d468e8b0adfb05733f5b87b3cff34829c4a8c1aea50c865aa8bdfe4bb150f`。历史 B075 确认结果登记的旧 Python executable hash 为 `1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118`，该旧 executable 当前不可用；但 Wav2Lip checkpoint（`ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8`）和 `inference.py`（`42cfc8d3060921e6714c8e08243e5508d2e637d29cd0be78760492c5d26380a9`）与历史记录一致。因此 B075 与本次 MFA-linear 的合并比较应作为描述性结果，而不是完全相同 runtime 的严格复现实验。

## Relations

- relates_to [[LRS3 natural-to-TTS bridge confirmation result]]
- relates_to [[MFA-linear 机制小样本诊断 2026-09-13]]
- relates_to [[优质静态图低强度 Bridge replacement 2026-09-13]]
- relates_to [[Bridge 自然视频 SyncNet 强度扫描 2026-09-13]]