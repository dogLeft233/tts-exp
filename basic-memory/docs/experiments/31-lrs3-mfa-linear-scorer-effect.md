---
title: 31-lrs3-mfa-linear-scorer-effect
type: experiment
permalink: tts-exp/docs/experiments/31-lrs3-mfa-linear-scorer-effect
status: concluded
date: '2026-09-24'
tags:
- lrs3
- mfa-linear
- wav2lip
- syncnet
- scorer-effect
---

# 31-lrs3-mfa-linear-scorer-effect

LRS3 上用同一批音频、同一人脸来源和已完成的 Wav2Lip 渲染，检验 SyncNet 是否系统性偏向 MFA-linear 合成音频。这里的 MFA-linear 是以 TTS 特征为条件、映射到自然语音时间轴并经 WavLM/HiFi-GAN 合成的候选音频；它不能代表所有 TTS。

## 协议与分母

原始 external-fit face-ready 清单有 194 条、23 个 source group。按固定的 SyncNet `min_track=50` 门槛，在评分前用 ffprobe 检查两个生成视频的源帧数，对称排除 45 帧的 `lrs3_6yR5OUVb2gY_00005` 和 46 帧的 `lrs3_6qqqVwM6bMM_00014`。最终 192 条、23 组，四格各 192 个，总计 768 个 SyncNet 分数；没有按分数筛样本。6 个历史候选 WAV 路径缺失，按原始生成环境重建，SHA-256 均与渲染清单一致。

两个独立因素是生成 Wav2Lip 视频帧所用音频 G（natural / MFA-linear）和最终评分音轨 E（natural / MFA-linear）。四格为 `G_N_E_N`、`G_M_E_N`、`G_N_E_M`、`G_M_E_M`。评分音轨以 16 kHz mono PCM16 严格 mux，视频流保持不变并校验。Sync-C 越高越好，Sync-D 越低越好；下文将 Sync-D 的下降记为正向“改善”。推断先在 source group 内求均值，再对 23 组等权平均；95% CI 为 10,000 次组级 bootstrap，正向 p 为组级精确 sign-flip 单侧检验。

## 四格分数

以下是 192 条记录等权的原始均值，用于看四格模式；正式效应量和区间用组级配对统计。

| 视频驱动音频 G | 评分音轨 E | Sync-C | Sync-D |
|---|---|---:|---:|
| natural | natural | 7.228 | 7.300 |
| MFA-linear | natural | 6.016 | 8.369 |
| natural | MFA-linear | 5.788 | 8.775 |
| MFA-linear | MFA-linear | 6.839 | 7.529 |

## 组级配对结果

| 对比 | ΔSync-C [95% CI] | Sync-D 改善 [95% CI] |
|---|---:|---:|
| 固定 natural 视频，E: MFA-linear − natural | −1.479 [−1.684, −1.287] | −1.504 [−1.700, −1.314] |
| 固定 MFA-linear 视频，E: MFA-linear − natural | +0.780 [+0.650, +0.912] | +0.807 [+0.688, +0.921] |
| 两种固定视频等权平均的 MFA-linear 音轨效应 | −0.350 [−0.491, −0.217] | −0.348 [−0.485, −0.232] |
| 固定 natural 音轨，G: MFA-linear − natural | −1.199 [−1.306, −1.086] | −1.055 [−1.147, −0.960] |
| 固定 MFA-linear 音轨，G: MFA-linear − natural | +1.060 [+0.954, +1.170] | +1.256 [+1.118, +1.400] |
| 生成视频差异随评分音轨改变的交互 | +2.258 [+2.055, +2.459] | +2.310 [+2.117, +2.511] |

固定 natural 视频时，23/23 组换成 MFA-linear 音轨的 Sync-C 和 Sync-D 均变差；固定 MFA-linear 视频时，换成 MFA-linear 音轨使 Sync-C 在 23/23 组改善、Sync-D 在 22/23 组改善。交互两指标均在 23/23 组为正，精确单侧 p=1.19×10⁻⁷。组均 MFA-linear 音轨主效应在两个指标上都为负，CI 不跨零。

## 结论与边界

这批数据**不支持 SyncNet 对 MFA-linear 音轨普遍加分**。同一视频的音轨替换效应会随视频驱动音频反向：natural 视频配 natural 音轨更高，MFA-linear 视频配 MFA-linear 音轨更高。即 SyncNet 分数强烈受生成视频与最终评分音轨的匹配关系影响；natural/natural 仍是四格中 Sync-C 最高、Sync-D 最低的一格。只拿 `G_N_E_M` 和 `G_M_E_N` 两个交叉格互比，会混淆生成因素与评分因素；完整 2×2 的交互才显示这个机制。

结果是当前 LRS3 external-fit、Wav2Lip、MFA-linear 和 SyncNet V2 的测量现象，不是 held-out 泛化或人眼口型质量判定，也不能推出 SyncNet 对所有 TTS 均无偏好。2026-09-13 的 22 条 MFA-linear native/natural replacement 实验使用另一批记录，不能与本轮效应量直接拼接；本轮在同一 192 条上重测了两个方向和两条对角线。早先严格 MFA 对齐中 `spn` 仍为未知语音，未改标成静音以扩大分母。

## 复现产物

- 运行根目录：`runs/lrs3_mfa_scorer_effect_20260923/`
- 冻结协议：`protocol.json`，SHA-256 `6f980d8371ee30dbcf98f71ba780679a4af49f4e1ef55c7b77115d6bcadcf332`
- 候选 WAV 修复：`01_repair/manifest.json`（6 条，SHA 与父清单一致）
- 完整评分清单：`02_scores/manifest.json`（192×4=768，complete）
- 分析与可读报告：`03_analysis.json`、`report.md`
- 代码：`scripts/experiments/lrs3_mfa_linear_replacement_mfa3_exploratory/run_scorer_effect.py` 和 `scorer_effect.py`
- 验证：协议/修复/评分/分析哈希链及 768 个格子核对通过；相关纯统计、评估与严格 mux 测试 10 项通过。

## Observations
- [status] concluded

- [result] 同一 192 条 LRS3 上，固定 natural 视频换 MFA-linear 评分音轨，组均 ΔSync-C=-1.479、Sync-D 改善=-1.504。 #syncnet
- [result] 固定 MFA-linear 视频换 MFA-linear 评分音轨，组均 ΔSync-C=+0.780、Sync-D 改善=+0.807。 #syncnet
- [result] MFA-linear 音轨两视频臂等权平均效应为负：组均 ΔSync-C=-0.350、Sync-D 改善=-0.348。 #scorer-effect
- [result] 评分音轨×生成视频交互为正：ΔSync-C=+2.258、Sync-D 改善=+2.310，23/23 组同向。 #scorer-effect
- [conclusion] 本轮不支持 SyncNet 对 MFA-linear 音轨普遍偏好；音轨与视频驱动条件的匹配主导分数。 #mfa-linear
- [boundary] 仅覆盖 LRS3 external-fit、Wav2Lip、MFA-linear 与 SyncNet V2；不是人类感知质量或所有 TTS 的结论。

## Relations

- extends [[LRS3 MFA-linear TFG native/replacement 2026-09-13]]
- relates_to [[SyncNet 短音频静默无输出]]
- relates_to [[MFA-linear existing-data diagnostic outcome]]
