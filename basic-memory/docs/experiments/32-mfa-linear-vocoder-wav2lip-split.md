---
title: 32-mfa-linear-vocoder-wav2lip-split
type: experiment
permalink: tts-exp/docs/experiments/32-mfa-linear-vocoder-wav2lip-split
status: concluded
date: '2026-09-25'
cohort_size: 10
artifact: runs/mfa_linear_vocoder_wav2lip_split_20260925/report.md
tags:
- mfa-linear
- wav2lip
- vocoder
- syncnet
- mechanism
---

# MFA-linear 重建声学与 Wav2Lip 口型响应拆分

## 协议

同一批 10 位 AISHELL-1 说话人的短视频：N 为自然音频及 Wav2Lip 视频；R 为自然音频经冻结 WavLM-L6 与 prematched HiFi-GAN 直接重合成，再用同脸、同 Wav2Lip GAN 与 `--nosmooth` 生成视频；M 为既有 Qwen 云端 TTS 经 MFA-linear/WavLM/同一声码器后的音频及视频。三路音频与视频在自然时间轴等长，25 fps。R 代表整个 WavLM+声码器重建链，不是纯 HiFi-GAN 消融。

N/R/M 音频走同一 x264/AAC MP4→AVI→16 kHz PCM 路径，逐样本解码采样数一致；N 音轨与新跑的官方流水线 PCM 哈希一致。新跑官方 SyncNet 人脸裁剪后做 3×3 音视频交叉评分。抽查 a1_002、a1_049 的 N×N 与 `run_syncnet.py` 完全一致（6.815、6.385）；90 个单元均无最佳偏移落边界。

## 结果

10 条配对样本的平均 Sync-C，行是视频，列是音轨；越高越好。

| 视频 / 音轨 | N 自然 | R 重建 | M MFA-linear |
|---|---:|---:|---:|
| N | 5.920 | 5.746 | 5.153 |
| R | 5.843 | 6.358 | 5.095 |
| M | 5.229 | 5.402 | 6.504 |

固定自然音轨，R−N 为 −0.076，M−N 为 −0.691，M−R 为 −0.614（M 低于 R：9/10 条）。M 视频使用自己的 M 音轨评分比使用自然音轨高 1.276，10/10 条为正；说明生成口型更跟随实际驱动的 M 音频。固定零偏移的 SyncNet 距离 N/R/M 配自然音轨依次为 12.363/12.451/12.752，越低越好。

Wav2Lip 实际 mel 的干净自然语音窗口平均 MAE：R−N 0.395，M−N 0.649；10/10 条 R 更接近 N。同帧嘴唇开度相关：R/N 0.891、M/N 0.774；按 N 开度动态范围归一的开度差 0.139/0.220；两项均 10/10 条 R 更接近 N。R 的 MFA 音素与 N 匹配 464/466 个，每样本音素中心误差中位数 5 ms；先前 M 同样 464/466、5 ms。排除 5 条自然语音存在未被 TTS 保留的长停顿后，剩余 5 条自然音轨 Sync-C 为 N 5.965、R 5.773、M 5.409；M−R 仍为 −0.364。

## 结论与边界

直接重建自然音频的 WavLM+HiFi-GAN 链本身不足以解释本批 M 视频的自然音轨失分。M 输入 Wav2Lip 的 mel 与自然音频差异更大，生成的口型也更偏离自然视频，但它能与自己的 M 音频取得较高 Sync-C。需要继续拆分 TTS 内容、局部停顿、韵律和 MFA-linear 特征映射；本实验不能唯一归因其中某项，也不能将 R 当成纯声码器消融。SyncNet 自轨高分不等于自然音轨迁移能力。样本量 10，仅作机制诊断。

旧 N/M 自然音轨均值 5.940/5.243，当前同协议为 5.920/5.229，量级与方向一致。

## 复现产物

- 完整报告与逐样本表：`runs/mfa_linear_vocoder_wav2lip_split_20260925/report.md`
- 原始交叉评分：`runs/mfa_linear_vocoder_wav2lip_split_20260925/scores.json`；汇总：`analysis.json`；声学/口型：`mel.json`、`mouth.json`；音素对齐：`mfa_alignment/`
- 代码：`scripts/experiments/mfa_linear_vocoder_wav2lip_split.py`；依次运行 `generate`、`render`、`prepare_clock`、`score`、`mel`、`mouth`、`align`、`analyze`（`align` 需要 MFA 的 `mandarin_china_mfa` 字典和 `mandarin_mfa` 声学模型）

## Observations

- [result] 固定自然音轨，N/R/M 视频 Sync-C 为 5.920/5.843/5.229；重建链对照接近自然基线，M 明显偏低。 #syncnet
- [result] M 自轨较 M 视频+自然音轨高 1.276，10/10 条同向；模型生成视频更跟随 M 驱动音轨。 #wav2lip
- [evidence] mel MAE R−N/M−N 为 0.395/0.649，嘴唇开度相关为 0.891/0.774，差异逐样本同向。 #mfa-linear
- [boundary] R 测试的是 WavLM+HiFi-GAN 整条重建链；目前不能单独指认 HiFi-GAN 或 TTS/MFA 的某一组件。 #mechanism

## Relations

- extends [[MFA-linear 逐音素口型与缺失停顿诊断 2026-09-25]]
- relates_to [[MFA-linear 声码器与 Wav2Lip 响应拆分 2026-09-25]]
- relates_to [[NAS 短窗口 10 说话人扩大样本试验 2026-09-24]]

## Changelog

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 完成 10 条 3×3 音视频交叉评分、mel/口型/MFA 对照和官方评分抽查 | September 25, 2026 | user（实验与 BM 请求）；agent（执行） |