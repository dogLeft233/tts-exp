---
title: MFA-linear 原样基线实验
type: task
permalink: tts-exp/tasks/mfa-linear-原样基线实验
status: done
priority: high
current_step: 6
completed: 2026-08-18
steps:
- 定位并核对已有 MFA-linear 生成脚本、输入 manifest、checkpoint 和输出目录
- 生成或确认 69 条 validation 的原样 MFA-linear 音频，并检查采样率、时长、hash 和失败数
- 为 MFA-linear 运行 candidate-audio self 评测
- 为同一批 MFA-linear 视频替换 natural PCM，运行 replacement 评测
- 以 global-resample、raw TTS 和 natural audio 为参照，报告两层 paired SyncNet 结果
- 通过基线检查后再设计 residual warper smoke
assigned_to: claude
tags:
- task
- mfa-linear
- baseline
- syncnet
- lrs3
---

# MFA-linear 原样基线实验

## Context

在开始可学习 MFA-linear residual warper 之前，先建立严格的 MFA-linear 原样基线。该基线不加入可学习 residual、不使用 online SyncNet 训练，只复现已有 MFA-linear 时间映射和冻结声码器重合成流程。实验应与当前 LRS3 验证集使用一致的 69 条样本、Wav2Lip/SyncNet 版本和文件级评测协议，并保留既有结果不覆盖。

## Steps

1. [x] 定位并核对已有 MFA-linear 生成脚本、输入 manifest、checkpoint 和输出目录
2. [x] 生成或确认 69 条 validation 的原样 MFA-linear 音频，并检查采样率、时长、hash 和失败数
3. [x] 为 MFA-linear 运行 candidate-audio self 评测
4. [x] 为同一批 MFA-linear 视频替换 natural PCM，运行 replacement 评测
5. [x] 以 natural audio 与同 cohort 的既有 RhythmWarp replacement 作为参照，报告两层 paired SyncNet 结果；raw TTS/global-resample 未在本轮重复评分
6. [x] 通过基线检查后再设计 residual warper smoke

## Constraints

- 不修改或覆盖旧 full-objective、online RhythmWarp、MFA-linear 结果和 checkpoint。
- 不将 Wav2Lip 或文件级 SyncNet 放入训练图；本任务只做独立下游评测。
- replacement 必须验证 video stream 与 natural PCM 完整保留，不做 atempo、裁剪或补静音。
- LRS3 当前按 source_group 解释，不能声称 speaker-disjoint generalization。
- API key 不写入脚本、日志、数据或 checkpoint metadata。

## Current state

MFA-linear 原样 baseline 已在隔离目录 `runs/lrs3_qwen_cloud_n500_20260818/05_mfa_linear_baseline/` 完成。69/69 音频生成成功，均为 16 kHz FLOAT、精确 natural sample count，hash/QC 通过；69/69 Wav2Lip 视频生成成功；最终 207/207 SyncNet score cells 有限且完整。replacement 69/69 保留原视频流与 natural PCM。

Aggregate：natural audio 为 C=6.91496/D=7.54843；MFA-linear self 为 C=6.69759/D=7.78459；MFA-linear video + natural audio 为 C=5.51804/D=9.09554。相对 natural，MFA-linear self 的 paired ΔC=-0.21736、ΔD=+0.23616，C/D/joint 改善分别为 23/69、19/69、12/69；replacement 相对 MFA-linear self 的 ΔC=-1.17955、ΔD=+1.31094，改善分别为 4/69、2/69、2/69。与既有 RhythmWarp replacement 比较，MFA-linear replacement 的 ΔC=+3.42145、ΔD=-3.38630，69/69 在两个指标上均改善。

结论：MFA-linear 原样 baseline 的 self arm 略低于 natural，replacement arm 明显优于既有 RhythmWarp replacement，但换回 natural audio 后仍明显低于其自身 audio-conditioned video。该结果支持继续做 MFA-linear 初始化的 bounded residual smoke；不支持直接宣称可学习 residual 已有效。现阶段不声称 speaker-disjoint generalization。

## Relations

- parent_task [[建立可学习音频对齐模型]]
- relates_to [[MFA-linear 冻结 VC 重合成与下游验证]]
- relates_to [[LRS3 SyncNet RhythmWarp 训练]]
## Observations
- [status] done