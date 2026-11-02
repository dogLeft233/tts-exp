---
title: MFA-linear 音素内部非线性时间映射 2026-09-25
type: experiment
permalink: tts-exp/experiments/mfa-linear-音素内部非线性时间映射-2026-09-25
status: concluded
date: '2026-09-25'
hypothesis: 音素边界已对齐但内部事件时刻不同；有界单锚点重映射应改善自然音轨口型同步
result: 27条 T−F Sync-C −0.013；10说话人等权95% CI [−0.065,+0.030]
conclusion: 有界单锚点音素内部时间映射没有可靠修复主要差距
report: docs/experiments/37-mfa-linear-intraphone-time-warp.md
inputs: runs/aishell1_qwen_mfa_linear_n100_20260816; prior 10+17 sample IDs excluded
outputs: runs/mfa_linear_intraphone_time_20260925
model: WavLM-L6, HiFi-GAN, Wav2Lip GAN, SyncNet V2
code_paths:
- runs/mfa_linear_intraphone_time_20260925/experiment.py
- runs/mfa_linear_intraphone_time_20260925/oracle.py
- runs/mfa_linear_intraphone_time_20260925/analyze.py
tags:
- mfa-linear
- intraphone
- time-warp
- wav2lip
- syncnet
---

# MFA-linear 音素内部非线性时间映射

从 n100 中排除先前 10+17 条样本，再固定 27 条、10 位说话人。五路 N/F/T/P/O 均以同一自然语音作 SyncNet 音轨；T 在音素内按谱通量事件设单锚点，P 反向，O 用自然/TTS mel 选择有界单锚点。GPU 渲染 135/135 视频，裁剪 135/135，逐条五路自然 PCM 完全相同。

## Observations

- [status] concluded
- [report] [[37-mfa-linear-intraphone-time-warp]]；运行目录 `runs/mfa_linear_intraphone_time_20260925/`。
- [result] N/F/T/P/O 自然音轨平均 Sync-C=5.948/5.494/5.481/5.469/5.518。预定主比较 T−F=−0.013；10 位说话人等权 bootstrap 95% CI [−0.065,+0.030]；12/27 条上升。51 个事件锚点只覆盖 23 条，活动样本 T−F=−0.010。
- [result] O 的纯发声 Wav2Lip mel MAE 比 F 低 0.0121、23/27 更近；Sync-C 仅 +0.023，区间 [−0.037,+0.065]。自然 N 仍比 F 高 +0.454。T 的事件目标窗口固定偏移 SyncNet 距离 T−F=+0.043，区间跨零。
- [evidence] 4 条零锚点的 F/T/P 驱动音频 SHA256 相同，但分别渲染后 T−F Sync-C 为 −0.042、−0.079、+0.008、−0.007；绝对差中位数 0.025。事后将零锚点视作恒等处理时，全 27 条 T−F=−0.008。该波动已足以覆盖主效应量级。
- [conclusion] 当前 15–60 ms、有界单锚点的音素内部时间重映射不能可靠修复 MFA-linear 的自然音轨失分；自然声学引导改善 mel 也不保证 Sync-C 上升。不能据此排除多锚点、大幅度或跨音素错位。
- [limit] 样本与旧批说话人重合、同一张脸与 Wav2Lip，O 读取自然语音声学信息仅作诊断；共用 N 人脸轨迹的裁剪是配对 SyncNet V2 协议，不是各路独立追踪。

## Relations

- extends [[MFA-linear 新说话人发声段残差复核 2026-09-25]]
- relates_to [[MFA-linear 声码器与 Wav2Lip 响应拆分 2026-09-25]]

## Changelog

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 27 条五条件全量 GPU 视频、自然音轨评分与音素内部时间实验结论 | September 25, 2026 | user（实验与 GPU 继续请求）；agent（执行） |
