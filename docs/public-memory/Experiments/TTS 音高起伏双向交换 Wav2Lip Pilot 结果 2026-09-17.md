---
title: TTS 音高起伏双向交换 Wav2Lip Pilot 结果 2026-09-17
type: experiment
permalink: tts-exp/experiments/tts-音高起伏双向交换-wav2-lip-pilot-结果-2026-09-17
status: concluded
execution_status: MANIPULATION_NOT_VALIDATED
protocol: tts_f0_swap_v1
run_id: f0spec_audit2
date: '2026-09-17'
tags:
- tts
- wav2lip
- f0
- causal-intervention
- pilot
- blocked
---

# TTS 音高起伏双向交换 Wav2Lip Pilot 结果 2026-09-17

本实验执行 [[TTS 音高起伏双向交换的 Wav2Lip 因果诊断 Spec]]，使用本地 AISHELL-1 natural/faster_qwen3 TTS 配对、各自真实 MFA、pyworld 0.3.5 和固定静态 portrait。实验只完成音频 pilot；按预注册规则未进入 Wav2Lip 渲染或 SyncNet 主统计。

## 方法与实现

冻结审计从 784 条 manifest 记录得到 172 个可配对候选，排除 220 条，固定 4 对 pilot、24 对 formal、8 个 formal speakers。每侧使用 harvest→stonemask、cheaptrick、d4c 和 5 ms WORLD；独立 QC 重提取使用 DIO→stonemask。8 个音频臂的统一重采样、RMS、共同峰值缩放、精确长度和 F0 目标/重提轨迹均落盘。实现额外修正了渐入渐出权重造成的 CONTOUR 加权均值残差，并记录修正量；这保持预注册的 CONTOUR 均值恒等式。

## Pilot 结果

4 对均完成音频生成和独立 DIO 测量，但只有 1/4 对在 N、T 两个接收方向都通过第 1–5、7 项数值 QC（S0906 pair）。因此 gate 为 `MANIPULATION_NOT_VALIDATED`，正式 24 对音频被阻断，未生成视频、未产生 SyncNet 分数；独立复算状态为 `NOT_AVAILABLE`，因为没有 score/analysis manifest。

| pair | N QC | T QC | 主要原因 |
|---|---|---|---|
| BAC009S0765W0312 | fail | fail | ID voiced mask/coverage；LEVEL/CONTOUR 测量覆盖或 mask |
| BAC009S0770W0414 | fail | fail | ID voiced mask/coverage；LEVEL/CONTOUR 测量覆盖或 mask |
| BAC009S0901W0487 | fail | fail | ID voiced mask/coverage；LEVEL/CONTOUR 测量覆盖或 mask |
| BAC009S0906W0401 | pass | pass | 无数值 QC 原因 |

通过的 S0906 pair 的 CONTOUR 目标剂量约为 N=1.534、T=1.451 半音 RMS，CONTOUR 均值恒等式误差约 `3e-15`；未通过 pair 也普遍有可辨识剂量，但独立 DIO voiced coverage 不足，不能用于正式因果评分。

## 判读与边界

这是操作有效性失败，不是“F0 无效”或“没有 TTS 增益”的阴性证据。主要瓶颈是 WORLD 重合成后独立 DIO 的 voiced mask/覆盖门：部分 pair 的 ID coverage 约 0.68–0.77，超过预注册的 0.80 coverage 或 0.10 mask 阈值。不能为通过 pilot 放宽阈值，也不能在看 SyncNet 后换样本。未完成固定评分音轨的生成端检验，所以无法回答自然音频是否改善、TTS 音频是否退步、CONTOUR 是否优于 LEVEL，亦不能解释 TTS 优势。

## 产物

- run 根：`runs/tts_f0_swap_f0spec_audit2/`
- 冻结输入：`inputs.json`（self-hash；candidate 172，pilot 4，formal 24）
- pilot QC：`pilot_qc.json`；详细行：`audio_qc.csv`
- 参数与原/目标/重提 F0、sp/ap、映射：`parameters/*.npz` 与同名 JSON
- 听检表：`pilot_quality_manifest.json`，状态 `QUALITY_NOT_ASSESSED`
- 阻断正式阶段：`audio_manifest.json`
- 工程校验：`validation.json` 为 `PASS`；报告：`report.md`
- 独立复算：`recompute.json` 为 `NOT_AVAILABLE`

## 后续公式审计纠正

September 17, 2026：复算旧 parameters/*.npz 发现，约3e-15的恒等式误差对应 taper 加权均值，不是原目标的全部有声帧普通平均。后者在8个接收方上偏移约−0.043627～+0.054398半音。1/4通过数是旧程序的历史结果，不能证明原普通均值契约通过；现有失败也尚未区分原音频上的提取器分歧与重合成新增变化。后续按 [[TTS 音高起伏交换的 WORLD 与测量链修复 Spec]] 修复公式与落盘测量、补RAW检测对照。旧run保持只读，本纠正不改变“正式阶段未运行、无法判断F0机制”的结论。

## Observations

- [status] pilot 完成，正式渲染被预注册 gate 阻断 #f0 #wav2lip
- [result] 4 对中仅 1 对双向通过 DIO→stonemask 数值 QC；不能下 F0 阴性结论 #negative
- [diagnosis] 失败主要来自 WORLD ID/候选输出的 voiced mask 与原 harvest/stonemask 不一致、原 voiced coverage 不足 #quality-control
- [invariant] 通过 pair 的 CONTOUR 加权均值恒等式误差约 3e-15，统一 RMS/长度/峰值契约通过 #audio
- [boundary] 没有 Wav2Lip/SyncNet 分数，不能判断生成端自然/TTS 方向效应或 TTS 增益原因 #scope
- [decision] 不放宽 QC 门、不按分数补选；若继续，应先设计独立的 WORLD/DIO 可行性修复或更换干预载体，并更新 spec 后重新冻结 #decision

## Implementation audit

- [implementation] 最终 runner 将 pilot/formal 音频阶段严格隔离，参数包保存逐帧 donor 时间、phone 索引、插值下标/权重和无效原因；Wav2Lip worker 支持显式 CPU/CUDA。代码契约测试 12 项通过，协议、pilot、阻断 manifest 和核心 JSON self-hash 一致。

## Relations

- implements [[TTS 音高起伏双向交换的 Wav2Lip 因果诊断 Spec]]
- follows [[TTS 时间辨识与同文本实例交叉实验 2026-09-17 结果]]
- relates_to [[18-tts-feature-control-validation]]
- relates_to [[TTS 声学变化重组轨迹与剩余项干预：实现与结果]]