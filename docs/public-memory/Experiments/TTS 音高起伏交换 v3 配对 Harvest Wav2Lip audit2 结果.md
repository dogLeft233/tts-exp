---
title: TTS 音高起伏交换 v3 配对 Harvest Wav2Lip audit2 结果
type: experiment
permalink: tts-exp/experiments/tts-音高起伏交换-v3-配对-harvest-wav2-lip-audit2-结果
status: concluded
execution_status: complete
protocol: tts_f0_swap_v3_paired_harvest
run_id: tts_f0_swap_v3_audit2
validation_status: PASS
scientific_status: INSUFFICIENT_SUPPORT
date: '2026-09-18'
tags:
- tts
- f0
- wav2lip
- syncnet
- paired-harvest
- repair
---

# TTS 音高起伏交换 v3 配对 Harvest Wav2Lip audit2 结果

## Question

在不把 Harvest/DIO 检测器分歧误判为操作失败的前提下，F0 level/contour 交换能否在冻结的 Wav2Lip/SyncNet 流程中完成并支持方向性因果结论？

## Protocol

- Protocol：`tts_f0_swap_v3_paired_harvest`。
- 输入严格复制 parent `runs/tts_f0_swap_f0spec_audit2/inputs.json`：4 pilot + 24 formal；没有按结果重选。
- WORLD 仍使用 Harvest 搜索 60–600 Hz、StoneMask/cheaptrick/d4c、5 ms、原有普通平均保持公式、RMS/长度/PCM16 门。StoneMask 精修后固定接受 50–600 Hz，不 clip。
- 生产验收使用 H_ID/H_LEVEL/H_CONTOUR；DIO→StoneMask 只作交叉诊断。SyncNet 每侧仍要求至少 50 个共同窗口。

## Results

- Pilot：4/4 pair 通过，formal audio 放行。
- Formal audio：24/24 pair manifest 完成，48 个 receiver QC 中 47 个通过。`aishell1_test_400__BAC009S0770W0267/T` 的 CONTOUR p90 目标误差为 3.650 半音，按预注册规则不生成该 receiver 的 CONTOUR 视频。
- Wav2Lip：190 个 QC 条件视频完成；无渲染失败。
- SyncNet：380/460 cells 完成。1 个 pair 因音频 QC 排除，8 个 receiver 因共同窗口少于 50 排除。
- 分析：`complete_pair_count=16`，`main_pair_count=14`，6 speakers；`g_N` mean −0.0262（95% CI −0.0586…0.0049），`l_T` mean +0.0014（95% CI −0.0248…0.0257），两者 Bonferroni 区间均跨 0。`scientific_status=INSUFFICIENT_SUPPORT`，`native_status=NATIVE_ADVANTAGE_UNCONFIRMED`。
- 验收：`validation.json=PASS`；独立复算 `recompute.json=PASS`，bootstrap 索引一致。

## Interpretation

这次修复确实跑通了从配对 Harvest 音频、Wav2Lip 到 SyncNet 的工程链路，也确认原 v2 阻塞主要来自检测器/StoneMask 量测契约和下游账本问题。它没有得到 F0 起伏导致 TTS 增益的科学证明：正式完整案例少于预注册的 18 pair/6-speaker 主支持，而且当前 g_N/l_T 都不显著。短句支持门和一个 receiver QC 失败是下一轮扩大或重新冻结样本时必须处理的限制。

## Relations

- implements [[TTS 音高起伏交换的配对 Harvest 测量修复与 Wav2Lip 执行 Spec]]
- amends [[TTS 音高起伏交换的 WORLD 与测量链修复 Spec]]
- follows [[TTS 音高起伏交换 WORLD 与测量链修复结果 2026-09-18]]

## Changelog

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 完成 v3 audit2 全阶段执行和独立复算；结果保留为支持不足，不作方向性结论 | September 18, 2026 | agent execution |
