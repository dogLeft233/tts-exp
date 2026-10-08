---
title: Wav2Lip ROI control failure diagnostic 2026-09-06
type: experiment
permalink: tts-exp/experiments/wav2-lip-roi-control-failure-diagnostic-2026-09-06
tags:
- wav2lip
- roi
- control
- diagnostic
- control-failed
---

# Wav2Lip ROI control failure diagnostic 2026-09-06

## Context

本诊断针对父实验 `Wav2Lip face-ROI replacement pilot 2026-09-06` 的门禁失败。它是 seen-fit 数据上的 CPU 离线复核，不生成新视频、不新增 SyncNet score、不运行 bridge，也不改变父 run。

## Execution

- 固定父入口：`runs/wav2lip_face_roi_replacement_20260906_host_fix5/`；fix5 指向 fix1 的不可变 GPU 资产，入口文件 hash、media/PCM 配对和 22 records / 22 source groups 均通过审计。
- 独立读取 198 个 score cells（154 个主 cell + 44 个 repeat cell），重建 `s[n]=n+1920*sin(2*pi*n/(L-1))`、共同候选行、31 点曲线、全部控制门禁和 source-group bootstrap。
- 诊断 run：`runs/wav2lip_roi_control_diagnostic_20260906_audit4/`。
- validator：valid；独立派生值与父历史值逐字段差异为 0。

## Result

- [status] concluded
- [result] 诊断终态为 `CONTROL_FAILURE_REPRODUCED`；历史科学终态保持 `CONTROL_FAILED`。
- [result] repeatability、baseline、generated-repeat、replacement-damage 均通过；A/B/O 均为 22/22。
- [result] inherited C 为 14/22，8 条失败记录全部由 `offset_error`（实际响应残差绝对值 > 1 帧）构成；`baseline_invalid=0`、`boundary_peak=0`、`unclear_peak=0`。失败 ID：`lrs3_6wk4dkYSrV0_00006`、`lrs3_6qqqVwM6bMM_00007`、`lrs3_73cTNHEQhkQ_00007`、`lrs3_796LfXwzIUk_00007`、`lrs3_7CIq4mtiamY_00007`、`lrs3_7DCofMA9eQA_00007`、`lrs3_6ydYeyNSQVY_00008`、`lrs3_79tRTivyMSM_00014`。
- [result] own-audio 的 Sync-C 95% CI 为 `[-0.170397, 0.186682]`，非劣性下界 `-0.10` 未通过；Sync-D CI 为 `[0.000202, 0.287374]`，offset 合格 22/22。CI 跨 0 本身不是失败条件。
- [result] own-audio 分解逐条满足 `own_C = median_change + own_D`，误差不超过 `1e-9`；这只说明评分曲线代数关系，不构成声学因果解释。
- [result] 本轮新增视频数为 0、新增 score cells 为 0、bridge 未执行、training/cross-model/generalization 均未授权。

## Conclusion and next step

当前最小下一步是对上述 C 失败记录做一次独立 SyncNet 局部峰复核，保持视频和音频不变，用来区分评分峰/实现问题与生成器响应不足；本诊断不自动执行该建议。控制门禁未通过前不运行原 bridge，也不宣称 replacement effect 或跨模型泛化。

## Report

- result：`runs/wav2lip_roi_control_diagnostic_20260906_audit4/result.md`
- final：`runs/wav2lip_roi_control_diagnostic_20260906_audit4/final.json`
- validation：`runs/wav2lip_roi_control_diagnostic_20260906_audit4/validation.json`
- OpenSpec：`openspec/changes/diagnose-wav2lip-roi-control-failure/`

## Observations

- [decision] 历史控制失败被独立复现，不通过扩大样本、换 seed、删离群点或修改阈值来追求 PASS。
- [insight] C 的可观测失败是响应 offset 残差，不是 baseline 无效、边界峰或峰不清晰。
- [insight] own-audio 失败的判定依据是 Sync-C 非劣性下界低于 `-0.10`，不是 CI 是否跨 0。
- [constraint] 后续任何 bridge 或训练都必须另立并通过新的控制门禁。

## Relations

- relates_to [[Wav2Lip face-ROI replacement pilot 2026-09-06]]
- implements [[openspec/changes/diagnose-wav2lip-roi-control-failure]]
