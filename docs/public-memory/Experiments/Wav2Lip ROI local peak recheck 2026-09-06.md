---
title: Wav2Lip ROI local peak recheck 2026-09-06
type: experiment
permalink: tts-exp/experiments/wav2-lip-roi-local-peak-recheck-2026-09-06
status: concluded
tags:
- wav2lip
- roi
- local-peak
- diagnostic
- peaks-reproduced
---

# Wav2Lip ROI local peak recheck 2026-09-06

## Context

本实验是对 Wav2Lip face-ROI replacement pilot 控制失败的定向评分诊断。上一轮 CPU 诊断复用了历史距离矩阵，已确认 8 条 C 失败均为 offset_error，但尚未检查历史矩阵的产生过程。本轮固定同一父媒体、音频、模型与权重，从 muxed media 重新 forward SyncNet，独立计算距离与局部峰，判断局部响应误差是否可由独立评分实现复现。

## Protocol and execution
固定父 run：`runs/wav2lip_face_roi_replacement_20260906_host_fix5/`，并绑定上一轮诊断的完整证据。按父 cohort 顺序选取 8 条历史 C 失败与前 2 条通过记录；每条仅评分 `(G_N,N,false)`、`(G_W,N,false)`，共 20 个新 cell。

使用 `[redacted-local-path]`、CPU、torch threads=4、batch=20。每个 cell 从原 muxed media 重新解码 JPEG/BGR 与 16k mono PCM，使用独立的官方 `SyncNetModel.S` forward、float32 embedding 和独立距离实现，保存 visual/audio embeddings 与 `[T,31]` distance matrix。旧 embedding/matrix 只用于对照，未作为新评分输入。

最终 run：`runs/wav2lip_roi_peak_recheck_20260906_review2/`。20/20 cell、10/10 record 完成；新生成视频 0。离线 validator 单独运行并返回 valid，OpenSpec 严格校验、ruff 和 focused pytest 均通过（9 passed）。
## Result and interpretation
终态为 `PEAKS_REPRODUCED`。独立新推理与历史矩阵的最大绝对差为 `0.000053406`，局部曲线最大绝对差为 `0.000006270`，均低于 `0.001` 容限；峰 offset、清晰度和逐条 C 判定全部一致。8/8 历史失败记录复现，2/2 历史通过记录复现。

8 条失败仍全部是历史可见的 offset_error：7 条 PLUS 残差约 `−1.15`（实际响应约 −4 帧、预期约 −2.85 帧），1 条 MINUS 残差约 `−1.86`；因此应称为局部响应偏差/误差，不能据此归因于生成器响应不足。该实验只说明独立评分实现复现了 SyncNet 所见结果，不能区分 SyncNet 表征限制、warp 后音频变化和生成器响应。

历史科学终态保持 `CONTROL_FAILED`；own-audio 未重测，bridge、训练和跨模型泛化均未授权。下一步如继续，只能针对具体评分差异另立单一修复实验；本线到此停止。
## Observations
- [status] concluded
- [result] 20 个新评分 cell、10 条记录全部完成；validator valid，独立差异数为 0。
- [result] `PEAKS_REPRODUCED`：最大矩阵差 `5.3406e-05`，最大局部曲线差 `6.2695e-06`，8/8 失败与 2/2 通过均复现。
- [insight] 8 条 C 失败是评分中可复现的局部响应偏差；复现结果不等同于生成器响应不足的因果证据。
- [constraint] 历史 `CONTROL_FAILED`、own-audio 未重测、bridge/训练/泛化未授权均保持不变。
## Relations

- relates_to [[Wav2Lip face-ROI replacement pilot 2026-09-06]]
- relates_to [[Wav2Lip ROI control failure diagnostic 2026-09-06]]
- implements [[openspec/changes/recheck-wav2lip-roi-local-peaks]]

## Report

- result：`runs/wav2lip_roi_peak_recheck_20260906_review2/result.md`
- final：`runs/wav2lip_roi_peak_recheck_20260906_review2/final.json`
- validation：`runs/wav2lip_roi_peak_recheck_20260906_review2/validation.json`
- OpenSpec：`openspec/changes/recheck-wav2lip-roi-local-peaks/`

## Execution note

首个正式 run `review1` 已完成 20 个 cell，但因 producer 漏写 validator 要求的 `location_name` 在收尾时被阻塞；修复后使用新 run `review2` 完整重跑并通过。父资产未修改。