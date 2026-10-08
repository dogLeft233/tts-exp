---
title: LRS3 Wav2Lip 局部时间传递诊断结果
type: experiment
permalink: tts-exp/experiments/lrs3-wav2-lip-局部时间传递诊断结果
status: concluded
date: '2026-09-05'
run_root: runs/lrs3_wav2lip_timing_transfer_20260905_v8
scientific_decision: GENERATED_RESPONSE_UNRESOLVED
engineering_decision: GO
reference_conditioned_audio_head_spec_eligible: false
tags:
- lrs3
- wav2lip
- syncnet
- local-timing
- diagnostic
- negative-result
---

# LRS3 Wav2Lip 局部时间传递诊断结果

## Context

本实验承接 [[LRS3 natural-to-TTS bridge confirmation result]]、[[LRS3 局部时间控制校准实验结果]] 和 [[LRS3 真实视频局部时间敏感性诊断]]。前两轮说明平滑音频局部 warp 的 endpoint 控制自身有效性不足，后一轮说明在真实视频固定、自然音频固定时，SyncNet 能检测离散视频局部 timing 扰动；本轮进一步问：已有 Wav2Lip 生成视频是否会把输入音频的局部 timing 变化传递到生成画面。

实验按 OpenSpec `diagnose-lrs3-wav2lip-timing-transfer` 执行，只使用已存在的 22-record / 22-source-group fit-only cohort、既有 real/Wav2Lip 视频、固定裁脸轨迹和官方 SyncNet V2。没有生成新的 TTS 或 Wav2Lip 视频，没有训练、MFA/DTW、heldout 访问、参数搜索或按结果重试。

## Protocol and execution

- 最终 run：`runs/lrs3_wav2lip_timing_transfer_20260905_v8/`
- 视频臂：R（真实视频）、G_N（已有自然音频 Wav2Lip 视频）、G_W（已有 warp 音频 Wav2Lip 视频）；音频臂：N（自然 PCM）、W（从既有 `LOCAL_WARP_120` 按冻结公式精确重建的 PCM）。
- 每条记录 6 个主 cell（3 视频 × 2 音频）和 2 个独立重复基线，共 132 主 cell + 44 repeat = 176 个 SyncNet cell。
- W 使用端点固定的 `s[n]=n+1920*sin(2*pi*n/(L-1))`、float64 线性插值和 nearest-ties-to-even int16；保留完整 PCM，不截断音频、不补视频。
- A/B/C/O 分别检验真实视频音频敏感性、生成视频 endpoint 敏感性、生成视频对局部音频 timing 的响应、生成视频在 W 音频下的自洽性。固定阈值为每项至少 18/22，重复性要求 22/22，两个 natural baseline 各至少 20/22。

## Results

- 工程终态：`engineering_decision=GO`；22/22 input audit、66 video streams、132 mux cells、176/176 scores 全部完成。
- 独立 validator：input audit、protocol、media、score、analysis、final 六个阶段均为 `valid`。
- 重复性：22/22；R/N baseline 22/22；G_N/N baseline 21/22。
- A（真实视频音频敏感性）：22/22，通过。
- B（生成视频 endpoint 音频敏感性）：21/22，通过。
- C（生成视频响应局部 timing）：0/22，失败。
- O（生成视频自洽局部对齐）：0/22，失败。
- 科学终态：`GENERATED_RESPONSE_UNRESOLVED`。共同窗口的描述性变化为 C(G_W/W)-C(G_N/N) mean -3.597，95% CI [-3.996, -3.209]；C(G_W/W)-C(G_W/N) mean -3.426，95% CI [-3.802, -3.058]；D(G_N/N)-D(G_W/W) mean -3.383，95% CI [-3.778, -3.007]；D(G_W/N)-D(G_W/W) mean -3.253，95% CI [-3.625, -2.892]。

## Conclusion and boundary

本轮支持一个较窄的工程诊断结论：在固定历史 Wav2Lip 视频、固定裁脸轨迹和官方 SyncNet endpoint 下，N→W 的局部 audio timing 变化能够被真实视频（A）稳定感知，也大体能被自然音频 Wav2Lip endpoint（B）感知；但没有证据表明已有 Wav2Lip 生成画面把这一局部变化按预期传递到视觉 timing（C=0/22），也没有通过生成视频自洽性检查（O=0/22）。因此不能据此声明 Wav2Lip 已建立可用的局部 timing response，也不能支持 TTS/bridge 唇形同步收益、reference-conditioned audio head、训练或部署。

共同窗口中 G_W/W 相对 G_N/N 的 Sync-C/D 变化明显为负，只是描述性证据，不能反转 C/O 的失败判定；它更提示生成视频存在较强的条件/内容变化或 endpoint 交互，需另立受控实验拆分，不能把分数下降解释为局部 timing 传递成功。

按历史计划，保持既有 `CONTROL_FAILED` 结论不变，`reference_conditioned_audio_head_spec_eligible=false`；下一步不直接进入训练或 audio-head，若继续应另立 OpenSpec，先设计能在生成端直接证明时序响应的最小控制。 

## Observations
- [status] concluded

- [decision] Wav2Lip timing-transfer scientific gate 未通过，终态为 `GENERATED_RESPONSE_UNRESOLVED` #wav2lip
- [result] A=22/22、B=21/22，但 C=0/22、O=0/22；生成视频 endpoint 的局部响应尚未建立 #syncnet
- [result] 工程链 22/22 records、132 主 cell、44 repeat cell 全部完成，独立 validator 六阶段均为 valid #reproducibility
- [insight] 真实视频和生成 endpoint 能看到音频变化，不等于生成画面实际传递了局部 timing #causality
- [decision] `reference_conditioned_audio_head_spec_eligible=false`，历史 bridge/control `CONTROL_FAILED` 不被改写 #project-policy

## Relations

- implements [[diagnose-lrs3-wav2lip-timing-transfer]]
- relates_to [[LRS3 natural-to-TTS bridge confirmation result]]
- relates_to [[LRS3 局部时间控制校准实验结果]]
- relates_to [[LRS3 真实视频局部时间敏感性诊断]]
- relates_to [[局部 timing 与 acoustic mismatch 分阶段诊断]]