---
title: LRS3 真实视频局部时间敏感性诊断
type: experiment
permalink: tts-exp/experiments/lrs3-真实视频局部时间敏感性诊断
status: concluded
date: September 5, 2026
tags:
- lrs3
- syncnet
- local-timing
- diagnostic
---

# LRS3 真实视频局部时间敏感性诊断

## Context

承接上一轮 `LOCAL_WARP_120` 控制自身有效性与替换敏感性失败。本实验按 OpenSpec `diagnose-lrs3-real-video-local-timing`，固定自然音频，仅改变真实视频的局部帧采样，诊断冻结 SyncNet forward score 是否能识别已知局部时间错配。

## Protocol

- 固定历史 22-record / 22-source-group cohort，保持原顺序与 source-group 结构。
- 三臂：`REAL`、独立重复编码的 `REAL_REPEAT`、`VIDEO_WARP_120`。
- `VIDEO_WARP_120` 使用端点固定、最大约 ±3 帧的离散视频采样映射；音频字节保持不变。
- 冻结官方 SyncNet V2、预处理、权重与 `vshift=15`；导出完整 distance matrix 及全局/局部 offset。
- 主要判据：局部 +3 帧区域的 offset 应移动约 -3 帧，局部 -3 帧区域应移动约 +3 帧；22 条中至少 18 条通过。

## Boundary

这是 endpoint 诊断，不是 TTS、TFG 增益、梯度、Wav2Lip 因果或可部署策略验证。工程不完整、重复性失败或基线局部峰不清晰时，按 spec 停止并报告原因。

## Amendment result

按 OpenSpec `fix-lrs3-real-video-tail-contract` 执行 `bounded_audio_tail_v2` 修订：在 25 fps / 16 kHz 下用整数定义 `d=N-640*F`，接受 `-640 <= d <= 1280`，保留完整 PCM 和视频帧，并在裁脸前审计全部输入。

- [result] 新 run `runs/lrs3_real_video_local_timing_20260905_tail_v2/` 的 input audit 为 22/22 通过、0 blocked；19 条为 `within_original_bound`，3 条为 `extended_audio_tail`：`lrs3_73jPh0eRPSY_00008` 为 +768 samples（1.2 帧），`lrs3_6qqqVwM6bMM_00007` 与 `lrs3_79tRTivyMSM_00014` 各为 +896 samples（1.4 帧）。最大音视频起点差 0 ms。
- [result] 三臂诊断完整完成：22 records / 22 source groups / 66 cells；`REAL` 重复性 22/22、基线清晰且对齐 22/22、局部恢复 22/22。
- [result] 描述性全局变化为 `C(REAL)-C(WARP)` 均值 +4.433，95% CI [+3.976,+4.883]；`D(WARP)-D(REAL)` 均值 +4.240，95% CI [+3.799,+4.682]。
- [conclusion] `engineering_decision=GO`，`scientific_decision=LOCAL_TIMING_DETECTED`。这确认在固定真实视频、自然音频不变、离散局部视频帧扰动下，SyncNet forward score 能检测该预注册的局部 timing mismatch。
- [validation] 独立 validator 对 input audit、protocol、media、score、final 全部返回 `valid`；旧 `20260905_v2` BLOCKED 终态仍只读保留，`20260905_tail_v1` 的 S3FD 相对权重路径阻塞也已修复后由 `tail_v2` 重跑，不修改科学参数或按结果重试。
- [limitation] `review_status=PENDING`，尚未完成盲审；本结果仍不证明 TTS/TFG 收益、Wav2Lip 因果、梯度可用性或部署策略，重复/跳帧与全局视觉运动变化仍可能混杂。

## Observations
- [status] blocked
- [hypothesis] 在自然音频固定时，SyncNet forward score 应能检测真实视频局部帧采样造成的已知 timing mismatch
- [protocol] OpenSpec: `openspec/changes/diagnose-lrs3-real-video-local-timing/`
- [result] 修正版 `20260905_v2` 在固定输入审计阶段阻断；未生成 media、score 或 review cell
- [input_contract] 22 条历史记录中有 3 条自然音频相对 25 fps 真实视频超过 1 帧：`lrs3_73jPh0eRPSY_00008` (+1.2)、`lrs3_6qqqVwM6bMM_00007` (+1.4)、`lrs3_79tRTivyMSM_00014` (+1.4)
- [implementation_note] `20260905_v1` 曾在恰好 1 帧的浮点边界提前阻断；已改为整数采样比较并以新 run 复核，未改变实验契约或科学结论
- [reason] Spec 要求自然音频与视频时长差不超过 1 帧，且禁止裁音频、补音频、替补样本或缩小分母；因此不能继续 materialize 或评分
- [conclusion] 工程状态 BLOCKED，scientific_decision=null；本轮没有检验 SyncNet local timing sensitivity
- [next_step] 若继续，需要另立 OpenSpec 明确容器尾部 padding 与时间契约的处理；在批准前不放宽当前 spec、不做参数搜索或重试
- [report] `runs/lrs3_real_video_local_timing_20260905_v2/final.json`
- [validation] 独立 validator 通过：final artifact valid

## Relations

- followed_by [[LRS3 Wav2Lip 局部时间传递诊断结果]]
- relates_to [[LRS3 局部时间控制校准实验结果]]
