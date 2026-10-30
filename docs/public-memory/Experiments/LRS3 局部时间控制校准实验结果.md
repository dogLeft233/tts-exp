---
title: LRS3 局部时间控制校准实验结果
type: experiment
permalink: tts-exp/experiments/lrs3-局部时间控制校准实验结果
status: concluded
tags:
- lrs3
- syncnet
- control-calibration
- negative-result
---

# LRS3 局部时间控制校准实验结果

## Context

本实验按 OpenSpec `calibrate-lrs3-local-timing-control`，在上一轮 LRS3 natural-to-TTS bridge confirmation 的固定 22-record / 22-source-group fit-only cohort 上，校准 Wav2Lip + SyncNet 是否能稳定、有效地响应局部音视频时间错配。该 cohort 已看过历史结果，因此本实验不是独立确认，也不重写历史 bridge 结论。

## Protocol and execution

- Run root: `runs/lrs3_local_timing_control_calibration_20260905_v5/`
- 历史 final SHA-256: `df0ca9767e70ccc384c86c1da23243c53fa609be12abd1dda20f9075b6732c6e`
- 审计：220/220 项通过，`audit_decision=NO_DEFECT_FOUND`，新评分读取数为 0。
- 唯一锁定分支：`SMOOTH_WARP`；控制臂：`LOCAL_WARP_120`，固定幅度 1920 samples（120 ms）。
- 新产物：66 个独立视频、88 个四-cell SyncNet 评分。
- 统计：按排序 source groups 做 10,000 次 cluster bootstrap，seed `20260904`，NumPy `default_rng/PCG64`。

## Results

- 重复性 gate 通过：`N_REPEAT-N` 的 Sync-C / Sync-D gap 均值为 0，95% CI 均为 [0, 0]，offset agreement 为 22/22。
- 控制自身有效性失败：`C(O)-C(B)` 均值 -3.230，95% CI [-3.666, -2.821]；`D(B)-D(O)` 均值 -3.020，95% CI [-3.467, -2.618]；offset agreement 仅 8/22。
- 替换敏感性失败：`damage_C` 均值 -3.040，95% CI [-3.421, -2.690]；`damage_D` 均值 -2.876，95% CI [-3.261, -2.534]；两个 damage 同时为正为 0/22。
- 终态：`engineering_decision=GO`，`scientific_decision=CONTROL_FAILED`，`reference_conditioned_audio_head_spec_eligible=false`。
- 独立 validator 为 `valid`；36 个相关测试、Ruff、compileall 和 OpenSpec strict validation 均通过。

## Conclusion and boundary

本轮没有发现数据、缓存、mux、Wav2Lip 或 SyncNet 评分链的工程缺陷；流程重复性可靠。但 `LOCAL_WARP_120` 的自身配对有效性和替换敏感性均未通过，且自然音频替换后的指标反而更好。因此不能把该控制解释为已验证的局部 timing sensitivity，也不能据此支持 TTS/bridge 的唇形同步收益。

按协议停止在控制校准结论，不切换到 `REPAIR_ONLY`、不搜索更大/更小扰动、不重测 bridge、不训练或部署。后续若仍需研究局部 timing 控制，应另建 OpenSpec。

## Observations
- [status] concluded

- [decision] 在审计无缺陷后只执行预注册的 `SMOOTH_WARP/LOCAL_WARP_120` 分支，未根据科学结果切换或重试 #protocol
- [result] 重复性通过，但控制自身有效性和替换敏感性同时失败 #syncnet
- [problem] 当前 endpoint 对该平滑局部时间扰动的响应不满足控制校准要求，不能支持 timing-sensitive 的科学解释 #control-calibration
- [insight] “流程稳定”与“控制有效且可检测”是两个独立条件；前者通过不等于后者通过 #methodology

## Relations

- relates_to [[LRS3 natural-to-TTS bridge confirmation result]]
- relates_to [[局部 timing 与 acoustic mismatch 分阶段诊断]]
- implements [[calibrate-lrs3-local-timing-control]]