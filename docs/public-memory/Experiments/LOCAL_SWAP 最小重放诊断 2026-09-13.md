---
title: LOCAL_SWAP 最小重放诊断 2026-09-13
type: experiment
permalink: tts-exp/experiments/local-swap-最小重放诊断-2026-09-13
status: concluded
date: September 13, 2026
openspec: diagnose-local-swap-minimal-replay
tags:
- local-swap
- syncnet
- replacement
- diagnostic
---

# LOCAL_SWAP 最小重放诊断 2026-09-13

## Context

本实验承接 [[LRS3 natural-to-TTS bridge confirmation result]] 中 22 条记录的 LOCAL_SWAP 反向评分现象，目标是区分历史数据问题、SyncNet endpoint 敏感性问题，以及生成视频没有充分传递驱动时序的问题。按 OpenSpec `diagnose-local-swap-minimal-replay` 执行；不生成新的 TTS/Wav2Lip/Ditto 视频，不训练，不重跑 bridge。

## Protocol

- 历史审计覆盖全部 22 条记录、44 份 SyncNet 日志、PCM 排列、mux 视频流、音频绑定和 manifest hash。
- 新评分固定 cohort 前三条：`lrs3_6WeS1bXRBOk_00006`、`lrs3_6ul2TSvUDog_00007`、`lrs3_6wk4dkYSrV0_00006`。
- A：旧生成视频 G 配历史 natural N / LOCAL_SWAP S，2 cells/样本，官方 pipeline fresh replay。
- B：真实视频固定一次官方 S3FD crop，构造 `R/N0`、`R/S0`、`Rs/N0`、`Rs/S0`；`Rs` 与 `S0` 同时按 A-C-B-D 排列，k 分别为 37、113、45。
- C：使用 A 的 `G/N` 官方 pipeline 实际选中的首条 crop，固定同一组帧和绝对时间支持，仅切旧 N/S 的准确音频片段。
- B/C 走官方 SyncNet forward，保存每个 cell 的完整距离矩阵、全局 C/D/offset、共同支持和局部 offset 曲线；局部 D0 剔除边界及感受野跨段窗口。

## Results

- 历史记录：22/22 通过；对同一 V_LOCAL_SWAP，mean[C(V_LOCAL_SWAP,N)−C(V_LOCAL_SWAP,S)]=+2.789，mean[D(V_LOCAL_SWAP,S)−D(V_LOCAL_SWAP,N)]=+2.994，22/22 同时 natural 更好。
- A：3/3 方向复现（G/N 的 C 更高、D 更低），C/D 数值均在 ±0.100、公差 offset ±1 帧内，状态 `REPRODUCED_3_OF_3` / `AGREE_3_OF_3`。
- B：3/3 样本的两个中间段均能区分 matched 与 wrong 配对，状态 `KNOWN_PAIRING_SENSITIVE`。两段的 `D0(wrong)-D0(matched)` 均为正：第一条为 6.176/5.802，第二条为 7.171/7.071，第三条为 10.666/6.500（顺序为 original/swapped；每条两个段共享同一有效窗口集合）。
- C：固定生成 crop 仍偏向 natural，`D0(Sc)-D0(Nc)` 分别为 +3.155、+1.471、+2.771，状态 `NATURAL`。
- 工程状态：`COMPLETE`；24/24 新 cell 完成，历史审计、距离矩阵 parity、PTS、输入 hash 和独立 validation 全通过；人工观看状态 `NOT_HUMAN_REVIEWED`。

## Conclusion

当前最小诊断支持：SyncNet 能识别已知真实音视频配对/错配，但同一份生成视频 crop 在固定输入下仍偏好 natural 音频。因此支持“生成视频中的驱动时序没有充分转移到 TTS/LOCAL_SWAP 音频”的解释，定位字段为 `SUPPORTS_INSUFFICIENT_DRIVER_TRANSFER`。这不是对 mouth leakage、full-frame box 或任何单一生成器缺陷的唯一归因，也不是对全部 22 条重新前向、全部 SyncNet 或 bridge 效应的总体证明；不授权 audio head 训练或部署。

## Artifacts

- 最终 run：`runs/local_swap_minimal_replay_20260913_v6/`
- 报告：`runs/local_swap_minimal_replay_20260913_v6/report.md`
- 机器状态：`runs/local_swap_minimal_replay_20260913_v6/final.json`
- 24 行分数：`runs/local_swap_minimal_replay_20260913_v6/scores.csv`
- 播放页：`runs/local_swap_minimal_replay_20260913_v6/playback/index.html`
- OpenSpec：`openspec/changes/diagnose-local-swap-minimal-replay/`；9/9 tasks complete，strict validation 通过。
- 相关实现：`scripts/experiments/local_swap_minimal_replay/`；相关回归测试 63 passed，focused tests 9 passed。

## Observations

- [status] concluded
- [result] 历史 22/22 审计可信，A 子集 3/3 方向和数值重放成功。
- [result] B 的真实已知配对控制通过，说明本轮 SyncNet endpoint 对该局部交换扰动有敏感性。
- [result] C 在锁定 A/G_N crop 后仍偏 natural；这是当前最强的驱动转移不足证据，但仅是三样本定位诊断。
- [limitation] 人工播放尚未完成；自动分数不能代替观看记录。
- [pitfall] B 两个中间段的局部值必须分段查看；合并两个段后可能因 A-C-B-D 对称排列出现数值抵消/交换，不能只看合并平均。
- [decision] 保持 bridge/control 的保守边界，不进入训练、audio head 或 deployment。

- [correction] September 13 复核发现 historical_audit.json 的 c_local_swap_minus_natural_mean / d_natural_minus_local_swap_mean 字段名与实际分差方向相反；数值来自正确原始分数，本文改用显式公式，不按字段名解释符号。第一条 C(N)=5.246、C(S)=2.086、D(N)=8.108、D(S)=11.421；无原始数据变更。
- [next] [[静态图片 Natural-to-TTS bridge 实验]] 已交付 spec，尚未运行；去掉动态参考后测试同一 B 波形，但仍不唯一定位旧生成器原因。

## Relations

- implements [[diagnose-local-swap-minimal-replay]]
- relates_to [[LRS3 natural-to-TTS bridge confirmation result]]
- relates_to [[LRS3 真实视频局部时间敏感性诊断]]
- follows [[LRS3 Wav2Lip 局部时间传递诊断结果]]
- motivates [[静态图片 Natural-to-TTS bridge 实验]]

## Changelog

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 保存结果解释边界，纠正 D 分差文字方向并注明产物字段命名问题，链接静态 bridge spec；原数值保留 | September 13, 2026 | user |
