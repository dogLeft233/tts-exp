## Why

LRS3 的 LOCAL_SWAP 与 LOCAL_WARP_120 连续控制失败，而最近一轮 220 项工程审计和重复性检查通过。需要先回答一个更小的问题：在真实视频上、音频完全不变时，SyncNet 能否识别已知的局部时间错位？这可以缩小失败来源，但不能单独证明 Wav2Lip 有问题。

## What Changes

- 复用历史固定 22 条 fit-only records，新增真实视频原样、独立重复、单一 120 ms 视频局部时间扰动三臂。
- 冻结同一真实人脸轨迹与音频，导出完整 SyncNet 距离矩阵、全局曲线及提前/滞后两段曲线。
- 用已知视频帧映射检验局部 offset 恢复；生成固定样本的盲看包作为辅助证据。
- 提供一个小 runner、离线 validator 和终态报告；判定规则、缺失处理及下一步边界预先固定。

## Capabilities

### New Capabilities

- `lrs3-real-video-local-timing-diagnostic`: 真实视频局部时间扰动的固定数据敏感性诊断。

### Modified Capabilities

无。

## Impact

预计新增 `scripts/experiments/lrs3_real_video_local_timing/` 及对应测试，产物写入独立 `runs/lrs3_real_video_local_timing_<id>/`。复用现有 PCM/hash 与官方 SyncNet 推理函数；历史实验、编号流水线与模型权重保持原状。

## Non-goals

本轮不运行 Wav2Lip、生成 TTS、训练、重测 bridge、搜索扰动强度或访问新 heldout/sealed 数据。只诊断固定真实视频上的前向评分敏感性，不验证梯度、替换收益或泛化。
