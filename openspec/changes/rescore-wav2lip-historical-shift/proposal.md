## Why

历史 23 条 SHIFT_200 的 replacement ΔSync-C=+0.298 可以复算，但共同内部窗口下为 +0.191（95% CI [-0.016,+0.380]），固定基线 offset 的距离收益也未确认。最新 global-shift v4 在另外 12 条记录和不同生成/评分流程中未确认收益。两轮不能直接作配对复现或证伪。

完成一次有限的处理流程复核：保持历史 N/SHIFT 视频和自然音轨不变，只把评分入口换成 v4 的整帧 SyncNet 路径。这样可以估计**相同媒体上的评分流程差异**，随后收口偏移分支。

## What Changes

- 固定历史 23 records / 23 source groups，复用 46 个旧生成视频。
- CPU 新评分 46 cells，另用 v4 固定 2 cells 校准评分器，总上限 48 cell-level forward jobs；零 TFG 生成、零训练。
- 在新旧矩阵共同绝对帧窗口上，配对比较自由 offset 分数、共享自然基线 offset 的距离收益和评分流程差值。
- 输出独立验收、简短报告和 BM 结论；所有科学结果均为回顾性诊断。

## Capabilities

### New Capabilities

- `wav2lip-historical-shift-rescore`: 对固定历史媒体做可溯源的同媒体评分流程复核。

### Modified Capabilities

None.

## Impact

新增小型实验包 `scripts/experiments/wav2lip_historical_shift_rescore/` 及对应测试；复用现有 CPU SyncNetScorer、FFV1 编码和 exact-PCM mux helper。历史 run 和 v4 只读。详细合同见 [design.md](design.md)，执行清单见 [tasks.md](tasks.md)。

## Non-goals

本轮不生成新视频、不重做 face detection/MFA/TTS、不搜索延迟或裁剪、不训练生成头、不解封数据。内容辅助实验另立 spec；本轮任何正分均不自动授权训练，也不证明普遍有效或普遍无效。
