# complete-wav2lip-reference-matched-control

## Why

旧 F46 参考交互在未实现配对延迟域的控制处终止，F46_C 未生成。E0 只证明 P/F0 的配对控制通过，尚未证明 F46 通过；需在正确控制下补完原 2×2 问题。

## What Changes

- 独立重算 F46 配对域控制，并在新 run 完整复验生成链。
- 控制通过后才生成 F46_C；用四 cell 共享自然 anchor 的差中差检验参考依赖。
- 保留原参考帧、候选、样本、阈值和历史阴性。

## Capabilities

### New Capabilities

- `wav2lip-reference-matched-control`: 独立续验/诊断命令、固定科学判定及可重算验收。

### Modified Capabilities

无。旧 change、旧 run 和主流水线保持只读；本实验在独立命名空间实现。

## Impact

仅新增本 change 对应实验包、对应测试与 run；新视频/评分上限 36/54。训练、TTS、下载模型均为0。执行契约见 [并行入口](../../parallel-next-experiments-20260910.md) 和本目录 design.md。
