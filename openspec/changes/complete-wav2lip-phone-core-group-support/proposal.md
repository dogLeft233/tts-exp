# complete-wav2lip-phone-core-group-support

## Why

phone-core 旧实验因两条记录无 U 曝光而停止，源码把原设计的组级曝光条件误作逐记录条件。保留原候选与端点，在独立复核组级支持后补完这个尚未得到评分的实验。

## What Changes

- 修复曝光门禁的统计单位；保留全部16条/8组，逐记录报告零曝光。
- 按原 PHONE_CORE / GENERIC_CORE 公式生成、固定自然音轨评分，并独立复算。
- 所有组支持合格才进入 GPU；不改音素边界、幅度、窗口或主假设。

## Capabilities

### New Capabilities

- `wav2lip-phone-core-group-support`: 独立续验/诊断命令、固定科学判定及可重算验收。

### Modified Capabilities

无。旧 change、旧 run 和主流水线保持只读；本实验在独立命名空间实现。

## Impact

仅新增本 change 对应实验包、对应测试与 run；新视频/评分上限 34/36。训练、TTS、下载模型均为0。执行契约见 [并行入口](../../parallel-next-experiments-20260910.md) 和本目录 design.md。
