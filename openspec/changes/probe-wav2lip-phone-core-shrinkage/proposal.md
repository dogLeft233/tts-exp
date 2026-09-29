## Why

全时域平滑没有已确认收益，但它与保留音素边界、仅约束音素内部变化的问题不同。用已存phone-core边界和同支持等范数平滑对照做一次固定检验。

## What Changes

- 新增独立小实验包、聚焦反例测试和真正独立的数值验收。
- 冻结候选/对照、输入身份、统计与停止条件，按公共交接资源上限运行。
- 保存同一BM实体及诚实的工程/科学结果；本提案尚未执行。

## Capabilities

### New Capabilities

- `wav2lip-phone-core-shrinkage`: 全时域平滑没有已确认收益，但它与保留音素边界、仅约束音素内部变化的问题不同。用已存phone-core边界和同支持等范数平滑对照做一次固定检验。

### Modified Capabilities

无。父run、旧spec与公共worker只读。

## Impact

只涉及 `scripts/experiments/wav2lip_phone_core_shrinkage/`、同名tests/runs、本change/tasks及BM `Wav2Lip phone core shrinkage probe 2026-09-09`。公共入口为 `openspec/parallel-replacement-next-20260909.md`。不训练或下载新模型，不声明泛化。
