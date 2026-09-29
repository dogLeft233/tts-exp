## Why

历史响度机制未确立且曾受Ditto重复噪声混杂。用确定性Wav2Lip、固定自然音轨和可精确实现的波形标量，测试驱动前端的增益响应。

## What Changes

- 新增独立小实验包、聚焦反例测试和真正独立的数值验收。
- 冻结候选/对照、输入身份、统计与停止条件，按公共交接资源上限运行。
- 保存同一BM实体及诚实的工程/科学结果；本提案尚未执行。

## Capabilities

### New Capabilities

- `wav2lip-waveform-gain`: 历史响度机制未确立且曾受Ditto重复噪声混杂。用确定性Wav2Lip、固定自然音轨和可精确实现的波形标量，测试驱动前端的增益响应。

### Modified Capabilities

无。父run、旧spec与公共worker只读。

## Impact

只涉及 `scripts/experiments/wav2lip_waveform_gain/`、同名tests/runs、本change/tasks及BM `Wav2Lip waveform gain probe 2026-09-09`。公共入口为 `openspec/parallel-replacement-next-20260909.md`。不训练或下载新模型，不声明泛化。
