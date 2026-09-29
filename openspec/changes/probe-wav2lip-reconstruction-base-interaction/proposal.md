## Why

masked条件中正确内容有价值，但同一增量加到完整natural无收益。固定增量不调参，只检验该响应是否依赖重建基底及其非线性交互。

## What Changes

- 新增独立小实验包、聚焦反例测试和真正独立的数值验收。
- 冻结候选/对照、输入身份、统计与停止条件，按公共交接资源上限运行。
- 保存同一BM实体及诚实的工程/科学结果；本提案尚未执行。

## Capabilities

### New Capabilities

- `wav2lip-reconstruction-base-interaction`: masked条件中正确内容有价值，但同一增量加到完整natural无收益。固定增量不调参，只检验该响应是否依赖重建基底及其非线性交互。

### Modified Capabilities

无。父run、旧spec与公共worker只读。

## Impact

只涉及 `scripts/experiments/wav2lip_reconstruction_base_interaction/`、同名tests/runs、本change/tasks及BM `Wav2Lip reconstruction base interaction 2026-09-09`。公共入口为 `openspec/parallel-replacement-next-20260909.md`。不训练或下载新模型，不声明泛化。
