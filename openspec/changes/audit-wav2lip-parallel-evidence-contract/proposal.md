## Why

最新A/B/C的validator没有独立重算主要统计；B还漏实现了spec已要求的已知延迟配对搜索域。先以缓存审计恢复可信证据，不重跑模型。

## What Changes

- 新增独立小实验包、聚焦反例测试和真正独立的数值验收。
- 冻结候选/对照、输入身份、统计与停止条件，按公共交接资源上限运行。
- 保存同一BM实体及诚实的工程/科学结果；本提案尚未执行。

## Capabilities

### New Capabilities

- `wav2lip-parallel-evidence-contract`: 最新A/B/C的validator没有独立重算主要统计；B还漏实现了spec已要求的已知延迟配对搜索域。先以缓存审计恢复可信证据，不重跑模型。

### Modified Capabilities

无。父run、旧spec与公共worker只读。

## Impact

只涉及 `scripts/experiments/wav2lip_parallel_evidence_contract/`、同名tests/runs、本change/tasks及BM `Wav2Lip parallel evidence contract audit 2026-09-09`。公共入口为 `openspec/parallel-replacement-next-20260909.md`。不训练或下载新模型，不声明泛化。
