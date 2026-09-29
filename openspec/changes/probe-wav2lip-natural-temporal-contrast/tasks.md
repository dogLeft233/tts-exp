## 1. 实现与锁定

- [x] 阅读公共契约、本change及指定历史BM；同一BM实体更新running。
- [x] 实现独立小包和合成测试，冻结输入/spec/代码hash；核对16条/8组。
- [x] 实现对称5点算子、双方向同幅度、端点与clip审计；先通过单测。

## 2. 执行与验收

- [x] 复现父控制，持GPU锁做2个replay和2个parity评分；独立control验收。
- [x] 按冻结协议生成32个候选并评分，保存完整PCM/PTS/矩阵证据。
- [x] 独立重算双contrast、99%CI和唯一终态，保存self-review。
- [x] 跑本包pytest及 `openspec validate probe-wav2lip-natural-temporal-contrast --strict --no-interactive`。
- [x] 更新并读回本BM实体，报告全部效应和真实预算；停止，不扩实验。
