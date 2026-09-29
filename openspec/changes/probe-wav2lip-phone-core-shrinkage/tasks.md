## 1. 准备与实现

- [x] 完整读取公共交接、本change和指定历史；BM先Router/实验指令，更新唯一实体running。
- [x] 核对冻结父资产、间接hash和精确ID连接；用CodeGraph定位接口，实现本包prepare及只读analyze。
- [x] 实现design的原数组计算、有限状态与独立validator；加入手算和篡改反例测试。

## 2. 执行与交付

- [x] 冻结输入/spec/代码与protocol；先通过本包聚焦pytest和阶段自审。
- [x] 按公共独立control验收通过后排GPU锁运行固定候选；控制失败不运行候选。
- [x] 从原数组第二次独立核算全部端点、组统计、hash与终态，生成validation/review后再写final及result.md。
- [x] 运行 `openspec validate probe-wav2lip-phone-core-shrinkage --strict --no-interactive` 与本包测试；BM写真实结果、当前单一status、Changelog并读回。
- [x] 交付工程/科学状态、所有contrast/诊断、实际预算和下一步边界；不自动训练或追加实验。
