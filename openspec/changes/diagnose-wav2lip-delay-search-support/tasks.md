## 1. 锁定和旧门禁复现

- [x] 1.1 阅读四份文档与指定BM；更新同一诊断笔记为running，写明新run路径。
- [x] 1.2 实现小CLI和输入锁定；验明16条/8组/32 cells、媒体/PCM/embedding身份及真实支持，冻结protocol。
- [x] 1.3 独立复现旧矩阵、13/16、三条异常集及anchor bootstrap；失败正确BLOCKED。阶段自审。

## 2. 固定搜索域诊断

- [x] 2.1 完成合成截断/符号/支持/anchor及边界判定测试，再计算全部16条的配对搜索域。
- [x] 2.2 保存两套lag坐标、逐条新旧offset、越界标志、embedding/曲线误差及峰间距；按固定规则计算两个布尔量和终态，不运行父候选阶段。

## 3. 验收与交付

- [x] 3.1 独立validator从原embeddings复算数值/统计/身份/预算；通过后写validation、self-review、final与result。
- [x] 3.2 聚焦测试、diff检查和OpenSpec strict通过；核对父文件hash不变、新forward/GPU/生成计数均0。
- [x] 3.3 更新并读回同一BM笔记，写真实结果及续跑修订资格；交付旧/新通过数、三条恢复情况、anchor和报告路径。
