## 1. 接续与输入锁定

- [x] 1.1 阅读本change四文件及design指定BM历史；更新同一实验实体为running。
- [x] 1.2 实现只读prepare：固定hash/532文件/133记录/23组/17+6 split，缓存结构检查与protocol锁定；不访问媒体或sealed资产。

## 2. CPU诊断

- [x] 2.1 实现旧exact-time和eligibility复验，复现115/133、21/23及缺组；不运行DTW或landmarker。
- [x] 2.2 实现三臂共同支持、固定资格、d_N/d_M/b及所有缺失原因；保存逐条索引。
- [x] 2.3 计算全部23组及17/6描述性分表的原分母界限，按固定规则给唯一诊断；全程零模型。

## 3. 自审、验收与交付

- [x] 3.1 完成design列出的合成数据回归；实现独立数值validator及不可篡改的父状态边界。
- [x] 3.2 运行all与独立验收，重验父hash；保存self-review、final、result.md。工程失败与科学阴性分开。
- [x] 3.3 更新同一BM实体并读回；报告observed数、L/U、诊断、预算和限制；不自动进入下一实验。
- [x] 3.4 `openspec validate audit-lrs3-visual-teacher-missingness --strict --no-interactive` 与新增包聚焦pytest通过后再勾完成。
