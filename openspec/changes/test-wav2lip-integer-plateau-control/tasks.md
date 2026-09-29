## 1. Freeze and implement

- [x] 1.1 读本change和父媒体/评分spec、BM路由及指定实验全文；更新同一BM为running。
- [x] 1.2 审计父hash/22条资产；冻结A/B协议、P/q/U/Q、全支持、预期cells与环境。任何窗口不足直接BLOCKED。
- [x] 1.3 实现最小runner/analysis/validate及必要测试；只复用纯工具和worker，不进入旧bridge分支。
- [x] 1.4 代码对spec自审，重点检查平移符号、源内容匹配、全31列支持与A→B门禁；运行聚焦测试/Ruff/OpenSpec strict。

## 2. Execute the fixed experiment

- [x] 2.1 CPU完成22个oracle流和66fresh cell，核对22cached cell；输出所有A统计与独立oracle_validation。
- [x] 2.2 若A科学失败，生成完整负终态并跳过GPU；若A通过且验收有效，宿主GPU生成22条G_P并完成44fresh评分。
- [x] 2.3 独立最终验收，检查阶段性计数、文件绑定、PCM/像素、矩阵、统计与唯一终态。

## 3. Close and review

- [x] 3.1 写中文result：关键数字、历史边界、实际算力消耗、所有失败门槛及继续/暂停建议。
- [x] 3.2 阶段自审确认没有把新estimand当旧gate修复，没有把控制通过当replacement收益；记录review结果。
- [x] 3.3 更新并读回同一BM：合法科学负结果记concluded；工程不完整明确blocked；交付run/validation/BM入口后停止。
