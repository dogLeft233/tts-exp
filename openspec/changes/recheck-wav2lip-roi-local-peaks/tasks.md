## 1. Prepare

- [x] 1.1 阅读本 change、固定父 spec，以及 BM Startup Router → 实验指令 → 两份父实验全文；按 2–3 个查询变体搜索后创建/更新本轮 Experiment planned 笔记。
- [x] 1.2 审计固定父资产，冻结 10-record/20-cell 协议、样本来源分组、设备、源码和 spec hashes。

## 2. Implement and execute

- [x] 2.1 实现独立 worker、汇总与离线 validator；覆盖距离方向/epsilon/零 padding、峰边界、缺失/错配 cell、篡改结果的有效测试。
- [x] 2.2 通过 focused tests 后完成一次 20-cell 正式重评分，保存 embeddings、距离、逐条局部峰与新旧差异。

## 3. Close

- [x] 3.1 离线 validator valid、focused tests/lint 和 `openspec validate recheck-wav2lip-roi-local-peaks --strict --no-interactive` 通过；检查未写父资产。
- [x] 3.2 生成中文 result.md/final.json，明确历史 CONTROL_FAILED、own-audio 未重测、因果归因边界和一个后续建议。
- [x] 3.3 subagent 更新同一 BM 笔记到真实终态，补充父诊断关联，读回核对目录/状态/数字；报告文件路径、命令、结果、BM permalink，然后结束。
