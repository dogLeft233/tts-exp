## 1. Freeze

- [ ] 1.1 阅读本 change 与继承 specs、BM Router/实验指令/父实验；搜索后创建或更新本轮 planned 实验笔记。
- [ ] 1.2 审计父 22 条资产与 hashes，冻结 q、共同掩码、88-cell 矩阵、CPU 环境、代码与阈值；只读检查不得写成实验结果。

## 2. Implement

- [ ] 2.1 实现小型 runner：V_ID/V_ORACLE 无损构造与逐像素验证，N/W 严格 mux，复用已复核 scorer 完成四-cell 路由。
- [ ] 2.2 实现统计、终态及独立 validator；覆盖 spec 所列映射/身份/门槛/篡改测试。不要复制旧 10 条选样或固定 N 音轨设置。
- [ ] 2.3 focused pytest、ruff 与 OpenSpec strict validation 通过后，冻结正式代码再执行。

## 3. Execute and hand off

- [ ] 3.1 在新 run 完成 44 个视频流 / 88 个 mux / 88 个新评分 cell；工程失败如实写 BLOCKED，修复后新 run；科学失败不重复搜索。
- [ ] 3.2 独立运行 validator，输出中文 result/final，报告所有 gate、历史 8 条失败的描述性结果及解释边界。
- [ ] 3.3 更新同一 BM 笔记到真实终态并读回；交付 run 路径、命令、validation、关键数字、BM permalink。完成即停止，由上游审阅，不自动进入 bridge/训练。
