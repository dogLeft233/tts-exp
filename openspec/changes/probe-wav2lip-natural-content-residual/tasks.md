## 1. 准备与锁定

- [x] 1.1 阅读四份spec与BM指定历史；更新同一BM实验为running，记录新run路径。
- [x] 1.2 锁定16 records/8 groups、144个缓存driver、三checkpoint同模型来源、父mask/自然mel/PCM/boxes及hash；不读取sealed数据。
- [x] 1.3 实现四臂、seed均值、K、排列/范数/clip审计与61440样本mel parity；锁定93帧、88行、U及真实支持。
- [x] 1.4 完成构造、坐标、统计符号、状态判定和resume/tamper关键测试；阶段自审。

## 2. 控制门禁

- [x] 2.1 实现小direct-mel worker；核对宿主CUDA kernel、模型hash、静态face来源、lossless编码和完整PCM。
- [x] 2.2 定位并重跑2个固定scorer parity cell；生成16个N与2个独立N_REPEAT，核对重复一致。
- [x] 2.3 完成16个同视频A_DELAY评分；按固定U/anchor/offset判定敏感性。
- [x] 2.4 运行独立control validator；写Stage A科学/工程状态和BM，失败时正确结束。

## 3. 候选与结果

- [ ] 3.1 阻塞：Stage A offset 敏感性仅 13/16 通过，按门禁未生成48个候选视频。
- [ ] 3.2 阻塞：Stage B 未授权，未计算候选相对N/对照的科学收益。
- [x] 3.3 独立验证 Stage A 实际产物、计数、数值和终态；生成 result/final/review，明确短支持与探索边界。
- [x] 3.4 聚焦测试及 OpenSpec strict 通过；BM 同一笔记已更新真实结果并读回。
