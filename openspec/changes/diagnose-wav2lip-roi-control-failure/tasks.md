## 1. 输入与边界

- [x] 1.1 阅读本 change 的 design/spec、父 ROI/timing spec 和 BM 路由/父实验笔记；确认这是 CPU 离线诊断，非重跑控制。
- [x] 1.2 实现新 CLI 与固定输入审计：入口 hashes、合法 fix1 symlink、22 条主键、198 矩阵、media/PCM 配对；冻结本轮协议。

## 2. 独立诊断

- [x] 2.1 独立重建 masks、曲线、全套父门禁和 bootstrap；与旧派生结果逐字段对照，不调用父统计函数。
- [x] 2.2 输出全部 22 条 C 的双段证据与可重叠失败标志；输出 own_C/own_D/median_change 分解和非劣性阈值距离。
- [x] 2.3 实现证据绑定、三种诊断终态、退出码和简短报告；给出一个有依据的后续建议。

## 3. 验证、执行与交付

- [x] 3.1 实现离线 validator 及 spec 要求的合成矩阵/边界/篡改测试；验证“历史派生偏差”与“源输入损坏”两个分支不同。
- [x] 3.2 跑对应 pytest 和静态检查；在新 run 上执行一次完整诊断，运行独立 validator，核对新增生成/评分均为 0。
- [x] 3.3 按 BM 指令记录诊断、纠正父 own-audio 失败理由；交付 result/final/validation 指针、结论和下一步建议。只有实际完成的任务才勾选，不运行 bridge。
