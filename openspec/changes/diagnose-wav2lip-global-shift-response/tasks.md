## 1. 准备与最小实现

- [ ] 1.1 读取本文档链与BM；检查固定输入、12个IDs及历史69cells；冻结protocol与预算。
- [ ] 1.2 实现精确±3200采样移位、动态／静态face+box、含face_mode的cell计划，复用已有worker和PCM mux。
- [ ] 1.3 实现历史parity、FULL/I、anchor和时序符号、视觉辅助量及配对bootstrap；补合同测试。
- [ ] 1.4 实现独立validator；自审关键边界并记录真实review方式，运行focused tests/Ruff/OpenSpec strict。

## 2. 执行与阶段核查

- [ ] 2.1 BM更新running；CPU prepare/history完成，确认parity及支持，通过后才检查宿主CUDA和磁盘。
- [ ] 2.2 完成两种face的N/N_REPEAT与同视频错配cells，核对PCM、时间轴和重复性；科学异常如实记录。
- [ ] 2.3 完成固定±200ms生成及own/replacement cells，不扫参数；全量应为96生成视频、192评分cells。
- [ ] 2.4 输出全部对比及视觉响应，独立验收，写final/result；阴性或uninterpretable也正常收口。

## 3. 交付

- [ ] 3.1 BM同笔记更新concluded/blocked，保留changelog并读回；记录数字、实际预算、解释边界及产物路径。
- [ ] 3.2 勾选已完成任务，交付简短结论与validator；停止，不自动运行语义辅助、训练、云部署或关机。
