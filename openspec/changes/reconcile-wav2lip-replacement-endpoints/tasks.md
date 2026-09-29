## 1. 输入与小实现

- [x] 1.1 按 design 阅读 BM 与父 manifest；将同一 BM 笔记置 running，确认本轮只读缓存、0 forward。
- [x] 1.2 实现小包及 audit CLI：13 个入口 hash、22 条 keyed join、176 矩阵/track 绑定、固定 FULL/I/U 支持；输出 protocol 与 input_audit。
- [x] 1.3 完成 PCM/生成/裁剪链差异表；区分容器与 PCM hash，保存 candidate LSB 差异，未知历史信息明确标记。

## 2. 重分析与验收

- [x] 2.1 实现历史 H FULL 与 S U parity；失败产出 discrepancy 并停止，不重新评分。
- [x] 2.2 实现 528 个 endpoint rows、固定对比、共用 group bootstrap 和逐记录算术分解。
- [x] 2.3 实现独立 validator：S embeddings→matrix、H cache/log 一致性、统计/分解/来源/终态；覆盖 design 列出的针对性测试。
- [x] 2.4 运行前完成并保存代码/spec 绑定自审；focused pytest、Ruff 与 OpenSpec strict 通过。

## 3. 执行与交付

- [x] 3.1 新 run-id 执行 audit→all；如遇工程阻塞，交付实际 discrepancy，不将未完成任务勾选或增加实验。
- [x] 3.2 完成独立验收与 result.md：解释原数字、窗口差异、共同支持差异及仍混杂的原因，保留父终态。
- [x] 3.3 BM 同一笔记更新 concluded 或 blocked、回读；交付 final/validation/result 路径与本轮停止边界。
