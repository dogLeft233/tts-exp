# A 实施清单（代码、GPU运行与阶段审计完成）

## 1. 冻结与支持

- [x] 1.1 创建独立包与CLI，读取快照分支A并生成protocol；绑定hash、缺输入非零和不调用模型的聚焦测试通过；16条/8组真实表待GPU `prepare`。
- [x] 1.2 实现原core/双候选公式及独立复算；短core、范数、边界与重叠反例测试通过；旧14条逐值复核随全量运行完成。
- [x] 1.3 将曝光按组聚合，保留16条记录并输出组级诊断；同组一条零曝光可保留、整组零曝光停止的反例测试通过；真实支持结果待 `prepare`。

## 2. 受控执行

- [x] 2.1 支持通过时执行公共控制并独立验收；A run `a_20260910_r3` 验证旧13/16、matched16/16、fresh replay/parity PASS，`control_validation.json` PASS。
- [x] 2.2 所有门禁通过时执行固定32候选和分析；A run 保留16条/8组分母，原N音轨逐样本复核通过，最终预算34视频/36评分，三项联合 gain 未达到主门槛。

## 3. 交付

- [x] 3.1 完成独立validate与反例测试；fixed binding、曝光契约、replay/parity 端点校验和重签拒绝均通过；聚焦pytest及本change的OpenSpec strict通过。
- [x] 3.2 已写 `final.json`/`result.md` 并更新唯一 BM 实体；证据 self-hash、34/36 实际预算、两条零曝光记录和下一轮 source-group confirmation 条件均已核对，全部授权保持 false。
