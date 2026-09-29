## 1. 只读输入审计与冻结

- [x] 1.1 阅读README/design/spec/input-bindings，验证22条队列与三个父清单file hash；核对旧云端M→原始云端TTS→自然reference链。
- [ ] 1.2 按ID+文本+reference+模型查本地产物，输出逐条可复用/需生成/不可接受原因；不得用跨数据集本地音频凑对。
- [x] 1.3 实现config/protocol/audit CLI，冻结实际backend与软件/权重/生成参数、调度表、矩阵、质量问卷及统计规则。

## 2. 配对音频与目标

- [x] 2.1 实现有界本地合成、一次输出、seed隔离、canonical/resume和失败台账；默认零云端新调用。
- [x] 2.2 同一MFA配置处理66个对齐输入，共享N，对双方同法生成44个M；保留phone/mapping/fallback/尾部证据。
- [x] 2.3 实现四臂N/B0/B_LOCAL/B_CLOUD；从PCM重算官方mel并冻结movement与scale诊断、共同几何及analysis lock。

## 3. 独立音质包

- [x] 3.1 导出88个匿名听评刺激、响度因子、blind mapping、问卷、CSV模板及评审分配，保证副本不进入模型链。
- [x] 3.2 实现ratings校验与raw/target配对统计；无人工数据输出NOT_ASSESSED而非伪造结果。人工听評收集是外部待办，不应阻塞自动渲染和评分实现。

## 4. 渲染与评分

- [x] 4.1 实现四臂两次独立渲染，固定face/几何/支持/seed/调度；176个新视频，repeat不允许复制。
- [x] 4.2 实现七cell枚举、N_REV、精确PCM mux及308个官方评分；支持、码流、音轨身份与parity全部有证据。
- [x] 4.3 实现严格resume，禁止按得分重试、动态缩短支持、选择更好track或重复。

## 5. 分析与独立验证

- [x] 5.1 实现r内配对、r平均、22组共同bootstrap、主deltaC、两项绝对gC次要区间、gD/offset/B0与所有控制。
- [x] 5.2 实现质量阶段差、记录×评审bootstrap、探索rho与缺失处理；来源优势、绝对增益和质量解释分别输出。
- [x] 5.3 实现独立validator：从原PCM重做bridge/N_REV，从原日志重算完整矩阵/差值/统计/终态；独立计算关键公式，不复用producer的判定函数。

## 6. 必须先通过的反例测试

- [ ] 6.1 provenance：同名不同reference/文本/模型、云端误标本地、file hash误作PCM hash、跨语言配对、缺失任一侧必须被拒绝。
- [ ] 6.2 audio：alpha0重建、相同target两臂完全相同、交换target只交换对应输出、无穷/静音/错误长度、int16边界、RMS/peak顺序；不得将波形插值实现为幅度谱插值。
- [ ] 6.3 alignment：不同原始时长、自然共享边界、未知speech/静音fallback、零长度phone、重复边界、<320尾部及过大尾部失败。
- [ ] 6.4 matrix：期望176/308、重复视频复制检测、错mux自身音轨/跨sample/不同支持/track/缺失cell、N_REV精确翻转。
- [ ] 6.5 statistics：合成数值验证两种deltaC公式相等；cloud/local标签互换时delta符号翻转；repeat不能增加样本量；一侧显著不能推出差异显著；云端较少退步不能被标正增益；使用未舍入阈值；各终态优先级。
- [ ] 6.6 quality：无ratings、pair缺rater、无效1–5范围、未盲化标签、常量rho、raw优势未传递到target、听评副本误入模型链必须被识别。
- [x] 6.7 跑 `PYTHONPATH=. pytest -q tests/experiments/lrs3_bridge_tts_quality`，以及实际修改公共组件的相关回归；Ruff/compileall检查实现；`openspec validate compare-lrs3-bridge-tts-quality --strict`。

## 7. 下游正式执行与交接

- [ ] 7.1 测试通过后按stage顺序执行固定22条，记录预算/真实推理/失败；无完整输入时如实INPUT_BLOCKED，不调设计补救。
- [ ] 7.2 交付原始日志、逐条paired.csv、控制、各来源C/D/offset、主差和CI、质量包/实际评分、分开的来源与质量状态、独立validation。
- [ ] 7.3 写面向新读者的report.md，明示既有云端历史、已见fit样本、TTS单次输出、质量关联而非因果、未评估项；更新对应Basic Memory planned实体为实际状态。

实现状态（2026-09-13）：1.1、1.3、2.1–2.3、3.1–3.2、4.1–4.3、5.1–5.3 已完成代码实现；Stage 00 输入审计已实际完成，6.7 的自动验收已通过。6.1–6.6 仍保留为反例覆盖清单，7.1–7.3 仍需在磁盘空间允许后执行正式 GPU 流水线并交付新实验数值。编写或实现本 change 不等于正式实验结果已经产生。
