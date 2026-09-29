# Luna 执行任务

U1 与 U2 可由两个 worker 并行实现；U2 用小合成媒体开发，真实审计等 U1 固定 pool。U3 由主 agent 串行集成。不要让多个 worker 同时编辑 protocol.py 或同一产物。

- [x] U1 来源 worker：新增 `fresh_source_inputs/acquire.py` 与对应测试；只写该文件、测试、acquisition/plan.json、source_manifest.json、pool.json 和来源 ledger。完成有界获取、历史排除、稳定排序和安全提取；不改历史扫描规则。缺来源写具体阻塞证据。
- [x] U2 视觉 worker：新增 `fresh_source_inputs/visual_audit.py`、对应测试；只写该文件、测试、视觉数组/预览、clip_audit.jsonl、visual_audit.json。输出实际 MediaPipe 测量与输入审查绑定；需要改公共接口时交给 U3。
- [x] U3 主 agent：最小改动 protocol.py/runner.py/validate.py，处理合法 FAIL、cohort 验收、真实状态文本和完整 execution_contract；校验两个 worker 的接口兼容性。
- [x] U3 测试：历史组换 clip 仍排除；24 组/前 8 clip 的固定顺序；FAIL 后同组下一 clip 可 PASS；跨 clip SHA 借用拒绝；低覆盖 FAIL 不破坏其他 PASS；NaN/伪造模型 SHA 拒绝；审计数字被篡改并重签 hash 仍拒绝；13 组阻塞、14 组稳定切为 12+2；覆盖缺失且 ≥14 组仍为合法 blocked；cohort-ready 不要求 C，但不等于 inputs-frozen；改变输入/代码后 resume 拒绝。测试放现有 `tests/experiments/fresh_source_inputs/`。
- [x] 阶段审查：U1 检查来源/预算/历史；U2 检查真实数组/输入审查/FAIL 语义；U3 检查独立验收/排序/哈希/旧 run 未修改。问题修复后只重跑相关检查。
- [x] 实际执行 U1 → U2 → prepare → cohort validator，保存 COHORT_READY 或具体阻塞原因。实际结果为 92 个历史排除、0 个可用新组，保留 `BLOCKED_SOURCE_ACCESS`，未改门槛或自动超预算。
- [x] 交付 candidate/freeze 命令，更新原 BM 实体；有完整实验执行任务时再接回原 A/B/C/D，并遵循用完关机。不得把此项当作科学实验已经完成。

实施后的检查：

```bash
python -m pytest -q tests/experiments/fresh_source_inputs
openspec validate unblock-fresh-source-inputs --strict --no-interactive
```

只有修改公共 P 接口影响下游时，再跑 fresh_source_replacement / visual / timing / tts_increment 对应测试。

## 2026-09-11 执行更新（r4）

上一轮 `r1` 的 0 新组阻塞仍保留为历史证据；本轮使用新的 LRS3 pretrain 来源重跑，未放宽门槛：

- [x] 获取 24 个新 source groups，历史排除 108 个；固定池保持 24 组。
- [x] MediaPipe 实际审计后接纳 14 组（12 formal + 2 smoke），`acquisition/final.json` 为 `COHORT_READY` / `integrity=GO`。
- [x] candidate 14/14 完成，freeze 状态为 `FROZEN`；独立 inputs validator 为 `status=GO`、`integrity=GO`、`errors=[]`。
- [x] 修复 validator 的 PCM16 身份读取与 pool 上下文解包缺陷；fresh-source 测试 16 passed，ruff 通过；未改变科学输入。
- [ ] GPU 生成器与 SyncNet 仍待运行：当前 AutoDL SSH 端口不可达；服务器启动后继续原 A/B/C 交接，完成后关机。不得把 CPU 阶段当作科学结论。
