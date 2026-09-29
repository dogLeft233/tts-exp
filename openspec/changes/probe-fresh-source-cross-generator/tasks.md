# Luna 执行卡

本次执行记录由主 agent 持有；`[x]` 表示已执行并有产物，`[blocked]` 表示按 spec 的硬门禁安全停止，不代表科学阴性。

## P：公共输入（一个 Luna）

只读 design.md；写 `fresh_source_inputs` 包及其测试、run/shared/。

- [x] P1 实现历史 source-group 提取/去重与元数据审计；产物：`runs/fresh_source_cross_generator_20260910/history_groups.json`。
- [x] P2/P3/P4 解冻后按冻结历史 source-group 清单得到12正式+2smoke；`run/shared/cohort.json`、`inputs.json`、审计与独立校验均为 GO。
- [x] P5 建立 C 的12组固定 DELAY PCM 与 source-index map；`run/C/delay_inputs.json` 为 READY（3200 samples / +5 frames）。
- [x] P6 保留所有输入 hash、成员顺序和来源绑定；未改写历史 run。

## A：两生成器与自然音轨（一个 Luna）

读design.md+A spec；进入D时再读D spec。写fresh_source_replacement包、测试、run/A与run/D；所有模型视频仍由本agent统一生成。

- [x] A1/A2 实现薄 wrapper、阻塞终态和独立验收；Ditto TRT 工程 smoke 成功（一次库路径修复重试）。
- [x] A3/A4 Wav2Lip 产生 50 个正式 N/C cells + 2 个 smoke，全部完成并经固定 SyncNet 评分；补充 24 个 seed42/43 DELAY 前向并评分。
- [blocked] A5 A 自然重复控制未过：两组 N42 repeat 的 C/D 差异超过 0.01；因此状态 `CONTROL_FAILED`，不宣称 Wav2Lip replacement signal。
- [blocked] A6 Ditto 为只读历史 ingest；当前冻结 PCM 与旧 manifest 不绑定，且缺少 seed42/43/repeat 合约，状态 `BLOCKED_CROSS_GENERATOR`。

## B：真实视觉与盲评（一个 Luna）

读 design.md + specs/fresh-source-visual/spec.md；写 `fresh_source_visual` 包及其测试、run/B/。

- [x] B1/B2 真实 R 轨迹固定校准 12/12，通过 identity/shift/reverse 指标校准；Wav2Lip 生成臂抽取并分析，11/12 组可观测，动态优势未过门槛。
- [x] B3/B3a 创建两套匿名盲评包（24 primary + 4 QC）；独立 validator GO，A 的 DELAY rows 被明确排除在 B 的 N/C 假设外。
- [ ] B4 `human_status=pending`、`visual_verified=false`；需人工提交 ratings 后才可闭合人工分支。
- [blocked] Ditto 视觉 cells 仍全缺失，报告为 missingness，不折算为零分。

## C：时间传递（P结束后启动一个Luna）

读design.md+C spec；写fresh_source_timing包、测试、run/C；不自行占用远端GPU。

- [x] C1/C2 生成并评分 Wav2Lip DELAY 前向；A/D/C 输入与 source-index map 保持固定绑定。
- [blocked] C3 因 A 自然控制 `CONTROL_FAILED` 且 Ditto 当前无 fresh DELAY forward，C 终态为 `CONTROL_FAILED`、0 scientific cells；独立 validator 对该零单元终态 GO，不把失败写成 timing negative。

## 主 agent：汇总与收尾

- [x] M1/M2 并行 Luna 实现完成；阶段审查后修复固定视频流抽取、DELAY manifest 归一化、C 控制失败终态哈希、D gate 与 B validator；当前 23 个相关测试、ruff、py_compile、OpenSpec 均通过。
- [x] M3 本轮仅用本地 V100；未启动新远端服务器，旧 Ditto 服务器保持已关闭状态。
- [x] M4 A/B/C/D 分开写终态；同一 BM 实体跟进，保留 `CONTROL_FAILED`、`BLOCKED_CROSS_GENERATOR`、`DEFERRED_MEASUREMENT` 与 human pending，不修改旧门禁。

协调顺序：P/A/B开发 → P冻结 → A自然控制与B真实校准 → C延迟诊断与A基础候选 → 满足D门槛才构造/生成M → 各路CPU分析 → GPU关闭 → 人工回收与汇总。C科学阳性不是D入口条件；GPU关闭无须等待任何CPU统计或人工评分。
