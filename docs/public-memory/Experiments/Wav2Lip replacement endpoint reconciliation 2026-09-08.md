---
title: Wav2Lip replacement endpoint reconciliation 2026-09-08
type: experiment
permalink: tts-exp/experiments/wav2-lip-replacement-endpoint-reconciliation-2026-09-08
status: concluded
date: '2026-09-08'
report: openspec/changes/reconcile-wav2lip-replacement-endpoints/
hypothesis: 历史微小正向与最新局部负向的差异是否对应评分支持变化，或仍来自音频与生成/裁剪链的混合差异
tags:
- wav2lip
- replacement
- endpoint-reconciliation
- cache-only
- concluded
---

# Wav2Lip replacement endpoint reconciliation 2026-09-08

根据用户要求，在最新 spectral structure replacement 结束后设计一次小型历史对账。最初 23 条 discovery MAG_075 ΔSync-C=+0.078；22 条确认轮约 +0.032、CI 跨零且 CONTROL_FAILED；最新固定 ROI/U 试验 MAG 为 −0.889、ENV 为 −0.042，终态 NO_USEFUL_GAIN_ESTABLISHED。当前谱迁移构造停止，本轮用于解释历史数字口径，不重新确认 gain 或启动训练。

设计时核查发现确认轮采用 constant_full_frame_fallback 生成框和 SyncNet 再检测裁剪，最新轮采用逐帧 ROI。旧 activesd.pckl 保留完整距离矩阵，因此可完全复用缓存；不能把两轮差异直接归因于窗口或 ROI。另须区分旧 audio manifest 名为 pcm_sha256 的容器 hash 与真实 decoded PCM hash。

固定共同 22 records/22 source groups，旧 N/N_REPEAT/BRIDGE_075 共66个矩阵，新 N/N_REPEAT/RT/MAG/ENV 共110个矩阵，音轨均为 N。每个矩阵固定计算 FULL、COMMON_INTERIOR、U，合计528个 endpoint rows；仅 CPU 读取现有音频/缓存，0 GPU、0新媒体、0模型forward。先复现旧 FULL 日志/统计与新 U 统计，再报告配对窗口差异与处理链混合差异。共用 seed=20260908、10000 group-bootstrap draws，95% CI仅描述性；原科学阈值和阴性终态不变。

## Observations
- [status] concluded
- [question] 原报告数字能否复现，且评分支持变化能解释哪些差异、哪些仍来自音频与生成/裁剪链混杂？
- [execution] 按 audit→all→独立 validation 完成 v8；全程 cache-only/CPU，0 新媒体、0 model forward、0 training。
- [audit] 固定 22 records/22 source groups；H=66、S=110、合计 176 个矩阵；每矩阵 FULL/COMMON_INTERIOR/U，合计 528 个 endpoint rows。13 个根入口 hash、keyed join、音频/容器与 PCM、track/媒体绑定均通过。
- [parity] H FULL 日志与历史统计 parity 通过；S U/矩阵重建、父报告比较 parity 通过，error_count=0。S parent 的 Stage A 使用 95% CI、Stage B 使用 97.5% CI，均按其实际 level 复算。
- [result] H BRIDGE_075−N：FULL benefit-C=+0.031431（95% CI [-0.035122, +0.104079]），benefit-D=+0.017023（[-0.033231, +0.068447]）；U 为 C=+0.097278（[-0.015484, +0.227955]）、D=+0.101117（[-0.016543, +0.231689]）。均值为正但区间跨零，不能当作确认。
- [result] S MAG−N：U benefit-C=-0.889497（95% CI [-1.141804, -0.638170]），benefit-D=-0.842897（[-1.112544, -0.589695]）；S ENV−N：C=-0.042199（[-0.087357, +0.003332]）、D=-0.036503（[-0.083226, +0.010413]）。
- [audio] N/N_REPEAT 身份一致；H BRIDGE_075 与 S MAG 的 decoded PCM 保留为不相同的 candidate confound，未调 alpha、未修复候选；容器 hash 与 decoded PCM hash 已明确区分。
- [interpretation] FULL→COMMON_INTERIOR→U 的变化只能说明支持窗口/边界与处理链口径共同变化；算术 decomposition 不是因果贡献。H 与 S 仍同时改变生成框、ROI、编码/时钟、SyncNet crop/frontend 等，不能归因于单一窗口或 ROI 原因。
- [validation] 独立 validator status=valid、error_count=0；focused pytest=5 passed；Ruff、git diff --check、OpenSpec strict 均通过。
- [terminal] engineering_decision=GO；diagnostic_decision=RECONCILED；scientific_decision=NOT_A_CONFIRMATION；next_action=STOP_CURRENT_SPECTRAL_CONSTRUCTION。
- [constraint] training_authorized=false、generalization_established=false、historical_gate_repaired=false；本轮不推出所有 TFG 或所有生成头不可行。
- [implementation] v1–v6 的中间阻塞均为实现/兼容性问题并保留 discrepancy：manifest hash 笔误、S 双行合并、H track/matrix 行数语义、mux 音频绑定、legacy pickle、处理链代表 arm、H/S parent schema 与 97.5% CI；修正后未改变输入数据或门禁。
- [report] openspec/changes/reconcile-wav2lip-replacement-endpoints/；最终产物 runs/wav2lip_replacement_reconciliation_20260908_cache_reconcile_v8/final.json、validation.json、result.md、parity.json、analysis.json。
## Relations

- follows [[Wav2Lip spectral structure replacement 2026-09-08]]
- relates_to [[LRS3 natural-to-TTS bridge confirmation result]]
- relates_to [[Wav2Lip integer plateau control 2026-09-07]]

## Changelog
| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 按用户要求设计纯CPU历史缓存对账OpenSpec；完成输入结构核查与strict校验，登记planned，尚未执行实验 | September 8, 2026 | user |
| 按新 spec 启动缓存对账实现与执行，先完成独立代码/测试，再运行固定 H/S 输入 | September 8, 2026 | user |
| 完成 v8 cache-only 对账、528 endpoint、H/S parity 与独立 validator；工程 GO、诊断 RECONCILED、科学结论 NOT_A_CONFIRMATION，停止当前 spectral construction | September 8, 2026 | user |