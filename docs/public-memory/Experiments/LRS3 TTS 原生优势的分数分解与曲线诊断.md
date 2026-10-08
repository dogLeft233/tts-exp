---
title: LRS3 TTS 原生优势的分数分解与曲线诊断
type: experiment
permalink: tts-exp/experiments/lrs3-tts-原生优势的分数分解与曲线诊断
status: concluded
date: '2026-09-15'
change_id: explain-lrs3-tts-syncnet-gain
tags:
- lrs3
- tts
- mechanism
- syncnet
- openspec
result: review_v15完成并通过独立validator：LRS3历史分解与LeapTalk 12来源曲线诊断均有效；INTERIOR ΔC=+1.252688，曲线控制按预注册规则完成。
conclusion: LRS3英文TTS原生Sync-C优势在历史数据和12来源曲线中均得到复核；INTERIOR ΔC显著为正，而最佳距离项区间跨零，曲线与FULL结果差异区间跨零，因此当前支持TTS优势存在及其主要表现为背景项变化，不能据此宣称声学或感知因果机制。人工感知仍为NOT_ASSESSED。
report: runs/lrs3_tts_gain_mechanism_review_v15/report.md
inputs: data/dataset_samples/video_manifest_250.json；Ditto/LeapTalk历史评分；固定input-bindings.json；LeapTalk
  ID151–162原生媒体
outputs: runs/lrs3_tts_gain_mechanism_review_v15/；CPU分解、24主曲线、4控制、分析图、人工包、report.md、final.json、validation.json
model: 历史Ditto与LeapTalk评分；LeapTalk ID151–162既有原生媒体；review_v15仅新增SyncNet前向
code_paths:
- scripts/experiments/lrs3_tts_gain_mechanism/
- tests/experiments/lrs3_tts_gain_mechanism/test_analysis.py
- openspec/changes/explain-lrs3-tts-syncnet-gain/
---

# LRS3 TTS 原生优势的分数分解与曲线诊断

用户指出英文同样有TTS优势，要求为下游编写快速、正确的机制实验spec。本次重新按multiset原manifest联结LRS3的50对历史评分：Ditto原生ΔSync-C=+1.124（45/50正向）、LeapTalk=+1.389（46/50正向）。旧LibriSpeech/HDTF阴性不能概括英文；CONTEXT.md已纠正。以上为历史record等权均值复算，不是新模型实验。

完整spec位于 `openspec/changes/explain-lrs3-tts-syncnet-gain/README.md`，含proposal/design/tasks、规范与input-bindings.json。新实验先CPU分解两个模型50对评分的C=B−D，再对已有LeapTalk视频固定12来源/24主cell补算完整距离矩阵，另2重复+2延迟控制，最多28个新SyncNet cell，零新TTS/TFG。人工评估包单独交付，无评分时NOT_ASSESSED。

本轮定位计算组成与边界/lag搜索影响，不把代数分解当声学因果归因。最佳距离下降不直接证明真实同步精度，背景上升也不直接证明评分偏差。N/T各自时钟不同，等窗口数量不是音素对齐。Masked TTS的modality正结果使用NAT_ONLY基线，应单列，不能混入raw原生增益。

## Observations

- [status] running
- [result] 已核对历史LRS3 50对：Ditto ΔC=+1.124、D改善+0.008；LeapTalk ΔC=+1.389、D改善+0.474。
- [requirement] manifest的speaker_key全部为lrs3，必须按video_local_path父目录恢复45个source groups；模型内先组均值再bootstrap，点估计与区间保持相同estimand。
- [requirement] 曲线队列按来源首次出现固定ID151–162，24个LeapTalk视频已绑定hash；不得按分数或缓存可用性换样本。
- [requirement] FULL/INTERIOR/EQUAL_COUNT分开，保存[T,31]，先时间均值再min/median；official_offset=15−j。
- [resource] 编写时根盘仅约447MiB空闲；下游先CPU交付，曲线阶段须重新检查峰值空间及GPU lease，资源不足RESOURCE_WAIT，不删除历史数据。
- [validation] OpenSpec strict与81个输入文件/媒体hash、50记录/45组、两模型历史均值、文档链接核验通过；实现已完成，review_v13 的新曲线前向因资源门禁未执行。
- [report] openspec/changes/explain-lrs3-tts-syncnet-gain/README.md

- [result] 最新运行 `runs/lrs3_tts_gain_mechanism_review_v13`：81个输入绑定全部通过；200个历史cell、100个模型内配对、45个source groups完成独立重算，Ditto ΔC=+1.1238、LeapTalk ΔC=+1.38894。
- [result] 45-group等权主估计：Ditto ΔC=+1.126011、LeapTalk ΔC=+1.409633；LeapTalk最佳距离项均值+0.475456、98.75%区间下界+0.154568，背景项均值+0.934178、98.75%区间下界+0.694730；Ditto最佳距离项区间跨零，背景项98.75%区间下界+0.895005。
- [status] 曲线阶段审计24个LeapTalk媒体并因资源门禁返回RESOURCE_WAIT：当前约51MiB可用，估计需约2.23GB（约2.08GiB，含1GiB保留）；GPU为Tesla V100 16GiB且无其他compute进程；未执行新SyncNet前向。
- [validation] review_v13 的独立validator通过200历史cell、48个绑定逐cell JSON交叉核对、12对人工包；人工评分为空，perception=NOT_ASSESSED，曲线状态被准确保留为RESOURCE_WAIT。
- [implementation] 新增专用runner/common/analysis/validate/syncnet_worker，固定FULL/INTERIOR/EQUAL_COUNT、C_5/D0/S/谷宽、28-cell重复/延迟控制和断点/lease逻辑；复审补强了Bonferroni、控制身份、worker媒体、矩阵dtype、PTS/video metadata、GPU中断清理和直接report续跑路径；13个新测试、Ruff、compileall、OpenSpec strict均通过。
- [report] 完整自动报告：`runs/lrs3_tts_gain_mechanism_review_v13/report.md`

## Relations

- follows [[跨数据集 TFG 测评（5×50 multiset）]]
- relates_to [[LRS3 masked TTS TFG confirmation]]
- relates_to [[LRS3 natural-to-TTS bridge confirmation result]]

## Changelog

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 按用户要求纠正英文证据概括，完成机制spec、输入绑定和校验；未启动新实验 | September 15, 2026 | user request |
| 实现并运行CPU分解、资源审计、报告和独立validator；曲线阶段按规范RESOURCE_WAIT，未启动新SyncNet前向 | September 15, 2026 | user request |
| 修复后复审：补强Bonferroni、控制/worker媒体绑定、矩阵dtype、PTS与资源中断清理；review_v13独立validator通过，曲线仍因磁盘RESOURCE_WAIT | September 15, 2026 | user request |
- [result] `review_v15` 完成并通过独立validator：200个历史cell、100个模型内配对、45个source groups、24个主曲线cell、4个控制cell均通过结构校验；自动实验状态complete。
- [result] LeapTalk 12来源的INTERIOR组等权 `ΔC=+1.252688`，95%区间 `[+0.765705,+1.716934]`，98.333%校正区间 `[+0.658135,+1.801447]`；`gain_match=+0.285715`，校正区间 `[-0.342583,+0.811103]`；`interior_minus_full_delta_c=-0.003162`，校正区间 `[-0.286653,+0.392245]`。
- [control] 两个repeat控制均为CONTROL_PASS，矩阵逐元素最大误差为0；两个200ms delay控制均检测到官方offset移动-5且原始最优位置距离上升，状态CONTROL_FAILED表示延迟注入触发了预注册检测规则，不表示运行失败。
- [implementation] 修复SyncNet worker向ffmpeg pipe输出裸PCM时缺少`-f s16le`的问题；补充repeat/delay控制记录的`matrix_shape`元数据。14个专项测试、ruff、compileall、独立validator均通过。
- [resource] 为满足曲线阶段门槛，清理了旧实验临时产物和安全缓存，保留模型权重、原始音视频与确认过的结果；review_v15运行期间GPU保持串行且约5MiB显存占用。
- [report] 有效报告与产物位于 `runs/lrs3_tts_gain_mechanism_review_v15/report.md`、`runs/lrs3_tts_gain_mechanism_review_v15/final.json`、`runs/lrs3_tts_gain_mechanism_review_v15/validation.json`。
| 清理安全缓存后完成review_v15；修复裸PCM pipe格式和控制矩阵形状元数据；24主曲线/4控制通过独立validator，延迟控制按预注册规则触发 | September 15, 2026 | user request |