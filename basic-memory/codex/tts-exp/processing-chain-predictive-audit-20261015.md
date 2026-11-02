---
title: processing-chain-predictive-audit-20261015
type: note
permalink: tts-exp/codex/tts-exp/processing-chain-predictive-audit-20261015
status: done
project: tts-exp
assigned_to: ROOT theory/BM; original GPT-6 Luna/max A/B/C
started: '2026-10-15T00:54:44.769473+00:00'
current_step: 6
execution_phase: completed
completed: 2026-10-15 03:26:24.812294+00:00
run_root: runs/processing_chain_predictive_audit_20261015_attempt01
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
---

# processing-chain-predictive-audit-20261015

## Description

全自动完成现有自然/TTS条件的处理链可辨识性与SyncNet配对分差预测审计，交付可复现、有适用边界的中文结论与proposal依据。主问题L1：七类处理链/声学/独立时间因素能否在来源分组样本外辨识N/T？L2：这些因素能否增量预测同一评分合同下的T−N Sync-C？阳性/零/负/共线不可辨识/UNAVAILABLE均允许。

## Current State

COMPLETED。2026-10-15按用户指出的LibriSpeech数据泄漏修正证据资格：排除其主结论/proposal支持/英语效果/机制/独立验证/跨域泛化用途，原计算仅处理链诊断存档；具体泄漏组件与样本本轮未核验，未重跑实验。正式报告 [[43-processing-chain-predictive-audit]] 已交付，研究地图与编号索引已更新。主/legacy所有可执行分析及独立QA均结束；LibriSpeech计算归档保留但独立证据资格已排除。source/coverage、固定protocol、CLI、features/scores/folds/OOF/stats/figures、原曲线端点、proposal段落和实际复现命令都在run/delivery_manifest_final_v2.json及正式报告。

主74对/13组generator：acoustic AUC0.971692，joint0.971363；分差七族R²−0.139984/MAE0.849871，均值−0.006278/0.742632。Legacy同74/13 scorer：jointAUC0.967164；R²+0.095474/MAE0.776358，relative+0.175059/0.718975；均值−0.034001/0.836558，blocknull p0.010，七族无校正正增量。LibriSpeech13/13 original为已排除诊断存档；旧AUC只保留原表，L2 source-score连接不可用。未做跨TFG/兼容LODO。

独立QA全部PASS：primary f4ad67b8…21checks；LibriSpeech诊断计算 f4d88e3c…18（非独立性确认）；legacy L1 4eae541f…16；legacy L2 39aaca90…10。最后worker2620235/ticks51876849 rc0/13.328秒，legacy_l2_patch_attempt01/readback_attempt01为最终合并产物；L1 JSON/OOF/foldmap逐值/字节复用。原72小contract/0L2fit和所有失败receipt保留。主workerOS wait rc仍null，不为了补字段重跑。

coverage_final_v2.json/.csv是最终覆盖。codec47组未知；multiset/Leap缺完整TTS来源；Report15的10自然原件hash合格但无配对链。缺失问题限定UNAVAILABLE/UNQUALIFIED，不继续恢复部署。统计与特征worker分别52.286/64.358秒；端到端和读取范围由交付manifest/resource侧记录。

跨TFG增强头研究继续paused，未执行finite6/旧GPU训练/Dev/Confirm/新TTS/ASR/TFG/SyncNet前向；其他HANDOFF/automation保持。不再有本任务实验待启动，后续研究须新的用户指示。

## Frozen Scope

只整理已有数据、提取物理特征和CPU统计分析。禁止新TTS/TFG/SyncNet前向、增强头/AV对齐训练、第三层干预及GPU占用。按明确manifest选择性读取local/LAN/B盘，不递归下载runs或全盘网盘扫描，不部署新环境。已压缩音轨不当原驱动WAV。

原始/生成驱动/成片音轨/实际scorer输入分层；native/统一PCM/replacement合同分开。同源自然音、其两臂/版本/模型派生共折；L1去重复增样。路径/ID/hash/condition不进特征，bestoffset/B/D不得泄漏进C预测；windowcount单列评价协议诊断。主logistic/ridge，唯一冻结浅非线性敏感性，训练折预处理和分组验证，族消融/组CI/置换。完整定义已在protocol_v1冻结，后续source扩展仅按实际资格另绑定；不把历史队列称未知预注册。

## Deliverables

- 紧凑protocol、来源覆盖/缺失表和“已有结论—可复用产物—本次缺口”表。
- 可复用CLI、features/paired scores/folds/OOF/stats JSON及少量分布/分类/分差/消融图。
- 独立QA、复现命令、中文正式报告/proposal段落与“论文方法—本项目实现—已有证据—仍缺证据”表。

## Observations

- [decision] 论文真假分类/去静音干预与本项目N/T原生配对评分不同；L1/L2预测关联不等于SyncNet依赖或因果解释比例。
- [decision] 已有编码/高频/电平/时钟干预只用于解释/复核，不重复；高AUC/R2/显著性不作结束门。
- [status] done；全部可执行分析、QA与正式报告已交付。
- [next_step] 按正式报告整理proposal；增强头保持暂停，不自动开启新的实验。

## Relations

- completed_by [[43-processing-chain-predictive-audit]]
- relates_to [[研究交接总览]]
- relates_to [[SyncNet 编码偏差的码率频谱与表征机制拆解 2026-09-26]]
- relates_to [[统一无损音轨后的跨模型剩余优势与自然音频替换审计 2026-09-26]]
- relates_to [[评分查询数量匹配的电平残余诊断 2026-09-27]]
- relates_to [[15-tfg-link]]