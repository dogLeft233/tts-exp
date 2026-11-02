---
title: 024-live-process-resource-capacity
type: decision
permalink: tts-exp/docs/adr/024-live-process-resource-capacity
status: open
project: tts-exp
decided: '2026-10-09T05:25:01Z'
tags:
- codex
- resource-gate
---

# 024-live-process-resource-capacity

## Context

正式V100 Wav2Lip slice在dispatch前发现，resource gate把probe的cgroup membership_path和leaf_path逐字符串固定，而Ubuntu每次SSH可能创建不同session-N.scope。新运行可具有相同GPU、运行库、输入及足够内存，却仅因临时会话名称改变而在row0之前被拒绝；尚无新科学分数。

## Decision

每次实际stage/row gate都重新解析当前运行进程的cgroup层级、有限limit/current/available与host RAM/SHM/output容量，依据同一冻结预算阈值决定能否继续。membership_path/leaf_path/session/PID保存到执行receipt作为实际观察出处，不作为跨SSH会话的稳定科学或runtime身份。

GPU UUID/型号、权重/源码、实际Python/运行库、输出文件系统和计划阈值继续绑定。读取失败、无限/有限limit解析矛盾、层级预算不完整或当前可用量不足都停止该次执行并保存失败。即使probe曾无有限limit，新进程落入有限cgroup也必须按当前层级有效可用量审核，不能沿用旧无限预算。

## Rationale

资源gate用于验证当前实际执行的进程有足够容量。临时session名称没有模型或输入语义，过早固定它会误拒绝合法的多会话恢复。逐行live检查保留容量限制；只读probe的结果不能代替执行时容量。

## Alternatives

继续固定临时路径会误拒绝正确执行。把probe、plan及所有后续任务锁在单SSH会话会增加脆弱编排，仍无法保证未来容量。取消资源gate会失去容量保护。上述方案均不采用。

## Consequences

执行source使用保留旧attempt的新窄修复。科学input identity、40context/240或720计划分母、ADR022目标/排序/阈值、ADR023支持规则不变。每次receipt记录当前实际cgroup路径及有效预算，便于诊断；预算不足或读取失败仍必须停止并记录。240的pilot和实际40音频时长估算只是测量场景，不是硬上界，仍需每stage/row检查。

影响正式W2L overlay中`_w2l_resource_observation`与后续Muse/Ditto容量gates；不允许伪造返回值或通过环境影子覆盖让gate强行成功。

## Observations

- [decision] 资源容量以当前实际进程的cgroup/host/output有效预算逐stage/row检查，临时SSH session路径只作receipt出处。
- [rationale] 稳定模型/runtime/input身份与易变进程资源观察用途不同，同一冻结阈值仍对每次实际运行生效。
- [alternative] 不采用固定session路径、长期单SSH编排或取消资源gate。
- [consequence] 保留旧source，不改科学input identity/完整分母/阈值；预算不足或读取失败保存失败并停止该次执行。

## Relations

- affects [[开发跨 TFG 的自然音频增强头]]
- relates_to [[022-fit-only-cross-tfg-native-score-refinement]]
- relates_to [[023-decoded-video-window-support-for-native-syncnet]]