---
title: 033-complete-pair-consumer-replay-after-parent-link-repair
type: decision
permalink: tts-exp/docs/adr/033-complete-pair-consumer-replay-after-parent-link-repair
status: open
project: tts-exp
decided: '2026-10-10T05:29:01Z'
tags:
- audio-enhancement
- native-execution
- complete-pair-replay
- provenance
---

# 033-complete-pair-consumer-replay-after-parent-link-repair

## 上下文

ADR032的最小Muse13/session05修复与同θ/RNG干净重跑已落实到actual package12/session07/coordinator20。首plus与minus三模型全部完成，每模型每符号8/8 official PASS，两个24行point terminal COMPLETE，六child实际rc0；但bootstrap在正式tell前rc2，coordinator UNKNOWN，事件0–5包含两个intent和两个terminal，没有PAIR_TOLD。旧UNKNOWN不重开。

ROOT逐一实际读回48行的336 NPZ数组、官方矩阵逐元素一致、score/delivery原自然PCM、generator共享PCM、N/clock、完整ledger/receipt链，全部PASS。权威QC为.codex/scratch/root_session07_complete_pair_actual_qc_attempt01/root_actual_qc.json，self4a3e09b4a985a8a0760c8ed7da17e5e7b31119225f4a0a5e09ec2ad400354d96。全12877文件382425736B已做实际哈希inventory，非媒体body67280058B已本地读回；生成视频等原件保留，B盘全文件归档正在执行。未归约新的科学effects。

Dev与ASR独立定位唯一失败conjunct：packet_builder05复制的spsa_execution_parent_plan是准确{path,bytes,sha256}，final-ingest05同样严格接受该3key形状；session05却与带self_sha256的4key _pin_link比较。真实parent path/rawSHA/2584596B完全相同，scope为Mapping且完整proposal身份相同；composed plan的canonical self另已在session输入加载时验证。该失败是consumer契约不一致，完整评分产物可保留。

## 决策

创建immutable session consumer06，仅把此parent预期改为与packetbuilder05/finalconsumer05相同的3key严格契约，继续保留composed raw+self实际重读及全部scope、theta、method、denominator、NPZ、原音时钟校验。绝不修改旧计划/receipt/source/pin，绝不热修旧session或伪造历史子进程rc0。

以同package12创建fresh初始selector receipt与独立session。对相同actual720以canonical torch_cpu诚实重放，校验初始选择、theta、完整PCG64/history和首15字段ask/post-RNG与旧session07逐字段相同。使用仅适配这个完整pair的一次性read-only recovery executor，从旧session07真实两个COMPLETE point及terminal event读取PointOutcome；其renderer入口必须拒绝调用，不重新生成48视频或评分。实际恢复执行前把新consumer源、旧事件/计划/包/NPZ/官方ledger/frame-map原音链及新旧输入关系封存为显式lineage并独立审核。

fresh receipt的durable_session源pin变化必须诚实记录。逐predicate核对旧point引用receipt02与fresh receipt03的关系，不把不同raw/self或source身份静默归一化；只有相同科学输入、选择、theta/RNG，且恢复lineage明确绑定两者，才能继续。如现有durable consumer拒绝该关系，应在新版本中明示范围有限的兼容验证，不能忽略检查或编辑旧point。新session正常ask/run_pending_pair独立写自己的intent/terminal/tell，只有官方完整两点归约成功且正常UPDATED才能从iteration1继续。

后续coordinator应以真实fresh session的已UPDATED首pair作为起点，沿既有API继续19步；明确首pair来自完整只读重放而非新native child，不制造历史launch/process成功receipt。实现优先最小一次性provenance绑定，不扩展通用跨package收养框架。中心另以独立center11 CLI处理：prepare传iteration=source_step、collector要求相同iteration，import最终匹配的session版本；runner08无需改变，当前20pair outer无需嵌入center调用。

## 理由与替代方案

本次两符号三模型已完整、评分相同包/参数/协议且ROOT实际资格通过，问题只在后置消费者。再渲染48条不会修复parent shape，且浪费已完成实验；放弃这批完整结果也不增加科学严谨性。直接改旧文件或在旧UNKNOWN coordinator中冒充rc0会破坏真实执行链。一次性显式完整pair重放保留原始执行事实，同时让修正后的正式消费者重建真实evaluation/tell。

ADR032关于旧不完整、失败point排除仍成立；本决定针对后续全新package12实际完整48行pair，refines其恢复策略，不将此前partial W2L8/Ditto8纳入本次更新。若重放过程中发现真实数组/输入/分母冲突，保留失败并修正版本，绝不降低科学门槛。

## 后果与影响

唯一科学上下文仍2160，完整48行只计一次；历史失败额外物理尝试另列。20步SPSA数学、四中心、同RMS/±3dB/原PCM时钟、FULL/GUARD20/31lags/anchor15、全部Dev/Confirm门槛不变。决定依据工程contract和独立产物资格，不依据局部effects选择参数。当前只完成决定、诊断和ROOT资格；source06/实际重放/首tell尚待实施，不能写成已更新或方法准入。

## Observations

- [decision] 用最小新session consumer与明确lineage的一次性完整pair重放，保留package12已完成48条真实官方结果，再由正常消费者写新的正式tell。
- [rationale] 评分产物、原音轨和scope已独立资格，失败仅为3key/4key parent contract；重复模型调用没有必要。
- [alternative] 不采用修改旧UNKNOWN/session/pins、伪造child成功、忽略初始receipt版本变化或无条件跨package收养。
- [consequence] fresh source与initial receipt诚实封存，逐字段theta/RNG/ask及全48官方输入重读，恢复后按真实UPDATED前缀继续19步；科学设计不变。

## Relations

- refines [[032-clean-native-rerun-after-post-render-contract-failure]]
- relates_to [[027-durable-spsa-journal-and-final-center-selection]]
- relates_to [[003-immutable-execution-repair-provenance]]
- affects [[开发跨 TFG 的自然音频增强头]]
- affects [[跨 TFG 自然音频增强头的训练与原音换轨验证]]