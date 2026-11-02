---
title: 003-immutable-execution-repair-provenance
type: decision
permalink: tts-exp/docs/adr/003-immutable-execution-repair-provenance
status: open
project: tts-exp
decided: '2026-10-01 18:24:51.843318+00:00'
tags:
- reproducibility
- execution
- provenance
git_sha: a4ddb5755146ed6a3572b821ce44b0ccbf51a7c7
---

# 003-immutable-execution-repair-provenance

## 决策

已生成实验产物的 plan、freeze 和科学输入封存保持原样。核验代码的执行修复使用独立的 root 授权文件，绑定原计划/封存的 raw 与 payload、原核验源码和修复源码的路径及 SHA，并声明修复范围。生产、独立核验和 GPU forward 各自通过自己的执行门。

## 背景与理由

800 个音素平均谱形×内部轨迹干预波形已完整生成。独立 QA03 在读取不存在的 plan 顶层 phone_inventory_payload_sha256 时失败；实际字段位于 input_bindings.phone_inventory.payload_sha256。直接改变已执行计划会破坏波形产物记录的来源关系。因此 QA04 只修复该字段合同，并通过独立授权验证新源码，同时保留原计划绑定的 QA03 身份。数值重建、音素集合、时钟、阈值和完整样本分母均保持原科学设计。

raw 文件哈希与 JSON payload 指纹分别记录。新 shape manifest 与其 parent WORLD manifest 分别绑定：QA input_report 指向新 shape，bound_world_manifest 指向原 parent。原 WORLD execution_plan 的 raw 文件哈希与 code_seal 中的 execution_plan_sha256 payload 指纹同样分开验证。

## 已考虑的方案与后果

- 改写已执行 plan 或 producer manifest：会改变实际生产时的来源证据。
- 仅跳过 QA 自身源码检查：会失去原计划到新核验代码的明确授权链。
- 当前方案增加一份小的授权文件和严格源码/字段核验；完整 800 波形保留，修复前失败记录保留，后续消费者和归档均需验证授权链。

这份决策记录执行规则。具体 QA04 的运行结果在实验笔记和完整 run 报告中更新，不从代码修复推断科学效应。

## 来源

- `openspec/changes/factor-world-phone-shape-mean-dynamics-20261001/execution-repairs/`：原 source04 plan/freeze 的逐字节副本及独立 QA 修复授权。
- `scripts/experiments/world_phone_shape_mean_dynamic_eval_independent_qa_20261001_attempt04.py`：原17科学角色、原QA03来源与新增独立授权检查。
- `.codex/scratch/world_shape_qa_execution_repair_source04_root_attempt01/root_authorization_receipt.json`：机器生成的 raw/payload 封存，不使用手抄哈希作为操作输入。
- 本地未推送 commit `ccc3c4ac44683015647f8c3109d7bfc743f7eb31`；六份相关 Git blob 与执行文件字节一致已核验。

## GPU 基线维度合同

GPU11通过包模块调用实际走到桥接检查后发现：Wav2Lip潜变量z为512维，SyncNet视觉embedding V为1024维，旧runner误将V要求为512维。只读核验全部160context、320组RT/SP数组逐文件匹配原all-files SHA，所有数组float32、有限、z帧数=V帧数+5，0失败。请求与完整response位于 .codex/scratch/world_phone_shape_mean_dynamic_gpu_attempt11/bridge_diagnostic_attempt02/；实际response SHA1c17187376758510986f1ec2810e7cc6845a2ba9257b8575a10a7f4763b0acae。修复使用fresh CODE/JOB/RUN，不改已归档的08源树，更新实际runner/wrapper身份并保留11执行源码与失败输出；此时仍无GPU forward授权。维度合同修复不改变音素、阈值、样本、波形或统计计划。

## Observations

- [decision] 核验执行修复通过独立授权绑定原计划与新源码，保持已生成产物的来源不变。
- [rationale] 将源码修复与科学输入变化分开记录，使完整波形重建和后续评分可准确复现。
- [consequence] 后续 GPU/独立消费者/归档校验纳入授权 raw/payload 及原、新 QA 源码链。

## Relations

- affects [[音素平均谱形与内部轨迹的因果分解]]
- affects [[解释 TTS 的 TFG 与 SyncNet 优势]]