---
title: 019-candidate-manifest-repair-and-execution-binding
type: decision
permalink: tts-exp/docs/adr/019-candidate-manifest-repair-and-execution-binding
branch: codex/cross-tfg-audio-head-20261003
git_sha: cbd47e2cd49d38653fa1f41db56c5afd4cf9890f
status: open
project: tts-exp
repository: dogLeft233/tts-exp
decided: '2026-10-08T12:45:38.9942935+00:00'
---

# 019-candidate-manifest-repair-and-execution-binding

## 决策

候选计划的元数据缺字段时，保留原回执，增加不同文件名的新回执并记录完整来源差异。冻结计划、行、方法、权重、配置和运行绑定不变。本次candidate manifest12缺status，新manifest13只增加status=FROZEN并重算manifest_sha256；原2491B文件仍留在原位。所有执行消费者必须明确使用新2513B回执的实际路径、原始SHA和自哈希。

冻结candidate13的metadata-only输出没有plan_manifest_raw_sha256字段。预检准入使用已审查调用参数与请求原始SHA、真实rc0完整响应、新回执create-only上传/完整读回和启动器独立重哈希。最终候选execution生产者确实输出该字段，完整180终态准入必须匹配新manifest13的实际rawSHA；缺失或不一致拒绝准入。这份规则只解决调用证据边界，不替代真实360/180终态、native回执、PCM/内容/视觉与全分母核验。

## 背景与理由

原candidate plan12已FROZEN，但CPU metadata rebinder产生的manifest12未emit status。冻结candidate13的validate_candidate_plan要求manifest.status=FROZEN，旧launcher04至06和ASR client03仍引用旧回执。直接运行会在元数据校验处失败，尚未涉及科学效应。ROOT实际完整比较两份回执，除status和自哈希外payload相同，并在新服务器实际create-only上传新文件、完整下载逐字节核对。fresh launcher/status07、ASR client04与repeat04使用新路径，原计划/行/raw pins不变。

原始文件SHA、canonical JSON自哈希和未来实际输出SHA承担不同角色。不能用计划的自哈希冒充原始文件SHA，也不能提前填入尚不存在的runtime结果。metadata预检返回字段有限时，把调用层的真实证据留在完整请求/响应中，随后用生产者真实execution记录闭合最终manifest链。

## 方案与后果

- 原位改写旧manifest：会改变已记录来源，旧文件rawSHA失效，因此保留原文件并新增manifest13。
- 改冻结candidate13以忽略status：会改变已封存验证代码；本次可通过最小元数据修复满足既有合同，因此保持生产者。
- 要求metadata-only JSON返回不存在的manifestSHA：与真实生产者不符，会稳定拒绝正确调用；改为调用证据和最终真实execution绑定。
- 实际消费者必须指定新物理文件名`plan_manifest_metadata_repair_attempt13.json`和SHA，不能运行仍引用旧文件的launcher06/client03。归档标准成员别名`stage/plan_manifest.json`及`producer/plan_manifest.json`保持原协议，元数据source_path/SHA则绑定新物理文件；不能把物理修复文件名误写成新tar成员协议。
- 无未来result哈希、无科学阈值/样本/Confirm变化。callers的SOURCE审查不代表实际180完成。

## 来源

- `.codex/scratch/dev360_candidate_manifest_metadata_repair_attempt13/`：原/new回执来源、两字段差异、actual upload/download response和2513B下载。
- 新manifest raw SHA `31d56651e48b6932382d51eedc3b81f96adb87c8c53e16cc8349b776bc22a4a7`，self SHA `6ea4dd85a5f88629f9663d71e284149a33d5749cd4c31f724af19f0d754297ba`。
- `.codex/scratch/root_candidate_manifest13_metadata_qc_attempt01.json`及`root_candidate_manifest13_actual_remote_readback_qc_attempt01.json`：ROOT实际最小payload差异与远端全字节读回。
- `.codex/scratch/dev360_candidate180_attempt07_independent_review_attempt01/independent_source_audit_attempt01.json`：27pins及真实生产者字段独立审查，P2 metadata证据边界。
- `.codex/scratch/root_asr_client04_delta_source_qc_attempt01.json`与`root_repeat04_delta_source_qc_attempt01.json`：继承原方法与元数据修复后的SOURCE核验。

## Observations

- [decision] 新回执显式补齐元数据，保留旧来源和冻结科学输入；最终真实execution闭合manifestSHA链。
- [rationale] 满足实际冻结生产者合同，完整保留执行修复来源与调用证据。
- [consequence] 所有caller更新到新物理文件和actual SHA；标准归档别名不变，未来runtime hashes保持null。

## Relations

- relates_to [[003-immutable-execution-repair-provenance]]
- affects [[开发跨 TFG 的自然音频增强头]]
- affects [[跨 TFG 自然音频增强头的训练与原音换轨验证]]