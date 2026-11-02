---
title: 032-clean-native-rerun-after-post-render-contract-failure
type: decision
permalink: tts-exp/docs/adr/032-clean-native-rerun-after-post-render-contract-failure
status: open
project: tts-exp
decided: '2026-10-10T04:28:39Z'
tags:
- audio-enhancement
- native-execution
- provenance
- deterministic-recovery
---

# 032-clean-native-rerun-after-post-render-contract-failure

## 上下文

用户已授权持续开发与必要重跑。联合SPSA initial720已完成，四候选NOT_ADMITTED，尚未取得第一个三模型双符号tell。session05旧W2L8真实结果由ROOT核验和全量归档；session06/coordinator19继承旧8并补跑Muse/Ditto后，Ditto8官方PASS/rc0，Muse0PASS/1FAILED/7INCOMPLETE，第一条生成后在校验中KeyError adapter_path。ROOT实际04:25:14Z读回所有旧PID已退出、GPU空闲，session仅INITIALIZED/PAIR_ASKED/POINT_DISPATCH_INTENT，0tell。

ASR actual package11源码确定Muse将科学协议generator传给要求运行descriptor的overlay校验函数，后者八个字段全在model.generator。另三份稳定session文件在point表中使用pt-*别名，session04按包含file_id的dataclass完全相等比较而拒绝同内容；未写terminal event并留下orphan point receipt。两项为执行合同错误，不是声学方法失败。

## 决策

以fresh Muse13、session05和新的immutable execution package/session进行完整重跑；不跨package继承任何旧point部分评分。保留session05/session06/coordinator19及所有W2L8、Ditto8、Muse部分原生文件、原pin和备份，明确将这些失败point的物理尝试排除在新canonical reduction之外。新session从同一已资格initial720输入按canonical torch_cpu诚实重算initial selector receipt，以当前session05源pin封存；不编辑旧receipt冒充新源。核对initial method/theta/PCG64/history及首ask的科学字段确定性一致，允许execution package、source、session等真实provenance变化。

Muse13仅保留source protocol self.generator并另取self.model.generator运行descriptor传给_validate_runtime_evidence，不改变生成输入、音频前端、原音评分或源资产。session05仅规范化相同role/declared_path的文件别名：local_path/bytes/rawSHA/pin_name完全一致且实际读回通过时保留全局已有canonical file_id；point文件保持自己的别名和self-hash。相同ID或payload/path任何冲突仍拒绝。四处独立控制器数学、SPSA迭代、损失及随机调度保持不变。

完整重跑plus24及minus24后才允许有效tell。实际preflight必须覆盖新包可达源、Muse八字段producer/consumer、真实首点file-table与session05合并、诚实initial receipt和fresh root无旧intent；通过后子代理可以直接启动一次，无需另加人工许可。严禁恢复旧UNKNOWN/有orphan的session06，或把超时当终局。

## 理由与替代方案

跨新旧package复用旧8需要额外兼容性证书、不同源package共同归约规则及耐久链映射，增加错误面并延迟研究主线。完整重跑成本有限且源身份单一、比较清晰。直接改旧pin/丢弃失败行/将Muse部分视频当PASS或放松评分合同会污染统计对象，均不采用。只修Muse而不修alias冲突仍会在新point汇总重复失败。重开session06不能解决已存在orphan receipt及事件不一致。

## 后果与影响

唯一科学上下文计划仍为initial720+20×48 proposals+4×120 centers=2160；最终720复用controls240与centers480。失败额外物理尝试另列，不宣称总物理渲染恰为2160，也不把旧结果重复计入新优化。initial720已资格结果仍可用于确定性initial选择；新Muse修复只影响后置证据传参，不静默替换历史模型输出。

Fit准入、FULL/GUARD20/31lags/anchor15、相同全局RMS、±3dB、原PCM/时钟、三模型完整分母、20步及四中心全部不变。Dev20全540与Confirm保持封闭；无真实admitted Fit中心前不读取新的Dev候选效果。恢复决定只依据执行故障，不依据部分point分数调参。

权威证据：.codex/scratch/root_coordinator19_actual_terminal_readback_attempt02/；.codex/scratch/root_muse13_session05_source_review_attempt02/两个actual diff（ROOT首版因复制错误报告SHA失败，第二版采用真实bytehash）；ASR/Signal独立source-only复核。Muse13为87354B/a4c42c92456f5ccc8e9754a0e637439a3dcf95dee7cdb3fbea622740c1dca8c0，session05为90811B/03acef33a1d38e874104a663f1a6d1e16a811190b111ebeb52febf6882fa2412。新包及fresh initial实际preflight尚待执行，此决定不是已成功运行的报告。

## Observations

- [decision] 新源包与session完整重跑首plus24，旧失败point全部保留并明确排除新canonical reduction，不做跨package partial adoption。
- [rationale] 单一新包的真实完整分母比增加跨包兼容框架更容易复现，修复针对已定位的后置descriptor和别名合同错误。
- [alternative] 不采用改旧pin/复开orphan session、放宽校验、丢弃失败分母或追加跨package继承框架。
- [consequence] 诚实重算并核对initial receipt/theta/RNG，保持2160唯一科学上下文，额外物理尝试单列；用户授权范围内preflight通过后直接launch一次。

## Relations

- refines [[027-durable-spsa-journal-and-final-center-selection]]
- relates_to [[030-lineage-bound-session-after-pre-native-control-failure]]
- relates_to [[022-fit-only-cross-tfg-native-score-refinement]]
- affects [[开发跨 TFG 的自然音频增强头]]
- affects [[跨 TFG 自然音频增强头的训练与原音换轨验证]]