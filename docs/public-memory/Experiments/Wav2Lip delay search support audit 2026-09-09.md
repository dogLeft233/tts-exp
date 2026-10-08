---
title: Wav2Lip delay search support audit 2026-09-09
type: experiment
permalink: tts-exp/experiments/wav2-lip-delay-search-support-audit-2026-09-09
status: concluded
date: '2026-09-09'
hypothesis: 已知音频延迟将峰值移出有限offset搜索域，能否解释上一轮全部三个控制异常？
report: openspec/changes/diagnose-wav2lip-delay-search-support/
tags:
- wav2lip
- control-diagnostic
- search-support
- concluded
---

# Wav2Lip delay search support audit 2026-09-09

下一步先做一次CPU控制诊断，再决定自然基底辅助增量是否具备续跑条件。历史直接谱迁移已有负结果，200ms shift的自由分数改善未转化为共享natural anchor收益；masked条件增量在完整natural上的候选实验仍未执行，不能因此判定内容方向无效。

## Observations

- [status] concluded
- [question] natural-content-residual Stage A的三个offset失败是否来自预期峰超出原31列搜索域？
- [evidence] 父run `runs/wav2lip_natural_content_residual_20260908_v2/` parity和重复性通过，anchor损伤mean=+1.162568、95%CI=[+0.844647,+1.482466]、8/8组正，但offset仅13/16，终态CONTROL_FAILED；Stage B未运行。
- [preflight] 只读检查发现三条异常的k_N=27、27、28，+5帧延迟预期列32、32、33超出0–30。32个缓存cell的worker/embeddings/matrix hash、shape和dtype已核对。设计期没有计算新搜索域结果，边界解释尚待实验。
- [protocol] 固定16条/8组/32个N与A_DELAY缓存cell；U=30..57不变。natural物理lag=-15..15，已知delay的配对lag=-10..20；从真实embeddings构造，offset分别15-j与10-j。保持旧自然anchor的未补偿错时损伤。
- [gate] corrected_control_pass沿用anchor规则和14/16；boundary_explanation_complete另要求三条越界异常恢复且原13条保持通过，即16/16。两项及独立验收通过才为SEARCH_SUPPORT_RECOVERED并允许建议续跑协议修订，否则CONTROL_UNRESOLVED；工程失败BLOCKED。
- [budget] 新TFG视频、新神经网络forward、训练、TTS与GPU调用均0；复用32个cell，计算32张派生U矩阵。
- [boundary] 新协议只诊断已知delay控制，不执行父候选阶段，不回写父CONTROL_FAILED，不确认历史shift、replacement、waveform head或跨模型泛化。若诊断恢复，下一步为显式修订原内容残差控制协议后再考虑原48候选；否则停止本次诊断，不扫参数。
- [implementation] 已完成 `scripts/experiments/wav2lip_delay_search_support/` 的runner/analysis/独立validate与测试；按tasks阶段自审并更新本笔记。
- [validation] OpenSpec strict、聚焦pytest与独立CLI validator均通过；实验产物和独立验收已完成。
- [report] `openspec/changes/diagnose-wav2lip-delay-search-support/{proposal.md,design.md,tasks.md,specs/wav2lip-delay-search-support/spec.md}`。

- [execution] 实验 run 为 `runs/wav2lip_delay_search_support_20260909_v1/`；只复用父run的32个缓存cell，未生成视频、未做新的SyncNet forward、未训练、未调用GPU。
- [result] 独立重建旧门禁为13/16；三条旧异常仍恰为预检集合：`lrs3_7JVTirBEfho_00039` 的 k_N=27/预期列32、`lrs3_7PwvGfs6Pok_00003` 的 k_N=27/预期列32、`lrs3_7c5t6FkvUG0_00001` 的 k_N=28/预期列33，均越出旧列0–30。
- [result] 使用 natural lag=-15..15 与 A_DELAY lag=-10..20 的物理配对搜索域后，16/16 条 offset差均通过[-6,-4]；三条异常全部恢复，且原13条保持通过。
- [diagnostic] 三条异常的配对最优数组列分别为27、27、28，offset差均为-5；`A_DELAY[q+5]-A_N[q]` 的最大绝对误差约4.77e-7，均值曲线最大绝对误差约6.81e-8，支持“延迟后搜索域截断”解释。
- [result] 未补偿 natural anchor 损伤 mean=1.162567632539，95% CI=[0.844646827451,1.482466255980]，8/8 source group为正；corrected_control_pass=true，boundary_explanation_complete=true。
- [decision] 终态为 `SEARCH_SUPPORT_RECOVERED`，`content_probe_revision_eligible=true`；独立validator PASS，OpenSpec strict PASS，父run五个锁定文件哈希保持不变。
- [boundary] 该结果只恢复了已知delay控制的搜索支持，不能宣称replacement效应、修复历史shift门禁、授权waveform head或证明跨模型泛化；父run原始`CONTROL_FAILED`保持有效不变。
- [next] 下一步需另建版本化 amendment，显式修订 natural-content-residual 的已知delay控制坐标，再决定是否运行原48个候选；本run不自动授权Stage B。

## Relations

- follows [[Wav2Lip natural content residual probe 2026-09-08]]
- relates_to [[Wav2Lip historical shift rescore 2026-09-08]]
- relates_to [[Wav2Lip spectral structure replacement 2026-09-08]]
- relates_to [[LRS3 masked TTS trajectory-specificity diagnosis]]

## Changelog

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 按用户请求回顾历史并设计下一实验；记录三条预期峰越界线索和缓存预检，创建并校验OpenSpec，实际诊断待下游执行 | September 9, 2026 | user |
| 开始执行 delay search support 诊断；先锁定父 run、缓存 provenance 和 legacy gate，再计算配对搜索域 | September 9, 2026 | user |
| 完成CPU诊断与独立验收；旧13/16、配对16/16、三条异常恢复，终态SEARCH_SUPPORT_RECOVERED；记录run、边界和下一步amendment资格 | September 9, 2026 | user |