---
title: Wav2Lip global shift response diagnostic 2026-09-08
type: experiment
permalink: tts-exp/experiments/wav2-lip-global-shift-response-diagnostic-2026-09-08
status: complete
date: '2026-09-08'
report: openspec/changes/diagnose-wav2lip-global-shift-response/
tags:
- wav2lip
- replacement
- global-shift
- diagnostic
---

# Wav2Lip global shift response diagnostic 2026-09-08

用户要求为下一阶段设计简洁、可正确实施的实验 spec。先诊断历史 natural 延迟200ms后 replacement 分数上升，再决定语义辅助研究。本次 v4 已在 GPU 恢复后完成生成、评分和独立验收，但结果仍是诊断性而非 replacement 确认。

历史 SHIFT_200 在23条记录上 C benefit=+0.298（95% CI [+0.131,+0.471]），D benefit=+0.164（CI跨零），评分offset仅变化0–1帧。这个分数观察未被后来局部warp或谱迁移直接推翻，也尚未证明真实口型改善。不能把旧CONTROL_FAILED等同于已证伪，亦不能当作生成头成功。

## Observations

- [status] complete
- [engineering] v4 在 GPU 恢复后完成：12 records、96 generated videos、192 score cells；独立 validator=`valid`，audio/face/video/score/hash 绑定全部通过。
- [controls] N_REPEAT 稳定；精确 ±200ms 音频移位可检测；生成画面对 shifted audio 的响应通过；这些是工程/敏感性控制，不是 replacement 证据。
- [endpoint] V_DELAY_200 的 DYNAMIC-STATIC 交互 C mean=-0.096（95% CI [-0.412,+0.198]），D mean=+0.011（CI [-0.247,+0.260]）；未出现预设正向信号。V_ADVANCE_200 也未通过正向信号门禁（C=-0.450，D=-0.394，方向并不支持目标结论）。
- [scientific] `diagnostic=COMPLETE`、`scientific_decision=NOT_A_CONFIRMATION`；`training_authorized=false`、`generalization_established=false`、`historical_gate_repaired=false`。本实验没有确认历史 natural DELAY_200 的 replacement 增益，也不把它单独宣布为已证伪。
- [next] 按 spec `STOP_AND_REVIEW`；下一步应先解释“历史 SHIFT_200 正向分数与本次受控动态/静态诊断不一致”的来源，再决定是否进入独立语义实验；不要训练生成头。
- [question] 全局偏移的正向评分是否保留在共同内部时间窗口和固定自然基线offset下，是否依赖动态人脸输入，生成画面是否跟随±5帧音频变化？
- [protocol] CPU复算历史23条/69矩阵；新实验从已有22条seen-fit ROI cohort按source_group/sample_id排序固定前12条，不按分数选样，不称独立确认。
- [budget] 两种face模式DYNAMIC/STATIC，每种N/N_REPEAT/DELAY_200/ADVANCE_200；上限96生成视频、192评分cell，另24个派生输入face流。
- [contract] STATIC同时重复第0帧与第0个box；±3200采样整数移位、补零不环绕；replacement始终整条untouched N PCM；同视频错配和own-audio只作诊断。
- [endpoint] FULL及共同内部窗口，先均值距离曲线再C/D/offset；增加固定自然baseline offset的距离和简单下半ROI像素lag。全部新CI探索性，不自动确认gain。
- [input-audit] 只读结构预检确认历史69矩阵/23条均有共同绝对帧内部支持，最短9行。6ZiN9ZJT294_00005的三条track从frame32开始；73rUjrow5pI_00005的N从0、SHIFT从20开始。spec已要求绝对帧交集映射，不假定track从0开始。此为资产结构事实，不是偏移机制结论。
- [decision] 完整验收后diagnostic=COMPLETE、science=NOT_A_CONFIRMATION、next_action=STOP_AND_REVIEW；旧谱迁移阴性不变，training/generalization/historical_gate_repaired均false。
- [future] 语义辅助单独立spec：完整natural加正确内容、错配内容、无辅助，必须胜过完整natural baseline；本轮不实施。
- [validation] 修订历史track坐标后，最终OpenSpec strict校验通过。只读结构预检不替代下游历史统计parity和独立validator。
- [report] openspec/changes/diagnose-wav2lip-global-shift-response/{proposal.md,design.md,tasks.md,specs/wav2lip-global-shift-response/spec.md}

## Relations

- follows [[Wav2Lip replacement endpoint reconciliation 2026-09-08]]
- relates_to [[Wav2Lip spectral structure replacement 2026-09-08]]
- relates_to [[LRS3 natural-to-TTS bridge confirmation result]]

## Changelog
| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 按用户要求编写全局±200ms偏移诊断OpenSpec，登记planned；新实验未执行 | September 8, 2026 | user |
| 历史绝对帧支持预检通过，最终OpenSpec strict通过，完成设计交付 | September 8, 2026 | user |
| 实验按 OpenSpec 启动，先执行 CPU prepare/history；状态更新为 running，尚无新生成结果 | September 8, 2026 | user |
| v3 CPU prepare/history 完成并通过；普通 Codex 沙箱 CUDA kernel assertion 阻塞全流程，未产生新分数；状态更新为 blocked | September 8, 2026 | user |
| GPU 恢复后以同一 protocol 完成 v4：96 视频/192 cells，独立 validator valid；未通过 replacement 正向门禁，科学结论保持 NOT_A_CONFIRMATION；状态更新为 complete | September 8, 2026 | user |
