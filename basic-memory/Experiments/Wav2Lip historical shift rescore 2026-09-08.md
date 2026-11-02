---
title: Wav2Lip historical shift rescore 2026-09-08
type: experiment
permalink: tts-exp/experiments/wav2-lip-historical-shift-rescore-2026-09-08
status: concluded
date: '2026-09-08'
report: openspec/changes/rescore-wav2lip-historical-shift/
tags:
- wav2lip
- replacement
- shift
- rescore
- concluded
---

# Wav2Lip historical shift rescore 2026-09-08

承接 global-shift v4 与 GPT Luna max 的只读审查，设计一次历史偏移分支的有限收口复核。旧23条 SHIFT_200 的 FULL ΔSync-C=+0.298 可以复算，旧共同内部窗口为+0.191（95% CI [-0.016,+0.380]），固定基线offset距离收益未确认。v4另12条的阴性诊断不能直接证伪旧观察，因为样本、生成和评分流程均不同。

本轮保持历史N/SHIFT生成视频及自然音轨不变，只更换评分入口。源码核对发现：v4的ROI用于Wav2Lip生成，SyncNetScorer实际读取完整224×224画面。因此FULL_FRAME_V4不额外裁嘴部；复用旧视频直接重新评分，比较同媒体下LEGACY_TRACKED与FULL_FRAME_V4的收益差。

## Observations

- [status] concluded
- [question] 同一批历史视频在共同绝对帧窗口和共享自然基线offset下的replacement收益，是否依赖评分处理流程？
- [protocol] 固定历史23 records/23 source groups；46个旧生成视频、46个无损派生视频/新mux、46个目标CPU评分jobs，加2个v4固定评分器parity jobs。零TFG生成、零CUDA、零训练。
- [support] J由历史I_H与新矩阵30帧内部支持取交集；保留旧track真实f0。共享anchor来自LEGACY_TRACKED的N/J；23组等权bootstrap，配对比较流程差值。
- [preflight] 执行时再次核对46视频hash、224×224/25fps及配对帧数；23条N的PCM身份与SHIFT精确补零延迟3200 samples全部通过。J全部非空，最短7行；历史 track 的非零起点被保留。
- [validation] run v7 engineering=GO、diagnostic=COMPLETE；独立 validator=valid。23 records/23 groups，46 target score cells，2 parity cells，fresh TFG/CUDA/training均为0。
- [result] 历史 FULL 旧口径精确复现：ΔC=+0.298，95% CI [+0.131,+0.471]；ΔD=+0.164，CI [-0.006,+0.330]。LEGACY 同 J：ΔC=+0.190，CI [-0.012,+0.376]；ΔD=+0.096，CI [-0.083,+0.265]；anchor benefit=-0.028，CI [-0.224,+0.169]。
- [result] FULL_FRAME_V4 同 J：ΔC=+0.341，95% CI [+0.195,+0.489]；ΔD=+0.179，CI [+0.039,+0.313]；anchor benefit=-0.099，CI [-0.278,+0.067]。两评分入口配对差值 C=+0.151、D=+0.083、anchor=-0.071，三者 CI 均跨零。
- [interpretation] FULL_FRAME_V4 的自由 C/D 正向信号没有转化为共享 natural anchor 改善；同媒体两入口差异未确认。结论仍是 NOT_A_CONFIRMATION，不证明 replacement，也不证明零效应。
- [decision] science=NOT_A_CONFIRMATION、next_action=CLOSE_SHIFT_DIAGNOSTIC；training/generalization_established/historical_gate_repaired均false；不再启动偏移扫描或生成。
- [report] openspec/changes/rescore-wav2lip-historical-shift/{proposal.md,design.md,tasks.md,specs/wav2lip-historical-shift-rescore/spec.md}; run=v7。

## Relations

- follows [[Wav2Lip global shift response diagnostic 2026-09-08]]
- relates_to [[Wav2Lip replacement endpoint reconciliation 2026-09-08]]
- relates_to [[Masked TTS reconstruction feasibility prototype]]

## Changelog

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 按用户要求设计CPU同媒体重评分OpenSpec；明确v4整帧评分入口，完成资产预检和strict校验；实验尚未实施 | September 8, 2026 | user |
| 已实现实验包并开始执行；准备阶段将先锁定23条历史记录和parity门禁 | September 8, 2026 | user |
| run v7 完成46个CPU重评分与独立验收；记录同J结果、anchor限制及偏移分支收口决定 | September 8, 2026 | user |
