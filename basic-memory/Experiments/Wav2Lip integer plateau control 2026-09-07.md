---
title: Wav2Lip integer plateau control 2026-09-07
type: experiment
permalink: tts-exp/experiments/wav2-lip-integer-plateau-control-2026-09-07
status: concluded
date: '2026-09-07'
hypothesis: 无局部插值的整数平台可通过源内容匹配的oracle控制，实际Wav2Lip生成链能否保留该控制有效性
report: openspec/changes/test-wav2lip-integer-plateau-control/
code_paths:
- scripts/experiments/wav2lip_integer_plateau_control/
- tests/experiments/wav2lip_integer_plateau_control/
tags:
- wav2lip
- replacement
- integer-plateau
- control
- planned
---

# Wav2Lip integer plateau control 2026-09-07

用户要求回顾历史、判断方向是否值得继续，并在可行时设计小实验spec。判断：replacement仍有研究价值，但通用生成头尚无成功前提；值得再做一次有止损的控制检查，不值得继续插值核/阈值搜索或投入训练。历史MFA严格replacement NO_GO、bridge确认控制失败且收益未确认、ROI局部C仅14/22；最近linear oracle虽timing为22/22/21、相对最近帧有改善，但own C均值−0.202、CI[−0.369,−0.062]，仍未通过。不能由这些结果唯一归因于Wav2Lip。

新方案固定原22条seen-fit数据，前半段源时间+5帧、后半段−5帧，直接索引原PCM和像素，远离拼接评分。5帧=200ms=3200采样点，是25fps视频、80Hz Wav2Lip mel和100Hz MFCC网格的最小共同正平移，不是旧正弦warp幅度搜索。以源内容匹配的Q行比较oracle own，同时完整报告目标时间U行的chronological own；这两个estimand不能混称，更不能把新规则当旧own gate修复。

Stage A仅CPU：22个oracle派生流、66fresh SyncNet cells和22cached baseline。核验共同平移曲线等价性、baseline/timing/own/damage。A通过并独立验收valid才执行B：用相同原face/框和P驱动冻结Wav2Lip，最多22个GPU视频、44个额外评分，并与oracle在相同音频/目标窗口比较。B沿用原评分轨迹，不重新检测或把face按源时间移动。

A失败暂停当前控制评估分支；A通过B失败暂停当前生成链控制分支，不能独归因于生成网络；都通过只支持另写适用域明确的候选replacement收益spec。本轮不运行bridge、训练、跨模型或sealed数据。原CONTROL_FAILED与LINEAR_OWN_AUDIO_UNRESOLVED保持。

## Observations
- [status] concluded
- [execution_status] concluded
- [decision] Stage A oracle 控制通过，但 Stage B generated-own 门禁失败；按止损规则暂停当前 Wav2Lip 生成链控制分支，不进入 bridge、训练或跨模型。
- [protocol] 固定22 records/22 source groups；Stage A为66fresh+22cached，Stage B条件性增加44fresh与22个GPU生成视频。
- [audit] Stage A v2 已完成：22/22 传输一致，baseline=22/22，B/C/O 时序=22/22/22，own 与 damage 门禁通过；chronological own C均值=-0.111947（描述性，不改判定）。
- [validation] OpenSpec strict validation通过；Stage A 独立 validator valid，independent_difference_count=0；已满足进入 Stage B 的条件。
- [boundary] Stage A 仅支持该固定整数平台控制在 oracle 中有效；all_v2 的真实 Wav2Lip 生成链未保留控制自身兼容性。结论不修复历史 CONTROL_FAILED，也不支持 replacement、训练或通用头。
- [engineering] all_v1 的磁盘阻塞已通过清理可再生成 pip/uv 缓存修复；all_v2 工程 GO，generated_validation valid，independent_difference_count=0。
- [result] all_v2：22/22 timing、damage C CI=[3.136270,4.035565]、D CI=[3.018645,3.753593]；generated-own C均值=-0.235518，CI=[-0.482542,-0.001515]；D均值=-0.124812，CI=[-0.343721,0.074844]，own gate失败。
- [report] `runs/wav2lip_integer_plateau_control_20260907_all_v2/result.md`；openspec/changes/test-wav2lip-integer-plateau-control/。

## Relations
- follows [[Wav2Lip oracle frame interpolation 2026-09-07]]
- relates_to [[Wav2Lip face-ROI replacement pilot 2026-09-06]]
- relates_to [[LRS3 natural-to-TTS bridge confirmation result]]

## Changelog
| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 按用户请求回顾历史、评估方向并设计有止损的小OpenSpec；尚未执行 | September 7, 2026 | user |
| 按用户请求开始执行；先实现并审计 Stage A，尚未产生实验结果 | September 7, 2026 | user |
| Stage A v2 正式完成并通过独立校验；按 spec 条件进入 Stage B | September 7, 2026 | user |
| Stage B 首次 run 因磁盘空间耗尽工程阻塞；清理 pip/uv 可再生成缓存后重试 | September 7, 2026 | user |
| all_v2 完成：工程 GO、独立验收 valid；科学结论 GENERATED_PLATEAU_UNRESOLVED，按止损规则暂停 | September 7, 2026 | user |
