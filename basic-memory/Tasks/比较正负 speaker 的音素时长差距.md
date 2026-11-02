---
title: 比较正负 speaker 的音素时长差距
type: task
permalink: tts-exp/tasks/比较正负-speaker-的音素时长差距
status: done
priority: high
current_step: 4
steps:
- 确认 clean MFA n25 是否已完成；若未完成，明确使用的可比 interim alignment，并记录限制。
- 按 paired sample 和 speaker 计算 TTS-natural 的 phone duration delta、绝对误差、相对误差及边界/总时长差。
- 将 duration discrepancy 与 MFA-linear 的 Sync-C/Sync-D 及音频异常指标做 speaker-block 探索性比较。
- 判断时长差距是否解释 speaker 间差异，并把结果与限制写入项目笔记。
tags:
- task
- aishell1
- mfa-linear
---

# 比较正负 speaker 的音素时长差距

## Observations
- [description] 检验相对于 MFA-linear 效果较好的 speaker，效果较差的 speaker 是否具有更大的 TTS 与 natural 音素时长差距。
- [status] done
- [assigned_to] claude
- [current_step] 4

## Steps
1. [x] 确认 clean MFA n25 是否已完成；若未完成，明确使用的可比 interim alignment，并记录限制。
2. [x] 按 paired sample 和 speaker 计算 TTS-natural 的 phone duration delta、绝对误差、相对误差及边界/总时长差。
3. [x] 将 duration discrepancy 与 MFA-linear 的 Sync-C/Sync-D 及音频异常指标做 speaker-block 探索性比较。
4. [x] 判断时长差距是否解释 speaker 间差异，并把结果与限制写入项目笔记。

## Context
- 当前项目为 AISHELL-1 预定义 n=25，5 speakers × 5 utterances，不使用 S0770。
- 需要优先使用同一 MFA 配置下 natural 与 TTS 的 tokens；历史 2.2.x/mismatched dictionary 结果不能当 clean 结论。
- MFA-linear 下游分数仅作探索性关联，固定 face 的 SyncNet 不能解释为真实 speaker 泛化或因果证据。
- 用户要求保留已有结果，不覆盖或删除旧产物。


## Result
- [completed] 2026-08-14
- [inputs] Historical strict tokens plus current resample-poly fixed-face1 SyncNet summary; output `runs/aishell1_mfa_linear_n25_resample_poly_20260814/phone_duration_gap_audit_20260814/`.
- [result] Good speakers S0765/S0913 and poor speakers S0901/S0912 had identical speaker-balanced phone MAE to four decimals: good 0.0265 s, poor 0.0265 s. Relative absolute gap was good 0.2277 vs poor 0.2184. Poor speakers had only 0.103 s more negative whole-speech duration gap on average.
- [decision] Do not treat larger per-phone TTS-natural duration mismatch as the explanation for poor MFA-linear speakers. Recompute after clean MFA-3 n25 is available.

## Context update
Full clean MFA-3 n25 tokens were not available at analysis time. The result is exploratory and does not justify a training decision; S0913 is a counterexample because it is a good speaker with the largest phone MAE.

## Relations
- project [[MFA-linear 冻结 VC 重合成与下游验证]]
