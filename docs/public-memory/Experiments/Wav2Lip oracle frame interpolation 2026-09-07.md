---
title: Wav2Lip oracle frame interpolation 2026-09-07
type: experiment
permalink: tts-exp/experiments/wav2-lip-oracle-frame-interpolation-2026-09-07
status: concluded
date: '2026-09-07'
hypothesis: 固定线性像素帧插值能否改善 oracle own-audio 并保留 timing/damage gates
report: openspec/changes/diagnose-wav2lip-oracle-frame-interpolation/
code_paths:
- scripts/experiments/wav2lip_oracle_frame_interpolation/
- tests/experiments/wav2lip_oracle_frame_interpolation/
tags:
- wav2lip
- oracle
- interpolation
- own-audio
- diagnostic
result: LINEAR_OWN_AUDIO_UNRESOLVED；interpolation_improvement_supported=true
conclusion: 线性混帧改善了相对旧 V_ORACLE/W 的分数，但 own gate 仍未通过，因此不能解锁 replacement、bridge、训练或泛化。
outputs: runs/wav2lip_oracle_frame_interpolation_20260907_linear_v1/
---

# Wav2Lip oracle frame interpolation 2026-09-07

承接 [[Wav2Lip ROI retiming oracle 2026-09-06]] 的最终 oracle_v4。父实验已 concluded、独立 validator valid；baseline/B/C_oracle/O_oracle 为 22/22/21/22，但 own-audio C/D 均值 −0.470/−0.361，95% CI 下界 −0.634/−0.508，未过 −0.10 门槛，终态 ORACLE_OWN_AUDIO_UNRESOLVED。

本轮已按 OpenSpec 完成实现、单样本 smoke、22 条正式实验和独立 validator；正式结果已写入 run。固定同一 22-record / 22-source-group seen-fit cohort、原 N/W 音频、连续时间映射、掩码及全部旧 gates，只把 oracle 最近帧取整改为相邻原始像素帧的 float64 线性混合，再逐通道 nearest-half-up 到 uint8。V_LINEAR 必须从父 V_ID 像素构造，不能从已取整的 V_ORACLE 或 embeddings 构造。

新增 22 个无损视频流、44 个 mux、44 个 CPU SyncNet fresh cell，引用并重新校验父 88 个 cached cell。主判定为 timing → own → damage 的固定优先级；另报 gain_C/gain_D 相对父 V_ORACLE/W 的配对改善，只有两项 CI 下界都 >0.10 才记 interpolation_improvement_supported。改善不能替代主 gates。线性混帧同时改变平滑和纹理，不能把正结果解释为纯量化因果效应。

正式执行已完成。规划审计确认父全部 22 条 q_float 坐标在 [0,F−1] 内；正式 run 生成 22 个 V_LINEAR 无损视频流、44 个 mux，并完成 44 个 fresh SyncNet cell。OpenSpec strict validation、focused tests、ruff 和独立 validator 均通过。

## Observations
- [status] concluded
- [execution] 正式 run 为 `runs/wav2lip_oracle_frame_interpolation_20260907_linear_v1/`；22/22 records、22/22 source groups、22 个 fresh video stream、44 个 fresh media、44 个 fresh score cell 完成。
- [validation] 独立 validator `valid=true`，`independent_difference_count=0`；像素、PCM、PTS、embedding/matrix、统计和终态均通过独立复算。
- [result] baseline=22/22；B/C_linear/O_linear=22/22/21。
- [result] own C mean=-0.202169，95% CI [-0.368965,-0.061888]；own D mean=-0.028129，95% CI [-0.173011,0.095964]；own gate 未通过。
- [result] damage C mean=4.688034，95% CI [4.227541,5.147804]；damage D mean=4.531433，95% CI [4.106740,4.941262]。
- [result] 相对旧 V_ORACLE/W 的 gain C CI [0.194518,0.338336]、gain D CI [0.262644,0.397308]；`interpolation_improvement_supported=true`。
- [decision] 唯一科学终态为 `LINEAR_OWN_AUDIO_UNRESOLVED`；父 `ORACLE_OWN_AUDIO_UNRESOLVED` 与历史 `CONTROL_FAILED` 保持不变。
- [conclusion] 线性混帧可能改善了量化/平滑/纹理组合下的 oracle 配对分数，但没有建立完整控制，也没有证明实际 G_W 响应或 replacement effect。
- [boundary] 不解锁 bridge、audio head、训练、跨模型泛化；一次固定插值方案完成后停止，不继续搜索插值核、幅度或阈值。
- [outputs] 结果见 `result.md`、`analysis.json`、`final.json`、`validation.json`；OpenSpec change 为 `openspec/changes/diagnose-wav2lip-oracle-frame-interpolation/`。
- [next_step] 需要下一轮决策时，先做项目级自我审查，评估是否值得继续控制诊断；当前不进入训练或 bridge。
## Relations
- relates_to [[Wav2Lip ROI retiming oracle 2026-09-06]]
- relates_to [[Wav2Lip ROI local peak recheck 2026-09-06]]

## Changelog
| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 按用户要求设计下一实验和下游 OpenSpec；状态 planned，尚未执行 | September 7, 2026 | user |
| 开始按 spec 实现；状态改为 running，尚未有科学结果 | September 7, 2026 | user |
| 完成正式 run 与独立 validator；状态改为 concluded，保留负门禁并记录 gain 结果 | September 7, 2026 | user |

## Validation recheck
- [validation] 收尾自审加强 validator，独立重算 own/damage 的 CI 与正向计数后重新验收；结果仍为 `valid=true`、`independent_difference_count=0`、`LINEAR_OWN_AUDIO_UNRESOLVED`，未重跑神经网络实验单元。
- [quality] ruff、聚焦 pytest（9 passed）、OpenSpec strict validation 与 `git diff --check` 均通过。
- [review] 阶段结论保持：线性混帧虽相对旧 oracle 有稳定改善，但 own gate 仍失败；下一步先做项目级自我审查，不进入 bridge、训练或泛化。