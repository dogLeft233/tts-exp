---
title: 固定真实视频 CEM 对齐验证
type: task
permalink: tts-exp/tasks/固定真实视频-cem-对齐验证
status: blocked
priority: high
current_step: 3
steps:
- Gate 0：完成 69 条 LRS3 validation 数据 join、visual/audio window 坐标、per-clip natural intrinsic
  lag 校准和 masked fixed-coordinate dense known-shift sweep；Gate 0 GO。
- Gate 1：完成确定性 bounded 一维 CEM known-shift recovery；Gate 1 GO。
- Gate 2：MFA-linear global residual pilot/full 及固定真实视频官方审计。
- Gate 3：合成 local-warp recovery，再比较 MFA-linear global+4 knots。
- Gate 4：仅在前置 gate 通过后做 train-group CEM pseudo-label distillation，随后可选 contextual-bandit
  fine-tuning。
- 在最后通过的 gate 上完成 held-out source-group 评估并记录成功或 NO-GO 结论。
tags:
- task
- syncnet
- cem
- mfa-linear
- fixed-video
blockers:
- Gate 2 MFA-linear global residual pilot normalized median gain=0.0238 < 0.05 promotion
  threshold; full cohort and later gates are intentionally stopped.
---

# 固定真实视频 CEM 对齐验证

## Observations
- [description] 在不运行 TFG 的前提下，使用固定真实视频、冻结 SyncNet 和现有 exact-length MFA-linear 音频，逐步验证低维无梯度 timing residual 是否能提升固定视频的绝对坐标同步 reward。
- [status] blocked
- [assigned_to] claude
- [current_step] 3
- [blocker] Gate 2 MFA-linear global residual pilot 的 normalized median gain 未达到 0.05 promotion threshold，因此按计划不扩展完整 69 条、不做官方 mux 审计、不进入 local knots 或 policy distillation。

## Steps
1. [x] Gate 0：完成 69 条 LRS3 validation 数据 join、visual/audio window 坐标、per-clip natural intrinsic lag 校准和 masked fixed-coordinate dense known-shift sweep；Gate 0 GO。
2. [x] Gate 1：完成确定性 bounded 一维 CEM known-shift recovery；Gate 1 GO。
3. [x] Gate 2 pilot：以 MFA-linear 为 candidate 做 bounded global residual CEM；实现、重复性、finite/exact-length 和 pilot reward 检查完成，但 promotion criteria 未通过。
4. [ ] Gate 2 full：由于 pilot normalized median gain 未达到阈值，暂不运行。
5. [ ] Gate 3：合成 local-warp recovery及 MFA-linear global+4 knots；被 Gate 2 promotion gate 阻塞。
6. [ ] Gate 4：CEM pseudo-label distillation/contextual bandit；被前置 Gate 阻塞。

## Context
- 代码库：`/home/wjj/tts-audio/tts-exp`。
- 数据 join artifact：`runs/lrs3_cem_fixed_video_20260819/00_manifest/manifest.json`；69/69 rows、6 validation source groups，原始视频/natural audio/MFA-linear audio hash 一致，MFA-linear exact natural length 全部通过。
- Gate 0 artifact：`runs/lrs3_cem_fixed_video_20260819/01_gate0_reward_sweep_full/summary.json`。所有 345 个人工 shift trials 在 40 ms 内恢复；zero-shift median absolute correction 10 ms；oracle boundary fraction 0；decision GO。
- Gate 1 artifact：`runs/lrs3_cem_fixed_video_20260819/02_gate1_cem_known_shift_full/summary.json`。69 rows×5 shifts、每 trial 256 queries、两次重复；within-one-frame=1.0、within-80ms=1.0、median dense-oracle gap 3.44 ms、max repeat difference 0 ms、median evaluation reward gain positive、decision GO。
- Gate 2 pilot artifact：`runs/lrs3_cem_fixed_video_20260820/03_gate2_mfa_global_pilot/summary.json`。12 rows、每 trial 256 queries、两次重复；positive normalized gain fraction 83.3%，normalized median gain 0.0238，bootstrap 95% CI [0.0120, 0.0312]，5/6 group medians positive，max repeat difference 0 ms，finite/exact-length=100%。由于 median gain <0.05，promotion decision blocked。
- Gate 2 中 natural audio 只用于 intrinsic lag calibration；candidate 是 MFA-linear，未运行任何 TFG，未做官方文件级 mux 审计。
- 现阶段应把结果解释为：CEM 能恢复人工 global shift，MFA-linear 上也有稳定但偏小的 fixed-video reward 改善；尚不能声称 full-cohort 或 downstream improvement。除非用户明确改变 promotion threshold/实验目标，否则不应继续加大 action space。
