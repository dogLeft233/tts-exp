---
title: 局部 timing 与 acoustic mismatch 分阶段诊断
type: task
permalink: tts-exp/tasks/局部-timing-与-acoustic-mismatch-分阶段诊断
current_step: 5
priority: high
status: done
steps:
- Gate 3A：在 synthetic natural-audio local warp 上验证 global+4 monotonic knots CEM 能否恢复已知局部
  warp。
- Gate 3B：在 12 条 MFA-linear pilot 上比较 global-only 与 global+4 local timing，保持相同 fixed-video
  reward、anchor split 和 query budget。
- Gate 4A：在时间轴固定的情况下做 acoustic-only oracle correction（先 global gain/spectral tilt，再平滑
  energy envelope），与 MFA-linear baseline 比较。
- Gate 4B：仅当 timing-only 或 acoustic-only 单独通过时，才做二者组合；否则停止并记录主瓶颈。
- 保存结果、更新项目记忆，并根据证据决定是否值得完整 validation 或训练 policy。
tags:
- task
- syncnet
- mfa-linear
- timing
- acoustic-mismatch
completed: '2026-08-20'
---

# 局部 timing 与 acoustic mismatch 分阶段诊断

## Observations
- [description] 在固定真实视频、冻结 SyncNet 的条件下，自动区分 MFA-linear 的剩余误差主要来自局部 timing 还是 acoustic mismatch；不运行 TFG，任一阶段 promotion gate 失败就停止升级。
- [status] done
- [assigned_to] claude
- [completed] 2026-08-20
- [gate_3a_result] Gate 3A synthetic local-warp pilot 已完成：12 条 LRS3 validation 样本、4 条 known residual paths、每条 512 次 vector-CEM queries；所有输出 finite/exact-length，但 30 ms 内恢复率=83.33%、evaluation reward recovery=83.33%，低于 90% threshold；local timing route NO-GO，不运行 Gate 3B。
- [gate_4a_result] Gate 4A acoustic-only full 已完成 69 条。global gain median gain=0.0018、positive=52.17%、bootstrap CI 下界<0；spectral tilt median=-0.0130、positive=44.93%、CI 下界<0；50 ms energy-envelope correction median gain=0.0989、positive=72.46%、bootstrap 95% CI=[0.0679,0.1378]、6/6 group medians positive。Gate 4A=GO，明确说明这是 natural-derived oracle upper bound，不是可部署 policy。
- [gate_4b_result] Gate 4B 12 条 combination pilot 未通过：gain+energy 相对 energy-only 的 median incremental gain=0.00076，bootstrap CI=[-0.0363,0.1185]；gain+tilt+energy median incremental gain=-0.00203，CI=[-0.1542,0.0283]。因此不跑 combination full，不训练 policy。
- [final_decision] 当前证据支持“acoustic envelope/domain mismatch 是真实瓶颈之一”，但只支持 natural-derived energy-envelope oracle 上界；没有证明一个不访问 natural audio 的可部署 correction policy。保持 MFA-linear global residual 和 energy oracle artifact，不继续叠加 local timing、spectral tilt 或 RL。

## Steps
1. [x] Gate 3A：synthetic local warp recovery；部分可识别但未过 promotion gate。Artifact：`runs/lrs3_cem_fixed_video_20260820/04_gate3a_synthetic_local_pilot/summary.json`。
2. [x] Gate 3B：真实 MFA-linear local timing；因 Gate 3A NO-GO 跳过。
3. [x] Gate 4A：时间轴固定的 acoustic-only oracle；energy-envelope 单因素通过，global gain/spectral tilt 不通过。Artifact：`runs/lrs3_cem_fixed_video_20260820/05_gate4a_acoustic_oracle_full/summary.json`。
4. [x] Gate 4B：12 条 combination pilot 未显示相对 energy-only 的可靠额外收益。Artifact：`runs/lrs3_cem_fixed_video_20260820/06_gate4b_acoustic_combination_pilot/summary.json`。
5. [x] 保存结果、更新项目记忆，并根据证据停止在“诊断上界”层，不训练没有可靠监督目标的 policy。

## Context
- 代码库：`/home/wjj/tts-audio/tts-exp`。
- 固定视频 manifest：`runs/lrs3_cem_fixed_video_20260819/00_manifest/manifest.json`；69 条 LRS3 validation、6 source groups。
- Gate 0：fixed-coordinate absolute reward 对已知 global shifts 可恢复，345/345 trials 在 40 ms 内，GO。
- Gate 1：global CEM known-shift recovery GO；每次 256 queries，重复差 0 ms。
- Gate 2 pilot：`runs/lrs3_cem_fixed_video_20260820/03_gate2_mfa_global_pilot/summary.json`；MFA-linear global residual 的 positive gain 83.3%，但 normalized median gain=0.0238 < 0.05，因此未跑 full。
- local timing action：global shift + 4 个 interior monotonic knots；首尾固定、exact sample count、局部 slope `[0.8,1.2]`；Gate 3A 失败说明该 reward 对局部 timing 方向不够稳定。
- acoustic-only 变换：不重采样、不改变 sample count/时间边界；global gain 不稳定，spectral tilt 无效，平滑 energy-envelope 具有稳定 oracle gain。natural-derived correction 只能用于诊断上界，不代表可泛化 policy。
- 评价主指标仍是 fixed-coordinate SyncNet reward；LRS3 split 是 source-group holdout，不声称 speaker-disjoint。
