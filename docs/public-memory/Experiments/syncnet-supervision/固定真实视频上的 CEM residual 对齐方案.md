---
title: 固定真实视频上的 CEM residual 对齐方案
type: report
permalink: tts-exp/experiments/syncnet-supervision/固定真实视频上的-cem-residual-对齐方案
status: active
tags:
- syncnet
- cem
- mfa-linear
- fixed-video
- alignment
- action-space
- acoustic-mismatch
---

# 固定真实视频上的 CEM residual 对齐方案

## Context
固定真实视频、冻结 SyncNet 的低维 CEM 验证已经证明 global shift reward 可搜索，但 MFA-linear global residual 的收益偏小。当前项目转入分阶段诊断：先区分局部 timing 与 acoustic mismatch，不把多个设计一次性叠加。

## 已完成验证
- Gate 0：69 条、345 个 known-shift trials 全部在 40 ms 内恢复，zero-shift median error 10 ms，无边界吸附，GO。
- Gate 1：69 条×5 shifts 的 deterministic bounded CEM 全部在 40 ms 内恢复，median dense-oracle gap 3.44 ms，重复差 0 ms，GO。
- Gate 2 pilot：candidate 为 exact-length MFA-linear，动作是 `r_ms ∈ [-120,120]` 的 global shift；12 条、每 trial 256 queries、两次重复。positive normalized gain=83.3%，median normalized gain=0.0238，bootstrap CI=[0.0120,0.0312]，5/6 group medians positive，finite/exact-length=100%，但 median gain 未达到 0.05 promotion threshold，停止 full validation。

## 分阶段诊断方案
1. **Gate 3A synthetic local timing**：对 natural audio 生成已知 global+4-knot monotonic warp，用 CEM 恢复 path；先证明 local action 和 optimizer 有能力，失败就不进入真实数据。
2. **Gate 3B real MFA-linear local timing**：在相同 12 条 pilot 上比较 MFA-linear baseline、global-only 和 global+4 knots；固定视频、absolute anchor、search/evaluation masks、seed 和 query budget，局部 timing 单独验证。
3. **Gate 4A acoustic-only oracle**：不重采样、不改变时间边界，只做 gain、spectral tilt、平滑 energy envelope 等低维 correction；natural-derived correction 只能作为 oracle upper bound，评估 acoustic mismatch 是否有可消除空间。
4. **Gate 4B combination**：只有 timing-only 或 acoustic-only 单独通过 promotion criteria，才测试两者组合；不直接上 phone-level 大动作空间或 PPO/SAC。

## 为什么 Gate 2 提升可能不高
- 单一 global shift 不能修复局部 phone duration、边界、pause 和 phone-internal trajectory。
- MFA-linear/WavLM→HiFi-GAN 可能改变 phase、transient、能量包络和频谱域；剩余差异未必是 timing。
- SyncNet/MFCC 是压缩表示，fixed-coordinate reward 可能在 MFA-linear 附近饱和；Gate 1 成功只证明 global shift 可搜索。
- natural intrinsic lag 与 MFA-linear acoustic domain 不同，natural 的最佳 coordinate 不一定是 MFA-linear 的最佳 coordinate。
- speaker/source-group 异质性使 global residual 的收益不一致；当前 LRS3 split 只能称 source-group holdout。

## 动作设计
- 当前 global action：`candidate = bandlimited_shift(MFA_linear, r_ms)`，范围 `[-120,+120] ms`，零 residual 是原始 MFA-linear。
- local action：`[global_shift, knot_1, knot_2, knot_3, knot_4]`，interior knots、首尾固定、strict monotonic、exact length、局部 slope `[0.8,1.2]`，初始 residual 为零。
- acoustic action：时间轴保持不变，先从 global gain/spectral tilt/平滑 energy-envelope 等低维可解释变量开始；不把 acoustic correction 伪装成 timing correction。
- 主 reward：保持绝对 visual window index 的 fixed-coordinate SyncNet reward；同时记录 C/D/offset、log-mel、energy、F0、VAD、phase 和 waveform integrity。

## 约束与当前状态
所有旧 runs、checkpoint、dataset 和结果只读；candidate audio embedding 不缓存；SyncNet frozen/eval；不运行 Wav2Lip、Ditto 或其他 TFG 生成。LRS3 local split 不声称 speaker-disjoint。当前任务已登记为 `局部 timing 与 acoustic mismatch 分阶段诊断`，从 synthetic local warp 开始，任一 promotion gate 失败即停止。

## Observations
- [decision] 先逐阶段验证 timing-only 与 acoustic-only，不同时加入复杂设计。
- [insight] Gate 0/1 排除了 reward 完全不可搜索和 CEM 不稳定；Gate 2 的小收益更像 action expressivity 或 acoustic mismatch 问题。
- [tradeoff] synthetic local warp 是额外成本，但能在真实 MFA-linear 之前验证 local action 的可识别性，避免把 optimizer 失败误判为 timing 假设失败。
- [problem] 当前还没有证据表明扩大 timing action 能超过 MFA-linear global residual 的小幅收益。

## Relations
- relates_to [[局部 timing 与 acoustic mismatch 分阶段诊断]]
- relates_to [[固定真实视频 CEM 对齐验证]]
- relates_to [[固定视频-sync-net-监督-gate-b-结果]]
- relates_to [[Direct waveform upper bound]]
