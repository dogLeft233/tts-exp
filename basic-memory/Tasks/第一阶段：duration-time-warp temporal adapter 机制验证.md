---
title: 第一阶段：duration/time-warp temporal adapter 机制验证
type: task
permalink: tts-exp/tasks/第一阶段-duration-time-warp-temporal-adapter-机制验证
status: done
priority: high
current_step: 5
assigned_to: claude
started: '2026-08-15'
steps:
- 固定当前 n25 local-TTS train/valid split，核验 TTS 音频、MFA tokens、cohort/reference 路径和输入
  provenance。
- 实现 known-warp 数据构造：从 TTS WavLM trajectory 和已知 target duration 生成可验证 target，不使用 natural
  完整 feature 作为 target；新增脚本和测试。
- 训练 identity-initialized duration-conditioned temporal adapter，并记录 identity、warp、velocity、cycle
  loss。
- 在 speaker-disjoint S0765 validation 上运行 synthetic warp/identity/audio-only gates；不使用
  SyncNet 选模型。
- 若机制 gate 通过，再记录 Qwen domain gap 和下一阶段独立 Qwen 训练池需求；若失败，停止扩大模型。
completed: '2026-08-15'
tags:
- task
- temporal-adapter
- duration
- time-warp
- mfa-linear
---

# 第一阶段：duration/time-warp temporal adapter 机制验证

## Observations
- [description] 用已有 AISHELL-1 paired natural/TTS 数据验证一个小型 duration-conditioned temporal adapter 是否能学习已知的 phone-level 时间变换；本阶段不调用云端 Qwen，不把结果解释为 Qwen-specific 或最终 hybrid 质量。
- [status] done
- [assigned_to] claude
- [current_step] 5
- [started] 2026-08-15
- [completed] 2026-08-15
- [result] 已实现 `scripts/train_duration_timewarp_adapter.py` 和 `tests/test_train_duration_timewarp_adapter.py`；新测试 5/5，相关回归测试 44/44 通过。
- [result] 使用当前 n25 local-TTS cohort：20 条样本训练、S0765 的 5 条样本 speaker-disjoint validation；生成 734 个 train segments 和 148 个 validation segments；1000 steps 完成。
- [result] train feature loss `0.015938→0.015249`，但 validation `0.012119→0.012701`；validation cosine distance `0.005946→0.006309`；只有 47/148 segments feature loss 改善、33/148 cosine 改善。
- [result] validation synthetic-warp feature gate 未通过；adapter 在训练 speaker 上拟合，但没有泛化到 S0765。不能运行 SyncNet，也不能把它推进 Qwen/RAMC。
- [result] 5 条 S0765 音频的 coarse/adapted/oracle 三种输出均 finite、exact natural length、0 clipping；adapted audio→oracle 的小幅变化不足以抵消 feature validation gate 失败。
- [decision] 不增加 temporal adapter 容量、不继续训练 steps、不调用云端 Qwen。当前监督/输入设计不能证明 learned adapter 优于确定性的 linear warp。

## Steps
1. [x] 固定当前 n25 local-TTS train/valid split，核验 TTS 音频、MFA tokens、cohort/reference 路径和输入 provenance。
2. [x] 实现 known-warp 数据构造：从 TTS WavLM trajectory 和已知 target duration 生成可验证 target，不使用 natural 完整 feature 作为 target；新增脚本和测试。
3. [x] 训练 identity-initialized duration-conditioned temporal adapter，1000 steps 完成并记录 feature/velocity/residual loss。
4. [x] 在 speaker-disjoint S0765 validation 上运行 synthetic warp/identity/audio-only gates；feature gate 未通过，未使用 SyncNet 选模型。
5. [x] 记录 Qwen/domain 含义和停止决定：当前 phase 1 失败，不扩大模型；若重开路线，必须先改变输入/监督设计并使用新实验目录。

## Context
本阶段的 target 是每个 TTS phone trajectory 经过已知 linear time warp 后的解析 target；粗 nearest-resample 作为输入，adapter 学习恢复解析 target。natural 只提供 phone frame count，natural 完整 WavLM 不作训练 target。该设计验证了训练链可以运行，但训练 loss 降低没有转化为 speaker-disjoint validation 改善。

失败原因更接近监督/表示泛化问题，而不是训练步数不足：模型只看粗 retimed trajectory 和 duration controls，且已知 linear warp 本身是确定性基线；在 train speaker 上学习的 residual 没有泛化到 S0765。之前 BM 中 native-TTS denoiser 的失败与本结果方向一致。

当前 Qwen cloud AISHELL n25 与 RAMC n25 仍保留为测试/诊断集，不混入本阶段训练；S0770 全程排除。下一次若继续，必须先设计能看到完整 source phone trajectory/context 的 conditional field，或回到显式 length-regulator + natural pause/prosody weak supervision；不能直接加大当前 adapter 或用 Qwen 测试集调参。

产物：
- `runs/aishell1_mfa_linear_n25_resample_poly_20260815/duration_timewarp_phase1_20260815/summary.json`
- `runs/aishell1_mfa_linear_n25_resample_poly_20260815/duration_timewarp_phase1_20260815/report.md`
- `scripts/train_duration_timewarp_adapter.py`
- `tests/test_train_duration_timewarp_adapter.py`

相关记录：
- [[Train learned phone trajectory adapter]]
- [[VC-inspired trainable speaker-robust rhythm transfer plan]]
- [[TTS Feature Alignment and Duration Model Research]]
- [[AISHELL-1 n25 Qwen cloud MFA-linear 2026-08-14]]
- [[RAMC n25 Qwen cloud MFA-linear 2026-08-15]]
