---
title: 训练 reference-conditioned alignment policy
type: task
permalink: tts-exp/tasks/训练-reference-conditioned-alignment-policy
current_step: 5
priority: high
status: done
tags:
- task
- syncnet
- policy
- distillation
- mfa-linear
assigned_to: claude
---

# 训练 reference-conditioned alignment policy

## Context

当前目标是在固定真实视频任务中，使用 paired natural reference audio、MFA-linear/TTS audio 和固定 visual embeddings，训练一个低维 correction policy。模型不生成 waveform，而是输出 bounded energy-envelope control points 与一个 global timing residual，再通过确定性 renderer 生成 corrected waveform。

## 已完成的数据与 oracle labels

- LRS3 共 500 条：train=347（30 source groups）、validation=69（6 groups）、test=84（7 groups）。当前只使用 train/validation，test 尚未使用。
- 2 条 train natural MFA TextGrid 缺失，已在 `runs/lrs3_qwen_cloud_n500_20260820/11_mfa_repair_missing_natural` 中单独修复，并合并出完整 repaired context/token artifact。
- train input manifest：`runs/lrs3_cem_fixed_video_20260820/11_policy_train_inputs/manifest.json`。
- train MFA-linear generation：`runs/lrs3_cem_fixed_video_20260820/12_policy_train_mfa_linear/summary.json`，347/347 exact-length、16 kHz mono、finite、hash 审计通过。
- train oracle labels：`runs/lrs3_cem_fixed_video_20260820/13_policy_train_oracle_labels/summary.json`。
  - 347/347 labels 生成成功；300 条 accepted，accepted fraction=86.46%。
  - 每条 16 个 envelope control points、2 次 global timing CEM、每条 512 queries。
  - 逐条 natural/MFA/cache hash、join、acceptance predicate 和顶层 provenance 全部通过。
  - 无 timing boundary hit；repeat difference median=0.176 ms、p90=6.27 ms。

## Policy 实现与 validation

新增：

- `scripts/experiments/syncnet_supervision/reference_conditioned_policy.py`
- `scripts/experiments/syncnet_supervision/train_lrs3_reference_conditioned_policy.py`
- `scripts/experiments/syncnet_supervision/validate_lrs3_reference_conditioned_policy.py`

模型输入：

```text
natural log-mel features
+ MFA-linear/TTS log-mel features
+ fixed visual embeddings
→ natural/TTS Transformer encoders
→ TTS queries cross-attend natural reference and visual stream
→ timing residual + envelope control points
```

输出约束：

```text
r_ms ∈ [-120, 120] ms
envelope gain ∈ [0.95, 1.0625]
```

训练流程已完成：

- 300 条 accepted labels；1000 steps；30 个 train source groups。
- 当前正式 run 使用 oracle-label loss；SyncNet 用于 validation，但 `sync_loss_weight=0`，尚未进行 frozen SyncNet 反向传播 fine-tuning。
- policy checkpoint：`runs/lrs3_cem_fixed_video_20260820/14_reference_conditioned_policy/checkpoint.pt`。
- 独立 validation artifact：`runs/lrs3_cem_fixed_video_20260820/16_reference_conditioned_policy_validation/summary.json`。
- validation 69 条重新加载 checkpoint，在不调用 CEM 的条件下独立重算；与训练阶段的 gain 最大差异为 0。

Validation 结果：

```text
mean gain vs MFA-linear       +0.04596
median gain                   +0.00727
positive clips                53.62%
bootstrap 95% CI              [-0.0118, 0.1049]
positive source groups        3/6
waveform integrity            100%
```

判定为 `NO_GO`，原因是正增益比例低于 65%、bootstrap CI 下界不大于 0、仅 3/6 source groups 为正。模型输出集中在近似平均修正：timing residual 约 +1.12 至 +5.50 ms，envelope mean 约 1.012 至 1.020，说明存在明显的平均输出收缩。

Content proxy 仅作诊断：spectral-shape cosine median=0.99394、minimum=0.97757，global log-mel cosine median=0.999995，active-frame change fraction median=0.0117。ASR、PER、speaker verification 依赖尚未准备，因此正式内容和 speaker preservation 仍未完成。

上下界对照：

```text
conservative envelope oracle:
  median gain=+0.01164, positive=82.61%

envelope + global timing oracle:
  median gain=+0.35155, positive=81.16%
```

这些 oracle 使用 natural-derived information，只能作为上界，不能视作 deployable policy。

## 当前决策与下一步

- 当前 reference-conditioned Transformer 链路、bounded renderer、provenance 和 fixed-coordinate SyncNet validation 已验证，但 label-only policy 泛化未通过；不运行 test 84 条。
- 不继续增加 local knots、phone-level timing、spectral tilt 或 RL。
- 下一步应从当前 label-supervised policy warm start，加入小权重 frozen fixed-coordinate SyncNet auxiliary loss，在完全相同的 69 条 validation 上重新比较；SyncNet loss 必须保留 waveform 梯度但冻结 SyncNet 参数，并继续保留 label、smoothness、magnitude 和 content constraints。
- 只有 validation policy 通过正增益比例、bootstrap CI、source-group 和 preservation gates 后，才允许对 test 84 条运行一次。

## Observations

- [decision] 当前 policy 只输出 global timing residual 和 bounded envelope curve，不输出 waveform 或 phone-level duration #policy
- [status] done
- [insight] oracle correction 在固定视频上明显有效，但 label-distilled cross-attention policy 只获得很小且不稳定的平均收益 #validation
- [problem] policy 输出收缩到近似固定的平均 correction，说明 natural/TTS/visual 特征尚未被有效转化为逐样本动作 #modeling
- [decision] validation NO_GO 后不使用 test、不降低阈值、不继续扩大 action space #evaluation
- [blocker] ASR/PER/speaker verification 尚未完成，且当前 SyncNet 还没有参与 policy 的反向传播 #preservation

## Relations

- relates_to [[固定视频 SyncNet 监督 Gate B 结果]]
- relates_to [[自然参考音频可在推理时获得]]
- relates_to [[Direct waveform upper bound]]
- [final-status] Task closed as done-with-NO_GO: the label-supervised reference-conditioned policy failed frozen validation and test remained unused.
- [superseded] Do not resume the proposed SyncNet auxiliary fine-tuning inside this task; future work requires the separately gated visual-teacher protocol.

## Updated Relations
- superseded_by [[TTS 视觉教师到 replacement-safe 音频控制链路]]