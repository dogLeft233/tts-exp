---
title: 执行 replacement audio-head 泛化与 TTS 贡献验证
type: task
permalink: tts-exp/tasks/执行-replacement-audio-head-泛化与-tts-贡献验证
status: open
assigned_to: downstream-agent
priority: high
started: '2026-08-30'
tags:
- task
- lrs3
- wav2lip
- syncnet
- validation
- generalization
- ablation
---

# 执行 replacement audio-head 泛化与 TTS 贡献验证

P1 原型已证明训练后 replacement 在过拟合记录上超过自然基线（F D=6.7253 < A D=6.79）。本任务在 LRS3 280-record policy cohort 上验证两点：适配器是否泛化到 unseen records（Exp A），以及 TTS cross-attention context 是否有实质贡献（Exp B 三臂 ablation）。完整协议见 worktree 内 `docs/specs/replacement-audio-head-validation-protocol.md`。

## Requirements

### Requirement: Exp A 泛化验证 SHALL 产出可判定结果
系统 SHALL 按 spec 的 Exp A 协议在 source-group 分离的 20/8/12 train/val/test split 上训练共享 adapter（200 steps，seed 0），在 val 上选 checkpoint，并在 12 条 test 记录上运行 official U/F 双 repeat 评分；test paired-gate favorable_rate ≥ 7/12 时报告 `GENERALIZATION_PASS`，否则报告 `GENERALIZATION_FAIL`，不得修改判据。

#### Scenario: 主场景
- **WHEN** 完成 Exp A 全部步骤且 test 的 paired-gate favorable_rate ≥ 7/12
- **THEN** `runs/exp_a_test_*/summary.json` 存在且 status=`GENERALIZATION_PASS`，且 split 的 train/val/test source groups 两两不重叠

#### Scenario: 失败场景
- **WHEN** favorable_rate < 7/12
- **THEN** 报告 `GENERALIZATION_FAIL` 并保留逐记录 official 数据，不以均值 D 改善等其他指标改写判定

### Requirement: Exp B TTS 贡献验证 SHALL 用三臂对比判定 TTS context 贡献
系统 SHALL 在同一 split 上分别训练 TTS-kv / NAT-kv / SHUF-kv 三个 adapter（各 100 steps，同一优化器与 seed），各自在 val 上选 checkpoint，并在 test 上比较 favorable rate；`rate_tts − rate_nat ≥ 0.2` 且 `rate_tts > rate_shuf` 时报告 `TTS_CONTRIBUTION_CONFIRMED`。

#### Scenario: 主场景
- **WHEN** rate_tts − rate_nat ≥ 0.2 且 rate_tts > rate_shuf
- **THEN** `runs/exp_b_test_*/summary.json` 输出 `TTS_CONTRIBUTION_CONFIRMED` 及三个条件各自的 favorable rate 与 beats-natural rate

#### Scenario: TTS 冗余场景
- **WHEN** rate_tts − rate_nat < 0.2
- **THEN** 报告 `TTS_CONTRIBUTION_NOT_CONFIRMED` 并给出三条件完整数值，不得以单独 beats-natural 指标替代判定

### Requirement: 产物与可复现性 SHALL 满足 spec 清单
每个实验 SHALL 在 `runs/exp_{a,b}_*/` 下产出 `split.json`、`train_summary.json`、`val_summary.json`、`test_summary.json`；val 选择规则为最高 favorable rate、平局取更早 step。

#### Scenario: 主场景
- **WHEN** 检查任意 run 目录
- **THEN** 四个 JSON 均存在且字段与 spec 示例一致，val_summary 记录每个候选 checkpoint 的 favorable rate 与最终选择

#### Scenario: 边界场景
- **WHEN** 某 test 记录渲染失败或数据缺失
- **THEN** 记入该记录 per_record 的 error 字段并从分母排除，同时写入 summary 的 excluded_records 列表，不静默跳过

## Observations
- [status] done
- [priority] high
- [assigned_to] downstream-agent
- [spec] `.claude/worktrees/lrs3-wavlm-resynthesis-50/docs/specs/replacement-audio-head-validation-protocol.md`
- [project] TTS 视觉教师到 replacement-safe 音频控制链路
- [started] 2026-08-30
- [context] 直接复用现有 `tmp/lrs3_policy_a1_200_20260828/policy_cohort`（280 完整记录、29 source groups）；排除 P1/P2 已用记录；渲染波动已知（A/U 的 D 波动 ~0.014–0.066，F 渲染字节确定），U/F 判定取双 repeat 方向一致。
- [exp_b_result] TTS_CONTRIBUTION_NOT_CONFIRMED — 2026-08-30 完成。split `data/splits/exp_b_split.json`（10/5/10，4/2/4 disjoint groups，含 shuffle_map）。三臂各 100 steps 训练于 `tmp/runs/exp_b_train_{tts,nat,shuf}_20260830_193435`；val 选择：TTS-kv step 20（1/5=0.2）、NAT-kv step 80（2/5=0.4）、SHUF-kv step 60（3/5=0.6）。official test（`tmp/runs/exp_b_test_20260830_193435`）10 条记录：TTS-kv 2/10（0.2）、NAT-kv 4/10（0.4）、SHUF-kv 4/10（0.4）；rate_tts−rate_nat=−0.2 < 0.2。结论：TTS cross-attention 无实质贡献，NAT-kv（自注意力）不劣、SHUF-kv（错配）同样不劣——kv 内容对观测效应无差异化作用。
- [report] 完整报告见 [[30-replacement-audio-head-validation]]（GENERALIZATION_FAIL 3/12 + TTS_CONTRIBUTION_NOT_CONFIRMED；P1 单记录 gain 判定为过拟合构造，未用替代指标改写判据）。

## Relations
- project [[TTS 视觉教师到 replacement-safe 音频控制链路]]
- continues [[Replacement-aligned audio-head fixed-data prototype]]
- relates_to [[tts-exp|Wav2Lip replacement is primary objective]]