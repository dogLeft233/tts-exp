---
title: MFA-linear 连续轨迹机制消融 support30 疏通 Spec
type: spec
permalink: tts-exp/research/mfa-linear-连续轨迹机制消融-support30-疏通-spec
status: implemented
date: '2026-09-16'
protocol: mfa_linear_trajectory_ablation_v2_support30
tags:
- mfa-linear
- spec
- support30
---

# MFA-linear 连续轨迹机制消融 support30 疏通 Spec

状态：✅ 已实现。面向下游 GPT Luna Max。本补充与 [[MFA-linear 连续轨迹机制消融实验 Spec]] 一起使用；仅以下条款覆盖原协议，其余按原 spec 执行。

## 1. 目标与唯一协议变更

原运行 `runs/mfa_linear_trajectory_ablation_20260916/` 已有 120 条音频、120 个视频，但 7 条样本的共同支持只有 31–47 个窗口，未达到 50，评分停在 120/225。

本轮将最低有效窗口数由 **50 改为 30**，完成同一批 15 条样本的 225 个评分 cell，回答原有机制问题。窗口数指 SyncNet embedding 起始位置数，不是独立样本数。30 是为解决当前长度限制选择的操作阈值，未证明统计稳定性；本轮属于探索性分析。

新运行目录：`runs/mfa_linear_trajectory_ablation_v2_support30_20260916/`。原运行保留 incomplete 状态和全部原产物。

## 2. 固定条件

- 沿用原运行全部 15 个 paired_key 及顺序、八臂、输入和模型 SHA、参考人脸、固定 score box、量化规则。
- lag 仍为 -15…+15。每条的 14 个机制 cell 共用原定义的 F 和 `range(15, F-15)`，支持数为 F−30。
- T_RAW 原生 cell 仍单独计算自身支持，也要求至少 30 个窗口。
- 使用所有有效窗口；**30 只是最低门槛，不是将每条截成 30 个窗口**。
- 不补齐评分窗口、不延长音视频、不缩小 lag 范围、不按 cell 单独选支持，不换样本或 crop。

## 3. 最小实现

修改现有 `scripts/experiments/mfa_linear_trajectory_ablation.py` 即可，不新增实验框架。

1. 给评分门槛增加一个参数，例如 `--min-common-windows`，默认仍为 50；本轮显式传 30。确保父进程、实际评分 worker 和报告读取同一个有效参数，禁止只改父进程检查。
2. 新目录保存输入清单和复用来源，记录原 run、协议名 `mfa_linear_trajectory_ablation_v2_support30`、门槛 30、代码版本。复用元数据可复制；音频、视频和特征可直接引用原绝对路径，不必复制大文件。
3. 复用前核验相关文件存在、身份与原清单 SHA 一致；模型及 score box 与原运行一致。缺失或不匹配要列出错误，不静默重生成或替换资产。
4. 只重跑 score、analyze，统一重新计算全部 225 个 cell；无需重跑 prepare/audio/render。所有新评分、曲线、分析和报告写入新目录，不能通过可写链接覆盖原文件。
5. 若任何 cell 仍失败，保留失败清单并标记 incomplete；不再自动降低门槛或删样本。修复实现错误后可以重跑。

交付时附实际可执行的命令；参数名称可沿用现有接口，无需为复用再设计复杂缓存系统。

## 4. 分析与判读

主分析使用全部 15 条，完全沿用原 spec 第 8–9 节：阳性参照 G_own、原生与固定自然音轨 R/E/I、C/D、2×2、keep 剂量曲线。bootstrap 按 utterance 配对抽样，seed=20260916，10000 次；置信区间及校正规则保持原样。

补一张简短敏感性表：阈值依次为 **30、35、40、45、50**，仅按每条 14 个机制 cell 的共同支持数筛选合格样本（支持数 ≥ 阈值），从本轮已算分数计算 G_own 和固定自然音轨 R/E/I 的均值。列出各阈值的 n、sample_id 和效应方向；不重新评分、不重新裁窗口、不挑最有利阈值，也不追加显著性检验。T_RAW 不参与这张表。

该表同时改变了样本组成，只能提示结论是否依赖短样本，不能证明窗口长度造成效应变化。原记录中阈值 50 对应 8 条机制样本，可作核对。原有 120 个 cell 不是额外观测，不能与本轮合并扩大 n。

若 G_own 均值 ≤ 0，完成报告但停止解释“为何保留增益”；CI 跨 0 时按原 spec 标记阳性参照不确定。所有结论保留单 speaker、历史已见样本、声码器/音质混杂的边界；未完成的听检继续明确标注，不编造结果。

## 5. 验收

- 原 run 未被覆盖；新清单明确记录门槛 30 和复用来源。
- `scores.csv` 恰有 225 个唯一且符合原定义的 cell，15 条每条 15 个，无失败，所有指标有限。
- 每条曲线包含 31 个 lag 点，可复算 C、D、median、k*；同条 14 个机制 cell 的支持一致，T_RAW 支持单独记录，均 ≥ 30。
- 重新评分得到的原 120 个合格 cell 与旧结果在相同精度下应一致；若有明显差异，检查输入、支持范围、模型与评分代码，不把它归因于门槛变化。
- 交付 `analysis.json`、`report.md`、keep 剂量图和敏感性表。报告首段写清“探索性 support30 补充运行”、阳性参照状态与主要 R/E/I。
- 在对应测试文件补一个有意义的边界测试：支持 29 在门槛 30 时拒绝、30/31 接受；默认 50 仍拒绝 31；变更门槛不改变同一合格输入的支持范围和分数。运行原协议测试并核验实际产物。
- 完成后更新 BM 实验结果和关联任务，链接原运行与本 spec；原运行的 incomplete 结论保留。

## Observations
- [status] implemented（support30 补充运行已完成）
- [decision] 最低评分支持由 50 降为 30，复用原音频与视频，另目录重跑全部评分和分析。
- [result] 新运行完成 225/225 score cells 与 225/225 曲线，原 run 的 incomplete 状态保留；G_own 均值为正但普通 95% CI 跨 0。
- [verification] 复用来源、输入/模型/score box SHA、唯一 cell、31 点曲线、原 120 cell 一致性和阈值敏感性表均已核验；py_compile、ruff、8 个协议测试、严格产物审计通过。
- [boundary] 阈值基于已观察到的可行性调整；本轮仅提供探索性机制线索，不能替代扩展 speaker 或真实视频真值实验。
- [report] 结果笔记：[[MFA-linear 连续轨迹机制消融 support30 2026-09-16]]；运行目录：`runs/mfa_linear_trajectory_ablation_v2_support30_20260916/`。
## Relations

- extends [[MFA-linear 连续轨迹机制消融实验 Spec]]
- depends_on [[MFA-linear 连续轨迹机制消融 2026-09-16]]
- relates_to [[实现 MFA-linear 连续轨迹机制消融实验]]

## Changelog

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 按用户要求编写简单疏通 spec；仅设计，尚未执行补充运行 | September 16, 2026 | user request / agent design |
| 执行并审阅 support30 补充运行；225/225 cell 完成，BM 结果与任务已回写 | September 16, 2026 | user request / agent execution |
