---
title: Ralph Step 2 - Phone Target Geometry Screening
type: experiment
permalink: tts-exp/research/ralph-step-2-phone-target-geometry-screening
---

# Ralph Step 2 - Phone Target Geometry Screening

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| Initial Document | August 12, 2026 | user |
| Restructured to Research/ with schema observations | August 12, 2026 | user |

本步新增并测试 `phone_alignment_target_audit.py`,比较三种 source-grid target:

1. 现有 MFA phone-local linear interpolation;
2. 每个 matched phone 的 TTS HuBERT L6 mean pooling,再重复到 natural phone frames;
3. 每个 matched phone 内的 constrained hard DTW path,再按 path 将 TTS frames 聚合到 natural frames。

## 实现约束

- hard DTW 只在同 label matched phone span 内运行;
- path 固定局部左上到右下;
- 使用归一化坐标 band ratio=0.5;
- 不跨 phone;
- 每个 natural local frame 至少被一个 path pair 覆盖;
- 保存每个 span 的 path、cost、frame count 和 strategy metadata;
- mean pooling 与 hard DTW 都只生成 source-grid feature target,不改变 waveform length。

新增测试:`tests/test_phone_alignment_target_audit.py`。验证:`9 passed in 3.58s`。

## K=1 真实初筛

使用 `/tmp/tts_two_stage_preflight/subset_k1.json`、同一 K=1 paired key、frozen HuBERT L6、direct waveform bounded residual=0.2,每种只优化 30 steps。

| target | initial gap | final gap (30 steps) | reduction | matched/total |
|---|---:|---:|---:|---:|
| MFA-linear | 0.3813 | 0.0883 | 76.8% | 232/416 |
| phone mean pool | 0.4634 | 0.1507 | 67.5% | 232/416 |
| phone-anchored hard DTW | 0.3875 | 0.0948 | 75.5% | 232/416 |

中间判断:当前 MFA-linear 优于简单 phone mean pool;hard DTW 与 linear 接近;30 steps 不是最终结论,需等长预算。

## 100-step equal-budget confirmation

| target | initial gap | final gap | gap reduction |
|---|---:|---:|---:|
| MFA-linear | 0.3813197 | 0.0335368 | 91.21% |
| phone mean pool | 0.4633910 | 0.0729741 | 84.25% |
| phone-anchored hard DTW | 0.3875121 | 0.0336940 | 91.31% |

Hard-DTW path 统计:43 spans,mean path cost 约 0.3464,horizontal 85 / vertical 5 / diagonal 104。

## Multi-sample equal-budget confirmation

固定 train split 前 4 个 utterance,相同 frozen HuBERT L6、100-step、lr 0.01、bounded residual 0.2。

| sample | target | initial gap | final gap | reduction |
|---|---|---:|---:|---:|
| 0122 | MFA-linear | 0.381320 | 0.033522 | 91.21% |
| 0122 | phone pool | 0.463391 | 0.072976 | 84.25% |
| 0122 | hard-DTW | 0.387512 | 0.033693 | 91.31% |
| 0129 | MFA-linear | 0.257308 | 0.030030 | 88.33% |
| 0129 | phone pool | 0.356807 | 0.085826 | 75.95% |
| 0129 | hard-DTW | 0.267552 | 0.028895 | 89.20% |
| 0137 | MFA-linear | 0.337569 | 0.042147 | 87.51% |
| 0137 | phone pool | 0.429824 | 0.098905 | 76.99% |
| 0137 | hard-DTW | 0.320753 | 0.036061 | 88.76% |
| 0144 | MFA-linear | 0.208240 | 0.027300 | 86.89% |
| 0144 | phone pool | 0.318304 | 0.090040 | 71.71% |
| 0144 | hard-DTW | 0.216300 | 0.029483 | 86.37% |

**Multi-sample decision**: phone pooling consistently inferior(最终 gap 是 MFA-linear 的 2.1–3.3 倍);MFA-linear 与 hard-DTW 在所有样本上接近,无稳定优势。保留 MFA-linear 为生产基线,hard-DTW 为 audit/control。

## Feature-residual predictor diagnostic

新增 `scripts/feature_residual_predictor_audit.py`,冻结 HuBERT L6 + MFA-linear target,只训练逐帧 identity-initialized residual MLP(hidden=128, lr 0.001, 200 steps)。

| subset | sample | initial gap | final gap | reduction |
|---|---|---:|---:|---:|
| K=1 | 0122 | 0.381320 | 0.009134 | 97.60% |
| K=4 | 0122 | 0.381320 | 0.050249 | 86.82% |
| K=4 | 0129 | 0.257308 | 0.028561 | 88.90% |
| K=4 | 0137 | 0.337569 | 0.023204 | 93.13% |
| K=4 | 0144 | 0.208240 | 0.043151 | 79.28% |

解释:K=1 可达极小 feature gap,target 可学习;K=4 有 cross-utterance generalization bottleneck;下一个优先级是检查 renderer 的 conditioning/preservation loss,而非引入 soft-DTW。

## 最终判断

1. phone mean pooling 明显损失 frame-level acoustic trajectory,不能作为直接替代;可能更适合 phone-level representation/predictor。
2. constrained hard DTW 的 direct upper bound 与 MFA-linear 几乎相同(差约 0.00016),没有证据说明 DTW target 优于现有线性映射。
3. hard DTW path 大多 diagonal/horizontal,当前样本线性映射已能表达主要 correspondence。
4. 不是全局结论:只验证了一个 K=1/K=4 子集、一个 HuBERT L6、一个 band ratio。
5. 下一步不引入 soft-DTW;更有价值的是 explicit duration/pause target 与多样本一致性检查。

## Observations
- [status] concluded
- [result] phone mean pool 最终 gap 为 MFA-linear 的 2.1-3.3 倍;hard-DTW 与 MFA-linear 无稳定差异 (Δ≈0.00016)
- [conclusion] 保留 MFA-linear 为生产基线;hard-DTW 作 audit;下一步检查 renderer conditioning 而非 soft-DTW
- [report] 记忆笔记, 研究步骤报告

## Relations
- part_of [[TTS Feature Alignment and Duration Model Research]]
- relates_to [[Ralph Step 1 - Controlled Baseline]]
