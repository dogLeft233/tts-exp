---
title: '31 - Replacement Salvage: Phone-Aligned Cross-Attention'
type: experiment
permalink: tts-exp/experiments/31-replacement-salvage-phone-aligned-cross-attention
status: in-progress
tags:
- experiment
- replacement
- wav2lip
- wavlm
- lrs3
---

# 31 - Replacement Salvage: Phone-Aligned Cross-Attention

- Status: in-progress (Week 1 诊断阶段)
- Date: August 31, 2026


## 背景

Exp 30 失败模式:
1. 泛化失败: 20 条训练 → 测试 3/12 favorable (< 7/12 阈值)
2. TTS 上下文未利用: NAT-kv (0.4) ≥ TTS-kv (0.2), SHUF-kv (0.4) = NAT-kv
3. 假设: 帧级 cross-attention 在未对齐特征上操作;TTS 与 Natural 的 WavLM 特征在帧空间过于相似

## 目标

验证 phone-aligned cross-attention 能否挽救 replacement 方案:
- Week 1: TTS 特征在 WavLM 空间是否与 Natural 可分 (CKA + 线性探针, go/no-go 门)
- Week 2: phone-aligned cross-attention 能否利用 TTS 上下文 (仅一个架构变量)

## 协议

- Spec: `docs/specs/replacement-salvage-protocol.md`
- 复用 Exp 30: splits (exp_a: train 20 / val 8 / test 12), loss, optimizer, 配对门, 评估
- 成功标准: Week 1 CKA < 0.7 AND probe acc > 0.7;Week 2 favorable ≥ 7/12 AND rate_tts − rate_nat ≥ 0.2

## Week 1 诊断

### Step 1.1: WavLM L6 特征提取
- [ ] 40 records (train+val+test) × natural/tts, 对齐切片
- [ ] 输出: runs/diagnostic_week1/features/

### Step 1.2: CKA
- [ ] 逐 record CKA, mean/std, 决策: <0.7 PROCEED, ≥0.9 STOP

### Step 1.3: 线性探针
- [ ] LogisticRegression 帧级分类, val_acc/val_auc, 决策: >0.7 PROCEED, <0.6 STOP

### Step 1.4: 注意力可视化 (回顾性)
- [ ] Exp 30 TTS-kv adapter (exp_b step 20) 注意力权重热图

## 结果
### Week 1 完成 (2026-08-31) → **PROCEED_TO_WEEK2**

**Step 1.1 特征提取**: 40 records (exp_a train 20 / val 8 / test 12) × natural/tts WavLM L6 (1024-dim, 50fps)。natural 统一 191 帧 (96 视频帧切片), TTS 183–1107 帧。输出 `tmp/runs/diagnostic_week1/features/` (80 npy + manifest.json)。

**Step 1.2 CKA** (linear CKA, truncate to min frames):
- mean CKA = **0.3828** (std 0.0501), 范围 0.28–0.51
- train 0.379 / val 0.395 / test 0.381 — 三组一致
- 判定: **CKA_PASS** (< 0.7) — TTS 与 Natural 特征在 WavLM 空间高度可分, 无冗余

**Step 1.3 线性探针** (LogisticRegression 帧级, train 20 → val 8):
- train_acc = 1.0000, **val_acc = 0.9411**, val_auc = 0.9871
- 判定: **PROBE_PASS** (> 0.7) — 帧级可分性极强

**Step 1.4 注意力可视化** (Exp 30 TTS-kv step 20, val record lrs3_6tSlMoMNSlY_00003):
- mean_row_entropy = **0.9978** (≈1.0 = 均匀分布)
- peak attention = 0.005 (415 keys, 均匀值 0.0024)
- argmax 列无对角线结构 (175, 59, 414, 58, ...) — 帧级注意力近似均匀, 未学到帧对应

**Week 1 决策**: `final_decision.json` → **PROCEED_TO_WEEK2**
- rationale: 特征可分 (CKA=0.38) 且线性可分 (acc=0.94), 说明帧级 cross-attention 失败源于对齐/利用方式而非特征冗余

## 结论

- CKA 与探针双证: TTS 特征携带大量 natural 中没有的可分信息 → replacement 在特征层面可行
- Exp 30 帧级注意力近似均匀 → 帧级 cross-attention 未能利用这些差异, 支持 Week 2 phone-aligned 假设
- 下一步: Week 2 MFA phone alignment → PhoneCrossAttentionAdapter 三臂训练

## 结论

- (待填)

## Relations

- extends [[30-replacement-audio-head-validation]]
- executed_by [[执行 Exp 31 Week 1 特征诊断 (CKA + 线性探针)]]
