---
title: 30-replacement-audio-head-validation.md
type: note
permalink: tts-exp/docs/experiments/30-replacement-audio-head-validation.md
---

# Replacement audio-head 验证实验（Exp A 泛化 + Exp B TTS 贡献）

**Status**: concluded
**Date**: 2026-08-30
**Verdict**: 两个判据均为阴性 —— replacement 效应不泛化（GENERALIZATION_FAIL 3/12），TTS cross-attention 无实质贡献（TTS_CONTRIBUTION_NOT_CONFIRMED，NAT-kv ≥ TTS-kv）。

## 背景

P1 原型在固定训练记录上证明 trained replacement 超过自然基线（F D=6.7253 < A D=6.79），但该记录同时用于训练与选择。本实验按 `docs/specs/replacement-audio-head-validation-protocol.md` 验证两个开放问题：适配器是否泛化到 unseen speakers（Exp A）、TTS kv context 是否有贡献（Exp B 三臂 ablation）。数据复用 `tmp/lrs3_policy_a1_200_20260828/policy_cohort`（280 完整记录、29 source groups）。

## 方法

- **Splits**（`data/splits/exp_a_split.json` / `exp_b_split.json`，seed 0）：source-group 完全 disjoint；排除 P1/P2 四组（6ORDQFh0Byw、6StqiaPoS2U、6VWPHKABRQA、6VnKV1sr5VQ）；组内优先 offset ∈ {-1,0} 的记录。Exp A：train 20（7 groups）/ val 8（3 groups）/ test 12（4 groups）；Exp B：10/5/10（4/2/4 groups），B 的 10 个组与 A 全部不同。
- **训练**：共享零输出 adapter，AdamW lr=1e-5、wd=0.01、clip 1.0，loss = `replacement_target_margin_v1` + log-mel trust，WavLM/HiFi-GAN/Wav2Lip/SyncNet 全 frozen。Exp A 200 steps（save 0/40/80/120/160/200），Exp B 每臂 100 steps（save 每 20）。
- **验证**：val 上 proxy（differentiable Wav2Lip + frozen SyncNet）paired gate（D_U−D_F>2q、C_F−C_U>2q、offset 不劣化、best_second_gap>2q、V_audio=0），选 favorable rate 最高、平局取更早 step。
- **测试**：official Wav2Lip 渲染 + 官方 SyncNet V2 评分；每条 test 记录 A×1、U×2、F×2（Exp B 三臂各 F×2），U/F 双 repeat 都过 paired gate 且 A 的 offset 校准通过才算 favorable；f_beats_a = F 双 repeat D 均 < A D。
- **Exp B 三臂**：TTS-kv（paired TTS）/ NAT-kv（kv = 自身 natural slice）/ SHUF-kv（kv = 同 split 内异组记录的 TTS，固定 seed 配对，存于 split json）。

## 结果

### Exp A（12 条 unseen test 记录）

| 阶段 | 结果 |
|---|---|
| val 选择 | step 40（4/8=0.5）；step 0/80/120/160/200 = 0/8、2/8、2/8、0/8、1/8 → 后期过拟合退化 |
| test favorable | **3/12**（lrs3_6ydYeyNSQVY_00005、lrs3_6weGCM3sWKc_00005、lrs3_6xtmm0MnaS0_00002）< 7/12 |
| test f_beats_a | **0/12**（无任何记录 F 双 repeat D 均低于 A） |
| 失败构成 | 4 条 D 反而变差（F 劣于 U）、6 条 C 无增益、1 条 offset 校准失败（6ydYeyNSQVY_00006） |

**Status: `GENERALIZATION_FAIL`**。F 在多数记录上有小幅 D 改善（+0.03~0.17）但 C 同步变差，paired gate 严格判据不通过；没有任何记录超过自然直接驱动。

### Exp B（10 条 unseen test 记录，三臂同 split 同 test）

| 臂 | val 选择 | test favorable | beats_a |
|---|---|---|---|
| TTS-kv | step 20（0.2） | 2/10 = 0.2 | 2 |
| NAT-kv | step 80（0.4） | 4/10 = 0.4 | 3 |
| SHUF-kv | step 60（0.6） | 4/10 = 0.4 | 2 |

rate_tts − rate_nat = 0.2 − 0.4 = **−0.2** < 0.2，且 rate_tts < rate_shuf。

**Status: `TTS_CONTRIBUTION_NOT_CONFIRMED`**。NAT-kv（自注意力）不劣于 TTS-kv，SHUF-kv（错配 TTS）同样不劣 —— kv 内容对观测到的（本就微弱的）效应无差异化贡献。

## 结论与边界

1. **P1 的 single-record gain 是过拟合构造**：同样的算法在 20 条训练记录上训练后，paired gate 在 unseen 记录上 3/12（25%），低于 >50% 的泛化门槛；val 的 proxy 选择也高估了 official 表现（4/8 → 3/12）。
2. **TTS cross-attention 不是效应来源**：自然自注意力臂持平或更好，错配 TTS 臂持平 —— 该 adapter 学到的更像 natural-feature 自身的微调，TTS context 未承载可迁移信息。
3. **claim boundary**：仅覆盖 LRS3 英文 + Qwen TTS + 当前 loss/规模；不排除换 loss、更大数据或架构改动后成立，但当前方案在验证协议下不成立。
4. 负结果本身是有效证据：spec 的失败场景判据按设计触发，未用任何替代指标改写判定。

## 产物

- Splits：`data/splits/exp_a_split.json`、`exp_b_split.json`（含 shuffle_map）
- Exp A：`tmp/runs/exp_a_train_20260830_193435/`（train_summary.json）、`exp_a_val_20260830/`（val_summary.json）、`exp_a_test_20260830/`（test_summary.json）
- Exp B：`tmp/runs/exp_b_train_{tts,nat,shuf}_20260830_193435/`、`exp_b_val_{tts,nat,shuf}_20260830_193435/`、`exp_b_test_20260830_193435/`
- 代码：worktree `scripts/experiments/lrs3_syncnet_finetune/` 新增 `generate_splits.py`、`prototype_replacement_validate.py`、`prototype_replacement_test.py`，扩展 `prototype_replacement_train.py`（split-file/save-steps/kv-mode）

## Relations
- continues [[Replacement-aligned audio-head fixed-data prototype]]
- relates_to [[执行 replacement audio-head 泛化与 TTS 贡献验证]]
- project [[TTS 视觉教师到 replacement-safe 音频控制链路]]
