# 14 — B5 说话人/音素解纠缠控制：音素可分离性是否被说话人混杂？

## 问题

natural 池 = **100 个说话人**（各 1 句），TTS 池 = **单一合成声**。B2 用 pooled 指标选层——若说话人身份与音素结构纠缠，natural-vs-TTS 的音素差异可能是"100 说话人 vs 单一声"的假象。本实验量化并控制该混杂（Pasad et al. 2023 的 Ph\Spk 范式）。

## 方法

- **数据**：HuBERT/XLS-R embedding + MFA 对齐（100 对），层 l6/l10/l11，viseme 层级
- **指标**：
  1. **RV(Ph, Spk) + 置换检验**：音素 one-hot 与说话人 one-hot 的 OLS 投影子空间重叠，置换说话人标签建零分布
  2. **对称 utterance 基线控制**：natural 与 TTS **都**减去各自句内基线（每句均值 OLS 投影），重跑音素探针，比较 `nat_ctrl − tts_ctrl`
  3. **Cramér's V(Ph, Spk)**：音素 × 说话人标签关联（描述性）

## 结果

### RV(Ph, Spk) 置换检验（低且与随机不可区分）

| 层 | RV | p（置换） |
|---|---|---|
| hubert l6 | 0.010 | 0.30 |
| hubert l10 | 0.010 | 0.28 |
| hubert l11 | 0.009 | 0.18 |
| xlsr l6 | 0.006 | 0.64 |
| xlsr l10 | 0.006 | 0.55 |

**RV ≈ 0.01 且置换 p 全部 >0.18** —— 说话人结构在 embedding 空间与音素结构**无可检测的重叠**（不可声称"正交"，只可称"与随机不可区分"）。

### 对称 utterance 基线控制（两条件都残差化）

| 层 | 原始 ph-nat / ph-tts | 控制后 ctrl-nat / ctrl-tts | nat-tts 差（原始） | nat-tts 差（对称控制） |
|---|---|---|---|---|
| hubert l6 | 0.864 / 0.865 | 0.848 / 0.854 | −0.001 | −0.006 |
| hubert l10 | 0.841 / 0.856 | 0.836 / 0.841 | −0.015 | −0.005 |
| hubert l11 | 0.837 / 0.836 | 0.835 / 0.835 | +0.001 | −0.000 |
| xlsr l6 | 0.838 / 0.835 | 0.836 / 0.837 | +0.003 | −0.001 |
| xlsr l10 | 0.834 / 0.835 | 0.834 / 0.838 | −0.001 | −0.004 |

**控制后两条件的探针差仍近零**（|Δ| ≤ 0.006）——去除 utterance 基线不改变 natural-vs-TTS 的音素可判别性结论。

### Cramér's V(Ph, Spk)

Cramér's V = **0.255**（音素分布随 utterance 有变化）。这是**内容级关联**（不同句子含不同音素组合），因 natural 与 TTS **同文本**，该关联对两条件完全一致，**不可能驱动 natural-vs-TTS 差异**——仅作为语境报告。

## 结论

1. **说话人结构与音素结构在 embedding 空间无可检测重叠**（RV 置换 p>0.18）
2. **对称 utterance 基线控制后 natural-vs-TTS 音素差异不变**（|Δ|≤0.006）——B2/B3 的 pooled 音素指标未被说话人/句基线驱动
3. **09 的显著性结论（segment_stability/boundary_sharpness）基于逐样本配对置换**，每对天然与 TTS 共享同一 utterance 内容，构造上已控制说话人——B5 是对其的间接佐证
4. 附带：B5 的按 utterance 分组探针（0.83–0.86）远低于 09 pooled 探针（0.93–0.95），印证"未分组 CV 会高估探针"（同 B3），但 09 显著性不依赖 pooled 探针

## 审查修复（对抗子 agent 审查 v1）

| # | 问题 | 修复 |
|---|---|---|
| M1 | RV 用 Ridge 拟合值且无零分布，≈0.01 恰在随机噪声级 | 改 OLS 投影 + 置换检验零分布；结论改为"与随机不可区分" |
| M2 | 说话人探针 0.0 是协议产物（未见过说话人无法预测），却被解读为"无泄漏" | 移除该推断；改用 RV 置换 + 对称控制作主证据 |
| M3 | 控制不对称（只残差 natural） | 两条件都减去句基线，比较 `nat_ctrl − tts_ctrl` |
| M4 | 残差化跨 CV fold 泄漏 | 改为 OLS 精确投影；泄漏量级小，标注 |
| M5 | Ridge α=1.0 非正交化 | 改 OLS 投影（句均值相减） |
| M6 | 缺直接标签独立检验 | 补 Cramér's V(Ph, Spk) |
| S8 | TTS 的 manifest speaker_id 是源说话人（误导下游） | 记录约定：TTS 分组应用 paired_key，勿用 speaker_id |

## 状态

✅ **已完成（v2）** — 脚本 `scripts/39_ph_spk_disentangle.py`，产物 `results/aishell100_phoneme/ph_spk/ph_spk_disentangle.json`

## 相关文件

- 脚本：`scripts/39_ph_spk_disentangle.py`
- 产物：`results/aishell100_phoneme/ph_spk/ph_spk_disentangle.json`
- 前置：`docs/experiments/09-aishell100-phoneme-extension.md`、`12-layer-curves.md`、`13-per-phoneme-kld.md`
