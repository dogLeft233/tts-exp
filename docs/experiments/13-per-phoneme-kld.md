# 13 — B3 逐音素 KLD + 逐类判别探针（HuBERT，MFA 音素边界）

## 问题

B2 确认了层选择规律与 early-layer 差异峰值。本实验把 natural-vs-TTS 差异下探到**音素类粒度**：哪些音素类驱动差异？逐音素 KLD 能否预测该类可判别性（Temmar et al., IEEE SMC 2025 模板：逐音素 KLD 排序 + 逐类真实/合成二分类 + KLD-准确率相关，元音相关最好）？

## 方法（v2，修正 v1 缺陷）

**数据**：HuBERT 帧级 embedding + MFA TextGrid 音素边界。

1. **逐音素池化**：每个音素 interval（TextGrid phones 层）内池化 HuBERT 帧 → 纯音素向量（不是整音节）。音素按去声调 base 符号分组。
2. **逐类 Gaussian KLD**：PCA→30d + Ledoit-Wolf 收缩协方差，对称 KL。
3. **逐类分组判别探针**：natural-vs-TTS 逻辑回归 + PCA（Pipeline 内 fit），**按话语分组**（GroupShuffleSplit，同话语 token 不跨 train/test），报告 accuracy 和 AUC。
4. **跨类相关**：Spearman(KLD, AUC) + 偏相关（控制样本量）+ 元音/辅音分组。
5. 送气对（b/p, d/t, g/k, zh/ch, j/q, z/c）检查。

**⚠️ v1 的两个根本缺陷（已重写）**：
- v1 池化整音节并给 initial/final/tone 三个视图复用同一向量 → "initial z" 实际以韵母为主，非纯音素
- v1 分类器用 token 级 StratifiedKFold（不按说话人分组）→ natural 是 100 个说话人 vs 单 TTS 声音，深层特征说话人主导，in-sample 可分（acc 1.0）但 out-of-sample 崩到 0.385（反机会），v1 的 Spearman 0.49-0.68 是伪影

## 结果

### 逐音素 natural-vs-TTS 判别力（AUC，越高越可分）

| 层 | 判别力最强音素 (AUC) | 判别力最弱 |
|---|---|---|
| **l1** | ow(0.95) kʰ(0.90) ɥ(0.89) tsʰ(0.88) k(0.87) xʷ(0.87) | p(0.69) tʲ(0.70) l(0.70) |
| l7 | t(0.72) tɕʰ(0.68) e(0.67) ʈʂ(0.66) | p(0.50) x(0.52) ow(0.53) |

**早层（l1）很多音素 AUC 0.87-0.95，深层（l7）降到 0.65-0.72** —— B2 的"early-layer 差异峰值"在音素层复现。

### KLD ↔ 判别力相关（修正后弱）

| 层 | Spearman(KLD,AUC) | 偏相关(n) | 元音 | 辅音 |
|---|---|---|---|---|
| l1 | 0.276 | 0.287 | **0.500** | 0.188 |
| l4 | 0.095 | 0.121 | **0.500** | 0.075 |
| l7 | −0.026 | 0.038 | −0.127 | −0.021 |
| l10 | 0.052 | 0.163 | 0.182 | 0.101 |

**v1 声称的强相关（0.49-0.68）在修正后消失**（l1 只有 0.28，深层趋零）。但**元音类相关（~0.50）一致高于辅音（0.08-0.19）** —— 对 Temmar"元音相关最好"的部分复现（n=11 元音类，p≈0.12，边际）。

### 送气对（l1，AUC 判别力）

k(0.87) t(0.82) j(0.82) z(0.82) 等不送气/送气成员在早层 AUC 0.82-0.87，可判别。

## 结论

1. **逐音素 natural-vs-TTS 差异真实存在且最强在早层**（l1 的 ow/kʰ/ɥ/tsʰ AUC 0.87-0.95），随层加深衰减——与 B2 一致，从音素粒度独立佐证
2. **Temmar"KLD 预测可判别性"在中文仅部分成立**：整体相关弱（l1 0.28），但**元音类相关高于辅音**（0.50 vs 0.19），方向支持元音敏感
3. **v1 结论撤回**：整音节池化 + 未分组 CV 的"KLD 强相关"是方法论伪影，修正后不成立
4. 判别力最强的音素（kʰ, tsʰ, ow, xʷ）提示**送气塞擦音与复合韵母**的声学差异是 TTS-vs-natural 的主要来源

## 审查修复（对抗子 agent 审查 v1）

| # | 问题 | 修复 |
|---|---|---|
| C1 | 池化单元是整音节，非音素；三视图复用同一向量 | 逐音素池化（MFA phone 边界），tone-stripped base 分组 |
| C2 | token 级未分组 CV → 说话人混杂，out-of-sample 反机会 | 按话语 GroupShuffleSplit + PCA 在 Pipeline 内 fit；报告 AUC |
| M3 | KLD/acc 被样本量混杂（Spearman(n,KLD)≈0.5） | 偏相关控制 n；同文本设计保证类内 n 平衡 |
| M4 | PCA 在组合数据上由说话人方差主导 | 分类器 PCA 在 fold 内；标注投影局限 |
| M5 | LR 未标准化、无收敛检查、无波动 | Pipeline + StandardScaler + accuracy/AUC ± std |
| M6 | 送气对模式是 n 伪影 | 同文本平衡 n；报告改为描述性 |
| M7 | 声调视图无功效（n=5 类） | 移除声调相关，保留描述性 KLD |
| M8 | "l1 最强"仅 2 层未公平比较 | 跑 4 层（1/4/7/10）确认趋势 |

## 状态

✅ **已完成（v2）** — 脚本 `scripts/38_per_phoneme_kld.py`，产物 `results/aishell100_phoneme/per_phoneme_kld/per_phoneme_kld.json`

## 相关文件

- 脚本：`scripts/38_per_phoneme_kld.py`
- 产物：`results/aishell100_phoneme/per_phoneme_kld/per_phoneme_kld.json`
- 前置：`docs/experiments/09-aishell100-phoneme-extension.md`、`12-layer-curves.md`
- 模板：Temmar et al., IEEE SMC 2025（DOI 10.1109/SMC58881.2025.11343334）
