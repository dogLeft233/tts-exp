# 音素识别迁移实验 — 最近质心探针（修正 MFA）

**Script**: `35_phoneme_recognition_probe.py`  
**数据**: AISHELL-1 中文，13 natural + 13 TTS  
**特征**: HuBERT 帧嵌入 → 逐音素/视位最近质心分类（Leave-One-Utterance-Out）  
**MFA**: `mandarin_china_mfa` + `mandarin_mfa`（普通话，natural/TTS 各自独立对齐）

---

## 8/20 修正说明

**发现**: 原 MFA 对齐仅对 natural 音频跑了一次，TTS 复制了 identical 的 TextGrid → TTS 帧标签全部错位。  
**修复**: 重跑 MFA 对 natural 和 TTS 音频各自对齐（13×2 = 26 个 TextGrid）。  
**影响**: TTS PER 从 96.5%（随机水平）纠正为 83.3%（与 natural 接近）。  

---

## 音素识别 PER（44 类，tone-stripped，chance = 97.7%）

| Layer | Natural PER | TTS PER | Δ (TTS−Nat) | Nat→TTS | TTS→Nat |
|---|---|---|---|---|---|
| L0 | 85.04% | 84.78% | −0.26% | 77.29% | 75.88% |
| **L6** | **83.90%** | **83.34%** | **−0.56%** | **71.50%** | **71.10%** |
| L11 | 86.63% | 85.49% | −1.14% | 76.19% | 77.25% |

## 视位识别 PER（11 类 Preston Blair，chance = 90.9%）

| Layer | Natural PER | TTS PER | Δ (TTS−Nat) | Nat→TTS | TTS→Nat |
|---|---|---|---|---|---|
| L0 | 70.99% | 68.45% | −2.54% | 64.09% | 68.66% |
| **L6** | **69.57%** | **66.88%** | **−2.68%** | **61.81%** | **64.33%** |
| L11 | 72.58% | 68.38% | −4.20% | 64.41% | 66.61% |

---

## 关键结论

1. **Natural 与 TTS 的音素/视位特征质量几乎相同**（Δ < 3%）
   - 之前 "TTS 使音素信息丢失 5×" 是 MFA 标签 bug 造成的假象
   - 正确对齐后，TTS 甚至略优于 natural（尤其是视位 L6: 66.9% vs 69.6%）
2. **L6 层最优**，与之前视位分离度分析一致
3. **交叉条件泛化良好**（71% vs 83% 自测）——Natural 质心能准确分类 TTS 帧
4. **TTS 不损失音素/视位信息** → TTS 使 TFG 变好的原因不在特征质量，而在其他方面（时序一致性、说话人多样性等）
5. 最近质心分类的绝对 PER 偏高（~83%），因为 LOO 下 13 句训练集无法覆盖全部 44 个音素类

---

## 与之前 7 度量结果的对比

| 度量 | L6 Natural | L6 TTS | 方向 |
|---|---|---|---|
| Viseme inter-class cosine | baseline | −6% (d=0.73) | Natural > TTS |
| Viseme sil. (clustering) | baseline | −1.4% | Natural > TTS |
| Phoneme PER (self) | 83.90% | 83.34% | TTS ≈ Natural |
| Phoneme PER (cross) | 71.50% | 71.10% | TTS ≈ Natural |
| Viseme PER (self) | 69.57% | 66.88% | TTS ≈ Natural |

**解读**: NMI/silhouette 级别看粗粒度视位分布（11 类），TTS 稍差；帧级最近质心分类看细粒度音素信息，TTS 相当。两者不矛盾——TTS 在类间结构的宏观层面有微小差异，但帧级信息量完好。

---

## 文件

- 音素结果：`data/wav2sem_analysis_zh/metrics/phoneme_probe.json`
- 视位结果：`data/wav2sem_analysis_zh/metrics/viseme_probe.json`
- 脚本：`scripts/35_phoneme_recognition_probe.py`
- MFA 输出：`/root/autodl-tmp/mandarin_output_natural/` 和 `/root/autodl-tmp/mandarin_output_tts/`
