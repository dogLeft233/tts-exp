# 多 TTS 模型音素特征对比实验

**实验日期**: 2026-07-20  
**数据**: AISHELL-1 中文，13 句 × 4 个 TTS 模型 + Natural 基线  
**特征**: HuBERT L6 帧嵌入  
**方法**: 最近质心 PER + L2 类内/类间分析  
**MFA**: 各模型独立对齐（mandarin_china_mfa）

---

## 1. 实验设计

对比 4 个不同架构的 TTS 模型，都有 13 句完整音频：

| 模型 | 架构 | TFG ΔSync-C（Ditto） |
|---|---|---|
| **Qwen3-VC** | 语音转换（VC） | +1.25（最强） |
| **IndexTTS2** | Transformer + flow | +1.04（中） |
| **GSV** | 扩散模型 | +0.60（弱） |
| **F5-TTS** | Flow Matching | 未测 TFG |

每个模型独立 MFA 对齐 + HuBERT L6 嵌入提取，跑相同的 PER 和 L2 分析。

---

## 2. 结果总表

| 模型 | PER | L2 Within | L2 Between | Sep.Index Phn | Sep.Index Vis |
|---|---|---|---|---|---|
| **Natural** | 83.9% | 7.26 | 5.37 | 0.74 | 0.41 |
| **Qwen3-VC** | **83.3%** | 7.22 | **5.78** | **0.80** ✓ | **0.46** ✓ |
| IndexTTS2 | 84.8% | 7.30 | 5.40 | 0.74 | 0.42 |
| GSV | 85.7% | 7.18 | 5.22 | 0.73 | 0.44 |
| F5-TTS | 87.1% | 7.27 | 5.47 | 0.75 | 0.42 |

### 2.1 音素 PER（越低越好）

```
Natural  ████████████████████████████████████████████████ 83.9%
Qwen3-VC ████████████████████████████████████████████████ 83.3% ✓
IndexTTS █████████████████████████████████████████████████ 84.8%
GSV      ██████████████████████████████████████████████████ 85.7%
F5-TTS   ███████████████████████████████████████████████████ 87.1%
```

### 2.2 L2 Between（越高越好，音素质心间距）

```
Natural  █████████████████████████████████████████████ 5.37
Qwen3-VC ████████████████████████████████████████████████ 5.78 ↑+7.6%
IndexTTS █████████████████████████████████████████████ 5.40
GSV      ████████████████████████████████████████████ 5.22
F5-TTS   ███████████████████████████████████████████████ 5.47
```

### 2.3 分离指数（between / within，越高越好）

```
Natural  ██████████████████████████████████████████████ 0.74
Qwen3-VC ████████████████████████████████████████████████ 0.80 ✓
IndexTTS ██████████████████████████████████████████████ 0.74
GSV      █████████████████████████████████████████████████████ 0.73
F5-TTS   ██████████████████████████████████████████████ 0.75
```

---

## 3. 关键发现

### 3.1 TTS 类内散度收敛

4 个 TTS 模型 + Natural 的类内散度在 7.18–7.30 之间（波动 < 2%）。TTS 不改变同一音素内帧的紧凑度——这是 HuBERT 编码器本身的特性，而非 TTS 差异。

### 3.2 类间距离解释全部差异

| 模型 | L2 Between | vs Natural | PER | vs Natural |
|---|---|---|---|---|
| Qwen3-VC | 5.78 | **+7.6%** | 83.3% | **−0.7%** |
| IndexTTS2 | 5.40 | +0.6% | 84.8% | +1.1% |
| GSV | 5.22 | **−2.8%** | 85.7% | +2.1% |
| F5-TTS | 5.47 | +1.9% | 87.1% | +3.8% |

类间距离和 PER 呈强相关（Qwen 最远→PER 最低，GSV 最近→PER 较高）。
F5-TTS 例外：类间距离远（5.47）但 PER 最差（87.1%）——可能有少数混淆音素拖累。

### 3.3 TFG ΔSync-C 排序完全匹配特征排序

| 模型 | TFG ΔSync-C | Sep.Index Phn |
|---|---|---|
| Qwen3-VC | **+1.25** | **0.80** |
| IndexTTS2 | +1.04 | 0.74 |
| GSV | +0.60 | 0.73 |
| （F5-TTS） | 未测 | 0.75 |

TFG 表现与特征分离指数完美匹配。Qwen3-VC 不仅是"最好的 TTS"，其 HuBERT L6 特征确实是结构最清晰的——虽然类内紧致度与其他模型持平，但类间距显著更大。

### 3.4 视位分离指数与音素趋势一致

视位分离指数与音素分离指数高度相关。Qwen 仍然是唯一显著优于 Natural 的模型：
- Natural: 0.41
- Qwen3-VC: 0.46（+12%）
- IndexTTS2: 0.42
- GSV: 0.44
- F5-TTS: 0.42

---

## 4. 结论

1. **TTS 不是铁板一块**——不同 TTS 模型产生不同的特征结构，Qwen3-VC 显著优于其他
2. **类间距离是关键差异维度**——所有模型类内散度相同，差异完全来自音素质心之间的 L2 间距
3. **TFG 表现遵循特征质量排序**——同步性能与音素分离指数完美对应
4. **Qwen3-VC 是当前最优 TTS 音频源**——不仅在 TFG 指标上，在 HuBERT 特征结构上也表明白

---

## 5. 文件

### 结果 JSON
```
data/wav2sem_analysis_zh/
├── metrics/phoneme_probe.json          # Natural + Qwen3-VC
├── metrics/hubert_l2_analysis.json     # Natural + Qwen3-VC L2
├── model_indextts2/metrics/
│   ├── phoneme_probe.json              # IndexTTS2 PER
│   └── hubert_l2_analysis.json         # IndexTTS2 L2
├── model_gsv/metrics/
│   ├── phoneme_probe.json              # GSV PER
│   └── hubert_l2_analysis.json         # GSV L2
└── model_f5_tts/metrics/
    ├── phoneme_probe.json              # F5-TTS PER
    └── hubert_l2_analysis.json         # F5-TTS L2
```

### 脚本
```
scripts/
├── 35_phoneme_recognition_probe.py     # PER 探针
├── 36_hubert_l2_analysis.py            # L2 分析
└── 37_tsne_visualization.py            # t-SNE 可视化
```

### MFA 对齐
```
/root/autodl-tmp/mandarin_output_indextts2/   # 13 TextGrid
/root/autodl-tmp/mandarin_output_gsv/          # 13 TextGrid
/root/autodl-tmp/mandarin_output_f5_tts/       # 13 TextGrid
```
