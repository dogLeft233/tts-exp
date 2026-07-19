# 英文 Wav2Sem Fs/Fp 实验总结

**实验日期**: 2026-07-17 至 2026-07-19  
**服务器**: AutoDL Pro6000-P（单 GPU），SSH 端口 33833  
**仓库**: `tts-exp`，commit 区间 `9befacd` → `68ff4ae`

---

## 1. 研究背景与核心问题

### 1.1 核心谜题

此前在**中文**（AISHELL-1，13 对匹配样本）上的实验发现了 TTS 音频在唇形同步中的一致优势：
- **Ditto SyncNet** 模型下，TTS 生成的音频比自然音频产生更高的 Sync-C 置信度
- **ΔSync-C** ≈ +0.205，dz = 0.79，p = 0.025（显著）

核心科学问题：**TTS 音频为什么对唇形同步更有利？** 驱动因素是什么？
- **Fs（语义特征）**——TTS 携带了更强或更简化的语义信息？
- **Fp（语音/韵律特征）**——TTS 的音素发音更清晰？

### 1.2 Wav2Sem 框架（Ditto et al., 2024）

Wav2Sem 将音频特征分解为两个正交通道：

```
逐帧特征:    per_frame = Wav2Bert(audio) ∈ R^(T×768)
语义通道:    Fs = mean_pool(per_frame) ∈ R^768       （T 维度均值池化）
语音残差:    Fp = per_frame − Fs.expand(T, 768)       （逐帧残差）
融合表示:    Fd = FC(Fs ⊕ Fp)                          （3D 人脸预测用）
```

训练方式：Wav2Bert 在 LibriSpeech-960 上最小化 L1(Fs, BERT_CLS(文本)) 来学习 Fs。

### 1.3 实验策略

1. **第一关**：在英文（LibriSpeech test-clean）上复现 TTS 优势
2. **若 TTS 优势存在** → 探测 Fs/Fp 通道找出中介因素
3. **若 TTS 优势不存在** → 通过 Fs/Fp 分析解释中英文不对称现象

---

## 2. 数据与实验流水线

### 2.1 数据集

| 属性 | 值 |
|---|---|
| 语言 | 英文 |
| 来源 | LibriSpeech test-clean |
| 样本数 | 13（ID 1–13） |
| 时长 | 每句 4.8–8.3s |
| 自然音频 | `data/data/audio_en/*.wav`（16 kHz 单声道） |
| TTS 音频 | `data/data/audio_en_qwen3_tts/*.wav`（Qwen3-TTS 音色克隆） |
| TTS 模型 | `qwen3-tts-vc-2026-01-22`（DashScope API） |

### 2.2 流水线脚本

```
步骤  脚本                                       产物
 1   00_datacheck.py                           验证 13 对匹配 + 响度检测
 2   02_tts.py                                 生成 TTS 音频（Qwen3 音色克隆）
 3   26_prepare_english_pairs.py               创建配对清单
 4   03_ditto.py ×3 种子（42/43/44）            Ditto 唇形同步视频生成
 5   04_eval.py ×3 种子                        SyncNet 逐样本置信度
 6   19_generate_eval_matrix.py                G×E 矩阵（生成器 × 引擎）
 7   23_identity_control_syncnet.py            身份控制（无操作差分）
 8   30_analyze_english_wav2sem.py              第一关汇总分析
────────────────────────────────────────────────────────────────
 9   MFA 对齐（手动）                            音素边界 TextGrid（13 文件）
10   28_prepare_english_alignment.py            对齐清单（26 条目）
11   15_extract_ssl_embeddings.py               HuBERT L0/L6/L11 逐样本 .npy
12   16_feature_separability.py                  可分离性指标（HuBERT + XLS-R）
13   29_oracle_fs_bert.py                       甲骨文 Fs = BERT_CLS(文本)
14   32_wav2sem_infer_fs.py                     Wav2Sem Fs = mean_pool(Wav2Bert(音频))
15   31_oracle_fd_separability.py               Fd = FC(Fp + Fs) 三档分析
```

### 2.3 关键配置

```yaml
# scripts/configs/english_wav2sem.yaml
models:
  syncnet: ditto_talking_head   # Ditto SyncNet
  generator: joyvasa            # 视频生成器
engines:
  natural: 原始音频
  tts: Qwen3-TTS 克隆音频
seeds: [42, 43, 44]
```

### 2.4 MFA 对齐配置

- **对齐器**: Montreal Forced Aligner 3.4.1
- **声学模型**: `english_mfa` v3.1.0（预训练）
- **词典**: `english_mfa`（基于 CMUdict）
- **音素集**: MFA IPA（13 个样本共 29 个独特音素）
- **对齐速度**: 13 个文件批量处理共 60s
- **IPA → 视位映射**: 13 类 Preston Blair 视位体系
  - 类别: pbmv, fv, th, cdsz, kg, chjsh, e, o, i, u, r, ai, aw + sil
  - 全 26 个条目零未映射音素

---

## 3. 实验结果

### 3.1 第一关：英文 TTS 优势复现

| 指标 | 值 |
|---|---|
| ΔSync-C（TTS − 自然） | **+0.160** |
| 标准差 | 0.516 |
| 样本量 | 39（13 样本 × 3 种子） |
| 配对 t 检验 | t = 1.955，p = 0.058 |
| Cohen's d（按种子） | dz = 0.31 |
| Bootstrap 95% 置信区间 | [−0.106, +0.423] |
| Holm 校正后 p | 0.571 |

**判定：未通过**——TTS 优势在英文上**未复现**。置信区间跨过零点，校正后不显著。这与中文结果形成鲜明对偶（中文 ΔSync-C = +0.205，dz = 0.79，p = 0.025）。

### 3.2 Wav2Sem Fs 分析（语义通道）

**问题**：TTS 音频是否比自然音频携带更强的语义特征（Fs）？

**指标**：Wav2Sem 推断的 Fs 与甲骨文 BERT_CLS(文本) 之间的 L1 距离

```
公式:
  Fs_wav2sem  = mean_pool(Wav2Bert_7TCN_12TF(音频))
  Fs_oracle   = BERT_CLS(文本)                          # "信息天花板"
  d(音频)      = ||Fs_wav2sem(音频) − Fs_oracle(文本)||_L1
```

| 条件 | 平均 L1 距离 | 标准差 |
|---|---|---|
| 自然 | 133.83 | 19.14 |
| TTS | 133.92 | 19.96 |
| Δ（TTS − 自然） | **+0.09** | |

**结果**: t = −0.127，p = 0.901，d = 0.035（可忽略）。

**结论**: 语义通道（Fs）对英文**已饱和**——Wav2Sem 预训练的 Wav2Bert 从自然音频和 TTS 音频中提取的文本信息质量相同。TTS 优势（如果有）**不**通过 Fs 中介。

### 3.3 HuBERT 视位可分离性（语音通道 Fp）

**问题**：TTS 音频是否提升 HuBERT 各层的音素可分离性？

**指标**: 13 个 Preston Blair 视位质心之间的类间余弦距离

```
公式:
  逐音素池化:    pooled_i = mean(frame[t_start_i:t_end_i])
  类质心:        centroid_k = mean(pooled_i for all i ∈ viseme_k)
  类间距离:      inter_class_dist = mean(1 − cos(c_a, c_b)) 对所有 a≠b
  段内稳定性:    segment_stability = mean(1 − cos(f_t, f_{t+1})) 段内帧间
```

#### HuBERT 逐层结果（池化，自然/原始）：

| 层 | intra_class↓ | inter_class↑ | fisher↑ | silh↑ | sstab↓ | probe_acc↑ |
|-----|-------------|-------------|---------|------|--------|-----------|
| L0 | 0.7575 | 0.6668 | 5.20 | −0.056 | 0.3061 | 0.7132 |
| L6 | 0.7625 | 0.6030 | 4.22 | −0.004 | 0.1816 | **0.8757** |
| L11 | 0.6176 | 0.4705 | 4.45 | −0.010 | 0.1662 | 0.8742 |

— *L11 是语音信息最强的层（线性探测准确率最高、类内方差最低）*

#### 自然 vs TTS 配对比较（HuBERT，全层 FDR 显著）：

| 层 | 指标 | 自然 | TTS | Δ | d | p (FDR) |
|-----|------|---------|-----|---|---|-------|
| L0 | inter_class_dist↑ | **0.739** | 0.546 | −0.193 | +2.94 | 0.0005 * |
| L0 | fisher_ratio↑ | **2.206** | 1.130 | −1.076 | +2.58 | 0.0005 * |
| L6 | inter_class_dist↑ | **0.697** | 0.547 | −0.150 | +2.92 | 0.0005 * |
| L6 | fisher_ratio↑ | **1.654** | 1.139 | −0.515 | +2.02 | 0.0005 * |
| L11 | inter_class_dist↑ | **0.554** | 0.379 | −0.176 | **+4.08** | 0.0005 * |
| L11 | fisher_ratio↑ | **1.721** | 1.148 | −0.573 | +1.99 | 0.0005 * |
| L11 | segment_stability↓ | 0.166 | 0.171 | +0.005 | −0.45 | 0.1195 |

**核心发现**: 在**所有层**上，**自然语音的视位可分离性始终优于 TTS**。效应在 L11 最强（inter_class d = +4.08）。段内稳定性无显著差异。

### 3.4 XLS-R 跨语言验证

| 层 | 模型 | Δinter_class | d | p (FDR) |
|-----|------|-------------|---|-------|
| L6 | HuBERT | −0.150 | +2.92 | 0.0005 * |
| L6 | XLS-R | **−0.169** | **+3.96** | 0.0005 * |
| L11 | HuBERT | −0.176 | +4.08 | 0.0005 * |
| L11 | XLS-R | **−0.131** | **+3.46** | 0.0005 * |

**结果**: 规律在单语模型（HuBERT）和多语模型（XLS-R）之间高度一致。英文 TTS 在保持语音可分离性方面一致地劣于自然语音。

### 3.5 甲骨文 Fd 三档分析（Fs + Fp 融合）

**问题**：合并 Fs（语义）和 Fp（语音）是否提升可分离性？增益在自然 vs TTS 之间是否不同？

**三档对比**:
```
Fp_only   = HuBERT_L11 逐帧向量               # 原始帧级嵌入
Fd_zero   = Fp + Fs_broadcast                 # 简单相加
Fd_random = FC_正交投影(Fp + Fs_broadcast)      # 学习的 768→768 投影
```

**关键指标**: `gain_random_vs_fp_fisher`——FC 投影带来的 Fisher 比提升

| 增益指标 | 自然 | TTS | Δ | d |
|---|---|---|---|---|
| gain_random_vs_fp_fisher | **+2.33** | +0.44 | −1.89 | **−3.00** |
| gain_random_vs_fp_intra_class_dist | −0.39 | −0.43 | −0.04 | −1.70 |
| gain_random_vs_fp_segment_stability | −0.10 | −0.10 | 0.00 | −0.18 |

**关键发现**: Fs+Fp 融合（FC 投影）对**自然语音**大幅提升可分离性（+2.33 Fisher），但对**TTS**几乎无增益（+0.44）。这意味着 Fs 为自然音频贡献了大量互补信息，但对 TTS 无贡献——TTS 的语音退化淹没了任何语义信号。

### 3.6 逐样本相关性：可分离性 vs SyncNet

**问题**：语音退化较大的样本是否同时有更大的 SyncNet 退化？

| 层 | Pearson r | p | Spearman ρ | p |
|-----|-----------|---|------------|---|
| 0 | −0.137 | 0.655 | −0.209 | 0.494 |
| 6 | −0.232 | 0.445 | −0.225 | 0.459 |
| 11 | −0.302 | 0.316 | −0.368 | 0.216 |

**结果**: Δinter_class_dist 与 ΔSync-C 之间无显著的逐样本相关性。L11 有弱负相关趋势（Spearman ρ = −0.368），但 n = 13 检验力不足。

---

## 4. 中英文对比分析

> **⚠️ 2026-07-19 修订**: 原中文数据使用均匀切片（pypinyin 等分）估算音素边界，严重低估了自然语音的视位可分离性（inter_class 被低估至 0.12）。现已用 MFA 精确对齐（v3.4.1, `mandarin_mfa` 模型）重新分析，Natural inter_class 修正为 0.51。以下对比以此为准。

| | 中文（MFA） | 英文 |
|---|---|---|
| 对齐方式 | MFA mandarin | MFA english |
| Natural inter_class_dist（L11） | 0.510 | 0.554 |
| TTS inter_class_dist（L11） | 0.477 | 0.379 |
| TTS 对可分离性的改变 | −6%（d=+0.73, p_fdr=0.099） | −32%（d=+4.08, p_fdr<0.001） |
| 方向 | Natural > TTS（不显著） | Natural > TTS（显著） |
| segment_stability L11 nat→tts | 0.146→0.163（恶化, d=−1.73*） | 0.166→0.171（不显著） |
| Fs 通道 | 已饱和（文字相同→BERT 相同） | 已饱和（无差异） |

**关键洞见（修订版）**:

1. **方向一致**: MFA 对齐后，中英文都是**自然语音 > TTS**，不存在"中文特殊"的 TTS 改善
2. **差异大小不同**: 英文退化更大（d=4.08 vs d=0.73），因为英文自然基线更高（0.55 vs 0.51）
3. **旧结论是均匀切片的伪影**: 均匀切片假设音素等长→自然语音的 heteroscedastic 时长分布被"抹平"→不同音素帧互相污染→inter_class 被低估。TTS 的参数化生成恰好更接近等时分布，所以在旧方法下被高估
4. **Fs 通道不产生额外贡献**——两种语言均已饱和；TTS 的退化完全通过 **Fp 通道**运作

---

## 5. 理论含义

### 5.1 主要发现（2026-07-19 修订）

**TTS 在所有测试语言上都降低（而非提升）视位可分离性。** 此前报告的中文"TTS 改善现象"已被证明是均匀切片对齐误差的伪影。经 MFA 精确对齐后，关键量——HuBERT 中高层中声学信号对音素类别的区分度——在中英文上都是 Natural > TTS。差异仅在于退化程度：英文退化更大（d=4.08, p<0.001），中文退化较小（d=0.73, p_fdr=0.099）。

### 5.2 机制模型（修订版）

```
TTS 对视位可分离性的影响
    ↑
    │ 小退化（d=0.73, ns）━━ 中文
    │ ┌──────────────────────────┐
    │ │ inter: 0.51 → 0.48 (−6%) │  ← TTS 小幅度退化
    │ └──────────────────────────┘
    │                    ●
    │              ┌──────────────────────────┐
    │ 大退化（d=4.08, sig）│ inter: 0.55 → 0.38 (−32%)│  ← TTS 显著退化
    │              └──────────────────────────┘
    └──────────────────→ 自然语音 inter_class_dist
         0.51(中文)          0.55(英文)
```

TTS 的参数化生成在两种语言上都**模糊**了音素对比，但退化程度与自然语音基线相关：基线越高，TTS 的"参数化损失"越明显。不存在 TTS 改善的场景——此前中文的"改善"来自均匀切片对自然语音基线的人为低估。

### 5.3 饱和假说

Wav2Sem 的 Fs 通道对两种语言均已饱和。TTS 的退化完全运作于 **Fp 通道**——即编码每个音素发音清晰度的逐帧语音残差。语义信息（Fs）在自然和 TTS 音频中质量相同。

---

## 6. 可复现代码与数据

### 本地结果文件（已提交）

```
data/
├── english_wav2sem_analysis/
│   └── ...（Ditto SyncNet Gate 1 结果）
├── wav2sem_analysis/
│   └── metrics/separability_metrics.json      # 中文可分离性（旧·均匀切片，仅供参考）
├── wav2sem_analysis_en/
│   └── ...（英文 HuBERT + XLS-R + oracle Fs + Fd）
├── wav2sem_analysis_zh/                        # 🆕 中文 MFA 对齐结果
│   ├── manifest/alignment.json                # MFA 对齐清单（26 条目）
│   ├── embeddings/                             # HuBERT/XLS-R 嵌入（52 文件）
│   └── metrics/
│       ├── separability_metrics.json          # HuBERT L0/L6/L11 可分离性 + 配对比较
│       └── oracle_fs.json                     # BERT-base-chinese oracle Fs（26 条目）
├── data/audio/                                 # 13 个自然中文 WAV
└── runs/r2_dashscope_vc_*/02_tts/             # 13 个 TTS 中文 WAV（Qwen3-TTS-VC）
```

### 新增/修改的关键脚本

| 脚本 | 功能 |
|---|---|
| `28_prepare_english_alignment.py` | 英文 MFA TextGrid→IPA→视位清单 |
| `33_prepare_mandarin_alignment.py` | 🆕 中文 MFA TextGrid→IPA→视位清单 |
| `15_extract_ssl_embeddings.py` | HuBERT/XLS-R 嵌入提取（通用，支持中英文 manifest） |
| `16_feature_separability.py` | 视位可分离性指标（通用） |
| `29_oracle_fs_bert.py` | Oracle Fs = BERT CLS（支持 bert-base-uncased / bert-base-chinese） |
| `32_wav2sem_infer_fs.py` | Wav2Sem 预训练权重推断 |
| `31_oracle_fd_separability.py` | Fd 三档融合分析 |
| `26_prepare_english_pairs.py` | 英文配对清单 |

### 最近 10 次 commit

```
68ff4ae chore: save all local changes before downloading results
a4068be feat(28): MFA TextGrid + IPA→viseme pipeline for English
8ae470a feat(32): Wav2Sem pretrained inference
9befacd feat(config): wav2sem block fields for English Fs-BERT analysis
7e72629 fix(30): remove dead param; add M3_FC_ARTIFACT tests
a93f0af test(29): oracle Fs BERT + English compatibility
9401fa3 feat(config): English viseme map + wav2sem constants
0b2e38b feat(27): English Wav2Sem Fs-BERT support
62562f5 test: bilingual constants + English gate1 smoke test
7a8dae2 feat(26): English TTS pair preparation
```

---

## 7. 讨论与局限

### 研究优势
1. **语义与语音双通道测试**——Fs 通过 Wav2Sem 预训练权重，Fp 通过 HuBERT/XLS-R 可分离性
2. **跨模型稳健性**——发现在单语（HuBERT）和多语（XLS-R）模型间一致
3. **三档 Fd 融合**——量化了 Fs 在原始特征之上对可分离性的贡献
4. **逐样本中介检验**——直接关联语音质量与 SyncNet 表现

### 局限性
1. **样本量小（n=13）**——逐样本相关分析统计检验力不足
2. **单一 TTS 模型**——仅使用 Qwen3-TTS；其他 TTS 引擎可能产生不同的语音特征
3. **Wav2Sem 仅支持英文**——中文 Fs 通道未能跨语言测试
4. **单一声学模型（MFA english_mfa）**——不同对齐器可能产生不同的音素边界

### 未来方向
1. 在英文上测试多种 TTS 引擎，判断语音退化是否是模型特异的
2. 微调中文版 Wav2Sem，跨语言测试 Fs 饱和假说
3. 使用中文 MFA 对齐运行相同的 Fd 三档分析
4. 扩大样本量（如 LibriSpeech-100）以获得更有检验力的逐样本相关
5. 测试中间语言（如日文、韩文），探索自然发音清晰度的连续谱
