# 12 — B2 逐层可分离性曲线：HuBERT / XLS-R 最优层标定

## 问题

09 号实验只用了离散层 {0,6,11,12}（HuBERT 实际存 0/6/11）。深度调研（10 号 A1）指出**层选择是系统性决定因素**：HuBERT 族音素信息集中在高层，wav2vec2/XLS-R 族在**中间层达峰、高层下降**（Pasad et al., ICASSP 2023）。本实验用稠密层网格标定每个模型的最优层，验证该规律。

## 方法

- **数据**：本地重新提取全部 400 个 embedding（200 样本 × 2 条件 × 2 模型）。HuBERT 层 0-11（全 12 层），XLS-R 隔层 0,2,...,22（12 层）。模型从 HF 缓存加载（HuBERT base 768d / XLS-R 53 1024d）
- **脚本**：`scripts/37_layer_curves.py` — 每 NPY 只读一次，按层池化 per-token 向量（声母/韵母/viseme 三种层级），复用 16 号的指标函数（intra/inter/fisher/silhouette + HuBERT 线性探针），输出每层 natural/TTS 曲线 + 峰值 + natural-tts delta
- **峰值方向**：intra_class_dist 取最小（越紧凑越好），其余取最大
- **退化层门控**：inter 与 intra 均 < 1e-3 判定为表征坍缩层，排除在峰值与 delta 之外
- **交叉验证**：本地提取与远端原始运行在重叠层（0/6/11）数值逐位一致

## 结果

### HuBERT probe 峰值（三个层级一致在中高层）

| 层级 | natural 峰 | TTS 峰 | 基线（多数类） |
|---|---|---|---|
| viseme | l7 (0.945) | l8 (0.959) | 0.843 |
| initial（声母） | l6 (0.765) | l6 (0.807) | — |
| final（韵母） | l6 (0.732) | l6 (0.770) | — |

### XLS-R separability 峰值（中间带）

| 层级 | fisher natural | fisher TTS | inter natural | inter TTS |
|---|---|---|---|---|
| viseme | **l10** (5.115) | l10 (6.274) | l8 (0.035) | l8 (0.039) |
| initial | l16 (6.436) | l16 (7.273) | l6 (0.071) | l6 (0.073) |
| final | l20* (7.341) | l20* (7.974) | l6 (0.116) | l6 (0.117) |

\* l20 的 fisher 高是 intra 分母缩小（0.061）的**比值假象**，不是真可分离性；稳健表述是"XLS-R 中间带 l8-16 达峰"。

### 顶层退化

- **XLS-R l22 完全坍缩**：intra≈5e-5、inter≈7e-7（所有 token 表征几乎相同），silhouette −0.16/−0.26。l20 部分坍缩（inter 0.005 vs 健康层 ~0.03）
- **HuBERT 无退化层**（l11 为可用最深层，正常）

### natural-vs-TTS delta 层分布

TTS 在所有层的 probe/fisher/inter 均 ≥ natural（方向一致），但 |Δ| 最大层因指标而异：

| 指标 | HuBERT | XLS-R |
|---|---|---|
| probe | 峰 l7-8，**Δ 峰 l0-2（早层）** | —（未算 probe） |
| fisher_ratio | 峰 l4-6，Δ 峰 l3-4 | 峰 l10-20，Δ 峰 l14-16 |
| inter_class_dist | 峰 l10，Δ 峰 l4 | 峰 l6-8，Δ 峰 l4-8 |

即：**检测 TTS-vs-natural 效应与最大化可分离性不是同一层**（尤其 HuBERT probe：绝对值峰值在 l7-8，差异峰值在 l0-2 早层）。

## 结论

1. **HuBERT 音素信息在中间-高层（l6-8）达峰，三个层级一致**——支持"HuBERT 族高层承载音素信息"，但非最顶层 l10-11
2. **XLS-R 可分离性在中间带（l8-16）达峰，顶层 l20-22 退化坍缩**——直接验证文献"wav2vec2 族中层达峰、高层下降"（Pasad et al. 2023）
3. **09 号实验的层选择需要修正**：XLS-R 用 l12（接近退化带）是次优的；应用中间层（l8-10）。HuBERT 的 0/6/11 合理但可加 l7-8
4. **TTS vs natural 的差异方向在所有层稳定**（TTS separability ≥ natural），效应检测的最佳层是早-中层，与绝对峰值层不同

## 审查修复（对抗子 agent 审查）

| # | 问题 | 修复 |
|---|---|---|
| C1 | XLS-R l22 退化层被当成 intra "峰值"（值 0.000） | 退化层门控（inter+intra<1e-3），排除在峰值/delta 外 |
| M1 | probe 峰值在 ~0.15pp 运行噪声内（l7 vs l8 差距≈噪声） | 报告多数类基线 + 用"平台 l6-8"表述，不宣称精确 argmax |
| M2 | viseme 4 类、84% 集中，probe 仅比基线高 +0.10 | 增跑 initial/final 层级（24/35 类）作为更严格证据 |
| M3 | 跨 100 说话人池化混杂说话人身份 | 绝对值峰值标注"可能说话人混杂"；delta 因条件配对而稳健 |
| M4 | 15 号脚本 metadata 记录请求层而非存储层（潜错标） | 修 15 号记录实际存储层；就地修补已有 JSON |
| M5 | max_abs_delta_layer 符号盲 + 未经检验 + 陈述过度 | 按指标分别报告 Δ 峰层；注明是点估计 |
| m1 | NaN 可进入峰值选择 | `_clean_metric` 过滤 NaN |
| m2 | 半帧池化偏移 | 内部一致，注明惯例 |

## 状态

✅ **已完成** — 脚本 `scripts/37_layer_curves.py`，产物 `results/aishell100_phoneme/layer_curves*.json`（viseme/initial/final）

## 相关文件

- 脚本：`scripts/37_layer_curves.py`、`scripts/15_extract_ssl_embeddings.py`（层元数据修复）
- 产物：`results/aishell100_phoneme/layer_curves/`（viseme）、`layer_curves_initial/`、`layer_curves_final/`
- 前置：`docs/experiments/09-aishell100-phoneme-extension.md`、`docs/experiments/10-next-steps-research.md`
