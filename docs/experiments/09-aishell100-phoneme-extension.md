# 09 — AISHELL-1 100 样本音素扩展实验

## 问题

原 Wav2Sem 中文音素实验只有 n=12 条配对语音（`data/wav2sem_analysis/`），统计功效不足，24 个检验仅 2 个 FDR<0.05 显著，且对齐曾依赖整句均匀切分。核心问题是：**扩充数据量后，natural-vs-TTS 的音素表征差异是否更稳定可见？**

## 方法

- **数据**：AISHELL-1 test split 100 条（16 位说话人），与自然语音**同文本**生成 100 条 TTS（`faster_qwen3` 0.6B ICL self-clone）
- **对齐**：MFA `mandarin_mfa`（声学）+ `mandarin_china_mfa`（词典），产出 natural/TTS 各 100 个真实词/音素边界 TextGrid；**未使用整句均匀切分**
- **表征**：HuBERT base（768d）+ XLS-R 53（1024d），层 0/6/11/12
- **指标**：intra/inter-class dist、fisher ratio、silhouette、phoneme probe（GroupKFold 逻辑回归）、boundary sharpness、segment stability；per-sample 配对 permutation + BH-FDR
- **范围**：仅音素/表征分析，不涉及 Ditto/SyncNet/TFG

## 结果

### HuBERT viseme probe（越高=音素/viseme 可解码性越好）

| layer | natural | TTS (faster_qwen3) | Δ |
|:---:|:---:|:---:|:---:|
| 0 | 0.849 | 0.863 | +0.013 |
| 6 | 0.939 | 0.953 | +0.014 |
| 11 | 0.932 | 0.948 | +0.016 |

每个层 TTS 探针都略高于 natural。

### 显著性

- 126 个 natural-vs-TTS 配对比较，**35 个 FDR<0.05 显著**
- 高亮：`segment_stability`（l0/l6，|d|≥0.55）、`boundary_sharpness`（l6/l11，d≈−0.59~−0.69）、`intra_class_dist`（l11）、`silhouette`（l0）
- 对比 n=12：2/24 显著 → 现 35/126 显著，方向一致

## 结论

**扩到 100 样本后，TTS 与自然语音在音素/viseme 表征上的差异更稳定可见**，方向与 12 样本一致。TTS 对音素信息的保持不弱于自然语音（probe 略高），但在边界锐度、段稳定性上显著改变。数据量不足（而非无效应）是原实验现象不明显的主因之一。

## 修复的问题

1. **probe 全 NaN**：`scripts/16_feature_separability.py` 用 `np.full(..., dtype=str)` 建 GroupKFold 分组，把 `aishell1_N` 截断成单字符 `a`，100 组全合并 → 改 `dtype=object`。
2. **XLS-R probe 过慢**（1024 维逻辑回归单条约 13 分钟）：`_compute_metrics_for_pool` 加 `do_probe` 开关，XLS-R 显式跳过探针（JSON 记 `null`），其余指标完整。

## 状态

✅ **已完成** — 产物已回传本地 `results/aishell100_phoneme/`

## 相关文件

- `results/aishell100_phoneme/REPORT.md` — 摘要
- `results/aishell100_phoneme/metrics/separability_metrics.json` — 42 结果 / 126 配对比较
- `results/aishell100_phoneme/mfa_full/out/{natural,tts}/` — 各 100 个 MFA TextGrid
- `results/aishell100_phoneme/alignment/manifest.json` — 200 条对齐 manifest
- `scripts/prepare_aishell1_100.py` — corpus 下载/选择
- `scripts/build_mfa_manifest_100.py` — TextGrid→manifest
- `scripts/configs/aishell1_100_zh.yaml` — 隔离配置（audio/fixed_image/canonical transcript）
