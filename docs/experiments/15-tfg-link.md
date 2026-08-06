# 15 — B4 表征→TFG 关联：Ditto + SyncNet 在 n=100 下的 TTS-vs-natural 唇形同步

## 问题

项目核心问题：**TTS 生成的语音是否比自然语音产生更好的唇形同步（TFG）**？09 号之前的 n=13 实验发现 TTS > Natural（Ditto ΔSync-C=+1.25）。本实验在 n=100（aishell1_100 配对音频）上完整跑通 TFG 流水线，验证两点：(1) TTS-vs-natural 的唇形同步差异在 n=100 是否成立；(2) 逐样本的可分离性/prosody delta 能否预测逐样本的 SyncNet-C delta（表征→TFG 连接）。

## 方法

- **TFG 推理**：Ditto talking-head（**离线 cfg** `v0.4_hubert_cfg_trt.pkl`，固定参考帧）生成 100 natural + 100 tts 视频（RTX 4080 服务器）
- **SyncNet 评估**：200 视频 → sync_c（置信度，越高越好）、sync_d（最小距离，越低越好）；**统一 min_track=25**（修复短视频 track 阈值问题）
- **配对检验**：逐样本 natural-vs-TTS（同文本同参考帧，仅音频不同），符号翻转置换检验 + Cohen's d + bootstrap CI
- **关联分析**：189 个可分离性/prosody 指标的逐样本 delta（tts−natural）× SyncNet-C/D delta（tts−natural），Spearman + 去重（104 个唯一向量）+ 标准 BH-FDR

## 结果

### 头号结果：TTS 唇形同步显著更好（n=100 稳健复现）

| 指标 | natural | TTS | delta | p | d | CI (tts−nat) |
|---|---|---|---|---|---|---|
| **Sync-C** | 5.067 | 6.089 | **+1.022** | **<0.0001** | **1.24** | [0.86, 1.18] |
| Sync-D | 7.808 | 7.738 | −0.069 | 0.37 (ns) | −0.09 | — |

- **Sync-C：TTS 在 92% 的样本更好**；对统一 min_track=25 稳健（混阈值重跑结果一致）
- **Sync-D 无显著差异**（53% 样本改善）；仅 **51%** 样本两个指标同时改善——Sync-C 的改善主要是"置信度"维度，而非偏移距离维度

### 关联分析：无指标在 FDR 后显著

104 个去重指标，**0 个 BH-FDR<0.05 显著**。最强的未校正信号：

| 指标 | rho | p | q | CI |
|---|---|---|---|---|
| hubert_l6 initial inter_class_dist | +0.59 | 0.006 | 0.23 | [0.23, 0.82] |
| xlsr_l12 initial inter_class_dist | +0.58 | 0.008 | 0.23 | [0.09, 0.88] |
| xlsr_l0 viseme fisher_ratio | −0.55 | 0.010 | 0.23 | [−0.77, −0.16] |
| dur_total_duration_s | −0.27 | 0.006 | 1.00 | — |

模式：**inter_class_dist 正相关**（TTS 增大类间距的样本 → SyncNet-C 改善更大）、**fisher_ratio 负相关**、**TTS 语速加快与唇形同步改善正相关**（dur_total_duration −0.27）。方向一致但均过不了多重检验校正。

## 结论

1. **TTS 唇形同步在 n=100 稳健优于 natural（Sync-C +1.02, d=1.24, p<0.0001, 92%）**——项目核心结论从 n=13 扩展到 n=100，且对评估阈值稳健
2. **改善主要体现在 Sync-C（置信度）而非 Sync-D（偏移距离）**——需谨慎表述为"TTS 产生更可信的唇形同步"，而非"更精确的对齐"
3. **无单个可分离性/prosody 指标能显著预测逐样本的唇形同步变化**（FDR 校正后）——表征可分离性与唇形同步质量之间没有简单的逐样本线性关系；最强的方向性线索（inter_class_dist 正相关、fisher 负相关）提示类间可分性可能与唇形同步改善相关，但 n=100 下功效不足（|rho|<0.3 测不出）
4. 与 Wav2Sem（CVPR 2025）对比：该工作的干预式"语义解耦提高可分离性→改善唇形几何"在项目内未观察到逐样本对应关系——项目是相关性证据，Wav2Sem 是干预证据

## 审查修复（对抗子 agent 审查）

| # | 问题 | 修复 |
|---|---|---|
| M1 | 评估混用 min_track=100/25（短视频 TTS 占比高） | 统一 min_track=25 全量重评 200 个（--no-cache）；头号结果不变 |
| M2 | Sync-C 改善但 Sync-D 无 → 表述过强 | 头号结论改为"Sync-C（置信度）显著更好"；并列报告两个指标 |
| M3 | 关联 195 检验含 140 死指标/27 唯一向量，FDR 过度惩罚 | 去重（104 唯一向量）+ NaN 样本过滤；最强信号 q=0.23 |
| M4 | 单个 NaN 样本通过 rankdata 毒化整指标 | 逐样本 NaN 过滤 |
| M5 | BH-FDR 方向错误（left-to-right） | 改标准 right-to-left BH |
| S7 | 无 CI、无提交聚合脚本 | 加 bootstrap CI；聚合数据提交到 results/ |

## 状态

✅ **已完成** — 脚本 `scripts/40_link_separability_to_tfg.py`，产物 `results/aishell100_phoneme/tfg_link/`

## 相关文件

- 脚本：`scripts/40_link_separability_to_tfg.py`
- 产物：`results/aishell100_phoneme/tfg_link/syncnet_100.json`、`separability_tfg_link.json`
- 服务器流水线：Ditto（离线 cfg）+ SyncNet（统一 min_track=25）于 Seetacloud RTX 4080
- 前置：`docs/experiments/01-tts-tfg-baseline.md`（n=13 原始结论）、`09-aishell100-phoneme-extension.md`、`12-layer-curves.md`
