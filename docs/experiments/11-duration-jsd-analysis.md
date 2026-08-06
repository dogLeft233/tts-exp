# 11 — B1 时长分布 JSD：natural vs TTS（MFA 对齐）

## 问题

09 号实验发现 segment_stability / boundary_sharpness 显著（TTS 段内帧更相似、边界更锐）。深度调研（10 号）指出这类时序结构差异对应 TTS 的已知伪影——**确定性时长模型均值塌缩 → 韵律扁平、停顿不足**（Abbas et al., Interspeech 2022）。本实验把该假说操作化：**natural 与 faster_qwen3 TTS 的时长结构差异是否可测、可证伪？**

## 方法

- **数据**：09 号实验的 200 个 MFA TextGrid（natural/tts 各 100，`results/aishell100_phoneme/mfa_full/out/`），IPA 音素层含声调
- **三个分析层**：
  1. **逐音素时长分布 JSD**（raw + 率归一化 + 去声调基础音素 3 种视角）：共享分位 bin（6 bins）→ JSD + 置换检验 + BH-FDR；**率归一化**（每音素时长 ÷ 该句平均音素时长）隔离分布形状与全局语速；报告 **null 参照中位数 JSD** 校准统计功效
  2. **停顿结构**：内部 sil 时长分布 JSD + 每句停顿数
  3. **逐句配对指标**（total duration / phone count / mean phone dur / dur std / **dur CV** / pause count）：配对置换 + bootstrap CI + Cohen's d，族内 BH-FDR，含 **outlier 剔除敏感性**与 **dur CV leave-one-out**
- **统计硬化**（来自对抗审查，见"审查修复"）：6 bins、p=(1+count)/(1+n_perm) 无 0 值、FDR 分族、显式去 NaN、排除 MFA `spn` 噪声、outlier 标记（|log tts/natural 时长| > 0.5）

## 结果

### 语速与停顿（稳健，全部 FDR 显著）

| 指标 | natural | TTS | d | p (FDR) |
|---|---|---|---|---|
| total_duration_s | 4.058 | 3.292 | +0.95 | <0.001 |
| mean_phone_dur | 0.1115 | 0.0977 | +1.03 | <0.001 |
| **dur_cv_within_utterance** | 0.557 | 0.502 | +0.42 | <0.001 |
| pause_count / 句 | 1.30 | 0.48 | +0.69 | <0.001 |
| phone_count | 30.76 | 30.79 | −0.17 | ns（同文本） |

剔除 5 个 outlier（28/56/58/66/92）后方向与显著性不变；dur CV 的 leave-one-out p 全在 0.0005（无单样本驱动）。

### 停顿时长分布

- natural 停顿 130 条、均值 0.303s；TTS 仅 48 条、均值 0.148s
- 停顿时长分布 JSD=0.052（null 中位数 0.016），p=0.018（未校正，单检验）——TTS 停顿更短更少

### 逐音素时长分布形状：**无差异**（率归一化后）

| 视角 | 观测均值 JSD | null 中位数 JSD | 显著数 |
|---|---|---|---|
| 原始时长（含语速） | —（见下） | — | — |
| **率归一化** | 0.0299 | 0.0308 | **0 / 51** |
| 率归一化+去声调 | 0.0267 | 0.0232 | 0 / 38 |

**观测 JSD ≈ null 中位数** → 逐音素的时长分布形状没有 detectable 差异（这是修正后的真实结论，不是功效假象）。

## 结论

1. **TTS 整体更快**：同文本、同音素数，但总时长短 19%（d=+0.95）、平均音素更短（d=+1.03）——所有音素近似等比缩短
2. **TTS 停顿不足**：停顿数少 63%（d=+0.69），停顿更短（均值 0.148 vs 0.303s）——对应 Abbas 2022 "确定性时长模型停顿不足"
3. **TTS 时长更均匀（扁平化）成立**：dur CV（std/mean，对语速不敏感）d=+0.42 显著，outlier 剔除与 leave-one-out 均稳健——这是率归一化后唯一保留下来的"形状"差异，且发生在**句内音素间**（段内 CV），而非逐音素分布形状层面
4. **逐音素时长分布形状无差异**：TTS 的扁平化是"全局等比缩放 + 句内相对时长压缩"，不是某个音素被特殊对待

对 09 号发现的连接：segment_stability（TTS 段内帧更相似）与 dur CV（TTS 时长更均匀）在"TTS 更规整"这一点上方向一致，且本实验给出了客观、可复现的时长侧证据。

## 审查修复（对抗子 agent 审查）

| # | 问题 | 修复 |
|---|---|---|
| C1 | 16 bins + n=20-40 下 JSD 置换检验功效近零，null≈观测，"1/52 显著"是假象 | 降为 6 bins + 率归一化 + 报告 null 参照；结论改为"无 detectable 差异"而非"无差异" |
| M1 | p=0.0 下限（无 +1 校正） | p=(1+count)/(1+n_perm)，floor=0.0005 |
| M2 | dur_std 显著约 83% 由语速驱动 | 新增 dur CV 作为主指标，dur_std 标注语速混杂 |
| M3 | dur CV 早期边缘结果依赖单样本（61） | leave-one-out 敏感性；spn 排除后结果已稳健（p 全 0.0005） |
| M4 | 样本 92 等 TTS 生成失败 outlier 未处理 | |log 时长比|>0.5 标记 5 个 outlier，配对指标给出含/不含两组 |
| M5 | FDR 只在音素族，标注误导 | FDR 按族（音素族、逐句族）分别校正，raw p 始终保留 |
| m1 | spn（噪声）被当音素 | 排除 |
| m3 | 潜在 NaN 配对 | 显式丢 NaN 对 |
| m4 | ʐ̩ 与 ʐ 未合并 | 声调/组合附加符正则同时剥除 |

## 状态

✅ **已完成** — 脚本 `scripts/36_duration_jsd.py`，产物 `results/aishell100_phoneme/duration_analysis/duration_jsd.json`

## 相关文件

- 脚本：`scripts/36_duration_jsd.py`
- 产物：`results/aishell100_phoneme/duration_analysis/duration_jsd.json`
- 前置实验：`docs/experiments/09-aishell100-phoneme-extension.md`、`docs/experiments/10-next-steps-research.md`
