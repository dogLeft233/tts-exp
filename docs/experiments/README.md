# 实验索引

项目入口见 `CONTEXT.md`。本目录按阶段编号的实验报告（01-15）与按主题的总结文档。

## 主实验线（按阶段编号）

| # | 实验 | 核心结论 | 状态 |
|---|---|---|---|
| 01 | [TTS-TFG 基线（Ditto）](01-tts-tfg-baseline.md) | TTS > Natural 唇形同步（n=13 中文） | ✅ |
| 02 | [TFG 跨模型对比](02-tfg-cross-model.md) | 效应跨模型：Wav2Lip > IMTalker > 其余 | ✅ |
| 03 | [因果机制分析](03-causal-mechanism.md) | 机制探索：单一声学特征无法解释 | ✅ |
| 04 | [英语复现](04-english-replication.md) | 英文不成立（LibriSpeech/HDTF 均 ns） | ✅ |
| 05 | [情绪/强度/多 TTS 模型](05-emotion-intensity.md) | 情绪与强度对效应的影响 | ✅ |
| 06 | [2026 TFG 模型调查](06-2026-tfg-model-survey.md) | 最新 TFG 模型候选综述 | ✅ |
| 08 | [参考帧选择](08-reference-frame-selection.md) | 固定中性参考帧方案（ADR-002） | ✅ |
| 09 | [AISHELL-1 100 样本音素扩展](09-aishell100-phoneme-extension.md) | n=100 音素可分性：35/126 FDR 显著，TTS 段更稳定/边界更锐；旧"类间 TTS 优"为均匀切片假象 | ✅ |
| 10 | [深度调研：下一步方向](10-next-steps-research.md) | 文献支撑的方向 + B1-B5 实验方案（带引用） | ✅ |
| 11 | [B1 时长分布 JSD](11-duration-jsd-analysis.md) | TTS 更快（−19%）、停顿少 63%、句内时长更均匀（dur CV d=0.42） | ✅ |
| 12 | [B2 逐层可分离性曲线](12-layer-curves.md) | HuBERT probe 峰 l6-8；XLS-R 峰中间带 l8-10、顶层退化；早层 delta 最强 | ✅ |
| 13 | [B3 逐音素 KLD + 判别探针](13-per-phoneme-kld.md) | 早层（l1）逐音素判别力最强（AUC 0.87-0.95）；KLD↔判别力相关弱、元音更敏感 | ✅ |
| 14 | [B5 说话人/音素解纠缠](14-ph-spk-disentangle.md) | RV(Ph,Spk)≈0.01 与随机不可区分——音素结论非说话人混杂 | ✅ |
| 15 | [B4 表征→TFG 关联](15-tfg-link.md) | **n=100 头号结论**：TTS Sync-C +1.02（d=1.24, 92%）；可分离性指标均不预测 lip-sync（FDR 后） | ✅ |
| 16 | [phone-local-warp 目标验证](16-phone-local-warp-validation.md) | **V2 No-Go**：raw TTS Sync-C +0.418，但 phone-local-warp −0.488；高 coverage 子集仍退化 | ✅ |
| 17 | [TTS 特征迁移与 natural 节奏对齐调研](17-tts-feature-rhythm-alignment-research.md) | **显式 duration-controlled acoustic transfer** 比 waveform splice 更有证据支持；属性解耦仍需 factorial 验证 | ✅ |
| 18 | [TTS 特征控制与 natural 节奏对齐验证](18-tts-feature-control-validation.md) | **n=10 Ditto/SyncNet：loudness 最接近 natural（ΔSync-C −0.058），但没有 control 恢复 raw TTS 优势；spectral 有 6/10 正向但均值 −0.226** | ✅ |
| 19 | [MDC English natural-vs-F5-TTS 音素与表征审查](19-mdc-english-representation-audit.md) | **50 对表征审查完成：TTS 多层 phone/viseme 可迁移性与时长规整性更强；B4 downstream TFG/SyncNet 仍阻塞** | ✅ |
| 20 | [AISHELL-100 跨条件音素/视位 PER](20-aishell100-cross-condition-per.md) | **补齐 100 对中文跨条件 frame-level PER：L6 phoneme natural→TTS 38.97%、TTS→natural 40.70%；TTS 略优且迁移对称** | ✅ |

## 专题/总结文档

| 文档 | 内容 |
|---|---|
| [summary.md](summary.md) | 多实验综合分析（n=9-12 旧数据，**部分结论已被 09-15 更新/推翻**） |
| [summary-7-31.md](summary-7-31.md) | TFG/THG 模型、HDTF 与多语言实验摘要 |
| [presentation_report.md](presentation_report.md) | 机制探究完整报告（含均匀切片假象修正） |
| [tts_tfg_mechanism_report.md](tts_tfg_mechanism_report.md) | TTS Enhancement of TFG 机制分析 |
| [tfg_model_comparison_summary.md](tfg_model_comparison_summary.md) | 多 TFG 模型对比总报告 |
| [multimodel_tts_comparison.md](multimodel_tts_comparison.md) | 多 TTS 模型音素特征对比 |
| [wav2sem_style_analysis.md](wav2sem_style_analysis.md) | Wav2Sem 风格特征分析 |
| [phoneme_recognition_results.md](phoneme_recognition_results.md) | 音素识别迁移（最近质心探针，MFA 修正） |
| [near_homophone_analysis_results.md](near_homophone_analysis_results.md) | 中文近同音词特征空间分析 |
| [english_wav2sem_fp_fs_summary.md](english_wav2sem_fp_fs_summary.md) | 英文 Wav2Sem Fs/Fp 实验总结 |
| [separability_metrics_formulas.md](separability_metrics_formulas.md) | 分离度度量公式参考 |

## 阅读建议

- **快速结论**：`CONTEXT.md` → `15`（n=100 头号结果）→ `09`（音素扩展）
- **机制理解**：`12`（层规律）→ `13`（逐音素差异）→ `15`（关联分析负结果）
- **方法学**：`11`（韵律指标）→ `14`（解纠缠控制）→ `13`（逐音素模板）
- **历史**：`01-08` + 各 summary 文档（注意部分结论已被 09+ 更新）
