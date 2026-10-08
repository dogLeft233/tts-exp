---
title: README
type: note
permalink: tts-exp/docs/experiments/readme
---

# 实验索引

项目入口见 `CONTEXT.md`。本目录按阶段编号的实验报告（01–31，07 尚未占用）与按主题的总结文档。近期实验的记忆笔记另见 `basic-memory/Experiments/`；本索引中的早期结论应按后续证据更新。

## 主实验线（按阶段编号）

| # | 实验 | 核心结论 | 状态 |
|---|---|---|---|
| 01 | [TTS-TFG 基线（Ditto）](01-tts-tfg-baseline.md) | TTS > Natural 唇形同步（n=13 中文） | ✅ |
| 02 | [TFG 跨模型对比](02-tfg-cross-model.md) | 效应跨模型：Wav2Lip > IMTalker > 其余 | ✅ |
| 03 | [因果机制分析](03-causal-mechanism.md) | 机制探索：单一声学特征无法解释 | ✅ |
| 04 | [英语复现](04-english-replication.md) | 历史 LibriSpeech/HDTF 队列未复现；后续 LRS3 英文队列出现优势，不能概括为“英文不成立” | ✅（历史范围） |
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
| 21 | [MVP waveform 失败诊断与 provenance 审查](21-mvp-audio-forensics.md) | **严格 paired cohort 与 natural alignment 50/50 通过；MVP Mandarin MFA 与逐样本 SyncNet ledger 仍阻塞，overall partial** | ⚠️ |
| 22 | [TTS feature supervision 与 feature-domain enhancement head MVP](22-tts-feature-supervision-and-mvp.md) | **监督改为 train-only phone-conditioned HuBERT prototype 的弱表征目标；不把 feature metric 写成 TFG/SyncNet 结果** | 🚧 |
| 23 | [Ditto-native feature head downstream 可行性验证](23-ditto-native-feature-head-feasibility.md) | **Ditto 原生 frontend 为 25 fps / 1024-D；native adapter 的 PyTorch 离线注入路线可完成渲染与 SyncNet 评估，但历史单样本仍远低于 raw TTS；online 非因果注入与泛化尚未验证** | 🚧 |
| 24 | [MVP waveform HuBERT 编码器特征审计](24-mvp-hubert-encoder-audit.md) | **严格 50 对中，MVP 的 HuBERT layer 0/6/11/12 geometry 均明显更接近 natural 而非 raw TTS；尚未完成 Ditto-native 1024-D 或 phone-level audit** | ⚠️ |
| 25 | [Single-encoder HuBERT waveform enhancer](25-single-encoder-waveform-enhancer.md) | **冻结 HF HuBERT 条件化的 identity-initialized waveform enhancer；HF 768-D 只用于本地监督，增强 WAV 再由 Ditto native frontend 重编码** | 🚧 |
| 26 | [AISHELL-1 400 训练数据 inventory 与 proximity telemetry](26-aishell1-400-training-inventory.md) | **恢复 400 → 392 → 391 的中国 paired waveform inventory；记录 speaker-disjoint split、hash、拒绝 gate，并定义零额外 encoder call 的 train/valid HuBERT proximity 诊断** | 🚧 |
| 27 | [AISHELL-1 HuBERT waveform enhancer training](27-aishell1-hubert-waveform-training.md) | **351 对 fresh strict rebuild 完成 3 epoch 训练；telemetry 零额外 encoder call，但 enhanced→TTS proximity 未显示稳定整体改善；无 downstream/heldout 泛化结论** | ⚠️ |
| 28 | [Chinese HuBERT training failure diagnosis](28-diagnosis-chinese-hubert-training.md) | **主因确认：acoustic reconstruction 与 TTS HuBERT style gradient cosine −0.737，style gradient 约小 15.7×；style-only 可改善 proximity，full objective 被竞争目标压制** | ✅ |
| 29 | [Two-stage feature-targeted waveform renderer](29-two-stage-feature-targeted-waveform-model.md) | **校正 Stage-1 loss scale 后，两 seed 的 local train/valid 直连 raw-TTS feature pull 为 3.39%–3.72%；冻结目标的 Stage-2 realization 为 2.46%–2.58%，但仍有强 realization/preservation gradient conflict；非 heldout/downstream 结论** | 🚧 |
| 30 | [Replacement audio-head 验证](30-replacement-audio-head-validation.md) | unseen test 上泛化门槛失败（3/12），TTS-kv 贡献未确认；不能把单条训练记录收益推广 | ✅ |
| 31 | [LRS3 MFA-linear 评分音轨效应](31-lrs3-mfa-linear-scorer-effect.md) | 192 条、23 来源组的四格交叉评分不支持 SyncNet 普遍偏好 MFA-linear 音轨；分数受驱动与评分音轨匹配关系影响 | ✅ |

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

- **快速结论**：`CONTEXT.md` → `15`（AISHELL-1 n=100）→ `30`（增强头泛化负结果）→ `31`（评分音轨交互）；跨语料结论另见 `basic-memory/Experiments/跨数据集 TFG 测评（5×50 multiset）.md`
- **机制理解**：`12`（层规律）→ `13`（逐音素差异）→ `15`（关联分析负结果）
- **方法学**：`11`（韵律指标）→ `14`（解纠缠控制）→ `13`（逐音素模板）
- **中国 waveform 训练恢复**：先读 `26`，再核对 `22`（feature-domain MVP）与 `25`（历史 single-encoder enhancer）；`27`/`28` 解释该旧目标失败，`29` 记录严格两阶段替代的 local train/valid 结果。不要把历史 13-pair corpus 与 AISHELL-1 400 合并，也不要把 `29` 写成 heldout 或 Ditto/SyncNet 结论
- **历史**：`01-08` + 各 summary 文档（注意部分结论已被 09+ 更新）