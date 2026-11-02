---
title: README
type: note
permalink: tts-exp/docs/experiments/readme
---

# 实验索引

当前研究导航见 [[研究交接总览]]，本页保留编号与文件入口，不以编号先后表示当前优先级。编号原件已到84（84个文件；07空缺、25两个历史同号文件），按完整文件名区分，不重命名冻结原件。2026-11-02模块化全量入口为 [[研究交接总览]] 与 [[实验台账与核验口径]]；七模块逐条列出本目录全部116份报告/专题以及Experiments等来源。下方旧表按其原整理范围保留，未逐项扩行不表示后续实验不存在。PC-Talk/DyStream全量以本机最终凭据为准，计算/自动验收完成、正式报告与完整媒体QA待补；HDTF TalkLikeYou暂停不变。

**近期关键更新（2026-10-22）：** 噪声训练 67 已完成；56 的低尺度训练正向先导未在 67 的无添加组保留同向 C 差异。TTS 训练 64 必须连同 66 的时序退化审计阅读。综合边界见 [[低尺度训练与噪声解释的证据边界]]，历史路线见 [[历史研究路线与停止边界]]。

本目录包含正式报告与主题总结。旧条目的“完成”表示该阶段交付，不代表假设获支持；实际修正与专题结果还需搜索 `Experiments/`。

## 最新协议复核（2026-10-31）

- [[82-talklikeyou-syncnet-protocol-controls]]：固定中文TalkLikeYou视频。共同13（含主要12）中100/25同轨迹同分；全20原MP4／PCM ΔC +0.493／+0.495，音轨效应−0.002、95%[−0.098,+0.086]。中间AVI为MP3且统一69.0625ms延迟；单指标p跨.05但Holm9敏感性未过。259阶段及完整读回通过，不是感知结论。

## 主实验线（按阶段编号）

| # | 实验 | 核心结论 | 状态 |
|---|---|---|---|
| 01 | [TTS-TFG 基线（Ditto）](01-tts-tfg-baseline.md) | TTS > Natural 唇形同步（n=13 中文） | ✅ |
| 02 | [TFG 跨模型对比](02-tfg-cross-model.md) | 效应跨模型：Wav2Lip > IMTalker > 其余 | ✅ |
| 03 | [因果机制分析](03-causal-mechanism.md) | 机制探索：单一声学特征无法解释 | ✅ |
| 04 | [英语复现](04-english-replication.md) | LibriSpeech因泄漏排除独立推断；HDTF仅历史队列；后续 LRS3 英文队列出现优势，不能概括为“英文不成立” | ✅（历史范围） |
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
| 25（机制） | [增益驱动与局部谱形相容性](25-tts-tfg-mechanism.md) | 与旧增强器报告同号；幅度、编码方向及固定自然评分音轨的机制证据，限定 Wav2Lip／所测队列 | ✅（单独文件） |
| 26 | [AISHELL-1 400 训练数据 inventory 与 proximity telemetry](26-aishell1-400-training-inventory.md) | **恢复 400 → 392 → 391 的中国 paired waveform inventory；记录 speaker-disjoint split、hash、拒绝 gate，并定义零额外 encoder call 的 train/valid HuBERT proximity 诊断** | 🚧 |
| 27 | [AISHELL-1 HuBERT waveform enhancer training](27-aishell1-hubert-waveform-training.md) | **351 对 fresh strict rebuild 完成 3 epoch 训练；telemetry 零额外 encoder call，但 enhanced→TTS proximity 未显示稳定整体改善；无 downstream/heldout 泛化结论** | ⚠️ |
| 28 | [Chinese HuBERT training failure diagnosis](28-diagnosis-chinese-hubert-training.md) | **主因确认：acoustic reconstruction 与 TTS HuBERT style gradient cosine −0.737，style gradient 约小 15.7×；style-only 可改善 proximity，full objective 被竞争目标压制** | ✅ |
| 29 | [Two-stage feature-targeted waveform renderer](29-two-stage-feature-targeted-waveform-model.md) | **校正 Stage-1 loss scale 后，两 seed 的 local train/valid 直连 raw-TTS feature pull 为 3.39%–3.72%；冻结目标的 Stage-2 realization 为 2.46%–2.58%，但仍有强 realization/preservation gradient conflict；非 heldout/downstream 结论** | 🚧 |
| 30 | [Replacement audio-head 验证](30-replacement-audio-head-validation.md) | unseen test 上泛化门槛失败（3/12），TTS-kv 贡献未确认；不能把单条训练记录收益推广 | ✅ |
| 31 | [LRS3 MFA-linear 评分音轨效应](31-lrs3-mfa-linear-scorer-effect.md) | 192 条、23 来源组的四格交叉评分不支持 SyncNet 普遍偏好 MFA-linear 音轨；分数受驱动与评分音轨匹配关系影响 | ✅ |
| 32 | [MFA-linear 重建链与生成器拆分](32-mfa-linear-vocoder-wav2lip-split.md) | 重建链不足以单独解释 TTS／迁移输入差异；重建不等于纯声码器消融 | ✅（限定方案） |
| 33 | [内容、停顿、韵律与映射](33-mfa-linear-content-pause-prosody-mapping.md) | 缺失停顿有局部可修复信号，不能解释完整迁移差距 | ✅（局部诊断） |
| 34 | [静音与停顿修复](34-mfa-linear-silence-pause-repair.md) | 延续局部映射缺口检查；与 36 的后续队列边界合读 | ✅（限定方案） |
| 35 | [发声段残余对齐](35-mfa-linear-speech-residual-alignment.md) | 定位自然与迁移音频的局部差异；不能唯一归因时间或谱形 | ✅（诊断） |
| 36 | [新说话人发声段复核](36-mfa-linear-heldout-speech-residual.md) | 部分局部差异可重复，整套修复与单操作作用分开；非通用修复器 | ✅（限定队列） |
| 37 | [音素内时间重映射](37-mfa-linear-intraphone-time-warp.md) | 所测有界映射未通过科学门，当前方案停止 | ✅（NO_GO） |
| 38 | [局部 TTS 谱形修复](38-local-tts-spectral-repair.md) | 未确认目标位置优于匹配对照；不推广为所有谱形编辑无效 | ✅（限定方案） |
| 39 | [固定RMS/自然时钟的原生因子与物理谱形筛查](39-native-input-factor-mechanism-screen.md) | native 9格无双人复现；physical6行0joint、meanΔC+0.001；两处C增益均D0恶化，独立mouth运动方向混合 | ✅（冻结剂量/两人探索） |

| 40 | [构音与模型响应时间的功能坐标](40-functional-response-coordinate.md) | 同144旧轨迹参考RMS时间后height/width六格Cp/Hp过描述门；旧head无共同收益；输出功能坐标候选，seenDEV探索 | ✅（4人/8句，小样本诊断） |

| 41 | [原生构音方向与独立功能响应](41-native-articulatory-response.md) | 18三臂screen门1/6格；后续Muse直接暴露0/14帧，定位输入行→输出事件坐标缺口，旧数值保留/解释收窄 | ✅（两句seen-DEV screen） |

| 42 | [输出所有者与共享前端坐标](42-output-owned-native-coordinate.md) | Muse两句口部功能双向控制通过，plus full C下降；编码传播及Whisper共享切片约束定位native→waveform缺口 | ✅（两句seen-DEV，限定native控制） |
| 43 | [处理链可辨识性与SyncNet分差预测审计](43-processing-chain-predictive-audit.md) | 主74对AUC0.972但分差R²−0.140；legacy有有限信号，无稳定单族；LibriSpeech因泄漏仅诊断存档 | ✅（现有数据观察性审计） |
| 44 | [同文本独立发声的交叉失配对照](44-repeated-take-crossover.md) | 8来源32视频128格；自然/TTS类内损失3.650/4.275，跨类3.963/4.708；普通实例失配已复现强下降模式 | ✅（单Wav2Lip/本地Qwen pilot；非纯偏好/泛化） |

| 45 | [固定视频局部时钟正控与交叉校正](45-local-clock-crossover.md) | 正控3/4；5合格对C全部+1.770～+5.495；19/24越界，完整来源0/4，停在4人 | ✅（2来源合格子集；未扩8） |
| 46 | [纯幅度通路与新20人独立验证](46-amplitude-gain-independent-validation.md) | 160/160完成；FIXED主+0.049、14/20正、95%区间跨零；预定+3dB+0.067、15/20正、四剂量同时区间为正；口部响应与分解单列 | ✅（单Wav2Lip；旧40/15复核与新20分开） |
| 47 | [SyncTalk音量前处理与真实权重审计](47-synctalk-amplitude-frontend.md) | AVE保留增益；HuBERT近似抵消且有epsilon残差；DS只验证前端；May条件入口通过 | ✅（0渲染/0训练） |
| 48 | [May预训练checkpoint纯幅度推理](48-synctalk-may-gain-inference.md) | 固定原PCM，−6/0/+6 C8.906/9.262/9.373；0重复exact，视频有变化 | ✅（单身份12秒；0训练） |
| 49 | [May官方HuBERT增益复核](49-may-hubert-gain.md) | 同12秒±6dB hidden变化约0.009%，重复0；整段归一化近似抵消 | ✅（仅编码器；0视频/0训练） |
| 50 | [May等预算训练电平×测试电平](50-synctalk-may-gain-training-pilot.md) | 3×60000步；9条val曲线平台筛查通过，九格测试固定test0 C9.199/8.909/8.568；原PCM与全量独立QA | ✅（已暴露单身份/单seed/缩减第一阶段） |
| 51 | [May固定视频评价音量27格](51-synctalk-may-eval-gain.md) | 原评价0逐位复现；评价±6平均ΔC−0.026/−0.008，存在驱动×评价交互；108矩阵独立QA | ✅（0训练/0视频，单身份单seed） |
| 52 | [主流语音编码器方向共性](52-encoder-direction-commonality.md) | 11新编码器+Wav2Lip缓存/663共同音素；固定RMS的TTS共性依赖编码器，纯增益响应依赖原生路径；0新TFG | ✅ |
| 53 | [输入归一化同权重消融](53-encoder-input-normalization-ablation.md) | 8模型/512前向；7个关闭输入归一化后恢复增益方向响应；W2vBERT由后续逐帧LN消除log增益，0新TFG | ✅ |

| 54 | [May同权重归一化训练/推理](54-may-encoder-normalization-pilot.md) | 四臂各12000成功更新，24格/48矩阵；同编码器输入norm开关及两轴交叉，单May/单seed短程先导 | ✅ |

| 55 | [May训练/推理音量×归一化](55-may-gain-normalization-factorial.md) | 8新×12000更新+4旧0gain；72格/144矩阵同权重交叉，小预算单seed | ✅ |
| 56 | [May后归一化输入尺度](56-may-postnorm-scale.md) | wav2vec2 单次低尺度 C +0.092；67 新训无添加组方向相反；不能认定稳定收益 | ✅（需连读67） |
| 57 | [May波形归零与AVE频谱下限核查](57-may-waveform-floor-audit.md) | 历史浮点波形无新增零点；AVE降6dB有新增mel截断，未证明降噪或训练收益因果 | ✅（CPU机制核查，0训练） |
| 58 | [编码器尺度残差与epsilon逐层定位](58-encoder-layer-epsilon-diagnostic.md) | 首层GN引入尺度残差，epsilon平方补偿恢复逐值一致 | ✅（冻结编码器，0训练） |
| 59 | [May wav2vec2弱响应逐层核查](59-wav2vec2-weak-signal-layer-audit.md) | 全部14块中弱预方差通道轻微压低；无统一逐层降噪证据，44冻结前向 | ✅（0训练/TFG/SyncNet） |
| 60 | [首层GroupNorm互换诊断](60-encoder-groupnorm-swap.md) | GN换参明显改变工作点，未将尺度响应方向简单互换 | ✅（冻结编码器，0训练） |
| 61 | [May已知噪声编码器四组对照](61-may-known-noise-encoder-pilot.md) | 整体减半噪声扰动近不变，只降添加噪声则减小；449冻结前向 | ✅（0训练/TFG/SyncNet） |
| 62 | [保持原基线的 GroupNorm 响应诊断](62-gn-baseline-preserving-diagnostic.md) | 关闭 epsilon 响应可消除所测编码增益变化；等相对扰动后两编码器末层仍有差距，未闭合训练收益因果 | ✅（冻结编码器，0训练） |
| 63 | [May已知噪声强度扫描](63-may-known-noise-strength-scan.md) | 19档SNR、566次冻结前向；弱剂量片段依赖，未见稳定硬截断，无训练 | ✅ |

| 64 | [May云端TTS净化训练对照](64-may-purified-tts-training.md) | test0 C5.703 vs自然8.909；66 查到重复／长冻结／跳帧监督，不能外推TTS训练有害 | ✅（需连读66） |
| 65 | [May四类噪声尺度扫描](65-may-four-noise-scale-scan.md) | 四类×三实例×两片段，746冻结前向；报告减敏与反向条件，无训练 | ✅ |

| 66 | [May TTS净化时序与加载审计](66-may-tts-alignment-loader-audit.md) | 长冻结/跳帧退化定位，AVE全量和真实loader通过，0重训 | ✅（审计；因果份额未定） |
| 67 | [May四类噪声训练交互](67-may-four-noise-training.md) | 18新模型已完成；背景讲话30dB低尺度C +1.383，其余7条件为负；无添加组反向，单seed | ✅（混合结果） |

## 近期正式报告入口

历史主表截至67；68–74按完整标题在本目录检索。本次新增如下：

| # | 实验 | 核心结论 | 状态 |
|---|---|---|---|
| 75 | [EARS40英语Wav2Lip九评分器](75-ears40-english-wav2lip-nine-scorers.md) | Sync-C +0.859、34/40提高；Holm9后Sync-C/SyncReward/Synchformer正向，其余六项未确认；回归器移音响应不足与分类器天花板另报 | ✅（评分器证据，非感知真值） |
| 79 | [TalkLikeYou中文20来源九评分器](79-talklikeyou-chinese20-nine-scorers.md) | 主要12九项Holm9均未确认；全20仅VocaLiST K5正差过校正；相对Echo主要12的Sync-C效应差+0.735、Holm18 p=.01818；616原生分数、六项控制通过 | ✅（评分器证据，非等效或感知结论） |
| 80 | [四TFG九评分器横向汇总](80-cross-tfg-nine-scorer-comparison.md) | PEAVS四中文模型负差方向重复；无同端点在至少两个同条件新TFG重复过Holm9；旧W2L中文q未知，英语另列；事后Holm27仅TLY−Echo主要12 Sync-C效应差过校正 | ✅（CPU汇总，81行/54对比） |

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

## 2026-10-31 PC-Talk补充

|编号|报告|结论与范围|状态|
|---|---|---|---|
|81|[[81-pctalk-chinese20-syncnet-official-comparison]]|同中文20输入，N7.266/T7.756/Δ+.490；历史SyncNet检查点，九项校正见83。自然低于官方HDTF2.104/MEAD.924，异协议不能直接归因评分错误；原矩阵/stdout/6478文件复核PASS，内部MP3及69ms边界单列|✅ SyncNet检查点|
|83|[[83-pctalk-chinese20-nine-scorers]]|父视频/SyncNet不变，补八项；主要12九项均未过Holm9，全20 CoG/PEAVS/MTD/Voca负差过校正、精确敏感性判定不变。MTD控制65/82未过，非嘴型真值；2339文件/25分层复核PASS|✅ 九评分器完成|
|84|[[84-pctalk-chinese-lips21-nine-scorers]]|新Chinese-LiPS21单draw/21HDTF人像，主要17仅Sync-C+0.569、q=.00675通过；全21+0.547、q=.00081，另PEAVS负差通过，Synchformer未确认均值优势。九项均21完整来源，2925内容/18统计分层复核PASS，非完整无泄露或人类质量证据|✅ 新队列九评分器完成|

## 阅读建议

- **快速结论**：[[研究交接总览]] → `46`（独立幅度响应）→ `52–53`（编码器与归一化）→ `56／67`（训练尺度与噪声结果对照）；跨语料现象另读 `15` 与 `Experiments/跨数据集 TFG 测评（5×50 multiset）.md`。
- **机制理解**：`12`（层规律）→ `13`（逐音素差异）→ `15`（关联分析负结果）
- **方法学**：`11`（韵律指标）→ `14`（解纠缠控制）→ `13`（逐音素模板）
- **中国 waveform 训练恢复**：先读 `26`，再核对 `22`（feature-domain MVP）与 `25`（历史 single-encoder enhancer）；`27`/`28` 解释该旧目标失败，`29` 记录严格两阶段替代的 local train/valid 结果。不要把历史 13-pair corpus 与 AISHELL-1 400 合并，也不要把 `29` 写成 heldout 或 Ditto/SyncNet 结论
- **历史**：`01-08` + 各 summary 文档（注意部分结论已被 09+ 更新）