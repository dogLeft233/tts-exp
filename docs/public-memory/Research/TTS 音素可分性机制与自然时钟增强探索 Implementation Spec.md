---
title: TTS 音素可分性机制与自然时钟增强探索 Implementation Spec
type: research_topic
permalink: tts-exp/research/tts-音素可分性机制与自然时钟增强探索-implementation-spec
status: in-progress
spec_status: implemented_exploratory
protocol: phone_separability_mechanism_v1
question: TTS音素可分性优势的来源，以及在自然时钟和停顿约束下构建可跨评价器迁移的优势
scientific_status: ATLAS_SUPPORTED_RULE_GATE_NEGATIVE
tags:
- tts
- phoneme-separability
- mechanism
- natural-clock
- implementation-spec
- exploratory
implementation_status: complete_for_v1_exploratory_branch
---

# TTS 音素可分性机制与自然时钟增强探索 Implementation Spec

## 1. Objective

设计并实现一个分阶段探索协议 `phone_separability_mechanism_v1`，回答：

1. 已观察到的 TTS 音素可分性优势，哪些部分依赖评价协议、音素时长/池化、录音与声码器域、音素谱形状或内部动态？
2. 在自然音频时间轴与停顿约束下，能否把这种优势构建成可播放音频，并在未参与优化的评价器上复现？
3. 这种可分性变化与历史 SyncNet 变化是否一致？这里只做关联/反例检验，不预设它解释了 TFG 增益。

本次交付是 spec 和已有资产的只读调查，尚未运行新可分性评分或训练。下游实现按本 spec 的条件分支执行，不把“科学阳性”作为工程完成条件。用户先前选择的主方向仍是增强 natural；paired-TTS 辅助与重合成仅作为明确标识的机制对照。最终默认候选允许 natural + transcript/MFA，推理时不读取同句 TTS。

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 初始探索协议；结合 BM 历史结果、当前实现及本地资产核验 | September 21, 2026 | user（设计任务）；agent（设计，未执行） |
| 复核加入历史学习增强器、梯度冲突与WORLD时钟控制 | September 21, 2026 | agent review |

### Observations

- [status] planned；spec ready_for_implementation；scientific_status=NOT_RUN。
- [question] TTS 的音素判别优势来自什么，能否在自然时钟下构建并跨评价器迁移？
- [evidence] LRS3 v1：HuBERT T−N=+0.08117，XLSR=+0.07354；当前 spectral_drc E−N≈−0.00003/+0.00120，均未支持改善。
- [constraint] 音素身份分类、natural/TTS 来源分类、人耳可懂度、SyncNet 时间匹配是不同目标。
- [decision] 优先重评分已有波形和缓存，再做有对照的局部干预；学习增强器是后续条件分支。

## 2. Repository Model

`既有manifest/波形/双侧TextGrid/SSL缓存 → 资产与泄漏审计 → 冻结音素测量协议 → 历史音频atlas → 机制干预 → natural-only候选 → 独立验证 → 探索报告/后续确认方案`

### 2.1 已有证据与本轮新增问题

| 已读 BM 证据 | 可靠信息 | 本轮怎么用 |
|:--|:--|:--|
| [[LRS3 音素优势与自然音频规则增强实现审计 2026-09-20]] | 40 source groups 上两套共享探针均支持 TTS 优势；规则近零；timing 未通过 | 复算锚点；检查干预实际覆盖和强度 |
| [[12-layer-curves]] | 音素判别的绝对最优层与 T−N 差异最大层不同；退化会抬高 Fisher 比值 | 固定主层；次要层单独列，不用峰值选主结果 |
| [[13-per-phoneme-kld]] | 早层能区分 N/T 来源；旧版本有整音节池化及分组泄漏问题 | 来源判别只作域偏移指标，不替代 phone 判别 |
| [[20-aishell100-cross-condition-per]] | 中文 L6 frame error：N self 40.97%、T self 38.38%、N→T 38.97%、T→N 40.70% | 复用各自 phones TextGrid；跨域探针矩阵；不与 LRS3 token-macro accuracy 拼成同一指标 |
| [[25-single-encoder-waveform-enhancer]] / [[28-diagnosis-chinese-hubert-training]] | 已有输入可反传的冻结HuBERT残差TCN；声学/目标梯度比约15.7、cos≈−.737导致目标被抵消 | 复用梯度实现模式；新训练必须检查目标冲突 |
| [[Scale 0.5 enhancer retrain and downstream evaluation]] | 50 valid音频已导出；更接近TTS特征但Sync-C较N为−.104 | 优先补测phone，检验feature proximity与phone identity是否分离 |
| [[15-tfg-link]] | 104 个去重表征/韵律指标无一个通过关联检验 FDR | 不再默认 phone 改善就解释 Sync-C；找不一致反例 |
| [[MFA-linear 连续轨迹机制消融 support30 2026-09-16]] | 固定 15 条 S0765，N/T 内部动态保留 1/.5/0 的音频已生成；TFG 结论不确定/负向 | 重评分可分度，检验内部动态是否必要；同声码器对照 |
| [[TTS 声学变化重组轨迹与剩余项干预：实现与结果]] | 共享模板语义诊断阳性；P/R 干预未支持 SyncNet 机制 | 测 waveform 中 phone margin 是否变动；分离“未操纵成功”与“下游无效” |
| [[TTS 音高起伏交换 v3 配对 Harvest Wav2Lip audit2 结果]] | F0 level/contour 交换工程完成，但支持不足，不能说 F0 已被排除 | N/T×RAW/ID/LEVEL/CONTOUR 重评分，扣除 WORLD roundtrip |
| [[自然时钟音频双目标小样本筛查 2026-09-13]] | EQ/DAC/FastPitch n=3 未达同步双目标；FastPitch 是单女声、近似 phone 映射 | 作为 codec/显式时长重合成探索臂；不能推广为技术不可行 |
| [[TTS 时间辨识与同文本实例交叉实验 2026-09-17 结果]] | 同文本 T1/T2 已生成；实例匹配与迁移未确认 | 估计 TTS 实例间波动与来源效应稳定性 |
| [[TTS 增强 TFG 的原因：证据、排除项与因果分叉]] | 评价端可读性、生成端效果、真实嘴型尚未闭合 | 约束因果语言，不沿用早期报告中的强归因 |

历史负结果只约束其 cohort、实现和终点；不能用 SyncNet 负结果否定同一音频可能有 phone 改善，也不能把旧来源 AUC 阳性当作 phone 改善。

### 2.2 可复用资产目录（September 21 只读快照）

以下路径均相对仓库根。当前核验等级在表中明确；下游仍需重新验证哈希、编码、文本和对齐绑定。只用显式注册路径，禁止遍历 sealed/test 目录补数据。

| key / 输入入口 | 当前核验 | 默认用途 / 限制 |
|:--|:--|:--|
| LRS3_BASE：`runs/lrs3_data_supplement_20260903/07_final_cohort/{manifest,tokens}.json` | 两文件 SHA 与 v1 一致；历史审计 240 pairs=200 train/18 source groups +40 evaluation/40 groups | 主 atlas；evaluation 已见，非新确认集；TTS 为既有 Qwen 云端来源，逐条报告 origin |
| LRS3_CACHE：`runs/lrs3_phone_rules_full_20260921_proxy/` | 已完成 v1；`01_features/{hubert,xlsr}`、`02_reference`、`04_rules`、`06_timing` 存在 | 优先缓存复算；额外层/池化改变需新键 |
| TRAJECTORY：`runs/mfa_linear_trajectory_ablation_v2_support30_20260916/audio_manifest.json` | 15×8=120 个 `records[].audio[].audio_pcm16` 全部存在且容器 SHA 匹配 | N_RAW/T_RAW/N_100/N_050/N_000/T_100/T_050/T_000；实际文件在 parent v1；S0765 单说话人 |
| PR_COMPONENT：`runs/tts_acoustic_reorganization_20260916/audio_manifest.json` | 10×7=70 引用中 60 PCM 文件存在、10 N_RAW 路径失效；未完成其余逐文件 hash 审计 | ID/R_DOWN/P_DOWN 可用性待审；N_RAW 仅允许用 hash/PCM 等价来源恢复；不盲替换 |
| F0：`runs/tts_f0_swap_v3_audit2/audio_manifest.json` | `manifests[].audio[N/T][RAW/ID/LEVEL/CONTOUR]` 24×8=192 WAV 均存在且容器 SHA 匹配 | 多说话人中文；保留历史 receiver QC 失败，不能只取旧视频评分完整的14对 |
| NATURAL_CLOCK：`runs/natural_clock_pilot_20260913_v2/audio_manifest.json` | 3 records，`audio` 中 N/N_REPEAT/S/ND/EQ/DAC/FASTPITCH；递归引用 WAV 均存在，尚未完整 hash 审计 | 极小样本机制筛查；S/ND 是损坏控制，不是增强候选 |
| BRIDGE：`runs/lrs3_natural_to_tts_bridge_confirmation_20260904/01_audio/audio_manifest.json` | `rows[].arms[].output` 22×4=88 WAV 均存在；另有22个旧来源路径失效 | N/N_REPEAT/LOCAL_SWAP/BRIDGE_075；parent cohort 提供 MFA 端点；不能复制历史全局归一化到新候选 |
| TTS_INSTANCES：`runs/tts_time_instance_20260917_v1/B/audio_manifest.json` | 12 records×N/T/T1/T2=48 路径存在；未完整 hash 审计；旧156/T1有MFA失败 | 同文本实例/不同 TTS 来源敏感性；不把48条当48独立样本 |
| AISHELL100：`runs/aishell100_zh_cross_condition_per_20260808/alignment_phone.json`；`results/aishell100_phoneme/{embeddings,mfa_full/out}` | embedding/既有统计存在；旧 `alignment/manifest_local.json` 的200个音频路径为失效 Windows 路径 | 恢复 phone tier 与真实 speaker 元数据后再用；旧 token JSON 是汉字/音节，禁止直接作 phone；缓存无法绑定则标 legacy exploratory |
| LEARNED_OLD：`runs/two_stage_hubert_aishell1_20260810/stage2_scale05_full_20260812/valid_enhanced_wav/enhanced_manifest.json` | `items`清单、50个WAV存在；未逐文件hash；部分provenance指向旧/tmp | 优先atlas；只读已导出valid音频，不用旧heldout或重训旧模型 |
| NOISE：`runs/tts_native_gain_attribution_implementation_20260915_v1/inputs.json` + `01_audio/<id>/<N,T>/manifest.json` | 已检查151/N schema，conditions 为 ORIGINAL/A0/GAIN/NOISE/DENOISE；未全量审计 | 可选早期录音域检验；只用 soundness audit 通过的 N/T，R 另列；不复制视频侧支持阈值 |

冻结的文件 SHA256：
- LRS3 manifest：`dd109c8dfdbde9419f6cc8eaa11e7b9f78dc5d41ac3d437036c9122d14f97304`
- LRS3 tokens：`ad2a54b0aa6f01323acc7fa7650b38d3508ba7edf4282f87f98d55731ac38509`

其它清单hash由P0写registry快照。manifest 内容发生变化时报告差异、冻结新版本；不能沿用这些 hash 宣称输入一致。manifest 中 `artifact_sha256`、文件 SHA、decoded PCM SHA 可能采用不同定义，必须分字段验证。

## 3. Code Anchors

CodeGraph 已先行查询；新未索引脚本未准确命中，随后读取当前磁盘源。HEAD=`158915603392a2b846cb4bfabe1821e9b4fec900`，有大量已有未跟踪/修改/删除文件；复现必须保存代码文件 SHA 与 dirty diff，不能只记录 HEAD。

| path / symbol | 当前职责 | 本次使用与改动 |
|:--|:--|:--|
| `scripts/experiments/lrs3_phone_rules.py::resource_snapshot, audit_inputs, freeze_protocol` | 资源、固定240对审计、协议快照 | 参考其编排；新 registry 自己审计可变 cohort；不调用硬编码240门槛处理其它集合 |
| 同文件 `_extract_model_records, _score_common_pair` | 双臂特征、共享标签评分 | 复用产物语义，不把多臂塞进 natural/tts 二选一；新 token identity 是原始 occurrence |
| `lrs3_phone_rules_metrics.py::derive_frontend_geometry, derive_frame_times, pool_phone_tokens` | 卷积前端中心、半开区间、无帧 token 保留 | 直接复用几何；新模块扩展 core/boundary/matched-1frame 池化，旧函数不改 |
| 同文件 `fit_reference_centroids, paired_bootstrap` | group 等权再 source 等权中心、配对 bootstrap | 原样复用 legacy replay；新多来源中心显式 weights，冻结后不按 arm 重拟合 |
| `lrs3_phone_rules_worker.py::load_ssl_bundle, extract_features, realign_audio` | 本地固定模型、推理、MFA | 推理/MFA复用；训练新增可微 wrapper，不能调用含 numpy 与 no_grad 的 extract_features 反传 |
| `lrs3_phone_rules_audio.py::build_edit_mask, render_rule_arm, make_gain_control, validate_waveform` | 边界保护、PCM干预、增益对照 | 保留旧契约；新 core/speech mask 显式版本；gain_control 仅在其匹配定义满足本 spec 时复用 |
| `lrs3_phone_rules.py::_timing_qc_from_waveforms` | MFA labels edit rate，zip匹配边界 | 仅作为历史参考。新实现用单调序列匹配＋分母审计，不复制 zip 造成的插入/删除后错配；实际实现未使用 silence_iou 配置，新 spec 必须实现 |
| `check_lrs3_phone_rules.py::recompute_statistics, verify_waveform_contract` | 独立复算/PCM检查 | 保留 v1 checker；新增 checker 不导入新 runner 的统计/决策函数 |
| `lrs3_phase_preserving_replacement_envelope/audio.py::phase_preserving_blend` | natural相位＋log幅度插值 | 参考 STFT/ISTFT；不能继承全局RMS/peak缩放及整段TTS混合 |
| `tts_acoustic_reorganization.py::_intervention_arrays` | 等幅P/R干预、均值保持 | 历史解释/审计模板；本轮先读已生成PCM，不能用conditioning空间改善替代输出音频重提取 |
| `scripts/waveform_hubert_enhancer.py::HuBERTWaveformEnhancer.encode_layers`、`feature_targeted_waveform_renderer.py::encode_layers` | 冻结参数保留输入梯度 | 参考retain_gradient分支与测试；新模型限24band，不复制旧大残差/4秒chunk推理或Stage1目标 |
| `tts_f0_swap.py::WorldParams, world_analyze, world_synthesize` | 5ms WORLD分析合成 | duration控制复用分析参数，按目标时钟映射后合成 |
| `natural_clock_pilot_audio.py::duration_grid, fastpitch_audios` | 显式时长/F0/能量 FastPitch | 只读已有音频；先解决 speaker/IPA映射/输出对齐混杂，不重跑同一个n=3实现 |
| `scripts/analysis/phoneme_recognition_probe.py`（历史报告锚点） | LOO self/cross帧错误率 | 仅复用既有产物/方法；新主指标不是legacy PER，不直接调旧CLI作为验收 |

新增包 `scripts/experiments/phone_separability_mechanism/`：
- `run.py::main, dispatch_stage, freeze_protocol, write_report`：单一入口与阶段状态。
- `inventory.py::load_registry, adapt_*_manifest, audit_asset, build_overlap_graph, freeze_splits`：每个注册 source 一个显式 adapter。
- `features.py::load_or_extract, pool_views, match_occurrences`：缓存、坐标、对齐支持。
- `metrics.py::fit_probe_bundle, score_frozen_support, summarize_groups, compare_domains, abx_score`：纯统计。
- `diagnostics.py::duration_analysis, mask_dose_audit, historical_contrasts, link_syncnet`。
- `audio.py::render_roundtrip, render_phone_shape, render_paired_shape, render_moment_match, render_duration_control`。
- `timing.py::audit_alignment_repeat, compare_boundaries, compare_pauses, evaluate_timing_controls`。
- `train.py::BoundedGainEnhancer, differentiable_ssl_forward, train_one_seed, select_checkpoint`。
- `check.py::verify_bindings, recompute_scores, verify_support, verify_audio, recompute_decisions`。
配置 `scripts/configs/phone_separability_mechanism_v1.yaml`；对应测试放 `tests/experiments/phone_separability_mechanism/`。

## 4. Reference Pattern

主模式来自 lrs3_phone_rules：配置冻结→资产绑定→按一个模型驻留提取→冻结探针→配对统计→独立 checker。不同处是本轮多 cohort、多臂、带探索选择，不沿用“Stage A 失败就全局停止”的逻辑；一个旧集合缺失只阻断相关对比。

波形模式来自 phase_preserving_blend，但新候选必须通过 natural mask 合成并在写盘后恢复不可编辑PCM。历史轨迹实验的“输入特征控制＋输出音频＋下游评分”启发双层验收：操纵了目标参数不等于实际输出提高可分度。

冻结声学模型提供语音增强监督有原始文献基础：[Phonetic Feedback](https://arxiv.org/abs/2003.01769)、[Recognition-model perceptual loss](https://arxiv.org/abs/2112.06068)。这些是降噪/识别研究，不证明本项目自然语音规范化一定成功。显式 duration/pitch/energy 条件可参考 [FastSpeech 2](https://arxiv.org/abs/2006.04558)，本仓库已有 FastPitch 试点，因此本轮优先审计旧输出而非再部署同类系统。

## 5. Invariants

1. 源 WAV/TextGrid/run/模型权重只读；所有转换、修正后的标签、结果写新 run。不能覆盖旧阴性结论。
2. 每个 arm 保留 `input_mode=natural_only|natural_text|paired_tts_oracle|resynthesis_control`、原始 `clock_owner` 和 `speech_identity`；MFA-linear 不冒充 raw TTS。
3. 自然增强主候选输出单声道16k PCM16，样本数相同；mask=0 的PCM逐样本完全等于自然输入；不补删停顿、不全局拉伸、不末尾裁切遮掩时长错误。
4. 原始TTS按自身MFA评估；自然时钟候选主评分使用冻结natural slots。重对齐是独立QC/次要端点，不准拿输出MFA重新挑容易的token来提高主分。
5. 严格共享支持：每个预定 contrast 在评分前按输入标签/几何冻结 occurrence 与 label；不可按输出预测正确与否、增强后对齐质量或SyncNet分数筛样本。
6. 原始label做NFC/strip，保留IPA、送气、长度等；英文/中文字典分开。allophone合并仅预先固定为次要敏感性，不能根据结果改映射。
7. HuBERT/XLSR模型、processor、层索引、frame geometry 和音频预处理全冻结。只能报告实际存在的 hidden_states 索引；0层非“第一个Transformer block”。
8. 全句输入保留原上下文。core池化不等于去除了Transformer长程上下文；需另做输入上下文干预才能支持该解释。
9. 拟合、PCA、标准化、模板、超参选择不能看目标评估group；同句所有TTS实例、历史变体、共享原视频/说话人不能跨允许的隔离边界。
10. 单说话人15句仍然是一个speaker；source_group不是自动验证过的speaker_id。禁止将token或多个TTS实例当独立统计样本。
11. 训练/调参不能用最终保留的XLSR结果；冻结教师参数但保留输入梯度。最终提升必须在导出的PCM重提取后成立。
12. 工程状态、操纵是否有效、音素提升、时序保持、音质、泛化、TFG效果分别记录，不能合并为一个PASS。
13. 所有空统计写null和原因，禁止NaN/Inf、空集满分、zero residual伪阳性。
14. 每个GPU/MFA/训练/批量特征阶段启动前检查GPU进程、空闲显存、RAM、磁盘；资源占用时标RESOURCE_BUSY，不终止他人任务。下载沿用已授权[redacted-ip]:7890代理与固定revision，推理仍本地加载。

## 6. Implementation Plan

### 6.1 P0：资产登记、来源隔离、资源与测量预检

先实现inventory、状态机和只读audit，不加载模型。

统一 `assets.jsonl` 字段：
`asset_id, dataset, language, corpus_key, sample_id, paired_key, occurrence_namespace, source_group, speaker_id, speaker_verified, parent_asset_ids, arm, input_mode, clock_owner, audio_path, container_sha256, pcm_sha256, sample_rate, sample_count, transcript_sha256, textgrid_path/hash, alignment_method, model_origin, historical_split, analysis_split, audit_status, missing_reason`。

asset_id由数据集＋原始pair身份＋arm＋PCM hash生成，不能只用1/151之类短ID。建立原视频/自然PCM/原始utterance/可靠speaker的overlap连通分量；同源跨run去重，一条音频可以有多个provenance别名但只算一次独立观测。重复文本另报文本泛化风险，不将所有同文本不相干说话人强行视为同speaker。

缺失资产先在本表已注册父子manifest中寻找容器hash或decoded PCM hash等价副本；每次修复保存 `path_resolution.json`。只有文件名相似不算等价。找不到则 `ASSET_MISSING`，继续其它队列。不得自动恢复用户删除的数据或扫描sealed目录。

划分：
- legacy replay保持v1原200/40不变。
- 新方法只在原18个train source groups上开发：按SHA256(`psm-v1-split|`+group)排序，前12组为FIT，后6组为DEV，组内全部记录随组。若可靠speaker使跨组相连，按连通分量重分，记录实际组数；不足FIT 9/DEV 3组时不训练。
- 原40 evaluation标为E_SEEN，仅冻结方法后的探索性外推，不称新确认；P1查看其atlas会进一步消耗其新颖性，明确记录。
- 所有历史N/T混合来源的模板只来自FIT。历史中文probe使用独立中文参考：优先有真实speaker映射的AISHELL100，按speaker分组5折（少于5说话人则leave-one-speaker-out）；对S0765历史干预时整个S0765排除出参考。元数据无法恢复则只给utterance-held-out描述性结果，禁止speaker-generalization结论。
- fresh confirmation见P5，不能从已有项目sealed/test偷取样本。其它历史数据不能偷偷扩入学习FIT。

资源预算：当前V100空闲（5MiB占用）、RAM可用21GiB、磁盘15GiB。默认CPU工作进程≤4，MFA jobs≤4，一个GPU任务/SSL模型。启动条件GPU无外来compute app、GPU空闲≥8GiB、RAM available≥8GiB、磁盘free≥`max(4GiB,1.2×stage_estimated_bytes)`。训练先4句profile估计显存与秒/step；预计显存超过空闲80%则microbatch1/gradient checkpointing；仍不足就阶段标资源不足，不能静默换模型/采样率。总新产物预算6GiB，保留至少4GiB磁盘；无需保留全部层全帧副本时只存必要层/池化与抽查帧，源缓存引用不复制。

### 6.2 P1：统一可分度 atlas 与测量可靠性

先从v1缓存独立复算原始Stage A与规则Stage B，数值误差≤1e−8（相同缓存）；模型重新前向只要求embedding max_abs≤1e−4且aggregate差≤1e−5，超过则报告环境漂移而非改阈值。legacy复算允许调用旧checker，作为新实现的对照，不是新checker的唯一依据。

主模型固定：
- HuBERT `facebook/hubert-base-ls960@dba3bb02fda4248b6e082697eee756de8fe8aa8a`，hidden_states[6]。
- XLSR `facebook/wav2vec2-large-xlsr-53@c3f9d884181a224a6ac87bf8885c84d1cff3384f`，hidden_states[10]。
- 次要层HuBERT [0,3,9,11]、XLSR [6,14]；只在前述主分析后按磁盘预算提取，不能挑最大差层替换主层。旧缓存只验证过的层可复用。

cache key包含PCM hash、processor fingerprint、revision/权重hash、实际层列表、frontend配置、全句/裁剪上下文政策；token缓存另加TextGrid hash、label-map hash、pooling/view。不以路径或sample_id命中缓存。

核心度量：
- pooled token vector=mean(frames)后L2；每个phone的FIT中心先group内均值归一化、再group等权、最后N/T等权，与v1一致。label至少每来源20 tokens、3 FIT groups；支持不足不降低阈值，输出coverage不足。
- 每对固定公共label集合内先逐phone算accuracy，再phone宏平均得A_i；同source group多句先均值，最终group等权。
- `margin(z,p)=cos(z,c_p)−max(q≠p)cos(z,c_q)`，同样宏平均；同时列正确类相似度、最强错误类相似度，避免只有整体缩放解释。
- source-conditioned confusion matrix；between-phone距离/within-phone离散度分别报，Fisher仅次要且检查退化；label count、majority baseline、token coverage必须同表。
- 来源二分类AUC单独命名 `domain_auc_N_vs_T`，禁止叫phone separability；FIT/DEV group隔离、Pipeline内fit PCA(至多30维)/scaler/logreg(C=1,max_iter=2000)，不收敛记失败。

探针偏好诊断：在相同FIT标签支持和相同evaluation token上另拟合N-only、T-only中心，形成 `reference={N,T,mixed} × evaluated_source={N,T,enhanced}`；域内自身中心准确率不作为增强终点。一个固定混合域L2 logistic probe(C=1，FIT-only scaler，max_iter=2000)作灵活判别器敏感性，不调C。若优势仅在T-reference存在，记录REFERENCE_DEPENDENT，不能称一般phone优势。

Occurrence匹配：先用word索引/转写边界约束，再phone序列全局单调edit alignment（match0，insert/delete/substitute1）；只保留所有最优路径中身份唯一、原label完全相同的matched occurrence。重复音素歧义、插删、无帧逐项记账，不用最近label强行填充。无法word锚定时用整句唯一匹配并注明。增强arm沿用natural occurrence ID。

分四个view独立拟合参考并固定支持：
- full：该phone全部帧；
- core：20%–80%时段的帧；
- boundary：前20%和后20%分别统计，不合并成两个独立样本；
- matched_1frame：每个N/T matched phone各取最接近自身中点、且实际落在phone内的一帧；取消长音素多帧平均的好处，不能补帧。
补充duration strata=[0,60)、[60,100)、[100,160)、[160,∞)ms；比较双方落同一bin且帧数相同的matched pairs。报告筛选前后支持，不称“控制掉时长的因果效应”。全句SSL上下文仍保留。

ABX作为第二类度量：在共同phone类及训练集混淆top10 phone-pairs上固定三元组，A/X同phone不同source_group，B为混淆phone；A/B由参考FIT取，X为目标group；frame序列余弦DTW，长度归一化cost，tie=0.5 error。按FIT中点时长差选择至多每对每group20个固定三元组，字典序打破平局，跨arm复用同一triplet ID；不足3groups/20triplets记不足。这不是人耳ABX；只检查token均值探针以外是否一致。

对齐敏感性：原始双侧MFA、相同设置fresh MFA、边界±10/20ms确定性jitter各自报告；jitter拒绝越界/交换次序，不修剪label。natural/identity完全相同WAV两次独立MFA作为基线。人工盲审待办包固定前10个hash排序pairs（每句至多20边界），不得假定MFA的confidence=1就是真值。未人工完成可继续自动探索，但结论标ALIGNMENT_DEPENDENT/PROVISIONAL。

atlas分language/cohort/reference/view/encoder列结果；不同phone库存、frame与token指标不直接比绝对数值。

### 6.3 P2：用已有音频定位机制

默认先跑LRS3缓存→TRAJECTORY→F0→NATURAL_CLOCK/BRIDGE→TTS_INSTANCES/LEARNED_OLD；PR_COMPONENT/AISHELL100/NOISE按资产可恢复性并行推进。一个分支缺失不阻断其余分支。所有历史音频必须从最终PCM重新提取HuBERT/XLSR，不能直接拿生成conditioning作成功证明。

| 假设 | 预定对比与预期证据 | 可排除的混杂/保留的限制 |
|:--|:--|:--|
| H0 规则没有效修改关键位置 | 旧arm的mask可编辑比例、实际changed samples、每phone残差RMS/原RMS、频带增益、guard/taper后的有效时长；按<30ms/30–60ms/>60ms分层 | 若难辨phone几乎没改，旧阴性只是该剂量阴性；不自动证明更强就有效 |
| H1 时长/多帧平均/边界贡献 | full→matched_1frame/core、同duration-bin；TTS_INSTANCES对应phone时长与margin联动 | 只给关联；SSL长程上下文和对齐误差仍可能参与 |
| H2 录音/声码器域 | N_RAW→N_100、N→DAC、N/T RAW→WORLD ID；可用NOISE/DENOISE各自减A0/GAIN | roundtrip本身若占大部分变化，后续任何改动必须减同链ID |
| H3 内部动态与稳定性 | TRAJECTORY中D_N(k)=A(N_k)−A(N_100)、D_T(k)=A(T_k)−A(T_100)，k=.5,0；交互D_T−D_N | 均值保持只在原conditioning中成立；PCM重新提取可检验是否仍成立 |
| H4 可重复phone轨迹 vs 剩余变化 | A(N_R_DOWN)−A(N_ID)、A(N_P_DOWN)−A(N_ID)，T同理，再做R−P等幅差 | 原S0765单speaker、donor只有5句；不伪装群体因果 |
| H5 F0并非谱形状 | N/T LEVEL−ID、CONTOUR−ID，两方向分别报 | 不使用旧SyncNet support>=50筛音频；保留receiver QC缺失原因 |
| H6 谱包络可迁移 | BRIDGE_075−N与MFA端点；N/EQ/DAC/FASTPITCH横看phone改善和时序QC | bridge含全局RMS与重建混杂，只生成下一阶段假设 |
| H7 TTS实例/来源一致性 | 每pair T1/T2/raw T对N；报告同来源seed波动和跨来源差异 | T1/T2不作为独立样本；现有云T与本地TTS不等价 |
| H9 学习目标错位 | LEARNED_OLD E−N的phone accuracy/margin/ABX，对照旧raw-TTS proximity；同组同源匹配 | proximity改善而phone不升提示目标错位；phone升而SyncNet不升提示下游连接不足 |
| H8 与下游终点脱钩 | 同一音频/同一pair的Δphone对历史ΔC、ΔD、曲线background/best-match变化 | 只按匹配的video/driver/eval-audio/supported-window条件join，相关不是中介因果 |

历史效果不论原来是否显著都纳入完整账本；不只挑“Sync-C改善”的arm。关联只预定HuBERT/XLSR主层margin两列×ΔC/ΔD，group bootstrap Spearman，BH校正；n<10独立groups只描述。同视频固定音轨与自身音轨结果不能混合；缺失的正向/负向组都报告。

单说话人轨迹/P-R分支使用句级配对不确定性只能标“该说话人内部”；群体置信区间为null。不把“15 utterances”写“15 independent speakers”。

### 6.4 P3：有对照的自然时钟构建（先规则/统计模板）

此阶段必须在看DEV候选分数前冻结arms与参数。P1/P2可提出下一协议版本的新假设，不能反复看E_SEEN调当前版本。

统一waveform算子：float64 STFT，16k，n_fft=win=512、hop128、periodic Hann、center=true、reflect；显式ISTFT原长度。得到候选v后：
`y_float=x+alpha*m*(v−x)`，alpha从1起按解析peak约束降到[0,1]；只缩残差，不全局缩输入。写PCM16后mask=0位置强制回填原PCM。峰值不可越[-1,1)，不能clip掩盖超限；alpha、残差剂量必须落盘。

两个预定mask：
- CORE：原build_edit_mask，边界guard10ms+taper5ms，作为最保守候选。
- SPEECH：只保护natural silence及语音/停顿边缘10ms，在连续speech内部不挖每个phone边界；相邻phone的频率增益在10ms内线性crossfade。允许编辑内部瞬态，但不移动时间坐标；声学边界仍必须独立验。
mask全部由输入natural MFA决定，不根据输出分数调整。

固定臂：
1. N_ID（PCM直通）、N_RT（完整STFT→ISTFT→mask零增益，允许≤1 LSB roundtrip误差）、旧spectral_drc（历史阴性）。
2. PHONE_SHAPE：从FIT学每phone平均谱形状差，不训练神经网。每帧log magnitude取实倒谱，仅保留|quefrency|≤1ms系数后还原；减频率均值获得shape。每phone在group内平均，再groups等权；T−N差为d_p(f)，每侧≥20 tokens/3 groups。NFK等额外映射禁用，稀缺phone保持原样。新帧shape=L_N+beta*d_p，beta={0.5,1.0}；增益clip到±6dB，以帧平方谱能量标量恢复原帧能量；natural相位不变。CORE/SPEECH均做（4候选）。
3. PAIRED_SHAPE（oracle）：同句匹配TTS phone内按相对位置插值平滑谱包络到natural STFT帧；增益同样±6dB、逐帧能量保持、natural相位；beta={0.5,1.0}、仅SPEECH（2候选）。单phone映射，不跨silence，不重定时natural。TTS phone插删/太短无两端包络支持则该phone不改且记coverage。
4. MATCHED_MOMENTS：TTS donor换成FIT其它group同phone平均模板；同beta/预算/SPEECH，仅beta1（1候选）。与paired比较，判断同类规范谱是否足够；实际就是PHONE_SHAPE之外的“直接朝T类模板移动”，公式 `L_new=(1−beta)L_N+beta*template_T_p`，随后去能量均值和限幅，不等同于差分d_p。
5. SHUFFLED_LABEL：将PHONE_SHAPE的FIT label模板按固定seed无自配排列，beta1/SPEECH（1负控制）；目标是同等预算但错误音素方向。
6. 每个候选配套GAIN_MATCH：在同mask上用单一非负gain通过固定二分法匹配候选可编辑speech RMS，容差0.1dB；无法同时满足peak则GAIN_CONTROL_UNAVAILABLE，不宣称排除响度。
7. EQ_MATCH：beta1/SPEECH的全phone平均d(f)，频带增益/能量约束相同（1控制），区分phone条件化和全局音色变化。

基础构建共9个新处理臂（4+2+1+1+1），加ID/RT/legacy和相应gain控制，固定清单去重后计数。不能把SHUFFLED_LABEL选成最终增强器，即使教师分数升高。每类错向控制不能按每个evaluation输入梯度寻找最坏方向。

有效操纵：每句报告“支持模板phone占比”“speech编辑时长占比”“实际残差RMS相对原speech RMS”“alpha”。若候选在>50% eligible句中speech编辑占比<20%或残差比<0.005，标MANIPULATION_TOO_WEAK，仍报告分数，但不能用阴性否定该机制。阈值是探索操作判据，不是自然常数。

可选H1干预（P1 matched_1frame使正的T−N优势下降≥50%、共同支持≥70%时触发）：以N/T声学来源×N/T目标phone时钟构成4臂，含两个self-map WORLD重建。使用`tts_f0_swap.py::world_analyze`固定5ms网格；按唯一匹配phone端点建立目标时间→来源时间的分段线性映射，对log谱包络/AP逐频插值、voicing最近邻、voiced内部log F0插值，禁止跨清浊边界插值log0。新WorldParams指定目标样本数，用同一WORLD合成；只允许≤一个5ms hop的合成尾长数值修正并记录。暂停区用目标时钟所属原音频的对应PCM，故这是联合时长/停顿重组而非纯时长因果隔离。双方phone必须唯一全匹配，时长比均在[.5,2]；否则整pair缺失并报告，不word回退。交叉臂对同来源self-map比较，并列F0/谱/音质操纵检查；不继承旧普通resample改变pitch/formant的混杂。若伪影、支持不足或时序QC失败，标DURATION_CONTROL_INCONCLUSIVE。该分支为resynthesis_control，非增强候选，不阻断主线。

### 6.5 P4：小型可微增强器（条件执行）

P3中任一有效自然候选HuBERT DEV margin>0、accuracy不下降，或paired oracle同时改善两套固定探针并通过初步timing门时，允许进入学习探索。前者不要求已达到最终显著性。若oracle有益而规则无益，记录“存在参考辅助可迁移信号”；若所有操纵有效且两套指标都无益，先输出科学负结果，不直接投入更大模型/RL。oracle的XLSR结果不用于选择训练超参；学习最终XLSR全程保留。

默认增强器是natural-text输入（text只用于冻结mask与训练label），无同句TTS推理依赖：
- 输入STFT的24个固定mel三角频带log power，FIT统计mean/std。
- Conv1d(24,64,k=5,pad2)+GELU；3个64通道residual Conv1d块(k=3,dilation1/2/4,same padding)+GELU；Conv1d(64,24,k=1)输出零初始化。
- 输出通过tanh限制±6dB，频率插值回257 bins，时间和频率各用[1,2,1]/4平滑一次；按原帧谱能量恢复；natural相位ISTFT后通过SPEECH mask。
- 不加duration predictor、不warp、不替换source F0、不用vocoder。训练后参数数目和感受野落盘。

HuBERT L6为唯一训练教师；mixed参考中心FIT上冻结。可微processor必须复现worker归一化（含epsilon与attention mask），用Torch计算，teacher.eval且参数requires_grad=False；对输入仍autograd开启。用同一句natural在wrapper与现有worker比较输出误差≤1e−4后才训练。完整句训练，batch1、累计4句、AdamW(lr1e−4,weight_decay1e−4)、gradient_norm≤1；不偷偷切句改变SSL上下文。24mel增益限制与mask属于模型输出合同。

loss（均先在phone/group内等权平均）：
`L = L_phone + 10*L_wave + 1*L_spec + 0.1*L_tv + 10*L_peak`。
- L_phone=mean softplus((0.05−margin)/0.1)；phone取natural固定core slots，不训练on输出重对齐标签。
- L_wave=mean((y−x)^2)/(mean(x^2)+1e−8)，仅speech。
- L_spec=mean|log(|STFT(y)|+1e−5)−log(|STFT(x)|+1e−5)|，仅speech帧。
- L_tv=mean相邻24band增益dB一阶差平方（时间/频率各半）。
- L_peak=mean relu(|y|−0.98)^2。
训练浮点输出无clamp；导出按P3残差headroom处理并记alpha，只有最终PCM有效。

梯度诊断：每100 steps在4条固定FIT句分别计算phone与加权保持损失梯度范数/cosine。连续3次中位cos<−.5、保持/phone范数比>10且DEV margin无改善，终止seed为OBJECTIVE_CONFLICT。固定schedule：前200 steps保持项乘.1、200–400线性升至1、之后全权重；L_peak始终原权重。checkpoint始终按正式QC，不降低质量门。这是基于历史冲突的待测规则，不在E_SEEN调权重。

训练预算：每seed最多1500 optimizer steps、每100 steps在DEV评一次HuBERT；seeds=[20260921,20260922,20260923]；先1 seed profile再余2。每seed最多2小时，先到即保存budget_limited与实际steps，不补跑至“显著”。无教师对照以完全同结构去掉L_phone训练首seed；零初始化应收敛/保持近ID，用于检查收益是否来自目标监督。

checkpoint选择只用DEV：先通过waveform/timing自动QC且accuracy≥N，再按HuBERT macro margin排序，平局取L_wave小、step早。没有合格checkpoint则NO_VALID_CHECKPOINT。跨seed报告全体，不只报最好seed；发布候选固定seed20260921，其它为稳健性，不能按E_SEEN/XLSR切换。

可选未来方向为更强生成式规范化或黑箱参数搜索；本v1不实现RL。若日后需要不可微综合奖励，先另定固定动作空间/预算及独立保留评价器，不能沿当前评估集持续刷奖励。

### 6.6 P5：时序、可分性、质量与确认判据

Timing分开报告：
- T0 waveform：exact length、mask外PCM identity、no clipping。
- T1 acoustic：用同配置对N、ID/RT、候选独立MFA；phone序列唯一单调匹配，报告matched coverage、edit rate、每句boundary median/p95/max；全句pool分位数不能代替每句。speech edit rate≤.05，匹配speech occurrence覆盖≥.90，每句median≤20ms、p95≤40ms。取满足全部门的句比例≥.90；所有失败句仍留主评分分母。
- pauses：natural所有≥50ms pause均需匹配；原PCM范围不变，输出MFA pause交并比每个≥.5、起止偏差各≤20ms；输出新增≥50ms pause也计失败。
- T2 independent：预冻结manual包（10pairs或全部若不足10）标注关键闭塞/释放/元音起点；尚未收集时 `HUMAN_TIMING_NOT_ASSESSED`，不宣称人类时序真值。不得假装模型自动置信分数代替听检。

测量校准控制：N_REPEAT应接近0边界差；N右移80ms并截尾、局部一个≥120ms phone区间移40ms的损坏控制应检出（只用于仪器敏感性、不进入候选）。若重复不稳或任一损坏控制未检出，则TIMING_MEASUREMENT_UNCALIBRATED，继续可分性探索但不能宣称保持时序。现有timing阴性不通过放宽阈值解决。

统计与门槛：
- 所有预定contrast按source group配对bootstrap，PCG64 seed20260921、10000 draws；每group内先句均值，严禁token bootstrap当主CI。
- 多臂探索同时列point estimate、95%CI、group sign-flip permutation p（≤16groups枚举全部，否则10000次）、同一cohort×主encoder×主view内全部预定候选contrast BH q。p采用双侧，候选阳性还须正方向；FDR家族与全部行落盘，不去掉阴性臂。
- P3候选只在DEV选：先满足T0/T1、HuBERT accuracy不降、margin>0，再按margin排序；oracle/control不得参选，平局按残差小、arm_id字典序。将候选/参数/code hash写selection.json后，才评E_SEEN。
- E_SEEN上支持至少30 source groups、每句≥5共同labels/≥10tokens/coverage≥.70。候选对N和对应gain-control的accuracy差均≥+.010且两套模型95%CI下界>0，margin同向，ABX error不恶化，才标 `EXPLORATORY_CROSS_ENCODER_GAIN`。GAIN不可用则 `GAIN_CONFOUND_UNRESOLVED`。
- 若达到上条且T0/T1通过，标 `NATURAL_CLOCK_GAIN_AUTOMATED`；人工未验/ASR未验在质量字段保留，不提升为可懂度或TFG成功。
- “构建了较强优势”的额外目标：对同一冻结support计算 `D_half=(A_E−A_N)−0.5*(A_T−A_N)`，两模型CI下界≥0；只在T−N>0时报告恢复比例。D_half不过但+.010过时称小幅改善。与TTS接近的非劣目标 `A_E−A_T≥−.010` 另列，不自动要求。
- 同时报告完整输入队列的失效率/有效率；对输出失败但baseline可评分的句，敏感性下界按candidate accuracy=0，上界=1；不能靠complete-case删除困难样本制造成功。
- 无阳性可为 `NO_GAIN_AT_TESTED_BUDGET`；操纵不足为 `MANIPULATION_TOO_WEAK`；测量不足为 `MEASUREMENT_INCONCLUSIVE`；不得统一成“目标不可能”。

音质/内容：输出N/RT/E/T随机盲听包，记录未提交评分。仅固定现有ASR若能核验revision和环境才加入WER变化，transcript只用于评分不用于ASR提示；无法核验则ASR_NOT_ASSESSED。声码器/说话人变化单列，不能因probe变好就声称清晰度提高。

新确认：本轮已有40组只能报告探索。成功后输出 `confirmation_plan.json`：新且未进入任何设计/训练的source groups（speaker核验优先），目标至少60组，每组最多1句；按DEV group差SD计算δ=.010、power .8、two-sided α=.05的样本量规划，注明近似及预计coverage损耗，若60不足则报告需求，不伪称有功效。候选/层/参考/门槛预冻结；不能访问项目已有sealed/test；新数据收集或额外大规模生成按后续任务范围另行执行。新数据不足不妨碍本轮探索交付。

最终不得将phone提升直接写成TFG增益。若有后续视频任务，固定natural评价音轨和源脸，比较N/E驱动的视频，同时报告own-audio及natural-replacement两种结果；该视频生成不属于本次spec的默认执行阶段。

### 6.7 CLI、产物与状态合同

计划CLI（代码尚待实现）：

    python -m scripts.experiments.phone_separability_mechanism.run --config scripts/configs/phone_separability_mechanism_v1.yaml --run-id <id> --stage audit
    python -m scripts.experiments.phone_separability_mechanism.check --run-dir runs/<id>

后续同入口用`--resume`，stage取atlas、mechanisms、construct、train、report。

`--stage all`执行所有就绪分支及条件训练，最后总报告；不自动确认/重生成视频。`--smoke`固定每source hash排序前2条，只验证工程，无科学判决。resume验证protocol/code/config/inputs所有依赖hash，变动则拒绝沿用旧缓存，要求新run-id。run-id拒绝绝对路径、..和越出runs目录。

新run目录（同名资产不得覆盖）：
- 00_inventory：assets.jsonl、registry_snapshot.yaml、overlap.json、splits.json、resources/、missing.csv。
- 01_protocol：protocol/config/code_hashes/hypotheses/support_plan。
- 02_features：cache_index.jsonl、必要frames.npz与token_records.jsonl、label_inventory。
- 03_atlas：per_token/per_pair/per_group、reference_matrix、confusion、duration/core/ABX、legacy_replay。
- 04_mechanisms：historical_contrasts、dose_audit、syncnet_linkage、measurement_checks。
- 05_construction：wav/<arm>/<asset_id>.wav、construction.jsonl、selection.json、support.json。
- 06_training：每seed配置/完整统计/DEV轨迹，仅留last+selected两个checkpoint。
- 07_validation：timing/pause逐句表、quality待办、checker.json、failure_bounds。
- 08_report：report.md、atlas图、mechanism_evidence.csv、decision.json、confirmation_plan.json。
表用CSV、结构与决定用JSON；不得把summary作为唯一证据。

status.json至少分 `execution, stage_states, measurement, manipulation, phone_gain, timing, quality, generalization, downstream, reason_codes`。execution COMPLETE可以伴科学NO_GAIN；资源/依赖状态不能冒充科学负结果。每张机制表必须列支持/反证/不确定、替代解释和可允许的下一步。

## 7. Expected Change Surface

### Must change

新增第3节包、配置、对应测试；BM中的本spec和执行后同主题最终实验笔记。下游实现时在新run冻结当前代码hash，包括复用的未跟踪脚本。

### May change

仅当复用helper存在直接阻断且有行为回归测试时，最小修补lrs3_phone_rules_worker/audio/metrics；先新adapter解决多cohort差异。中文元数据恢复写新inventory，不修原legacy JSON。资源模型缺失可按已授权代理下载公开固定资源，不覆盖现有权重。

### Should not change

旧v1配置/决策、编号00–05流程、TTS provider、TFG/SyncNet权重、旧源数据、用户现有dirty worktree、CONTEXT/HANDOFF内容。不重构整个实验框架，不为本任务修改AGENTS/skills，不提交或删除用户文件。

## 8. Validation Plan

按静态→单元→集成→smoke→正式运行：

1. `ruff check scripts/experiments/phone_separability_mechanism tests/experiments/phone_separability_mechanism` 与py_compile，期望通过。
2. `pytest -q tests/experiments/phone_separability_mechanism`。必须有下列行为测试：
   - 各manifest schema adapter输出一致规范；短ID冲突、PCM重复、失效路径、hash不符分开处理。
   - 持久划分对排序不敏感；speaker/同source跨run联通；FIT/DEV/E重叠必须失败。
   - 改arm分数不改变冻结support；插入phone不会使后面所有边界zip错配；重复label歧义正确标missing。
   - 400sample receptive field / 320stride的中心应为199.5/16000而非10ms；core无帧不补零/不偷邻帧。
   - source AUC高但phone准确率机会水平的合成数据，不能输出PHONE_GAIN。
   - 给某类整体缩小所有向量会导致Fisher比值误导时，collapse检测/分项距离揭示问题。
   - 同group重复多句不会扩大独立n；swap N/T使差值与CI符号反转；空值为null。
   - 原始TTS时轴与natural不同，复制TextGrid会被binding拒绝；增强候选重MFA不得覆盖natural主support。
   - STFT identity/alpha0、mask外精确PCM、短phone/全silence/尾段、增益匹配失败、peak residual衰减。
   - 错模板保持预算但不能被参选；oracle读取TTS必须被natural-only导出契约拒绝。
   - 教师参数梯度全None、输入和enhancer梯度非零；训练wrapper与worker一致。
   - 序列编辑、pause缺失/新增、80ms控制以及局部40ms控制，timing门必须能抓住。
   - checkpoint选择不读取XLSR/E_SEEN；输出失败敏感性不能删分母。
3. 现有回归：`pytest -q tests/experiments/test_lrs3_phone_rules*.py tests/experiments/test_check_lrs3_phone_rules.py tests/test_ssl_embedding_shapes.py tests/test_feature_separability.py`；不需重复整个项目所有TFG测试。
4. CPU synthetic端到端fixture须覆盖至少3labels/4groups/2cohorts/3arms，使用测试专用阈值，不污染正式配置；一条故意missing只阻断所属contrast。
5. 真数据smoke每source2条，先资源检查，控制scope=engineering_only；至少真实前向一个encoder并输出可播放ID/候选和重复对齐差。
6. 正式phase结束独立checker：直接从每token预测/中心向量重算accuracy、margin、group权重、bootstrap/多比较，从PCM重算mask/length/hash；不import runner.decide函数。分层hash排序抽取每arm至少1条、总≥10条重新前向；其余验证cache绑定。报告抽查范围，不能说全部重新前向。
7. 文档验收：atlas含所有注册source的available/missing/legacy状态；结论有数据域/独立group数/测量定义/对照/CI/限制；每个阴性能够区分无效干预、无改善、测量不足；无未完成质量评估被写成通过。

## 9. Risks and Edge Cases

- 旧缓存往往层名/时间中心不同；当前“L6”与旧第6个存储槽不一定相同。必须用sidecar和实际shape核验，不能仅按文件名。
- 旧metadata中的speaker_id如aishell1_8是样本占位符，不是真speaker；优先BAC009Sxxxx原始键恢复。
- tone-stripped中文phone与英文IPA不共享一个探针；跨语言仅比较各自标准化的paired effect方向。
- 旧PR N_RAW路径丢失、bridge来源别名丢失；存在输出不等于全部来源已恢复。若缺少可信hash，不能按听起来相同替代。
- 音素内部去动态可能改善token均值同时损害真实过渡；因此frame/ABX、内容检查与输出PCM都需要。
- 逐句gain/headroom、声码器重建、single-speaker TTS会改变domain；必须ID对照，不把结果全部归给发音。
- 神经模型可利用教师盲点；限制低维平滑增益、跨encoder保留、输出音质检查不能保证完全消除，结论需如实限定。

## 10. Assumptions / Unknowns

- VERIFIED：v1优势/规则阴性已落盘；mask保护phone边缘；LRS3仅18训练groups，40 evaluation已见。
- VERIFIED：TRAJECTORY120/F0192容器hash匹配；其余资产核验等级见2.2。旧learned50 WAV存在，未完整审计。
- VERIFIED：旧resource_snapshot缺RAM门；timing用zip且未应用silence_iou，新模块不能照搬。
- LIKELY：父manifest能恢复多数跨run绑定；必须实际核验hash，文件名不是证据。
- UNKNOWN：优势中时长、边界、录音域、上下文各占多少；P1/P2回答。
- UNKNOWN：中文speaker/旧路径/层坐标能否恢复；不足不阻断LRS3。
- UNKNOWN：规则和学习能否跨encoder构建优势、V100完整句反传是否可承受；分别实验与profile验证。
- UNKNOWN：人工时序、可懂度、TFG迁移；本轮不以代理指标代替。

## 11. Handoff Contract

下游先按第3节具体锚点实现P0/P1与独立checker，复算v1并建立资产atlas；然后执行所有可用历史机制对比，再按预定门槛构建候选与训练。遵守第5节全部不变量，仿照第4节冻结/审计模式，补丁限制在第7节范围。

完成标准是得到可复查的机制证据表、生成音频的真实可分度/时序结果和明确下一分支，而非必须训练或必须阳性。发现局部资产缺失继续独立分支；若仓库证据与核心协议冲突（如无可靠pair绑定、模型层定义不同、无法建立隔离划分），停该依赖分支并在报告说明，不换数据/标签/门槛蒙混执行。不做无关重构、不把探索结果升级成确认、不把下游视频成功作为本轮未授权的默认扩展。

## Implementation Status（2026-09-21）
- [implementation] 已落地并运行协议 `phone_separability_mechanism_v1`：显式资产 registry/哈希/重合图/固定 FIT-DEV-E_SEEN 划分；HuBERT layer 6 与 XLSR layer 10 的 full/core/boundary/matched-1frame atlas；严格 FIT-only reference 与 DEV、E_SEEN 分开评分；自然时钟 waveform 构造；候选规则重抽取与 DEV gate runner；独立 checker。
- [resource_check] 全量 atlas 与候选启动前均检查 GPU、RAM、磁盘；候选正式重跑时 GPU 空闲且无竞争 compute app，HuBERT/XLSR 候选重抽取完成，未终止任何外部任务。
- [audit] LRS3 registry 通过：240 对、480 assets、480 available、0 missing；FIT/DEV source-group 隔离，E_SEEN 40 groups 只作已见评估，不参与选择。
- [evidence] 全量严格 atlas run=`runs/phone_separability_mechanism_full_v2_20260921/`。HuBERT E_SEEN T−N accuracy：full +0.09260（95% CI [+0.06777,+0.11913]）、core +0.10517（[+0.07762,+0.13438]）、matched_1frame +0.10576（[+0.07415,+0.14027]）；margin 分别 +0.01303、+0.01471、+0.01332，均为 40 source groups。XLSR 对应 +0.08999、+0.09459、+0.07670，CI 均不跨 0。DEV 6 groups 方向一致，但 boundary_end 支持不足（0/4 eligible），不作强结论。
- [mechanism] 来源域 AUC（N vs T nearest-domain-centroid）为 HuBERT DEV/E_SEEN=0.704/0.715、XLSR=0.664/0.683；ABX error 也偏向 TTS（HuBERT E_SEEN N/T=0.0988/0.0638；XLSR=0.1288/0.1150）。这支持“域/声学实现差异参与优势”的假设，但不等于音素因果机制。
- [construction] FIT-only phone spectral-shape template 得到 71 个可用标签；natural-clock 构造生成 1050 条 arm WAV，10 arms、DEV/E_SEEN 候选，全部 waveform contract 通过；输出保持自然样本数、mask外 PCM 和自然时间轴，E_SEEN 未参与选择。
- [candidate_gate] 候选重抽取完成：HuBERT primary 在 DEV 对 10 个 arms 评分，6 个 selectable rules 均未通过预注册 gate（至少5 groups、accuracy CI low>0、margin CI low>0、达到 TTS core effect 的50%）。最佳 selectable arm `PHONE_SHAPE_BETA1_SPEECH` 的 core accuracy +0.003893（95% CI [-0.001155,+0.009166]）、margin +0.000718（CI [+0.000425,+0.001010]），而 TTS core baseline 为 +0.076801；XLSR 作为 held-out cross-encoder 也未触发选择。因无入选 arm，E_SEEN 未被用于选择或锁定评估。
- [training] `06_training/decision.json` 为 `SKIPPED_NO_RULE_PASSES_DEV_GATE`；没有无依据地进入神经训练。训练输入快照已记录实际 candidate gate，而非保留 construction 阶段的 pending 占位。
- [validation] 独立 checker：480 asset bindings、1050 construction rows、atlas statistics 均 PASS；`pytest -q tests/experiments/phone_separability_mechanism` 为 14 passed；candidate 输出与 report 已写入 run。
- [artifacts] 严格 atlas/机制/构造/候选报告：`runs/phone_separability_mechanism_full_v2_20260921/08_report/report.md`；候选明细：`07_candidates/selection.json`、`scores_hubert.jsonl`、`scores_xlsr.jsonl`；独立 checker：同目录 `08_report/checker.json`；代码入口：`scripts/experiments/phone_separability_mechanism/candidate.py`。
- [decision] 当前最稳结论是：LRS3 原始 TTS 的音素可分性优势在去除边界、使用中段/单帧和跨 HuBERT/XLSR 后仍存在；本轮自然时钟规则未构建出接近该优势的 phone probe 提升。该负结果只否定本轮预注册规则/预算/谱形假设，不证明自然音频增强不可行；后续仍应作为新的探索协议处理，不把 phone probe 提升写成 TFG/SyncNet 因果解释。
### Relations

- extends [[LRS3 音素优势验证与自然音频规则增强 Implementation Spec]]
- follows [[LRS3 音素优势与自然音频规则增强实现审计 2026-09-20]]
- part_of [[TTS 增强 TFG 的原因：证据、排除项与因果分叉]]
- reuses [[MFA-linear 连续轨迹机制消融 support30 2026-09-16]]
- reuses [[TTS 声学变化重组轨迹与剩余项干预：实现与结果]]
- reuses [[TTS 音高起伏交换 v3 配对 Harvest Wav2Lip audit2 结果]]
- reuses [[自然时钟音频双目标小样本筛查 2026-09-13]]
- reuses [[TTS 时间辨识与同文本实例交叉实验 2026-09-17 结果]]
- reuses [[20-aishell100-cross-condition-per]]
- reuses [[Scale 0.5 enhancer retrain and downstream evaluation]]
- constrained_by [[28-diagnosis-chinese-hubert-training]]
- constrained_by [[15-tfg-link]]
