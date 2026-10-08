---
title: 自然时钟受约束音素增强与机制对照 v2 Implementation Spec
type: research_topic
permalink: tts-exp/research/自然时钟受约束音素增强与机制对照-v2-implementation-spec
status: concluded
spec_status: implemented
protocol: phone_separability_enhancement_v2
measurement_version: signed_margin_fixed_support_v2
question: 在自然时序和受限失真下构建跨编码器音素可分性提升，区分逐句标签辅助优化、固定增强器泛化及机制对照
scientific_status: EXPLORATORY_DIRECT_RULE_GAIN_LEARNED_NO_GAIN
implementation_status: implemented_formal_run_checker_pass
tags:
- phoneme-separability
- natural-clock
- constrained-enhancement
- implementation-spec
- exploratory
- mechanism
---

# 自然时钟受约束音素增强与机制对照 v2 Implementation Spec

## 1. Objective

实现探索协议 `phone_separability_enhancement_v2`：在保留自然音频时间轴、停顿和受限波形失真的条件下，直接构建音素可分性提升，再用少量对照定位有效修改。工程完成与科学阳性分别验收；不要求先解释完整的 TTS 优势。

本次文档交付不运行实验。后续实现必须完成测量校准、逐句优化、小型增强器训练、冻结候选跨编码器验证和机制对照；旧规则没有过门槛不能成为跳过优化或学习的理由。

两个主问题分别报告：
- Q1：给定 natural + transcript/MFA，逐句调整参数是否能在同一音频上提高跨编码器音素可分性？这是标签辅助的局部可行性实验。
- Q2：只在 FIT 训练的增强器，固定参数后在 DEV/E_SEEN 上前向，是否仍能提升？推理只需 natural 与自然停顿 mask，不能读取同句 TTS 或按待测音素标签优化参数。
- 机制问题：有效修改是否需要时间变化；统一编解码重建是否改变 N/T 差距。两者均不预设为 TTS 原因的最终证明。

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 初始设计；仓库锚点、真实训练接线、指标纠偏、分支预算和冻结验证 | September 21, 2026 | user（spec任务）；agent（方案，未执行） |
| 实现交付；v2包、测试、smoke独立checker与full audit/calibration完成；正式优化/训练受磁盘安全门阻断 | September 21, 2026 | agent（2026-09-21） |
| 正式 run 完成；DIRECT/XLSR 探索性评价、LEARNED 阴性、report 与独立 checker PASS；机制/质量/TFG 仍有未运行项 | September 21, 2026 | agent（2026-09-21） |

### Observations

- [status] concluded
- [progress] implemented_formal_run_complete；scientific_status=EXPLORATORY_DIRECT_RULE_GAIN_LEARNED_NO_GAIN。
- [question] 能否在自然时序约束下构建跨编码器音素可分性提升，并区分逐句可行性与增强器泛化？
- [evidence] 父实验原始 TTS 的 E_SEEN core accuracy 优势为 HuBERT +0.10517、XLSR +0.09459；最佳规则 DEV accuracy +0.003893，CI 跨零。
- [audit] 父代码 margin 是预测 top1−top2，不是正确标签 signed margin；旧 ABX 可能 A=X；旧来源 AUC 在评估 split 内拟合中心。本轮必须校准，旧辅助数值不得直接作为新门槛。
- [decision] 逐句优化与学习独立进入；XLSR 在本轮所有候选、checkpoint、机制参数冻结后统一打开。
- [constraint] 现有 E_SEEN 与两编码器均有历史暴露，本轮只能声称探索性跨编码器迁移。
- [constraint] phone proxy、音质、时序、泛化、TFG 效用分别判定。

## 2. Repository Model

当前可用流程为：

`父 registry/双侧TextGrid/缓存 → FIT音素中心 → HuBERT/XLSR评分 → 固定自然时间轴规则 → DEV规则门 → stage_train记录跳过 → 报告`。

父 run：`runs/phone_separability_mechanism_full_v2_20260921/`。只读核验得到 240 对、480 资产；FIT=135句/12 source groups，DEV=65句/6 groups，E_SEEN=40句/40 groups。source_group 是来源视频键，不自动等于唯一说话人。

新流程：

`审计与冻结 → 指标/梯度/音频校准 → OPT_STATIC与OPT_DYNAMIC逐句优化 + LEARNED训练 → DEV固定选择 → E_SEEN生成并封存 → 独立XLSR评分/时序质量验证 → 两类机制对照 → 独立checker/报告`。

### 2.1 发现的实现差距

1. `run.py::stage_train` 当前只产生 skip decision，尚未把真实波形重建、冻结探针、DEV选择接入训练。
2. `train.py::train_one_seed` 的 keep 项取 batch 内 candidate（缺失时直接 source），未保证它是本步生成波形；diagnostics 在 backward 后调用 autograd.grad，存在计算图已释放风险。不能原样接线。
3. `train.py::differentiable_ssl_forward` 的直接模型分支未复现 worker processor 的归一化。需要 Torch 可微预处理与 worker 数值对齐。
4. `metrics.py::_predict_prepared/score_frozen_support` 的 margin 为预测 top1−top2，错误且自信的分类也可能很高。旧“正确音素间隔变大”解释需纠偏。
5. `metrics.py::abx_score` 在同类只有一个 occurrence 时会 A=X，且各臂重新抽样；不可作为固定配对辨别证据。
6. `metrics.py::domain_auc` 用被评分 split 的 N/T 拟合中心，属于 split 内描述，非跨样本泛化 AUC。
7. `audio.py::build_speech_mask` taper 后需重新施加硬保护区，才能保证全部自然停顿及 guard 精确不变；不能只验证“所生成 mask 外一致”。
8. `timing.py::compare_boundaries` 仅按 pause IoU 检验停顿，未逐一强制起止偏差，也不是完整的一对一 pause 匹配。
9. 父 checker 从 summary/group effects 重算部分统计，不足以排除原预测或 support 定义错误；新 checker 须从向量/标签和 PCM 独立重算。

这些是源码检查结果，不代表已运行缺陷复现实验。v2 中增加对应测试并重评分；原准确率结果不因 margin 命名错误自动作废。

## 3. Code Anchors

新包路径缩写 `P=scripts/experiments/phone_separability_enhancement/`；下表的新增符号为明确实施接口。

| path | symbol | 当前职责 / 所需变化 |
|:--|:--|:--|
| scripts/experiments/phone_separability_mechanism/inventory.py | load_registry, build_overlap_graph, check_resources | 复用资产解析和重合检查；v2继承实际split，不重洗；资源包装补齐自身PID白名单、GPU UUID及磁盘余量 |
| scripts/experiments/phone_separability_mechanism/features.py | load_bundle, load_token_records, pool_views, match_occurrences | 固定模型加载、缓存、视图和序列匹配；复用读取，v2显式冻结support与occurrence key |
| scripts/experiments/phone_separability_mechanism/metrics.py | fit_probe_bundle, paired_group_bootstrap, sign_flip_p, benjamini_hochberg | 复用FIT中心与统计工具；不复用旧margin、自动complete-case、旧ABX为新主判据 |
| scripts/experiments/lrs3_phone_rules_worker.py | extract_features, frontend_config_from_model, processor_fingerprint, read_pcm16, write_pcm16, realign_audio | 固定非可微评价器/PCM/MFA基准；保持公共接口不变 |
| scripts/experiments/phone_separability_mechanism/train.py | BoundedGainEnhancer, phone_margin_loss | 复用小型24band TCN及softplus形式；不用原train_one_seed、默认QC为True的select_checkpoint |
| scripts/experiments/phone_separability_mechanism/audio.py | build_speech_mask, _quantize_masked | 参考时间轴、mask和headroom模式；Torch重建与硬停顿保护放新包，避免NumPy断梯度 |
| scripts/experiments/lrs3_phone_rules_audio.py | make_gain_control | 复用能量匹配思想；v2必须实际保存并评分GAIN_<arm>.wav，不仅记录available |
| scripts/experiments/phone_separability_mechanism/timing.py | compare_boundaries, evaluate_timing_controls | 参考单调匹配；新包补全pause一对一匹配、speech分母、局部时移校准 |
| scripts/experiments/lrs3_dac16k_codec_identity/audio.py | load_dac_model, _model_forward | 复用冻结DAC加载及右侧pad/crop；新adapter分别处理N/T，不调用旧run_stage01 |
| scripts/experiments/lrs3_dac16k_codec_identity/config.py | DAC_SOURCE_COMMIT, DAC_CHECKPOINT | 已固定的源码/权重定位；显式核验父protocol哈希，不重用旧全局RUN_ROOT |
| P/config.py, P/run.py | FrozenProtocol, stage_audit, stage_calibrate, stage_optimize, stage_train, stage_lock, stage_evaluate, stage_mechanisms, stage_report | 新schema、阶段依赖和run目录；真实执行状态机 |
| P/data.py | inherit_registry, select_pilot, freeze_support, build_manifest | 隔离划分、选择、自然侧occurrence及路径权限视图 |
| P/metrics.py | signed_phone_scores, build_abx_triplets, score_fixed_support, decide_gain | 新测量语义、固定分母、配对统计及结论 |
| P/teacher.py | FrozenPhoneTeacher, normalize_waveform, encode_with_input_grad | 可微Torch归一化、冻结HuBERT、固定core池化及signed margin |
| P/audio.py | make_protected_mask, BandGainRenderer, export_pcm, build_gain_control | 可微重建、硬约束、最终PCM审计 |
| P/optimize.py, P/train.py | optimize_utterance, train_enhancer, select_dev_candidate | 逐句有限预算优化、真实训练、DEV-only锁定 |
| P/quality.py, P/mechanisms.py | audit_timing, export_blind_pack, static_dynamic_contrasts, codec_contrasts | 时序/质量检查与两类机制对照 |
| P/check.py, P/report.py | check_run, write_report | 独立核算、缺失上下界、分维度结果、下游交接 |
| scripts/configs/phone_separability_enhancement_v2.yaml | 全配置 | 新建唯一参数源，冻结默认值与资产绑定 |
| tests/experiments/phone_separability_enhancement/ | test_data, test_metrics, test_teacher, test_audio, test_optimize, test_train, test_quality, test_protocol, test_check | 覆盖第8节合同，不以旧14项测试代替新验收 |

## 4. Reference Pattern

- `phone_separability_mechanism/features.py::extract_asset_views`：保留整句上下文、明确层索引、前端感受野中心、PCM/model/TextGrid绑定；梯度分支不能继承其no_grad/NumPy路径。
- `phone_separability_mechanism/rescore.py::score_atlas_rows`：FIT-only reference、DEV/E分开落盘；v2主统计必须使用显式support，不能依赖候选各自eligible交集。
- `phone_separability_mechanism/candidate.py::_select_dev`：选择与最终评价分离；不用规则必须达到TTS的50%才训练的门槛。
- `phone_separability_mechanism/train.py::BoundedGainEnhancer`：24band、64channel、dilation=[1,2,4]残差TCN、零初始化输出。复用网络定义，重写真实训练闭环。
- `scripts/train_waveform_hubert_enhancer.py::hubert_features`：冻结教师但保留输入梯度；其旧目标/归一化/中文数据划分不可照搬。
- `lrs3_dac16k_codec_identity/audio.py::_model_forward`：可复现重建及pad/crop元数据。旧hash字段有container/PCM含义混用风险，新包统一分开命名。
- 对“逐句有约束优化→可复用增强器→冻结跨模型评估”的完整流程，没有可直接调用的现成runner；采用新包adapter而非全仓重构。

## 5. Invariants

1. 原数据、v1配置/run/科学决策不可覆写；新协议和新run。冻结未跟踪复用代码的内容哈希，不能只记git commit。
2. FIT/DEV/E_SEEN保持父划分。PCM重复、同source、已知同speaker跨split要拦截并报告；speaker无法核验时只声称source隔离。
3. 探针中心只从原始FIT N/T计算；不从增强音频、DEV、E更新。主mixed与辅助natural-only中心均在校准阶段冻结。
4. HuBERT L6是唯一优化与选择教师；XLSR L10的新候选分数在本轮一次性锁定后才评价。P0/P1可读原始N/T的XLSR缓存以冻结支持、校准旧测量，但不得提前评本轮OPT/LEARNED/机制臂。既往查看过的XLSR/E_SEEN不能被改称全新盲测。
5. 各候选使用同一自然侧support；不根据候选置信度、输出对齐或改善幅度换token、换句或换分母。
6. 逐句优化可用该句正确标签，这是已知标签条件；增强器测试时不得按该句标签反传。二者的generalization标签不可混写。
7. 主候选输出16kHz/mono/PCM16、样本数完全一致；全部自然silence及其guard、音频边缘硬保护区PCM逐样本一致。输出后重对齐仅用于验证，不重写训练/主评分标签。
8. PCM写盘后重新读回评分。float波形得分、未导出checkpoint、重建前增益上限不等同最终有效干预。
9. T0波形不变时钟不代表T1声学边界不变；无人工标注不得写“人耳确认”。
10. 不用当句TTS作主方法目标或输入；FIT TTS仅用于冻结mixed探针、TTS比较和独立机制对照。
11. 无GAN、自由逐采样扰动、文本重合成、时长伸缩或RL；首轮使用低维平滑增益和梯度法。改变变换族须新revision，不能因阴性自动加码。
12. 缺依赖/资源/支持不足与科学NO_GAIN分开；不得默认QC通过、把SKIPPED记为COMPLETE、按效果删除失败样本。

## 6. Implementation Plan

### 6.1 P0：冻结资产、support与预算

配置与父绑定：
- protocol_id=`phone_separability_enhancement_v2`，measurement_version=`signed_margin_fixed_support_v2`。
- parent_run为第2节目录；原manifest/tokens路径与哈希继承父protocol并重新核验，registry新增绑定当前container/PCM/TextGrid/token_signature。
- HuBERT：facebook/hubert-base-ls960，revision=dba3bb02fda4248b6e082697eee756de8fe8aa8a，hidden_states[6]。
- XLSR：facebook/wav2vec2-large-xlsr-53，revision=c3f9d884181a224a6ac87bf8885c84d1cff3384f，hidden_states[10]。
- natural core定义为音素时间区间20%–80%；full及matched_1frame为次要视图，不根据结果选层/选视图。
- 保持父FIT12/DEV6/E40组。pilot从每个FIT组按sha256("pse-v2|pair_id")取前2句（最多24），每个DEV组取前4句（最多24）；E保持40句全体。只在自然基线格式/标签/帧支持合格后确定资格，保存所有未入选原因。不能按TTS优势或候选效果抽样。
- 探针FIT使用全部135句及其TTS，不限于pilot；学习训练同样使用全部FIT合格句。主DEV选择使用冻结pilot24句；其余DEV仅留档，不事后扩充选择分母。
- natural主support先冻结每个(pair_id, token_id, view, encoder)与label、帧索引。共同标签取FIT两编码器合格label交集；每label各域≥20 tokens/≥3 FIT groups。每句≥5labels、≥10有效tokens、speech occurrence coverage≥0.70。两编码器共同support在生成候选前由原始N构建，不能看TTS分数决定入选。
- 短音素无帧标missing，绝不补邻帧。source→候选一一绑定；TTS对比另冻结N/T可唯一匹配的共同occurrence support，主E−N分母不因TTS缺失而缩小。
- resource：每次重计算/下载前检查GPU UUID、compute PID、RAM、磁盘；CUDA任务要求无外部compute进程、空闲显存≥12GiB、可用RAM≥8GiB；锁内本任务进程白名单，不能把自己当竞争进程。CPU任务不因GPU忙被阻断。
- 磁盘要求：预计本阶段新产物×1.2加2GiB余量小于当前free，且free≥4GiB。新产物总上限3GiB（下载额外资源也纳入预算）；缓存引用父run，不复制整套模型/特征，流式写结果；只留selected/last checkpoint与必要PCM，checkpoint中不保存冻结教师。
- 资源不足记录RESOURCE_BUSY/DEPENDENCY_UNAVAILABLE，保存可resume状态；不杀别人进程、不自动清旧run、不改物理约束。缺失公开资源可经http://[redacted-ip]:7890下载到固定缓存，固定revision/hash后才能执行。
- smoke最多2条、仅FIT、engineering_only；先CPU合成测试，再真实最长FIT句的完整前向/反向profile。禁止静默裁句、分块或AMP换语义来通过OOM；允许等价gradient checkpointing但需数值复验。
- 默认active GPU预算16小时：校准/评价/机制共≤4h，逐句优化≤8h（FIT/DEV≤4h，E≤4h），学习≤4h。先到阶段时间或step上限就记录BUDGET_LIMITED，不补跑到显著。smoke外的运行时间全计入。资源等待不计active预算。

依赖开放范围：优化进程仅获得自然波形、固定标签/mask和HuBERT中心；评价进程可读TTS/XLSR。锁文件之外的E评分不得被优化/选择读取。预先注册12条E_SEEN hash顺序样本用于盲听和后续TFG交接，不按结果更换。

### 6.2 P1：测量与梯度校准（主分支必需）

为每个encoder/view保存原始冻结中心、标签表和hash。复用fit_probe_bundle的group-equal/condition-equal规则。

正确margin定义：
`s_k=cos(z,c_k); m_y=s_y-max_{k!=y}s_k`。
预测argmax按排序label表确定性处理平分；另外保存legacy_top1_gap以解释历史差异，不能用它优化或判定成功。unit-normalize池化后的embedding，而非把已归一化帧再平均；与当前worker一致。

主统计：每source group内按label等权，再对该label全部固定occurrence求均值；最后group等权。accuracy和signed margin采用相同权重，不混用token-weighted margin。序列重复不能增加独立n。主CI为source-group paired bootstrap，PCG64 seed=20260921，10000 draws；bootstrap全臂共用抽样表。

ABX：在natural固定support上生成并保存occurrence三元组，A与X同label但必须不同occurrence，B不同label；每组各合格label均衡轮转最多20组triplets，按seed/hash排序。N/E复用完全相同三元组；distance=1−cos，平分计0.5 error。本轮这是within-source token ABX，不宣传为跨说话人ABX。另报告可用triplet覆盖；无足够实例则NOT_ASSESSED，不能A=X补齐。TTS ABX只在独立N/T共同匹配support上重建，不能和原始全体N/E值直接拼接。

来源AUC若保留：N/T中心只在FIT训练，DEV/E逐组评分后等权；按phone平衡后单列。旧split内AUC仅作为legacy描述，不用于分支门槛。

校准输出：
1. 从父token向量重算T−N accuracy、新signed margin、新固定ABX，写入本run的legacy_metric_audit；旧文件保持不变。同一旧support/旧accuracy定义应复现父值至1e−6；v2共同support变化另列，不能强求不同support完全相等。
2. 仅在父DEV上针对父最佳规则及N/RT重新提取必要向量，核对旧margin趋势是否仍成立；缺缓存就CPU/GPU顺序提取，不伪称已核实。
3. Torch processor：完整句float32、与固定feature extractor相同均值/方差/epsilon与attention mask；不得detach输入或通过NumPy/HF processor完成训练预处理。先比较input_values，再比较hidden_states[6]/池化向量；同设备相同精度相对L2≤1e−4，cos差≤1e−5。保存实际误差。
4. teacher.eval、requires_grad=False；waveform及变换参数对非平凡phone loss有有限非零梯度；teacher所有grad为None。零初始化最后层只保证输出层初始有梯度，不能错误要求第一步所有TCN层都非零。
5. STFT_ID读回PCM应与N完全一致（必要时zero-gain直接返回source PCM），通用float重建max error≤2e−6；输出评分差在数值容差内。
6. 合成错误自信分类样例必须signed margin<0但legacy_gap>0；固定ABX禁止A=X；候选坏向量不改变support；模拟dropout/模型train状态必须拒绝。

P1失败时停止依赖该测量/梯度的分支，并产生可复现失败报告；独立codec资产审计仍可继续。真实结果不可绕过这些工程门。

### 6.3 P2：统一受约束可微变换

新BandGainRenderer输入x、固定自然mask、24band增益场，输出float waveform及约束元数据。sr=16000，n_fft=win_length=512，hop=128，periodic Hann，center=True、reflect padding、显式ISTFT length=len(x)。长度≤256样本不进入正式实验，记录SHORT_AUDIO。

24频带节点：mel=2595log10(1+f/700)在0–8000Hz等间隔取24点，对FFT频点做相邻节点线性插值，插值权重每频点和为1。网络输入为同一节点对应的三角加权log-power能量（权重频率和归一化，floor=1e−8），随后用FIT自然speech帧均值/标准差标准化；统计不使用DEV/E。

`g_db=gmax*tanh(u)`，gmax固定候选{3,6}dB。节点插值及5帧三角核[1,2,3,2,1]/9时间平滑后，乘原复STFT的10^(g_db/20)，原STFT相位不直接改写。最终mask/headroom会改变实际相位，因此只称“原相位STFT增益变换”。

硬mask：自然全部silence区间（包含短停顿）及两端各10ms保护；首尾speech边缘10ms保护；保护区外向可编辑区内部作5ms taper，最后再次把硬保护区置0。其余speech内部允许编辑，保留自然内部音素边界位置不代表边界声学必然不移动。

残差r=mask*(ISTFT−x)。按下式约束，所有操作可微（除离散mask/固定索引）：
- rho=sum(r²)/max(sum(x²),1e−8)，两者限mask>0；alpha_energy=min(1,sqrt(0.01/max(rho,1e−12)))。
- alpha_peak为保证最终范围[-1,32767/32768]的最大[0,1]残差缩放；沿用残差缩放而非全句归一化。
- alpha=min(alpha_energy,alpha_peak)，y=x+alpha*r。记录缩放前后rho、alpha、gain剂量、改变样本比例；允许缩放导数，禁止把y整个detach。
- 不做free waveform residual、不clip饱和制造结果；PCM量化最后进行，硬保护区直接复制原PCM。
- distortion QC：speech residual SNR≥20dB、speech RMS变化绝对值≤1dB、无新增饱和sample；这些是操作性预算，不是可懂度/自然度保证。
- exported PCM若不满足预算，按同一规则减alpha并重验；最终scores只针对重读PCM。若alpha<0.05或最大样本改变量<1 LSB，标MANIPULATION_TOO_WEAK（仍保留结果/分母）。

### 6.4 P3：逐句直接优化与对照

固定实验臂：
- N_ID、STFT_RT：原音与零增益重建。
- OPT_STATIC_{3,6}：每个speech occurrence一个24维可优化向量，区间内常量；短/不支持目标的speech也保留mask但其参数固定0。
- OPT_DYNAMIC_{3,6}：每个speech occurrence设K=max(1,ceil(duration/0.08))个均匀中心节点，每节点24维；线性插值到帧，端点用最近节点。允许音素内变化；静态与动态使用相同band、mask、平滑和能量预算。
- LABEL_PERM_6：与OPT_DYNAMIC_6相同预算，优化目标用FIT label表的固定无不动点置换；真实评分始终用正确标签。只用于FIT/DEV，不能参选；证明是否存在教师目标驱动，不以它音质受损反证主候选。
- GAIN_<arm>：每个正式候选对应实际输出能量的mask内常量gain控制，落盘且评分，mask外PCM不变；能量相对误差≤1e−3、同peak约束，否则GAIN_CONTROL_UNAVAILABLE且相关机制结论不成立。

优化：全部u从0开始，Adam(lr=0.05)，clip_grad_norm=1，固定120 steps；在0/25/50/75/100/120保存标量检查点。以导出PCM通过T0/distortion且该句HuBERT signed margin最高的检查点作为该次算法输出，平分取rho更小、step更早；允许选step0并记录IDENTITY_RETURNED。这一逐句内选择规则对FIT/DEV/E相同，不能拿XLSR、E聚合效果或输出MFA来选step。

loss（对固定可评分token先label等权）：
`L_phone=mean softplus((0.05-m_y)/0.1)`；
`L_keep=sum((y−x)²)/max(sum(x²),1e−8)`；
`L_tv=mean(diff_t(g_db/gmax)²)+mean(diff_f(g_db/gmax)²)`；
`L=L_phone+1*L_keep+0.01*L_tv`。
损失系数是本轮预注册工程假设；硬投影负责主要失真约束，避免重用旧过强保持损失。L_keep必须从本步y计算。所有臂包括静态采用同一公式；无可用目标token的句为BASELINE_INELIGIBLE，不假装零loss训练成功。

每25步在backward前对同一完整参数列表计算phone与保持/TV梯度；unused参数补同形零，不能分别拼接不同长度向量；记录norm、cos、投影触发率和实际PCM剂量。非finite立即失败；目标无梯度为ENGINEERING_FAILURE；连续3次cos<−0.5、keep/phone>10且signed margin无改善时标OBJECTIVE_CONFLICT，保存当前结果并停止该句，不自行调权重。

先完成FIT/DEV全部4个主臂和置换对照。任务顺序按split、source group轮转并在组内arm轮转，避免全局预算耗尽仅覆盖前几个组；在锁中保存完整队列。每个样本每臂总时限120s，超时记录实际step，不能用快速样本替代。资源/时间造成未生成项单列，并输出完整队列界限；未达到覆盖要求不作科学阴性结论。

### 6.5 P4：学习可复用增强器（不依赖P3阳性）

在P1与renderer通过后启动，P3是否成功只影响解释，不影响启动。复用BoundedGainEnhancer(24,channels=64,max_gain_db=6)，网络接6.3的固定logband特征，输出g_db，再经相同renderer得到y。推理时不输入音素label、TTS、HuBERT特征；可读取由natural MFA产生的停顿mask，明确依赖transcript/MFA，不能称无文本全自动系统。

训练FIT全部合格句，batch1完整句，source-group均衡采样：每epoch打乱groups，再每组循环取句；AdamW(lr=1e−4,weight_decay=1e−4)，梯度累积4句，clip=1。相同L_phone+L_keep+0.01L_tv；两种导出剂量为训练输出原值与0.5倍（6/3dB），不额外训练四套网络。

seeds=[20260921,20260922]；每seed≤1500 optimizer updates或2h。step0及每100 updates在DEV24句评HuBERT，导出PCM后检查。每100 updates对固定4条FIT哨兵做梯度诊断；diagnostics在backward前或独立重前向，不在释放图后复用。每seed保留last和selected；明确记录实际完成updates，BUDGET_LIMITED不等于收敛。

同结构NO_PHONE对照移除L_phone，只跑首seed的100 updates或15min（计入学习总预算），预期从零初始化停在identity附近；若明显改善/改变，查seed/训练数据/评分泄漏。不能用no-phone对照替代第二个正常seed。

checkpoint只读DEV HuBERT、T0及distortion；先剔除硬约束失败，再按signed margin降序、accuracy降序、rho升序、step升序。所有QC必须显式给出True，不接受missing=pass；不要求已经达到TTS50%或显著性。step0可被选择，但标NO_LEARNED_IMPROVEMENT。首seed为发布候选，第二seed仅稳健性；不能按E/XLSR挑seed。每个seed的3/6dB按同一DEV规则选一个。

真实集成验收需模拟至少101次update，证明y依赖模型参数、L_keep梯度连通、teacher冻结、保存恢复optimizer/RNG无遗漏、DEV评价不更新模型。不能将旧train_one_seed返回COMPLETE视作已接通。

### 6.6 P5：锁定与一次性迁移评价

在打开本轮XLSR候选分数之前写selection_lock.json，包含：
- protocol/config/code/dependency/input/probe/support/hash，暴露历史说明；
- DIRECT主候选：4个OPT臂中按DEV的T0/distortion通过、signed margin、accuracy、rho、arm_id排序选1个；
- LEARNED主候选：首seed固定checkpoint与剂量；第二seed单独封存；
- OPT_STATIC/OPT_DYNAMIC机制对照预先固定6dB，不按最终效果换剂量；
- E40输出计划、逐句内优化规则、ABX表、QC阈值、所有对比与缺失规则。
没有合格臂就selected=null并如实结束该方法；identity step0合格但不称增强成功。任何涉及XLSR、E结果的设计修改都需新protocol revision，当前所有已看结果转为开发资料。

E阶段：
- DIRECT按冻结算法对E每句使用正确标签做局部优化，保持HuBERT教师；这是目标音频自适应，标TARGET_LABEL_ADAPTATION，不能称无标签泛化。
- LEARNED两个seed只前向，不更新任何参数/归一化统计，标FIXED_ENHANCER_SOURCE_HELDOUT。
- 主臂与各自GAIN控制全部先生成并保存PCM hash，再顺序加载HuBERT/XLSR评分；一次只驻留一个大模型。
- 两模型主view为core；full与matched_1frame做方向检查。输出逐token pooled embedding、完整cosine scores（float32矩阵分别保存npz）与标签/occurrence索引，方便独立checker，不在JSON逐行复制高维embedding。
- 主N/E冻结support；候选失败不得从分母移除。可评分但QC失败仍评分并标失败；向量缺失的token在保守主accuracy中计0、signed margin计−2，同时报告complete-case和accuracy上界1/margin上界2。科学提升须在保守下界分析也成立；失败覆盖不足则MEASUREMENT_INCONCLUSIVE，不能把失败惩罚写成观测声学下降。
- 统计主族为2种方法×2模型×(对N、对GAIN)的8个accuracy contrast；双侧group sign-flip p，统一BH q。预注册臂缺失列null/p=1纳入族大小，不删除。主CI为paired group bootstrap；DEV6组只供探索选择，不宣传充分功效。
- 每个方法报告结果独立，不把DIRECT优势当LEARNED结果。

分级判据：
1. TEACHER_ONLY_GAIN：仅优化教师正向；这是局部目标可操纵，非可迁移增强。
2. EXPLORATORY_CROSS_ENCODER_GAIN：E预冻结基线有效支持≥30 groups、生成覆盖≥90%，两编码器对N的accuracy提升≥0.010、95%CI下界>0、q<0.05；对GAIN accuracy CI下界>0且q<0.05；signed margin同向；固定ABX error差值E−N的95%CI上界≤0.010（非劣容差，不能称证明不变）。ABX有效覆盖不足则只能PROBE_GAIN_ABX_UNRESOLVED。
3. STRONG_PHONE_GAIN：在2成立基础上，在冻结N/T共同support分别检验D_half=(A_E−A_N)−0.5*(A_T−A_N)，两模型CI下界≥0。T−N≤0时恢复比例不定义；比例不用作训练启动门槛。
4. NATURAL_CLOCK_GAIN_AUTOMATED：2及第6.7的T0/T1通过；质量与人工时序另列。仅固定增强器符合第2条才能说“新source上的固定增强器有效”；不能声称全新说话人或新数据集泛化。
5. 完成充分预算/覆盖但未过门槛为NO_GAIN_AT_TESTED_BUDGET；预算中断、操纵几乎为零、测量不足分别标BUDGET_LIMITED/MANIPULATION_TOO_WEAK/MEASUREMENT_INCONCLUSIVE。都不证明增强不可能。
6. 第二seed给全体结果；若跨模型效果反号则SEED_UNSTABLE，保留主seed估计但不称稳健增强。

### 6.7 P6：时序、内容与音质

T0：样本数、sr、格式、finite、硬保护区PCM、预算、无新增饱和，逐句检查；任何失败阻止该句成为有效候选，不能因support未覆盖就忽略。

T1：固定本地MFA环境、词典与声学模型hash，对N重跑及输出各自对齐；不直接拿旧N TextGrid与新版本候选对齐比较。natural旧标签仍是主phone评分support，N重跑仅时序验证。speech occurrence通过唯一单调匹配，不用zip；匹配覆盖≥0.90、speech序列edit rate≤0.05、每句边界median≤20ms、p95≤40ms。全部自然≥50ms停顿与候选停顿按时间顺序一对一匹配，每个IoU≥0.50、起止偏差均≤20ms，新增/缺失长停顿判失败。主队列≥90%句通过全部T1条件，失败句仍留phone分母。

时序仪器先在FIT固定8句校准：重复N、全局右移80ms且截尾、单个≥160ms音素内部局部40ms位移但不移其他区域。要求重复N全部通过，全局控制≥7/8检出，局部控制≥6/8检出；局部无eligible phone明确不足，不换成更容易的全局控制。未校准时TIMING_MEASUREMENT_UNCALIBRATED，继续phone评价但不能给自然时序保持结论。

内容/质量：
- 对预定12句导出N/E/GAIN/T随机盲听包、匿名映射与评分模板，评清晰度、自然度、内容错误和停顿位置；人工未提交即HUMAN_NOT_ASSESSED，不编造听检。
- 本轮默认加入固定独立ASR检查。P0选择已经部署且可固定权重/processor的英文ASR；若无，允许下载faster-whisper base.en，首次审计时解析并冻结确切模型revision/文件hash及运行环境后才产生任何候选。固定language=en、beam_size=5、temperature=0、vad_filter=False，无reference提示，无previous-text跨句上下文。算法选择不依据候选ASR结果。
- WER使用同一自然转写reference和固定tokenizer，对N/E/GAIN/T全体比较；group bootstrap的E−N WER 95%CI上界≤0.01且point≤0.005作为自动内容非劣门。无ASR/转写不可靠时CONTENT_NOT_ASSESSED；无正确文本时不能拿模型自转写当真值。
- SNR/RMS/谱差只是预算；不将它们命名为perceptual quality。ASR/MFA不是人工可懂度或时序真值。
- 下载或依赖不足只影响该质量维度；自动phone结果仍可交付，但不能宣称“保持自然度且更清晰”。

### 6.8 P7：两类机制对照（无须主臂阳性才能报告）

A. 静态与动态：
- FIT/DEV按相同预算比较OPT_DYNAMIC_6−OPT_STATIC_6，两者参数自由度不同，因此差异首先说明“允许更丰富时间变化的变换族更有效”，不能直接等于自然发音动态的因果份额。
- 对OPT_DYNAMIC_6最终增益场做两种不再优化的消融：PHONE_TIME_MEAN（每个自然音素内取时间平均后广播）；PHONE_TIME_REVERSE（每个音素内增益帧序反转）。重用原renderer/mask。
- 消融作用于最终选中step的投影前g_db场，随后重新执行renderer。剂量匹配仅枚举scale=[0,0.25,0.5,0.75,1,1.5,2,3,4]，缩放场后clip至±6dB，按最终PCM测rho；选不超过原动态臂rho且最接近者，平分取小scale。须匹配至相对误差5%，达不到标DOSE_UNMATCHED。零残差单独记；不能通过放宽预算来配剂量。
- 在DEV与E预定全体执行静态/动态及消融；它们仅诊断，不回流候选选择。比较两模型accuracy/signed margin、T1及实际剂量，BH族单列；若动态优于静态且均值/反转消融在剂量匹配后丢失收益，可写“收益依赖所构建增益的时间组织”，仍不能写“已解释TTS机制”。

B. 编解码通道：
- 固定DAC16k tag=0.0.5、source commit=408235a9dcd2983684c87615a1bc2a8954f6eb47；权重sha256=95ab7176b67137d4d4c6c54b8d6ef3cea797faec228cb03ad084badcad570b4d，路径/源码manifest取父codec protocol核验。
- 对同一DEV24与E40的N/T分别做一次原样DAC encode/decode：N、T、DAC(N)、DAC(T)。使用所有codebooks、固定16k配置、内部右pad至下一hop再加一hop、输出仅右crop到各自输入长度；不为得分重对齐/调延迟，不把T裁成N长度。
- 本分支是全音频重建机制对照，允许改变停顿波形；不得冒充满足自然clock主候选T0。记录原样重建的固定延迟/边界偏差；若原始slot评分差异伴严重时序偏移，归因标TIMING_CONFOUNDED。
- 每侧沿自己原TextGrid/frame slots评分；N/T差距使用预冻结共同occurrence支持。主contrast为[(DAC(T)−DAC(N))−(T−N)]，并必须同时列DAC(N)−N与DAC(T)−T；不能只说差距缩小。
- 差距缩小可能是N改善、T损坏或两者共同改变；只有N改善且T没有明显退化/时序异常，才支持“统一重建可缓解部分差异”的候选解释。codec本身改变多种因素，不代表已隔离录音噪声或已证明声码器是唯一原因。
- 旧DAC音频只有输入PCM、源码、权重、参数、crop与输出hash全部一致才复用；旧cohort不一致就重新渲染当前条目。先12条DEV smoke/资源profile再全体；分支缺依赖记明，不更换codec刷结果。

### 6.9 P8：报告、下游接口与执行合同

每个方法的报告必须回答：教师是否提高、是否跨模型、增益规模、是否超越GAIN、是否保持T0/T1/内容、是否固定模型推理、哪些机制解释得到支持/反证/仍不确定。附原始队列失败率和上下界、实际预算、seed、support、全部阴性臂。旧规则最佳臂作为历史参照，不当成新公平训练基线。

若跨编码器且T0/T1通过，输出`tfg_handoff.json`，固定第6.1预定12句的N/E路径与hash、自然音轨及同源脸/源视频绑定，推荐后续Wav2Lip配对比较V(N)/A(N)、V(E)/A(N)，own-audio作为次要结果。标明映射缺失；不改用别的脸或其他句。默认本轮不生成TFG视频，downstream=NOT_RUN，避免把代理提升直接称最终唇形收益；报告明确下一实验入口。

若阳性，输出新确认计划：新source且核验speaker不重合，冻结方法/参数/评估器；按DEV group差SD估算δ=0.01、power=0.8、双侧α=0.05所需样本，列出多比较与coverage损耗。现有40组不能变成新确认集；本轮不访问项目sealed/test。

计划CLI：
```bash
python -m scripts.experiments.phone_separability_enhancement.run --config scripts/configs/phone_separability_enhancement_v2.yaml --run-id <id> --stage audit
python -m scripts.experiments.phone_separability_enhancement.run --config scripts/configs/phone_separability_enhancement_v2.yaml --run-id <id> --stage all --resume
python -m scripts.experiments.phone_separability_enhancement.check --run-dir runs/<id>
```
stage固定audit/calibrate/optimize/train/lock/evaluate/mechanisms/report；quality由calibrate的仪器校准与evaluate的T0/T1/ASR子步骤完成；all按依赖执行全部就绪分支，train不能因optimize科学阴性被跳过。XLSR机制/候选统一在lock后。smoke禁止生成科学GO。run-id只允许字母数字下划线横线；resume核验schema、config、代码、模型、input与support hash，变动拒绝缓存复用，要求新run。

目录合同：
- 00_audit：registry、splits、overlap、pilot、support_plan、resources、dependencies、exposure_ledger、budget。
- 01_calibration：probe/<encoder>/<view>、support.jsonl、abx_triplets、legacy_metric_audit、teacher_parity、gradient_check、timing_controls。
- 02_optimize：FIT/DEV arm PCM、gain_fields、step_metrics、failures、realized_dose；不存每步整句隐状态。
- 03_train：各seed history、selected/last、DEV outputs、no_phone；optimizer/RNG及版本可恢复。
- 04_lock：selection_lock、eval_plan、code/config/input/probe/support hashes。
- 05_evaluation：E PCM、GAIN PCM、predictions/score matrices、group effects、contrasts、failure_bounds、family_tests。
- 06_mechanisms：static_dynamic、ablations、codec绑定/音频/四臂contrast。
- 07_quality：T0、每句T1/pauses、ASR/WER、盲听包及未评状态。
- 08_report：report.md、decision.json、checker.json、mechanism_evidence.csv、tfg_handoff.json、confirmation_plan.json。

status.json至少分execution、engineering、stage_states、measurement、optimization、phone_gain、timing、content、human_quality、generalization、mechanism、downstream、budget、reason_codes。任何optional dependency失败必须有对应明确终态；execution=COMPLETE只代表就绪分支完成且其他分支有可审计终态，不代表全部科学问题解答。

关键表每行包含protocol、measurement_version、pair_id、source_group、split、arm、method/input_mode、seed/checkpoint、parent/output container/PCM hash、reference/support hash、effective_edit、status/reason；不能只存summary。

## 7. Expected Change Surface

### Must change

新增第3节P包、v2 YAML与对应测试目录；新run由执行阶段生成。维护本spec执行状态，并在完成时写一篇同主题Experiments结论笔记，链接本spec；保留负结果和指标纠偏。

### May change

若现有公共helper确有直接阻断且adapter不可解决，可最小修补helper并补相应回归测试；默认优先新包实现v2测量，保留legacy行为用于复现。只有实际重评分后才能修订父实验margin/ABX结论；修订经BM先读原文、保留changelog、注明新measurement版本，不覆写父run。

### Should not change

v1 configs/原始数据/旧run/主00–05流水线/TTS providers/TFG或SyncNet权重、AGENTS、CONTEXT、HANDOFF、用户其他dirty changes。当前任务只交付spec，不提交代码、不启动GPU任务、不删除缓存。

## 8. Validation Plan

按以下次序验收：

1. 静态：`python -m compileall -q scripts/experiments/phone_separability_enhancement`；项目可用ruff时对新包/测试定向检查；`git diff --check`。
2. `pytest -q tests/experiments/phone_separability_enhancement`：
   - split/PCM/source重合、确定性pilot、hash变化resume拒绝、路径越界、缺字段QC失败。
   - 错而自信样例：signed margin为负而top1_gap正；与独立NumPy手算一致；zero/nonfinite vector记录失败。
   - ABX同类单例不能A=X，N/E同triplets，平分0.5，arm顺序不影响样本表。
   - 删除最差候选token不能增加主accuracy；追加同group重复句不增加独立n；group与label权重可手算复现。
   - 池化20–80%与感受野中心一致，短phone不偷帧，N/E用同slot，输出MFA不能改变主support。
   - Torch processor对齐；冻结参数grad=None、非零输入/参数梯度；禁用normalize的错误wrapper应被校准拒绝。
   - renderer zero-gain identity，STFT短句边界，硬pause/guard在taper后仍精确保护，能量/peak投影，PCM导出后约束。
   - OPT最大step/时间终止、逐句checkpoint只用teacher/T0预算；E标签访问仅DIRECT，LEARNED推理不得按label更新。
   - 真实101步toy训练、keep连到本步y、梯度诊断不复用释放图、unused grad维度一致、保存恢复一致。
   - XLSR分数/E聚合数值改变不能影响selection_lock；封存后任何候选参数修改必须拒绝评分。
   - pause一对一匹配、起止超限但IoU合格仍失败、插入音素后不zip错位、局部/全局时移校准。
   - DAC四臂共同支持、T被损坏造成差距缩小时不能称N改善；原始slot时序混杂正确标记。
3. 回归：`pytest -q tests/experiments/phone_separability_mechanism tests/experiments/test_lrs3_phone_rules_worker.py tests/experiments/test_lrs3_phone_rules_audio.py tests/experiments/test_lrs3_phone_rules_metrics.py`。只在修公共helper时扩展对应回归，避免无关全项目GPU测试。
4. CPU合成端到端：至少3labels/4groups/2split，fake frozen teacher，显式test-only阈值；覆盖工程成功、科学阴性、失败样本不删分母、真实train不能写skip。无需用tiny fixture证明正式统计显著。
5. 真数据smoke：资源检查后完整HuBERT前后向、原worker parity、OPT导出回读、TCN至少一次更新、MFA重复与时移控制。评价XLSR的smoke只用FIT，不能看DEV/E候选趋势。
6. 正式独立checker：
   - 不调用runner的score/decide函数；从保存的embedding/centroids/labels重算cos、signed margin、support、group聚合、bootstrap/BH；从PCM重算mask、length、剂量和hash。
   - 核验所有ABX A/X不同且跨臂key相同；核验所有失败/缺失在冻结队列出现。
   - 对各主方法/seed、两模型分层hash抽至少10条输出重新前向，数值容差明确；报告抽查范围，不说全部重新推理。
   - 篡改summary、向量、support、正确label、PCM保护区、selection hash的fixture必须被发现。
   - 用源码独立实现最小公式；共享只读hash/PCM I/O可接受，不能共享同一判定函数“自证正确”。
7. 文档验收：明确历史已见、Q1与Q2的输入权限、实际训练steps、旧margin纠偏、工程/科学/音质/时序分别结论；无人工评分时仍标未评。至少一个有效负例/缺失案例能完整追溯到原始行。

## 9. Risks and Edge Cases

- 优化教师的盲点可能提高得分却不改善发音；跨编码器、固定ABX和质量检查能减轻但不能排除。音频对抗研究说明波形小改动可显著改变语音模型输出：[Carlini & Wagner, 2018](https://arxiv.org/abs/1801.01944)。不能把同模型指标当唯一终点。
- 探针分数受评价设计影响；保留冻结reference与control tasks思想，不把高accuracy直接写成机制解释：[Hewitt & Liang, 2019](https://aclanthology.org/D19-1275/)。
- E_SEEN标签用于DIRECT内循环是方法定义，不是无监督迁移；每句独立参数不能归为训练好的通用增强器。
- XLSR历史暴露以及同源架构意味着跨模型一致仍有限；本轮不为“新”盲测额外挑第三模型，未来确认需新数据/评估冻结。
- 6个DEV组使选型易波动；限制候选、固定seed、全列DEV分布，不无限扩超参。
- 20dB残差预算可能太紧或不足以保证听感；它是本轮测试空间。低剂量/投影饱和需单列，不能把无效干预解释为普遍不可能。
- 增益改变爆破/元音起点即使长度完全不变也可能使发音边界移动；T1必需且先校准。
- 动态模型自由度更大；均值/反转消融和剂量匹配帮助定位，但不构成唯一因果解释。
- TTS的全句时长不同；不得给TTS复制natural时轴或用同一frame index。
- codec差距缩小可能纯粹来自TTS损坏，必须四臂分解；不能以DAC重建说明所有声码器等价。
- 可用GPU与磁盘会变化；数值未知时资源fail-closed，允许其余只读/CPU分支继续。禁止自行租机、删除用户资产。
- 若新校准发现accuracy也受support错误影响，先报告差异与定位，再决定该测量分支是否可用；不可为保留旧结论篡改指标。

## 10. Assumptions / Unknowns

- VERIFIED：父registry含FIT135/DEV65/E40句与12/6/40 groups；原TTS有accuracy优势，规则主门失败。
- VERIFIED：旧训练stage未接通真实执行；旧margin/ABX/AUC及训练loop的上述源码问题存在，实际影响大小未重算。
- VERIFIED：本地已有固定revision HuBERT/XLSR接口、小型BoundedGainEnhancer、DAC源码/权重绑定记录、MFA worker；记录存在不等于本轮环境一定可运行。
- LIKELY：V100完整短句反传可行，需最长FIT句profile确认；不能假设显存空闲。
- LIKELY：自然时间轴低维变换可提供比固定规则更丰富干预；能否得到可迁移收益未知。
- UNKNOWN：正确signed margin重评分后旧规则弱正信号是否仍在。
- UNKNOWN：在固定20dB预算、3/6dB增益和既定训练时间内，优化/学习能否跨模型成立。
- UNKNOWN：自动ASR/MFA是否校准充分、人工自然度和时序是否保持、TFG是否获益；分别报告，不用代理值填空。
- UNKNOWN：source与speaker重合映射完整性；不能据source组数宣称40个独立说话人。

## 11. Handoff Contract

实现者按第3节锚点新增v2包，先完成指标纠偏、固定support和真实梯度测试，再按P0–P8执行。继承已核验资产与固定划分，保留第5节不变量；模仿第4节局部模式，不复制旧skip gate、unsigned margin、自由complete-case或未接线训练loop。不做无关重构。

下游最小可验收交付：新代码与定向测试；可复查的legacy_metric_audit；真实逐句优化及两seed学习的运行/终止记录；可播放PCM与GAIN；冻结候选的HuBERT/XLSR配对结果；独立T0/T1/内容状态；两类机制证据表；独立checker与BM结论。无科学阳性仍可工程完成，资源/依赖阻断必须保留具体未完成项。

若仓库证据与核心协议冲突，停止该依赖分支、保存复现证据并继续独立分支；不私自改数据、层、预算、阈值或把已看E重新命名盲测。未提交人工评价、未运行TFG不得写成已验证。当前仅交付spec，待实现时再运行计算。

## Implementation follow-up — September 21, 2026

- [engineering] v2 包、YAML、独立 checker、状态机和 20 项定向测试已实现；既有 phone-separability-mechanism 回归 14 项通过。
- [run] 当前代码哈希下的 full audit/calibration 运行目录为 `runs/phone_separability_enhancement_audit_v2_20260921`；audit 与 calibration 均 PASS。
- [calibration] processor parity max_abs=4.768e-7、cosine=1.0；waveform gradient norm=17.2787；teacher parameter gradients=0；zero-gain PCM max error=0。
- [smoke] `runs/phone_separability_enhancement_smoke5_20260921` 独立 checker PASS，14 个 PCM；仅有 smoke 不做科学 E 评价的警告。
- [fixes] 修复 support row-map 的 O(N²) 冻结性能、训练 history tensor JSON 失败，以及 BLOCKED stage 无法 resume 的状态机缺陷。
- [boundary] 正式逐句优化、两 seed 训练、lock 后 XLSR/E、质量/机制/TFG 尚未运行；科学状态为 `NOT_DETERMINED_RESOURCE_BLOCKED`，不产生 phone gain 阳性或阴性结论。
- [resource] 当前根盘约 4.1 GiB 可用；协议要求至少 4 GiB 且 formal optimization 会继续产生大量 PCM/field/checkpoint，因此按资源门停止，不降低阈值、不删除父 run/模型缓存。

## Formal execution follow-up — September 21, 2026

- [run] 磁盘恢复后完成 `runs/phone_separability_enhancement_audit_v2_20260921` 的 audit → calibrate → optimize → train → lock → evaluate → mechanisms → report；独立 checker 为 PASS（336 个 PCM，errors/warnings 均为空）。
- [lock] HuBERT DEV 预注册规则锁定 `OPT_DYNAMIC_6` DIRECT；LEARNED 锁定 seed=20260921、step=1300。优化 336 行；E_SEEN=40 句/40 source groups，其中 1 句无有效固定 support，主配对统计为 39 句。
- [Q1 DIRECT] HuBERT accuracy 0.649969→0.848565（Δ+0.198596），signed margin 0.038730→0.105424（Δ+0.066694）；XLSR accuracy 0.562609→0.644031（Δ+0.081422），signed margin 0.000780→0.011145（Δ+0.010364）。这是使用本句正确标签的逐句局部可行性结果，不是固定增强器泛化。
- [Q2 LEARNED] HuBERT accuracy Δ+0.005229、signed margin Δ−0.001428；XLSR accuracy Δ−0.003903、signed margin Δ+0.000215，未形成稳定 phone gain。两 seed 都通过 T0/失真门，但当前预算下未复现 DIRECT 的收益。
- [constraint] DIRECT 40/40 PCM/保护约束通过（平均残差能量比 0.008929、SNR 20.495 dB）；这不等价于 T1 时序、ASR 内容或人工自然度通过。MFA/ASR/人工听检、DAC 实际机制对照、TFG/SyncNet 均未运行。
- [interpretation] 当前最强可辩护结论是“低维标签辅助规则可以强行提高目标教师上的局部可分度，并在有历史暴露的 XLSR/E_SEEN 上出现探索性同向迁移；尚不能构建无标签可复用增强器，也没有解释 TTS 优势”。classification=`EXPLORATORY_ONLY_HISTORIC_E_EXPOSURE`。
- [next] 新 source groups 上冻结 DIRECT/LEARNED 参数做 confirmation；实际执行 static/dynamic、time-mean/reverse 与 DAC 四臂机制对照；补齐 T0/T1/ASR 后再决定是否进入 TFG。

### Relations

- extends [[TTS 音素可分性机制与自然时钟增强探索 Implementation Spec]]
- follows [[TTS 音素可分性机制 LRS3 严格 Atlas 与自然时钟构造 2026-09-21]]
- relates_to [[TTS 增强 TFG 的原因：证据、排除项与因果分叉]]
