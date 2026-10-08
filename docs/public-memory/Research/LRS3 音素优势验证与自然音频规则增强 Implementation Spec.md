---
title: LRS3 音素优势验证与自然音频规则增强 Implementation Spec
type: research_topic
permalink: tts-exp/research/lrs3-音素优势验证与自然音频规则增强-implementation-spec
status: active
question: LRS3 TTS是否具有音素可分性优势，规则增强natural能否保持时序并获得接近TTS的优势？
spec_status: ready_for_implementation
protocol_version: 1
tags:
- lrs3
- phoneme-separability
- rule-enhancement
- implementation-spec
---

# LRS3 音素优势验证与自然音频规则增强 Implementation Spec

## 1. Objective

先在固定 LRS3 配对队列中验证原始 TTS 相对 natural 是否有音素可分性优势；只有预注册门槛通过，才对 **natural** 做无学习参数的规则增强，检验是否提高可分性、缩小与 TTS 的差距，同时保持自然音频的时间轴和停顿。

用户于 September 20, 2026 明确选择“增强自然音频”，不是将 TTS 重定时。主产物为自然音频增强 WAV、逐样本证据与独立验收报告。本文件是实施规范，**尚未运行本实验、没有 LRS3 音素优势结论**。固定参数是本次提出的实验假设，不是已验证有效的算法。

本轮不训练增强网络、不做 RL、不重生成 TTS、不生成 TFG 视频、不扫描参数寻找评估集最优值。音素探针准确率不等同于人耳可懂度，也不等同于 Sync-C；不得相互代替结论。

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 初始实施规范；明确 natural 为规则输入 | September 20, 2026 | user（任务与输入方向）；agent（实验设计，待执行） |

## 2. Repository Model

实验数据流：

`固定240配对 + 双侧MFA → provenance/支持度审计 → 冻结SSL特征与参考探针 → Stage A优势门槛 → [通过] natural规则增强 → 冻结探针评估 + 独立时序QC → 独立checker → 报告`

首选且唯一默认队列：

- `runs/lrs3_data_supplement_20260903/07_final_cohort/manifest.json`
- SHA256：`dd109c8dfdbde9419f6cc8eaa11e7b9f78dc5d41ac3d437036c9122d14f97304`
- `runs/lrs3_data_supplement_20260903/07_final_cohort/tokens.json`
- SHA256：`ad2a54b0aa6f01323acc7fa7650b38d3508ba7edf4282f87f98d55731ac38509`

manifest 的 `records` 是列表；tokens 的 `records` 是 sample_id 键控对象，内含 `natural/tts` 的 TextGrid、SHA256、tokens。不要混用两种结构。按完整 sample_id 联结，禁止用数组位置或短编号关联。

队列已有 200 train / 18 source_group，40 evaluation / 40 source_group。train 只建立支持标签与参考探针，evaluation 做配对统计。source_group 表示源视频分组，**不是经过验证的 speaker_id**。该数据历史上属于 fit-only 研究队列，评估集不是未经接触的确认性测试集；本次结果属于固定划分的探索性复现实验。不得访问其他 sealed/test 划分来补样本。

TTS 元信息为 DashScope Qwen `qwen3-tts-vc-2026-01-22`，包含已有云端音频来源；必须报告 `tts_audio_origin` 分布。不要把本实验描述为所有 TTS 的普遍规律，或与另一批 LRS3 50 条实验混为一谈。

已做只读核验：480 个源 WAV 存在且 SHA256 与 manifest 一致；tokens 引用的 480 个 TextGrid 存在且 SHA256 一致。尚未验证全部音频编码、对齐误差、模型资产或实际探针分数。

所有新产物写入新 run，例如 `runs/lrs3_phone_rules_<UTC时间>/`；源 run、原音频、原 TextGrid、旧结果均只读。

## 3. Code Anchors

以下现有文件以 September 20, 2026 的工作树为准；HEAD 为 `158915603392a2b846cb4bfabe1821e9b4fec900`，但工作树存在用户改动，不能只靠 HEAD 复现。

| path / symbol | 当前职责 | 本任务用法 / 所需变化 |
|:--|:--|:--|
| `scripts/15_extract_ssl_embeddings.py::load_model`、`extract_frame_embeddings` | 本地只读加载 HuBERT/XLSR，提取指定 hidden states | 复用低层提取思路或经 importlib 调用；新增 worker 负责固定预处理、严格层检查和时间坐标；不改旧脚本 |
| 同文件 `_compute_frame_stride`、`_frame_to_token_pooling` | stride 推导、半开区间均值池化 | 复用半开区间原则；不能继承“半stride即帧中心”的隐式约定 |
| `scripts/35_phoneme_recognition_probe.py::nearest_centroid_accuracy` | 最近类中心识别 | 参考计算形式；新模块建立一次共享冻结中心，而非各条件分别拟合 |
| 同文件 `assign_frame_labels`、`_evaluate_loo` | 帧标注、句级留一 | 不直接用其 CLI：它重建零偏移帧时间、且按句而非 source_group 分组 |
| `scripts/16_feature_separability.py::fisher_ratio`、`confusable_pairs` | 类内/类间与混淆统计 | 可作次要诊断；不能替代主门槛；旧 `_pool_frames_for_layer` 时间约定不直接沿用 |
| `scripts/manifest.py::validate_manifest` | 通用 natural/tts manifest 验证 | 保持原 schema；本实验独立 artifact schema 使用 `arm`，不要把增强 natural 冒充 tts |
| `scripts/experiments/lrs3_mfa_linear_replacement/mfa_alignment.py::parse_textgrid`、`validate_tokens`、`build_mfa_command` | 英文 MFA 解析与命令构造 | 可复用；新增审计检测空隙、尾长、音频绑定；不调用固定24条的 `prepare_mfa_corpus/run_mfa_alignment` |
| `scripts/experiments/lrs3_phase_preserving_replacement_envelope/audio.py::phase_preserving_blend`、`write_pcm16` | 自然相位 STFT 合成、PCM 写盘 | 仿照原长度逆变换与原子写盘；不复制全局 RMS/peak 缩放、TTS 幅度替换或旧实验常量 |
| `scripts/tfg_feature_common.py::half_open_span_mask` | 时间区间归属 | 复用半开区间原则；新 worker 提供显式 frame_times |

新增位置和明确职责：

- `scripts/experiments/lrs3_phone_rules.py`：`main, audit_inputs, freeze_protocol, run_stage_a, run_stage_b, write_report`。仅编排、状态管理、产物契约；执行入口。
- `scripts/experiments/lrs3_phone_rules_metrics.py`：`normalize_phone, derive_frame_times, pool_phone_tokens, fit_reference_centroids, score_pair, paired_bootstrap, decide_stage_a, decide_stage_b`。纯 NumPy/标准库逻辑，无音频生成副作用。
- `scripts/experiments/lrs3_phone_rules_audio.py`：`build_edit_mask, spectral_emphasis, upward_compression, render_rule_arm, make_gain_control, validate_waveform`。确定性规则，不读取 TTS 或探针分数。
- `scripts/experiments/lrs3_phone_rules_worker.py`：`extract_features, realign_audio`。重依赖隔离；一次只驻留一个 SSL 模型；MFA subprocess 参数使用列表。
- `scripts/experiments/check_lrs3_phone_rules.py`：`main, verify_artifact_graph, recompute_statistics, verify_waveform_contract`。独立读取产物并复算，不调用 runner 的决策函数，不信任 summary 中的 passed。
- `scripts/configs/lrs3_phone_rules_v1.yaml`：下文所有常量与路径、seed、模型 revision、门槛。以该配置加冻结 protocol 为唯一实验参数来源，不改全局 config.yaml。

## 4. Reference Pattern

主要参考 `lrs3_phase_preserving_replacement_envelope/audio.py::phase_preserving_blend`：保留自然相位、显式 ISTFT length、严格数值与编码 QC。区别是本实验输入只有 natural，且所有修正必须通过自然时间轴上的 edit mask；不能在最后全局归一化，否则停顿不再逐样本保持。

对齐参考 `lrs3_mfa_linear_replacement/mfa_alignment.py::parse_textgrid/build_mfa_command`。只借用解析和命令构造，不借用跨音频 frame mapping，不拉伸 TTS，不复制固定队列数量，也不将最后一个音素延伸填满 SSL 尾部。

分类参考 `35_phoneme_recognition_probe.py::nearest_centroid_accuracy`；拟合必须改成 train 上的同一个冻结共享参考。旧脚本自身域内可分性与新固定探针分数是不同 estimand，应在报告解释，不直接拼表宣称同一指标提升。

## 5. Invariants

1. A 不通过就不生成规则候选音频；B 的所有候选与参数在 A 的 evaluation 打分前冻结。不得看到失败结果后换层、换 TTS、换样本或调参继续冒充同一实验。
2. 同一 pair 的 natural、TTS、规则音频共享文本来源；每个资产记录内容哈希。sample_id、source_group、原划分不变。所有拟合仅使用 train，不能将 evaluation 自然音频加入参考中心。
3. 对每个评估模型，所有 arm 使用同一模型权重、预处理、层、词表、距离、参考中心和打分标签集合。增强结果绝不参与重新训练探针。
4. IPA 标签仅做 Unicode NFC 和首尾空白清理；保留长度符号、送气、附加符号与 allophone 区别。不套中文去声调规则，不随意将 IPA 折叠成 ARPABET。
5. silence 从原 `silence` 字段显式转义；空标签、非语音、未知标签不得成为 speech class。强制对齐不是人工真值，应报告局限。
6. B 输出 sample_rate、声道、采样点数与 natural 相同；不做重采样、time stretch、移位、裁剪、拼接、补停顿或改变播放速率。主运行要求源数据为 mono 16 kHz PCM16；不满足则预检报告并停止，不隐式转换。
7. 输出在所有原 silence 样本、音素边界保护区及其他 mask=0 位置，PCM 数值与 natural **逐样本相等**。编码头可不同。停顿能量不强制清零，而是保留原背景。
8. 旧脚本公共接口、原 run、旧分数、用户工作树改动不变。缓存仅凭完整输入及协议指纹复用，不能凭文件存在或 mtime 命中。
9. 工程验收通过不代表科学正结果；NO_CLEAR_ADVANTAGE、NO_IMPROVEMENT 也是完整有效实验结果。缺失/NaN/模型失败不能变成0或自动跳过。
10. 报告至少同时列样本数、source_group 数、标签数、token覆盖率、逐组效应和置信区间；不得仅列有利的平均值。

## 6. Implementation Plan

### Step 1 — 预检、冻结与产物结构

先实现 CPU-only `--stage audit`。读取固定两个源 JSON，校验 SHA256、全量资产、split互斥与分组数、文本哈希，输出：

- `00_audit/cohort.json`：240条逐条来源、音频编码、时长、原始和解析后的 token哈希、TextGrid 来源、tts_audio_origin、排除/阻断原因。
- `protocol.json`：schema_version=1、spec SHA256、配置、源manifest/tokens哈希、实际代码文件哈希、git HEAD与dirty标记、依赖版本、随机种子20260920、模型/processor/MFA资产哈希、主次指标与全部阈值。
- `status.json`：阶段、状态、reason_codes、是否允许下一阶段。原子写盘；冲突 run 不覆盖。

检查 staged MFA 原音频/文本来源；原 `05_mfa_alignment/input_manifest.json` 声明 dictionary=english_us_mfa、acoustic_model=english_mfa。某些条目可能来自替补对齐目录，必须按 tokens 的实际 TextGrid 路径追溯，不能把初始 input_manifest 一律当成最终来源证明。如果无法建立某侧 TextGrid 与其 WAV/转写的绑定，在新 run 中使用固定 MFA 资产重新对齐该侧；无法重对齐则 INPUT_INVALID，不借用另一侧的 span。

语音 span 必须有限、正长度、不重叠；内部未覆盖空隙>1个采样点视为无效。最后边界距实际时长允许≤20ms的尾差，但明确记为未标注区域：不扩展最后speech token、不纳入评估、B保持原样。更大误差阻断。TextGrid 与 tokens.json 解析不一致必须解决来源，不静默择一。

源文件完整性错误是 INPUT_INVALID；模型/MFA环境缺失是 DEPENDENCY_BLOCKED。不更换模型顶替，不擅自扩云资源、发起付费 TTS 或修改旧环境。模型下载如需发生，必须固定revision与校验；没有已解析revision则报告依赖缺口。

### Step 2 — 固定 SSL 坐标与特征

主指标固定 HuBERT base（facebook/hubert-base-ls960）hidden_states[6]；HuBERT [0,11] 仅诊断。跨编码器检查固定 XLSR（facebook/wav2vec2-large-xlsr-53）hidden_states[10]。序号是 HuggingFace hidden_states 索引，0不是第一个 transformer block。不得根据结果改层。

使用与模型同revision的本地 processor 配置，记录 do_normalize 等实际预处理设置；按其契约对每条完整音频预处理且所有arm一致。不要将旧 extractor 的 raw waveform 直传误认为已经做了 processor 归一化。eval/no_grad；batch逐条，无拼接长音频或跨音素裁剪；一次一个模型。

从卷积 frontend 的 conv_kernel/conv_stride 推导 frame中心：
初始 receptive_field=1、jump=1；每层更新 receptive_field += (kernel-1)*jump，再 jump *= stride；
`t[i] = (i*jump + (receptive_field-1)/2) / sr`。
标准400样本感受野/320stride时首中心=199.5/16000秒。校验卷积padding/dilation配置，非预期结构报错。此坐标是 frontend 中心，不表示 transformer 只看局部。

禁止混用旧 probe 的0ms、旧 extractor 的10ms与本规范的中心。缓存内显式存 frame_times、层索引、模型/processor指纹；新结果不复用旧缺失时间契约的数组。

每个 speech token 对 `start <= t < end` 的帧做均值，再 L2归一化得到一个token向量。没有中心的短token记 `no_frame`，不重复邻帧或插值补造。零范数/非有限值为技术错误。序列长度差不能让慢语速自动获得更大权重。

### Step 3 — 一次性冻结支持集与参考探针

对每个模型分别取 train：标签在 natural 和TTS各≥20个有效token，且每侧各来自≥3个source_group，才能进入模型支持集。主/跨编码器共同支持词表取两者交集，之后固定；至少10类，否则 INSUFFICIENT_SUPPORT。

对每个label、condition、source_group先平均归一化token向量；再在该label的各group间等权平均；最后 natural/tts 两侧各权重1/2，得到共享中心并L2归一化。预测用最大cosine，数值相等按NFC标签排序打破tie。同一模型只拟合一次，保存中心、词表、train来源及哈希。

评估每个pair使用两侧都有有效token的共同受支持标签集合 K_i。逐label统计token识别正确率，再对K_i等权平均得 A_i(N)、A_i(T)。不强行一一匹配不同allophone的token序列。每pair要求 K_i≥5且每侧参与token≥10；两侧受支持且有帧的speech token覆盖率各≥70%。覆盖分母为原全部speech token，不能只以成功池化者为分母。

资格仅由标签/时间与train支持决定，不由预测对错决定；在 A 打分前冻结合格pair与每侧token ID集合。主/跨编码器使用同一批合格pair。至少30个evaluation source_group通过，否则 INSUFFICIENT_SUPPORT。保留全部40条审计清单，不能隐瞒排除条目。reference/train 与eval分组交集必须为空。

补充诊断：全支持词表macro recall、micro token accuracy、混淆矩阵、各标签覆盖率、HuBERT其他层。没有资格的pair不以0填充。

### Step 4 — Stage A门槛

主估计量 `D_A = mean_group(A_i(T)-A_i(N))`。同一source_group若有多clip先等权平均，再组间等权。本固定eval预期一组一clip，仍实现通用组聚合。

固定 NumPy PCG64 seed=20260920、10000次source_group配对bootstrap，等概率有放回重采样组，所有arm共享重采样索引；95% percentile区间。数组稳定排序、分位数linear方法。所有数值内部用0..1，报告用百分点。

Stage A仅在以下全部满足时输出 ADVANTAGE_SUPPORTED：
1. 数据/支持度/特征与探针审计通过；
2. HuBERT L6 的 D_A≥0.020，且95%CI下界>0；
3. XLSR L10 的 D_A>0（跨编码器方向性护栏，**不是第二项显著性证据**）。

其他有效数据结果输出 NO_CLEAR_ADVANTAGE，并报告“效应太小/区间跨零/跨编码器方向冲突”的具体原因；不能将它写成证明TTS绝无优势。Stage A结果只支持本队列、该标签口径和冻结表征。

产物 `01_features/`、`02_reference/`、`03_stage_a/{per_token.jsonl,per_pair.json,statistics.json,decision.json}`。decision引用各输入哈希。未通过时全流程到此科学完成，B状态=SKIPPED_GATE_NOT_PASSED，B音频目录不得出现候选WAV。

### Step 5 — Stage B确定性自然音频增强

主规则固定 `spectral_drc`，另外两个消融 `spectral_only`、`drc_only` 仅解释机制，不竞争主成功名额。主运行只需处理 A冻结合格的evaluation natural；规则没有训练阶段。不读取 TTS波形、音素类别身份或分类梯度。音素边界/静音mask允许来自 natural MFA，因此结论限定“转写与离线MFA可用的规则处理”，不能宣称无监督实时可用。

通用浮点PCM映射 x=pcm/32768，float64 CPU计算，指定torch STFT实现与版本。

1. edit mask：每个natural speech span映射到样本 n/sr；span首尾各10ms置0，再向内各5ms raised-cosine从0升到1/降到0，中间为1。过短不能容纳两个guard加taper的span全0。所有silence、未标注区域和越界区域为0；相邻speech不能合并取消边界保护。最终生成 `y=x+m*(z-x)`。
2. spectral：STFT n_fft=512、win_length=512、hop=128、periodic Hann、center=True、pad_mode=reflect，ISTFT同参数且length=N。频率增益dB为≤300Hz=0；300–1000Hz半余弦升至+2；1000–4000Hz=+2；4000–6000Hz半余弦降至0；≥6000Hz=0。相位不变。逐帧将加权谱重新缩放到原帧能量，能量用rFFT单边权重（DC/Nyquist=1，其余=2），epsilon=1e-12；全零帧保持零。这里只是规则假设，不能称为完整SSDRC复现。
3. drc：从该分支输入波形计算居中20ms矩形窗RMS（边缘reflect），取原speech mask内RMS的75分位作为阈值T；dB floor=-80。上行压缩gain_dB=clip((1-1/1.5)*(T_dB-RMS_dB),0,3)。用奇数801点归一化Hann平滑gain_dB，再转线性增益。全speech近静音则失败并报告。spectral_drc按 spectral→drc 顺序；消融省略相应步骤。
4. 所有分支最后应用同一个edit mask。不用全局响度归一化或limiter。峰值安全通过整个残差的单一系数 alpha∈[0,1]：解析求满足PCM可表示范围[-1,32767/32768]的最大alpha，取min(1,所有约束)。不得硬削波。记录alpha、输入/输出峰值、编辑样本比例、speech RMS变化和mask哈希。alpha=0或量化后零变化为有效实现但“无有效增强”，不能算实验成功。
5. 按 round-to-nearest-even量化PCM16。对mask=0样本直接写回原始整数，避免浮点往返误差。源文件不要覆盖；写完重读验证。

增加 `identity` 编码对照，必须与源PCM逐样本相同。

为主规则增加 `gain_control`：仅做 `x+m*(g-1)*x`，用一个常数g≥0匹配主规则的最终量化前speech能量。以a=sum((m*x)^2)、b=2sum(x*m*x)、c=sum(x^2)-target_energy求 a*(g-1)^2+b*(g-1)+c=0，求根取最接近1且非负者；求和范围为speech。检查峰值与量化后能量相差≤0.05dB；没有可行根/峰值越界不得偷偷归一化，记录 CONTROL_INVALID。此控制不证明排除了所有响度因素，但能识别简单增益是否解释主要改善。

所有分支均保存WAV及 `04_rules/artifacts.jsonl`：parent natural SHA、arm、参数、sample_count、输出SHA、maskSHA、alpha、QC状态、gain_control的g。主规则必须全体合格pair生成成功，不能只报告易处理样本；消融故障单独披露，不替换主规则。

### Step 6 — 冻结标签评估、时序验收与B结论

规则arm和identity使用 **natural原始token时间/标签/池化索引**，不重新对齐后再计算主可分性，确保没有通过移动标注改善分数。K_i沿用A；同一冻结参考中心；TTS分数直接引用A。重提取特征采用同一processor；禁止重新fit。

定义每组：
- `D_E = A(E)-A(N)`：增强增益；
- `D_C = A(E)-A(gain_control)`：超越简单增益；
- `D_T = A(E)-A(T)`：与TTS差距；
- `D_half = (A(E)-A(N))-0.5*(A(T)-A(N))`：是否弥补至少一半TTS优势。

全部使用与A同样的组级bootstrap，明确B是A条件筛选后的探索性分析，不能称为独立确认。

时序验收分两层：
- waveform硬约束：所有pair采样率/长度一致，mask=0 PCM逐样本一致，无NaN、越界、削波；若identity的PCM或主特征/分数不一致，IMPLEMENTATION_INVALID。
- 声学边界检查：在新目录对 natural和主增强音频均运行同版本MFA（同文本、词典、声学模型），两侧同环境重跑以控制aligner版本差。仅用于时序QC，不替换主标签。用完整speech标签序列的确定性Levenshtein回溯建立匹配（match→substitution→deletion→insertion打破tie），仅同标签match计算边界差；不把重复标签按出现次数任意配对。报告编辑率、匹配率、边界绝对偏移分布、新增/丢失≥50ms静音。每pair要求speech编辑率≤5%，匹配token首/尾边界偏移中位数≤20ms、P95≤40ms；不得出现无法与原停顿以≥50%交并比匹配的新增/丢失≥50ms静音。任一不满足，TIMING_NOT_ESTABLISHED，不以波形等长替代发音时序证明。对齐失败为TIMING_UNVERIFIED，不冒充通过。

B的主科学结论需同时满足：
1. 主规则所有pair波形和时序验收通过，gain_control有效；
2. HuBERT L6：mean D_E≥0.010且95%CI下界>0；D_C的95%CI下界>0；
3. XLSR L10：mean D_E>0。
满足则 NATURAL_ENHANCEMENT_SUPPORTED，否则按原因报告 NO_IMPROVEMENT、GAIN_ONLY_NOT_EXCLUDED 或时序/QC状态。固定主规则失败而消融显著，只能记“消融探索性线索”，不能改写主结论。

在主结论成立基础上，再单独判断：
- D_half 95%CI下界≥0：证据支持“弥补至少一半观察到的TTS优势”。
- D_T 95%CI下界>−0.010：在事先规定1个百分点容差内“不劣于TTS”；不是“等价”或“超过TTS”。
- D_T 95%CI下界>0：在本口径下超过TTS。
- 不满足以上也可仅报告确有改善，不能写接近/追平。间隙恢复比例只作描述，不作为不稳定的比值检验。

输出 `05_stage_b/` 逐token/逐pair预测与统计、`06_timing/` MFA/QC及绑定证据、`07_report/report.md`。报告表须同时呈现natural、TTS、identity、gain_control、主规则和两个消融，不过滤负面结果。

### Step 7 — 独立检查、CLI与恢复

计划新增CLI（下游必须实现，不是当前已存在命令）：

```bash
python scripts/experiments/lrs3_phone_rules.py --config scripts/configs/lrs3_phone_rules_v1.yaml --run-id <new_id> --stage audit
python scripts/experiments/lrs3_phone_rules.py --config scripts/configs/lrs3_phone_rules_v1.yaml --run-id <new_id> --stage all
python scripts/experiments/check_lrs3_phone_rules.py --run-dir runs/<new_id>
```

支持stage=a、b；b必须验证已完成A、ADVANTAGE_SUPPORTED及全部指纹。没有允许越过门槛的force开关。支持 --resume，仅恢复同指纹未完成阶段；配置/代码/模型/源音频改变要求新run。--smoke固定取排序第一条train与第一条evaluation的natural/TTS双侧，输出专门smoke子目录；smoke只验工程管道，永远不能产生科学pass或授权完整B。规则CPU unit tests可用合成输入，不受真实A门槛限制。

状态至少区分：PENDING、RUNNING、COMPLETE、INPUT_INVALID、DEPENDENCY_BLOCKED、INSUFFICIENT_SUPPORT、IMPLEMENTATION_INVALID，以及各阶段science_decision。退出码0=工程正常完成（包括科学负结果），2=输入/依赖阻断，3=独立验收失败；不能通过单个exit code判断科学成功。

checker从逐token预测/标签重建准确率，从逐组数据独立重算CI和门槛，从WAV重验长度/PCM保持，从资产图重验SHA；不导入runner的decide函数，避免同一bug自证。能够识别被篡改的summary和孤立B产物。report明确 `engineering_pass` 与 `science_decision`，并引用checker结果。

## 7. Expected Change Surface

### Must change

仅新增以下文件及本规范：

- `scripts/experiments/lrs3_phone_rules.py`
- `scripts/experiments/lrs3_phone_rules_metrics.py`
- `scripts/experiments/lrs3_phone_rules_audio.py`
- `scripts/experiments/lrs3_phone_rules_worker.py`
- `scripts/experiments/check_lrs3_phone_rules.py`
- `scripts/configs/lrs3_phone_rules_v1.yaml`
- `tests/experiments/test_lrs3_phone_rules.py`
- `tests/experiments/test_lrs3_phone_rules_metrics.py`
- `tests/experiments/test_lrs3_phone_rules_audio.py`
- `tests/experiments/test_lrs3_phone_rules_worker.py`
- `tests/experiments/test_check_lrs3_phone_rules.py`

### May change

新增小型合成fixture目录；必要的实验依赖说明。若numeric脚本导入需要适配，在本实验worker局部用importlib，不重命名已有脚本。确有新依赖时先说明必要性，优先使用现有 NumPy、torch、标准库 wave；不要借机替换全项目环境。

### Should not change

不改 `15/16/35` 旧脚本行为、不改通用manifest schema、不改全局 `scripts/config.yaml`、不改TTS provider、不改TFG/SyncNet、不覆盖旧数据和对齐、不改CONTEXT/HANDOFF、不清理dirty worktree、不创建进度日志。新结果只能放新run；正式结果后续写BM Experiments，而不是覆盖只读旧编号镜像。

## 8. Validation Plan

按静态→纯单测→小集成→已有回归→完整实验顺序。

1. 静态：对五个新增脚本及metrics/audio模块运行 `python -m py_compile`（按实际新增清单逐项），确认CPU audit/help不加载GPU模型或启动MFA。
2. `pytest tests/experiments/test_lrs3_phone_rules_metrics.py -q`：
   - 人造易分类类簇得高分，随机标签近机会水平；N=T时差为0、CI=[0,0]；交换N/T效应取反。
   - token变长不改变其计数权重；某group复制clip不改变组权重。
   - fit只看train，eval改值不影响中心；不混淆condition域分类与phone识别。
   - 帧中心199.5/16000与半开边界样例准确；无帧短token有明确覆盖损失。
   - IPA diacritic/长度符号保留；silence不参与；B沿用K_i，不因效果重选标签。
   - bootstrap排序/seed稳定，百分数单位与阈值无100倍错误；各门槛边界与所有负面状态有测试。
3. `pytest tests/experiments/test_lrs3_phone_rules_audio.py -q`：
   - 合成speech/pause/短音素/头尾silence/峰值附近PCM；长度不变，mask=0整数全等，identity全等。
   - spectral_only、drc_only、combined参数固定、重跑一致；正弦/STFT往返重建误差有界，rFFT能量权重正确。
   - 全零、非有限、短于reflect padding、非法编码明确拒绝；alpha衰减与无有效修改有记录。
   - gain_control求根和量化后0.05dB容差；无可行峰值解不偷偷限幅。
4. `pytest tests/experiments/test_lrs3_phone_rules.py tests/experiments/test_lrs3_phone_rules_worker.py -q`：
   - 用mock特征/MFA完成正门槛、负门槛两条全流程；A负结果必须零B候选WAV。
   - 源hash变化/错配TextGrid/不同source_group划分/重复pair/缺失tts/旧缓存/NaN阻断。
   - 每次仅一个模型；非法layer拒绝，processor配置记录；传递实际frame_times。
   - 新MFA目录与命令资源固定；不得调用旧固定24条corpus runner。
   - smoke不产科学pass，resume指纹不同拒绝。
5. `pytest tests/experiments/test_check_lrs3_phone_rules.py -q`：
   - 篡改单个停顿PCM、WAV SHA、逐token预测、summary CI、A门槛、B父资产哈希均被抓到。
   - checker正常接受科学负结果并区分工程通过；不依赖runner决策函数。
6. 现有回归：
   `pytest tests/test_ssl_embedding_shapes.py tests/test_feature_separability.py tests/test_manifest.py tests/experiments/lrs3_mfa_linear_replacement/test_mfa_alignment.py -q`。
   环境缺依赖时如实列blocked，不声称已通过。无授权不要为跑全仓库测试清理或改动用户其他实验。
7. 实数据audit需重验240pair/200+40/18+40和全部哈希；真实smoke核对模型层/shape、显式时间、identity、PCM和MFA通路，不从2条样本推科学结论。
8. 完整执行A；若通过执行B及checker；否则交付完整A负结果与B跳过原因。最终附精确命令、配置/代码/输入指纹、耗时、依赖、合格样本清单和结果文件路径。

## 9. Risks and Edge Cases

- 两种录音的强制对齐质量可能不同；TTS更容易对齐可能贡献表观优势。原标签+共同支持并不能消除该混杂。报告每侧短token/未标注/对齐异常率；差异异常时降低解释强度，不强称发音因果机制。
- 标注是MFA phone/allophone inventory，不自动等于语言学合并后的phoneme集合。本实验“音素可分性”应明确操作性定义，不在结果后合并不利标签。
- 较短英文clip可能达不到共同标签覆盖门槛；不得降低预注册门槛临时放行。INSUFFICIENT_SUPPORT与没有优势不同。
- 共享参考探针衡量相对同一表征空间的可解码性；不构成与模型无关的声学定理。XLSR只是方向检查。
- 冻结参数可能没有增益，这是有效结果；不把消融最优者改成主臂。后续调参需新的协议版本和独立验证设计。
- 修改幅度谱不保证可懂度或边界稳定；等长、保留相位都不能代替时序检查。MFA边界检查本身也只是自动证据，不宣称人耳发音时序完全不变。
- 极短phone在guard后没有编辑空间可能导致弱增强；报告实际编辑speech比例和alpha分布。不得删除保护区换取分数。
- 原PCM采样点完全保留与全局响度归一化冲突；只能按本规范做局部残差与匹配对照。
- source_group并非speaker标签；模型曾接触LRS3相关语料的预训练重叠也未排除。外推范围必须收窄。
- 旧分析脚本会在缺文件时warn+skip，且frame坐标不一致；不能直接串旧CLI后把输出视为本规范结果。
- workspace为dirty，新增规范/代码不得覆盖用户已有更改；若同名新文件已由其他agent出现，先读差异再协调，不盲写。

## 10. Assumptions / Unknowns

- VERIFIED：用户选择 natural规则增强；不是TTS重定时。
- VERIFIED：指定队列存在、240pair，200train/40evaluation，18/40源组；源WAV与引用TextGrid的480+480哈希检查无异常。
- VERIFIED：tokens为双侧独立MFA时序，标签含IPA/allophone，不能依序强行对应；manifest记录云Qwen provider。
- VERIFIED：旧SSL extractor和probe/pooling函数具有不同frame-time默认值；旧MFA orchestration含固定24条约束。
- VERIFIED：尚未运行此规范的音素统计或生成规则音频。本规范中门槛与处理参数不是经验结果。
- LIKELY：该队列足够支持英文常见phone的共享参考探针；仍以实际支持审计为准。
- LIKELY：温和谱强调与弱段上行压缩可能增强辅音线索；是否提高冻结SSL可分性未知，不能保证正结果。
- UNKNOWN：本机可用的HuBERT/XLSR模型资产位置、精确revision和processor版本；默认HF目录未查到不代表全机没有。下游预检解决，不换架构。
- UNKNOWN：全部WAV编码、最终替补TextGrid与音频的完整绑定、MFA词典/声学模型具体资产哈希、对齐质量。
- UNKNOWN：LRS3本队列TTS是否优于natural、增强是否有效、是否满足时序与TTS接近判据；只有执行才能回答。

## 11. Handoff Contract

从 Step 1 与对应CPU tests开始，按上述Code Anchors与Reference Pattern实施；保持全部Invariants，补齐每个新脚本的测试，严格限定Expected Change Surface，不做无关重构。

必须交付：代码与测试、冻结配置/协议、输入审计、A完整结果、条件允许时的B音频/指标/时序QC、独立checker结果和简明报告。报告第一屏必须写“工程验收状态／A结论／B是否运行及结论”，不能只说脚本跑完或生成音频成功。

如仓库证据与规范矛盾（例如对应关系不成立、受限测试集、源数据需要重生成、数据不足导致需要另选队列），停止该阶段并报告具体矛盾与最小所需决策；不要自行扩大数据、云资源或实验范围。模型/依赖缺口不授权更换主模型或降级门槛。

### Observations

- [status] active
- [question] LRS3中TTS是否具有稳定的phone可分性优势，以及固定自然时间轴的规则增强能否获得该优势？
- [decision] 用户明确选择对natural做规则增强；先A验证，后条件B实施。
- [constraint] 本文是实施规范，不是已完成实验或正面结果。
- [requirement] 停顿PCM保持、冻结共享探针、source_group配对统计与独立验收缺一不可。

### Relations

- relates_to [[tts-exp]]
