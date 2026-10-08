---
title: LRS3 音素增强的静态肖像 TFG 验证与 MFA 条件增强 Implementation Spec
type: research_topic
permalink: tts-exp/research/lrs3-音素增强的静态肖像-tfg-验证与-mfa-条件增强-implementation-spec
status: planned
spec_status: ready_for_implementation
protocol: phone_gain_static_tfg_mfa_v1
question: 标签辅助音素增强是否改善无视频输入的静态肖像TFG，训练和推理均输入MFA能否构建可复用收益
scientific_scope: exploratory_seen_sources
tags:
- lrs3
- phoneme-separability
- static-image
- tfg
- mfa-conditioned
- implementation-spec
---

# LRS3 音素增强的静态肖像 TFG 验证与 MFA 条件增强 Implementation Spec

## 1. Objective

实现探索协议 phone_gain_static_tfg_mfa_v1，回答三个问题：

1. 已生成的标签辅助逐句增强 DIRECT，是否在没有目标视频输入的情况下改善 Wav2Lip 生成嘴形与原自然音轨的匹配？
2. DIRECT 及新增固定增强器的音素可分性是否达到 TTS 水平？在相同 occurrence、探针与聚合口径下比较，区分点估计、非劣、等效和优势。
3. 训练和推理都提供自然音频 MFA 音素身份及时间区间，能否得到比仅音频特征、仅边界/时长条件更有效的固定增强器？

本次交付为 spec，不执行训练或 TFG。下游实现应完成真实训练、真实推理、评分与独立验收；不能用注册表、空结果或 NOT_RUN 占位宣称实验已完成。研究允许阴性结果。E_SEEN 有历史暴露，全轮明确标 EXPLORATORY，不重新命名为盲测或独立确认。

### Observations

- [status] planned；spec_status=ready_for_implementation。
- [question] 探针可分度提升是否能转化为静态肖像 TFG 的 natural-audio replacement 收益？MFA 条件是否帮助固定增强器？
- [requirement] 推理禁止读取 LRS3 原视频、逐帧人脸/嘴部轨迹；只允许冻结的外部单张肖像 PNG。
- [requirement] 学习臂必须真实包含训练和推理均输入 MFA 的条件模型，并配训练预算相同的对照。
- [boundary] MFA 标签用于 FIT 监督属于方法定义；推理标签来自 N+已知转写的音频对齐。不能称无文本系统。

### Changelog

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 仓库核验后制定静态输入防泄漏、MFA 条件学习、自然音轨交叉评估与独立验收合同 | September 22, 2026 | user request；agent design |
| 自检明确局部边界扰动的波形构造、目标边界检出标准及ASR聚合口径 | September 22, 2026 | agent review |

## 2. Repository Model

历史资产：

- P2 = runs/phone_separability_enhancement_audit_v2_20260921。
- P1 = runs/phone_separability_mechanism_full_v2_20260921。
- P2/00_audit/registry.json：240 对音频，FIT 135句/12 source groups，DEV 65句/6组，E_SEEN 40句/40组；pair 中有 sides、tokens、transcript、video_path。
- P2/00_audit/pilot.json：历史 DEV pilot 24句；继承该选择，不重新挑选。
- P2/04_lock/selection_lock.json：DIRECT=OPT_DYNAMIC_6；历史 LEARNED seed=20260921、step=1300。
- P2/05_evaluation/summary.json：DIRECT 与历史 LEARNED 的逐句输出路径、PCM hash、HuBERT/XLSR 分数。文件很大，导入时流式投影必要字段，禁止全量复制到每个子进程。
- P2/01_calibration/probe/{hubert,xlsr}/e_seen.json：N/T 基线。已核对历史三臂相同 39 个 group、2256 个 support key：HuBERT N/T/D=0.649969/0.742155/0.848565；XLSR=0.562609/0.622082/0.644031。只作为历史重放验收值，未做 D−T 显著性/等效检验。
- 当前训练 _loss_for_sample 已用自然 MFA tokens 计算 phone loss，但 model_input 仅为24频带 log-power；“新增 MFA 条件”指把标签/时间作为网络输入，并非首次使用监督标签。
- 当前 v2 quality 的 MFA/ASR runner 未接通，正式 run 的 T1/内容/人工质量及 TFG 均未评。不能继承工程 PASS 就认为这些分支已完成。
- 当前 freeze_support 通过 N/T 匹配构建 support，与原 spec 的纯自然主 support 不完全一致。本轮保留历史 matched-support 重放，另建自然侧主 support，不能静默把历史值当作新口径。

新流程：

冻结输入/划分/配置/肖像与资源预算 → FIT 测量校准 → 三种条件模型训练 → DEV-only checkpoint 锁定 → E 固定前向及历史 DIRECT 复用 → 音素/质量评价 → 静态 TFG 与交叉音轨评分 → 独立复算/报告。

已冻结 DIRECT 的 TFG 可在协议锁定与 FIT 校准后先执行；它不依赖新学习臂阳性。学习臂的 checkpoint 选择进程只能读取其 DEV 结果，不能读已产生的 E/TFG 结果。

## 3. Code Anchors

新包记为 P=scripts/experiments/phone_gain_static_tfg_mfa/。优先通过 adapter 复用旧代码，保持旧 run 可复现。

| path / symbol | 当前职责 | 必须实现或复用的内容 |
|:--|:--|:--|
| phone_separability_enhancement/data.py::pair_index, get_side, load_feature_store | 父资产定位/缓存读取 | P/assets.py::import_parent_assets 流式导入，形成无视频字段的训练/推理 manifest；校验 PCM 与划分 |
| 同文件::freeze_support, _fit_centroids | N/T matched support、FIT中心 | P/support.py::freeze_phone_support 分离 natural_only 与 matched_nt；复用已冻结中心，不重拟合 E；保留 legacy support |
| phone_separability_enhancement/teacher.py::FrozenPhoneTeacher, normalize_waveform, pool_hidden | 可微 SSL 与 core 池化 | 直接复用前向；P/train.py 按新冻结 FIT slots/labels 计算损失，避免训练静默换 token |
| phone_separability_enhancement/audio.py::BandGainRenderer, make_protected_mask, export_pcm | 24带增益、相位保持、硬保护/PCM | 直接复用 renderer；P/quality.py 在回读 PCM 上独立复核饱和、失真与保护 |
| phone_separability_mechanism/train.py::BoundedGainEnhancer | 64通道、dilation 1/2/4、零初始化输出 | P/model.py::MFAConditionedGainEnhancer 仿其结构，43输入通道、24输出；不改旧类 |
| phone_separability_enhancement/train.py::_balanced_order, _loss_for_sample, train_enhancer | 已有 FIT 监督训练 | P/train.py::train_arm 实现三个 input_mode、公平更新数、完整续跑；不照搬其 microstep 命名和缓存 |
| phone_separability_enhancement/metrics.py::signed_phone_scores, score_fixed_support | cosine 分类与 signed margin | P/phone_eval.py 调用评分；accuracy 必须 argmax 分类，不能用 margin≥0 代替 tie-breaking |
| static_image_bridge/render_worker.py::chunk_mels, encode_ffv1, main | 单 PNG Wav2Lip 静态渲染 | P/render.py::render_cell 以严格 manifest 调用；新增 P/render_worker.py 做输入断言与流式编码，可复用纯函数 |
| static_image_bridge/images.py::select_detection, generation_box, score_box | 静态脸框与固定评分 crop | 只在固定 PNG 上检测一次；冻结 bbox/score crop，不逐臂重新检测 |
| static_image_bridge/score_worker.py::audio_embedding, read_video, crop_zero_padded | 官方 SyncNet 前端和距离矩阵 | P/score_worker.py 仿前向，额外保存音频 embedding，使用预冻结窗口；只接受生成视频 |
| static_image_bridge/frontal_probe.py::generator_support, frozen_support, metrics | 修正后的前端支持与固定窗口分析 | P/sync_support.py 仿其几何公式并做源码/边界验证；不导入硬编码22句 cohort、资源轮询或 analysis |
| tts_time_instance.py::_mfa_command, run_mfa | 实际 MFA 调用与缓存核验模式 | P/mfa.py::align_batch 参数化 isolated corpus、模型/词典 hashes；不复制自动删除旧目录逻辑 |
| phone_separability_enhancement/quality.py::audit_timing, pcm_contract | 时序/PCM 统计辅助 | 概念复用；修正见第6节，不能直接用当前全部逻辑当完整验收 |
| 新 P/config.py, run.py | 无 | 唯一配置、两级锁、依赖状态机、资源审计、失败台账 |
| 新 P/analyze.py, check.py, report.py | 无 | 配对统计、独立验证、可播放对比及 BM 结论 |

每个新模块只负责表中边界。旧主流水线、TTS provider、TFG/SyncNet 权重不改；不为本轮整理全仓库。

## 4. Reference Pattern

- 静态渲染仿 static_image_bridge/render_worker.py：固定一张 PNG，在各时刻重复相同视觉张量，仅音频 mel 变化；source_frame_indices 全0、生成 ROI 外像素不变、25fps、FFV1 无音轨 master。
- 仿 static_image_bridge/frontal_probe.py 的 input hash、真实前端支持及 ND 控制。不要复制原 static_image_bridge/analysis.py 从 finite mask 事后选 W，也不要复制原 validate.py 与 runner 共用分析函数后自称独立验收的模式。BM“静态图片 Natural-to-TTS bridge 实验”已记录这些缺陷。
- 训练仿 phone_separability_enhancement/train.py 的完整句反传、冻结教师、current-y 保持损失、DEV选择。旧历史 checkpoint 只作历史参照；本轮 A/B/C 都重新在同一设置训练。
- MFA 仿 tts_time_instance.py 实际创建 WAV/LAB 并调用 align；缓存键增加 PCM、转写、MFA可执行文件/版本、词典、声学模型与参数 hash，不凭 TextGrid 存在就复用。

没有可以直接完成本轮“三条件训练+无视频推理+完整交叉评价”的单一现成 runner，需新建上述薄适配包。

## 5. Invariants

### 数据与标签权限

| 进程 | 允许输入 | 禁止输入 |
|:--|:--|:--|
| 训练 A/B/C | FIT自然 PCM、FIT自然MFA、冻结FIT中心、训练配置；DEV仅独立选型 | E样本参与更新、同句TTS目标、任何视频/视觉特征、SyncNet损失 |
| 推理 A | 自然PCM、预生成停顿保护mask、checkpoint | 音素身份/内部边界作为网络条件、TTS、teacher、测试时优化、视频 |
| 推理 B | A输入 + 自然MFA的音素区间/时长 | 音素身份作为网络条件、TTS、测试时优化、视频 |
| 推理 C | B输入 + 自然MFA音素身份 | TTS、teacher、测试时反传/优化、视频 |
| DIRECT | 复用P2冻结音频；丢失时只能按原冻结算法重建并单列 | 新看TFG后调参数/选step |
| TFG worker | 冻结外部单张PNG、指定驱动PCM、固定Wav2Lip权重/静态框 | LRS3视频、目标帧、逐帧crop/landmarks、动态参考、MFA、评估音轨 |
| 评分 worker | 本轮生成视频、指定评分PCM、固定几何/SyncNet | 把LRS3原视频冒充生成结果、以评分结果改增强器/渲染 |

1. 原视频路径可以留在只读 provenance 中，但不得出现在 train/infer/render worker 请求中；只传白名单字段，禁止将完整 registry 下传。
2. 使用自然音频及已知 transcript 的 MFA，是 declared transcript-assisted 条件。A 也依赖 MFA 保护mask，不称完全 audio-only 系统；A 名称为 AUDIO_FEATURES。
3. conditioning 标签、training supervision 标签、evaluation 标签是三个独立参数对象。消融只改 conditioning，不能顺带改评分答案或保护mask。
4. source_group 隔离继承父划分；不把40 source groups写成40说话人。若发现真实 speaker 跨分割重合，记录并限制结论；不根据 E 分数重划分。
5. 所有 N-clock 输出16kHz/mono/PCM16，样本数与N完全一致；停顿/guard保护区PCM一致。长度保持不是实际音素时序保持，T1另验。
6. 仅 FIT 统计可拟合中心、词表、归一化；DEV用于冻结规则选择；E/XLSR/TFG不回流。
7. 原 TTS 保持自己的时长与 MFA；不得把 N 时间区间复制到 T，亦不得直接将未校时 V(T) 配 N 宣称 TTS replacement 基线。
8. 原run/资产/用户dirty changes只读；新结果写独立 run。不可删除缓存、旧run或用户媒体腾空间。
9. state_dict、模型/前端/输入/支持/肖像/几何/代码/spec/config均hash绑定。协议不兼容变动创建新run，不能覆盖或伪装resume。
10. 工程 PASS、phone gain、TFG gain、时序、ASR、人工质量分别给状态；一个维度不能代替另一个。

## 6. Implementation Plan

### 6.1 P0：审计、资源、协议锁与资产权限

唯一配置 scripts/configs/phone_gain_static_tfg_mfa_v1.yaml。新run目录 runs/phone_gain_static_tfg_mfa_<run_id>/。先完成资源只读检查，再加载任何GPU模型：nvidia-smi GPU利用率/显存/compute PIDs，磁盘可用、RAM、模型/依赖存在性。每个训练/渲染/评分GPU阶段前重复检查；有他人进程占用时等待/记录RESOURCE_WAIT，禁止杀进程或擅自换服务器。CPU审计可继续。等待轮询≤30秒，可恢复且持续报告状态。

资源门：训练可用显存≥12GiB；渲染/评分≥5GiB；RAM≥8GiB；无外部GPU计算进程且利用率≤10%。同一GPU串行执行本轮任务，保存lease/PID与开始结束快照。磁盘基础余量≥4GiB，并额外满足 smoke 测得剩余产物估计的1.25倍及下一cell峰值；不能仅沿用父3GiB总产物预算。资源未知按不足处理。模型缺失允许经 http://[redacted-ip]:7890 下载，解析并锁定实际revision/hash，不按候选效果换模型。所有模型权重与环境先冻结后评分。

导入 P2/P1 manifest、PCM、tokens、中心、selected_direct；路径与hash逐条复核。保留每个旧E的排除/失败记录，禁止按 DIRECT提升筛样本。DIRECT40句即使1句没有phone support仍可参加TFG。FIT全量135句用于训练，不限于原24句优化pilot；DEV固定原24句；E保持原40句。若实际数量不同，明确资产合同冲突，暂停对应分支并报告，不能自行替代cohort。

创建 training_manifest、inference_manifest、render_manifest 三种白名单schema（unknown field reject），worker请求禁止 video_path、landmarks、frame_sequence 等字段。音频推理要求WAV实际PCM格式；静态输入检查PNG magic及解码后单帧，不只看后缀；symlink resolve 后核对允许文件hash。图像与音频都通过明确路径传入，禁止从pair_id自动寻找原视频。

主肖像固定 data/data/image/3.png；敏感性肖像固定6.png、9.png。P0核验三张均独立静态资产、记录已知来源与未知身份，不从当前LRS3片段提帧。PNG像素hash、容器hash、尺寸、检测器hash/阈值、generation_box与score_box全部冻结。只允许固定静态几何检查；任一肖像检测失败记录该肖像阻断，不依据SyncNet改选肖像。主肖像缺失则主TFG分支阻断，其他可继续。外部静态参考排除目标视频时序泄漏；预训练数据是否重叠另记UNKNOWN，不声称消除所有训练数据污染。

主TFG对全部40句×主肖像；两张敏感性肖像对E按 sha256("phone-tfg-v1|pair_id") 排序前12句全交叉，不按增益更换。三张图不是新增独立样本，敏感性结果按source组聚合。

P0冻结协议参数、源绑定、数据选择、所有预定contrast、词表、窗口规则、图像和预算至 00_protocol/protocol_lock.json。checkpoint尚未产生的字段放第二级 model_lock，不能在P0伪填完成。

### 6.2 P1：测量支持与历史值重放

复用P2固定HuBERT layer6、XLSR layer10、revision、FIT mixed centroids及label集合。teacher.eval，参数requires_grad=False；输入waveform梯度必须存在。校准processor最大误差≤1e−4、cos差≤1e−5；zero-gain导出回读PCM须逐样本等于N。上述校准失败阻断训练，不能放宽容差。

显式输出三个support：

- legacy_matched：精确继承P2键、中心、group/label权重，仅重放历史数值；从原逐token分数独立复算N/T/D accuracy，误差≤1e−6。旧的2256/39不是所有后续分析强制分母。
- natural_primary：只根据N、固定label集合、N的frame位置及原输入质量构建；主E−N/A−N/B−N/C−N不依赖T是否可匹配。core为自然区间20%–80%，短音素无帧不偷邻帧。冻结每encoder的具体frame_indices及occurrence键。
- matched_nt：N/T一一、单调、无歧义匹配，要求两侧原始向量有效；在生成新候选前冻结共同occurrence。D/A/B/C沿N slots，T沿自身slots；只用它回答相对TTS是否达到水平。

每句资格要求≥5labels、≥10tokens、自然语音occurrence覆盖≥0.70；不满足保留BASELINE_INELIGIBLE。两个encoder各自frame几何不同，分别存frame索引，共同occurrence主集合取两侧原始N有效交集。新增模型不得改变support/label集合。

训练FIT support同样只由自然音频冻结，且覆盖全部合格FIT句；DEV checkpoint选择使用相同自然主support。分类为L2归一化token向量对冻结中心cosine argmax；并列按排序后的label表取首项。signed_margin=正确label cosine−最大错误label cosine。先group内label等权、label内occurrence均值，再group等权；训练损失也在句内label等权。不得把margin>=0代替accuracy。

失效候选不移除token：主保守accuracy计0、margin计−2，同时给可评分子集结果；缺失与真实低分分列。音素向量/分数矩阵用NPZ，JSONL只存索引，禁止再生成数百万行重复cosine JSON。

### 6.3 P2：三条件网络与训练/推理接口

冻结六个主音频臂：

| arm | 定义 | 推理有无本句音素身份 |
|:--|:--|:--|
| N | 原始自然PCM | 不适用 |
| D | P2锁定OPT_DYNAMIC_6的逐句标签辅助输出 | 历史生成时有 |
| A | 新训练AUDIO_FEATURES固定增强器 | 无；仅停顿保护mask |
| B | 新训练BOUNDARY_TIME固定增强器 | 无身份；有内部区间/时长 |
| C | 新训练MFA_PHONE_TIME固定增强器 | 有；训练与推理相同自然MFA输入 |
| T | 原始同文本TTS | 仅比较与独立评分 |

A/B/C为相同43输入通道的TCN：24频带log-power +16维音素embedding +3时序通道（phase、归一化log-duration、speech标志）。input Conv1d(43,64,k5,p2)，三残差block各含两层Conv1d(64,64,k3,dilation=1/2/4，对应padding)+GELU，仿旧结构；output Conv1d(64,24,k1)权重/bias全0，6*tanh输出增益。实现常规nn.Module，不借修改旧24→24模型参数含义实现条件输入。

- A将全部19条件通道置零。
- B将16身份通道置零，仅提供3时序通道。
- C提供全部条件。三个模型保留同样参数形状、初始化及训练预算；禁用通道的有效自由度不同，报告此限制。
- 词表只取FIT自然tokens经现有normalize_phone的标签，外加SIL和UNK；不使用E/DEV/TTS拓展。C的SIL/UNK embedding固定零，OOV仍有真实speech/time通道，记录覆盖率；禁止给不同未知标签临时分配新ID。
- STFT采用现renderer的512窗/128hop/center=True；第k列条件时刻k*128/16000，不使用HuBERT帧时刻或视频帧号。归属[start_s,end_s)，共享边界归后一个phone。时刻超PCM真实支持置SIL/0；区间重叠、负长、NaN或超长不是自动修复，预审记录错误。
- phase=(t−start)/duration，speech=1；duration取clip(log(duration_seconds),log(0.01),log(2.0))后按FIT统计标准化；静音/gap三通道全0。24带mean/std只由FIT自然帧估计，std下限1e−6，统一保存hash。
- 不给网络绝对pair_id、speaker_id、occurrence序号或整句缓存索引。保护mask从真实N预生成，对所有臂一致。
- 新接口 infer_pcm(source_pcm, protected_mask, conditioning, checkpoint) -> PCM+metadata；纯前向不需要teacher、centroids、评分label或optimizer。A/B/C分别校验允许的condition字段。推理时model.eval+inference_mode，前后state_dict hash一致。
- labels进入增强器，TFG仍只接受最终PCM与PNG；不向Wav2Lip注入标签、重训Wav2Lip或使用原视频监督。

P0可保存自然MFA已有tokens作为主条件；必须核验来源为N与固定转写的音频对齐。转写是已知文本辅助，不能来自视觉识别或待测TTS重新识别。输出重跑MFA只用于质量验证，绝不回写conditioning或主评分slots。

### 6.4 P3：真实训练、预算、锁定与消融

A/B/C各两个seed：20260922为发布seed，20260923为稳健性seed；禁止按E或TFG从两seed选优。每seed同一FIT采样顺序/初始公共权重，按source组均衡轮转；每epoch重新打乱，保存数据游标与全部RNG状态。完整句batch1、累积4句，AdamW(lr=1e−4,weight_decay=1e−4)，clip_grad_norm=1。每模型最多400次optimizer.step（1600 microbatches），或2小时训练墙钟预算，先到即停并保存last，6个训练job最多12 GPU小时。checkpoint每25 optimizer updates及step0保存，禁止把microstep写成optimizer updates。两小时不含后续DEV评价，DEV耗时独立记录；所有臂评价相同候选频率。超时臂标BUDGET_LIMITED，不宣称公平完成400步。

损失：按固定FIT slots计算 m_i=correct−max_wrong，L_phone为softplus((0.05−m_i)/0.1)的句内label等权均值；L_keep=sum((y−x)^2)/max(sum(x^2),epsilon)，全部用本步实际renderer输出；TV为gain/6的相邻时间及频带平方差均值。L=L_phone+1.0*L_keep+0.01*TV。renderer执行与父一致的硬投影，质量在PCM回读后再次检查。无有效FIT目标句预先排除留理由，不用零loss伪装训练。

固定renderer参数：n_fft=win=512, hop=128, bands=24；自然silence、10ms guard、5ms taper沿父合同；可编辑区残差能量比≤0.01、SNR≥20dB、|RMS变化|≤1dB、无新增PCM饱和。输出max gain固定6dB；本轮不再按E试3dB/其他剂量。记录projection触发率与实际能量，不把SNR当人类音质。

DEV对所有checkpoint做确定性纯前向与HuBERT主support评分；先要求100%输出T0/失真合格，再按signed margin降序、argmax accuracy降序、残差能量升序、optimizer step升序选择。step0可以被选中，必须如实标NO_LEARNED_IMPROVEMENT。DEV质量合格率或分数缺失不等于通过。第二级 model_lock记录每arm/seed checkpoint、训练实际更新数、DEV完整选择表与hash。XLSR/E、ASR、TFG不参与选型。

锁后在E生成并封存两个seed的A/B/C音频；主TFG只用发布seed，第二seed全量phone评价并在预定12句×主肖像进行C的TFG稳健性检查，不能据其效果增加/换seed。

C的诊断消融在预定12句做固定前向，主phone评分答案保持真实：
- C_ID_PERM：FIT词表按固定排序循环偏移一位，SIL/UNK不变，时间不变。
- C_BOUNDARY_JITTER：对相邻speech之间的内部边界，用seed=20260922按occurrence hash产生±20ms偏移；clip使两相邻区间各≥10ms。真实silence及保护mask不变；短到无法扰动者记skip。不改真实evaluation spans。
只评phone/剂量/质量，非选择臂，不纳入主TFG预算。A/B/C与消融的差异解释为条件可用性/依赖性证据，不等于分解TTS因果机制。

训练意外失败仅允许同配置工程重试一次，记录原因及续跑点。不能因效果差扩大网络/改loss继续挑选，修改方法需新协议。

### 6.5 P4：时序、内容与可分性评价

N/D/A/B/C全部E做T0；T单列原始参照，不对T套自然长度门。独立T0用int32/int64处理PCM极值，明确饱和属于pass条件；修正当前quality.pcm_contract中int16 abs溢出及新饱和未纳入总pass的问题。identity无残差时SNR写null+IDENTITY，不写非有限JSON值。

实际接通MFA：固定 [redacted-local-path]

MFA仪器在固定FIT8句校准：P0按固定hash顺序、每source至多一句，优先选择含两个相邻speech区间各≥160ms的句子；不足8句不放宽条件，记录局部校准不足。控制包括重复N，以及音频真正右移80ms并截尾。重复8/8通过、整体偏移≥7/8检出才可把重对齐用于全局时序门。

局部控制固定选择该句最早满足条件的相邻phone边界b，区间为[a,b)、[b,c)，将前段线性插值重采样到原长度+640 samples、后段到原长度−640，再拼回[a,c)，区间外PCM逐样本不变；np.interp使用两端点对齐的均匀网格。保存原[a,b,c]、预期新边界b+640及实际PCM，不只修改TextGrid。目标检出单独比较这一边界：相对重复N边界移动须同向且≥20ms，至少6/8句检出；不能依赖整句median/p95，因为单个真实局部偏移可能被总体分位数掩盖。未达标时局部时序保持标UNCALIBRATED；全局通过也不代替局部校准。MFA各split/各音频条件隔离corpus，不共享由E估计的适配统计回FIT，条件构造与输出复核不能修改原标签。

候选T1：唯一单调occurrence匹配，禁止zip；ambiguous不作合格。覆盖≥0.90、speech edit rate≤0.05；边界绝对误差median≤20ms、p95≤40ms；≥50ms自然停顿一一保序匹配，IoU≥0.5且起止误差各≤20ms，新增/缺失长停顿判失败。不要直接复用当前one_to_one_pause_match的全局贪心；实现保序最大匹配并记录未配对项。全队列至少90%通过方可给自动时序保持支持，失败句保留所有phone/TFG分析中并打标。

独立ASR使用固定faster-whisper base.en（P0解析确切revision/权重hash），language=en、beam=5、temperature=0、vad_filter=False、condition_on_previous_text=False、无reference prompt。每句独立转写N/D/A/B/C/T；使用同一自然参考文本，固定小写/去标点/空白分词规则，保存归一化前后文本。先在每source内用总编辑数/总参考词数得到WER，再source等权平均；空参考文本记CONTENT_UNSCORABLE，不置零。相对N点估计增幅≤0.005、95%CI上界≤0.01才给自动内容非劣。forced alignment不能代替ASR内容检验。

phone：两个encoder分别报告natural_primary的N/D/A/B/C accuracy、signed margin、coverage、paired delta；matched_nt报告N/T/D/A/B/C。旧top1−top2不再称正确音素间隔。原始softmax/cosine分数不因条件标签输入而更换。再用固定N侧triplets做ABX：A/X同label但必须不同occurrence，B不同label；triplets在P0冻结，每组每label最多20组，cos距离、tie=0.5，全臂相同键。无三元组则NOT_ASSESSED，不把A=X补齐。

D和C相对T“达到水平”的判定见6.8。HuBERT受直接优化，XLSR主迁移证据；两者都不能代替人类可懂度或TFG。

导出预定12句N/D/A/B/C/T匿名盲听包和评分表，映射独立存放；未获得人工评分一律HUMAN_NOT_ASSESSED。质量分支失败不取消已注册TFG诊断，但阻止“保持自然时序/内容的实用增强”声明。

### 6.6 P5：静态 TFG、延迟校准与音轨交叉

冻结Wav2Lip GAN与SyncNet：
- Wav2Lip checkpoint: third_party/Wav2Lip/checkpoints/wav2lip_gan.pth；sha256=ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8。
- SyncNet: third_party/syncnet_python/data/syncnet_v2.model；sha256=961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442。
- 独立环境参考 [redacted-local-path]}/bin/python；P0核验可用，不把路径存在当作前向成功。
- 25fps，generation batch4，SyncNet batch20，seed42，eval模式与确定性设置；固定静态人脸框，所有臂完全相同几何。
- FFV1 master无音轨，音视频评分分别指定路径；播放MP4不用于主评分。新worker流式编码帧，避免累积整段512×512帧导致内存峰值。保留master与实际评分crop/embedding/hash。

先在FIT按固定hash顺序取4个不同source groups，各用三张肖像运行N、独立N_REPEAT。对同一个V(N)分别配N、N_DELAY_+200ms、N_DELAY_-200ms：
- 延迟控制以真实PCM移位±3200 samples、空位填0、保持长度构造；使用专属预冻结窗口避开填零及所有前端感受野。
- 重复所有cell的C/D/D_anchor差≤1e−4、lag一致；若当前环境无法满足，记录测量校准失败，不凭效果调容差。
- d(t,k)=distance(v_t,a_{t+k})。音频晚200ms时期望k相对N增加5，早200ms减5；每肖像至少3/4符合±1帧，且正确自然lag的平均距离变差≥0.10。控制必须先通过，才解释TFG分数。
- 保存每个cell完整lag曲线；lag符号不得混用官方打印offset，两个字段都存并注明映射。

主矩阵：同一PNG生成V(N)、V(D)、V(A)、V(B)、V(C)，每个增强E∈{D,A,B,C}保留四cell：
[V(N),A(N)]、[V(E),A(N)]、[V(N),A(E)]、[V(E),A(E)]。
共享N,N仅算一次，共13个自然时钟评分cell/句/肖像。主结论来自V(E),A(N)−V(N),A(N)；另两cell用于判断评分音轨效应与交互，不能替代主比较。

T仅另生成V(T)并配A(T)，作为native TTS参照（第14个cell）。N,N与T,T另在各自时间轴上按normalized-duration十等分bin先均值再句均值给描述性结果，报告时长及有效窗口数；禁止未对齐T与N直接做主replacement比较、禁止截短T至N以凑时长。T−N native差异不是纯生成端因果证据。

主肖像40句×6视频=240；敏感性两肖像×预定12句×6=144；第二seed C在12句×主肖像加12视频及相关cell；另计FIT校准/重放。所有数量是计划上限，不把缺失填成已完成。先profile最长FIT校准cell估算空间/耗时，保留逐cell资源状态；暂停后同hash续跑。

视频防泄漏验收必须包含：
1. worker参数/请求无目标视频路径；图像decode为一帧；静态视觉tensor每时刻相同，保存输入tensor hash与source_frame_indices全0。
2. 输出frame数由驱动音频mel chunks决定，不由目标视频长度决定；静态ROI外逐像素等于PNG。
3. CPU契约测试对目标视频读操作设为抛错，静态renderer仍可执行；更改原视频路径/内容不应影响渲染manifest或输出缓存键。
4. 真数据smoke记录文件访问审计（可用strace限定媒体打开或等价隔离沙箱）；只允许指定PNG/WAV、模型和依赖文件。全正式cell校验请求白名单与hash；不凭元数据声明已证明无泄漏。
5. 原视频真实嘴形、SyncNet real-video特征、逐帧框、任何face sequence都不能用于挑帧、推理或校时。不要以“遮住下半脸”为理由允许动态视频进入参考分支。

### 6.7 P6：SyncNet前端、固定支持与评分合同

采用官方SyncNet V2的BGR数值/224×224 crop、5视频帧、python_speech_features MFCC的20帧/步长4，与现score_worker一致；不擅自RGB交换/归一化。模型、MFCC参数/版本、Wav2Lip音频前端源码hash与ffmpeg版本均入锁。

在评分前，仅根据实际PCM长度、生成mel分块规则和冻结前端几何，生成W；不能读距离矩阵后按finite/好坏筛窗口。N/D/A/B/C同长度，主全部13 cells用同W；T有独立W_T。支持要求每个lag k∈[-15,15]：
- 视频5帧的所有生成mel真实输入支持在[0,L)内；
- MFCC 20帧及preemphasis支持在[0,L)内；
- 无末端mel重复chunk、反射pad或延迟填零样本进入。
参考现formula：generator [floor(3.2t)*200−401, (floor(3.2(t+4))+15)*200+400)，audio [(t+k)*640−1,(t+k)*640+3440)。必须对当前锁定前端逐边界核验，不未经验证照抄常数。时刻t+4涉及尾端重复chunk时整个query不合格。区间统一半开，写明最大样本索引。

每句主W≥20 queries才评主TFG，资格在任何候选评分前冻结；不合格句保留BASELINE_SUPPORT_INELIGIBLE，其他句不补入。FIT控制用独立W_delay，包含±移位后原始有效区域。W内任一NaN/Inf为CELL_FAILURE，不能缩W后再算。

保存V/A embeddings、d矩阵、lag列表、W/hash、逐lag均值。距离公式保持官方torch.nn.functional.pairwise_distance语义（包括eps）；独立验证器按锁定公式重算，不悄然换成平方距离/余弦距离。
dbar(k)=mean_{t∈W}d(t,k)；D=min_k dbar(k)；C=median_k dbar(k)−D；lag=argmin（tie取升序首项）。另保存D0=dbar(0)。

每pair/肖像的k_N只从V(N),A(N)在同W取得并固定给所有自然时钟候选；D_anchor(E,N)=dbar_E,N(k_N)。这个基线锚只用于评价，不用于重定时、选模型/候选。报告lag变化与C/D/D_anchor一起，避免仅靠背景距离升高宣称同步改善。不得平移/拉伸生成视频或音轨以提高主结果。

### 6.8 P7：预注册统计与结论状态

统计单位source_group；同组多句先等权聚合，多肖像敏感性先在组内聚合；主肖像与敏感性分别报告，不把帧、音素、肖像当独立n。group bootstrap PCG64 seed20260922、20000 draws；全对比复用同一group抽样表。预先列完整对比，点估计与95%CI总是报告。以下工程阈值为本轮预注册选择，不是已有公认MCID。

主TFG family三项：D−N、C−N、C−A，全部使用原A(N)及主肖像。
- 每项ΔC>0.05，Bonferroni三比较的98.333333%双侧CI下界>0才支持TFG metric gain。
- 同时要求D改善（baseline D−candidate D）及D_anchor改善点估计≥0、各95%CI下界≥−0.05；相对该baseline的|lag−k_N|增加不超过1帧的句子≥90%。不过关标C_ONLY或TIMING_TRADEOFF，不称可靠同步增强。
- C−B、B−A、第二seed、其他肖像与交互为预注册描述性诊断，不能替换失败的主endpoint。
- practical_success另要求T0全通过、T1与ASR自动门通过；无人类评价只能称模型指标收益。Wav2Lip与SyncNet有关联，不称跨TFG泛化或人工同步已证实。

主phone family为XLSR的C−N、C−A、C−B，natural_primary，同样使用98.333333%CI；增益需accuracy差point≥0.01且区间下界>0、signed margin同向。HuBERT平行报告但属于训练目标，不独立确认。C>N成立但C≤B时，只能说条件模型有效，不能归因于音素身份输入。ABX恶化或质量未评限制解释。

相对TTS独立family：在matched_nt上分别计算D−T、C−T，两encoder都报告，四个对比使用98.75%双侧CI。预设accuracy容差ε=0.02：
- 下界>−0.02：支持该探针口径“非劣于TTS”。
- 整个区间在[−0.02,+0.02]：支持该口径“近似等效”。
- 下界>0：支持该探针口径“更高”。
- 其余INCONCLUSIVE；不能把p>0.05当等效。
“跨编码器达到TTS”要求两encoder均通过相应条件、signed margin没有相反下降证据。D对HuBERT的优化暴露始终注明。原历史点估计优势不能直接代替新CI检验。

交叉cell诊断：Δ_gen=C(E,N)−C(N,N)，Δ_audio=C(N,E)−C(N,N)，interaction=C(E,E)−C(E,N)−C(N,E)+C(N,N)；对D可同样分解，方向定义单列。正interaction不自动证明更自然的嘴形。

缺失合同：phone候选缺失使用6.2保守计分；TFG无合理有限C/D惩罚界，不能编造0分或删除失败句。报告所有预定group的成功/失败/原因，complete-pair效果明确标条件性；任何主比较候选特有缺失时默认TFG主判定INCONCLUSIVE_MISSING，仍给完整对子集数值及成功比例。主合格组<30同样INCONCLUSIVE_SMALL_COHORT；预冻结基线不合格与候选特有失败分开。T1不合格也不许为了好看删出分母。

解释矩阵必须能区分：phone↑/TFG↑、phone↑/TFG无明确↑、phone无明确↑/TFG↑、两者无明确↑。“无明确↑”不是证明无效。即使双阳性，也只支持该音频干预与TFG收益共现，不足以证明音素可分性是唯一中介。逐组phone delta与TFG delta的Spearman相关仅探索性附表。

### 6.9 P8：状态机、产物与报告

CLI预定：
python -m scripts.experiments.phone_gain_static_tfg_mfa.run --config scripts/configs/phone_gain_static_tfg_mfa_v1.yaml --run-id <id> --stage audit
同入口 --stage all --resume；独立检查：
python -m scripts.experiments.phone_gain_static_tfg_mfa.check --run-dir runs/phone_gain_static_tfg_mfa_<id>

stage枚举audit/calibrate/train/lock/infer/quality/phone/render/score/analyze/report；all执行所有就绪分支，无“规则没过门则跳过训练/TFG”逻辑。每stage分别COMPLETE/PARTIAL/RESOURCE_WAIT/DEPENDENCY_BLOCKED/ENGINEERING_FAILURE。report即使其他分支阻断也可生成真实状态；all不得无条件写execution=COMPLETE。锁和hash变化不得复用旧cell。

目录：
00_protocol（protocol_lock/资源/输入/划分/肖像/support/曝光账本）；
01_calibration（phone/renderer/SyncNet repeat+delay/MFA控制）；
02_train（arm/seed checkpoint/history/DEV表）；
03_lock（model_lock）；
04_audio（推理PCM/manifest/ablation；D可指向只读父资产）；
05_phone（逐token NPZ/JSONL/ABX/配对表）；
06_quality（T0/T1/TextGrid/ASR/盲听）；
07_tfg（每cell无音轨视频/worker sidecar/访问审计）；
08_sync（embedding/matrix/支持/曲线/交叉表）；
09_report（报告/统计/状态/独立check/播放索引）。

每个cell key含pair_id/source_group/arm/seed/portrait_id/video_audio_arm/score_audio_arm；绑定PCM hash、静态像素hash、模型/配置/代码/支持hash。resume只接受完整且hash一致产物，partial写独立attempt路径，工程失败允许一次同配置重试。

报告开头直接回答三个问题，分别列旧D、新A/B/C、原T，注明E曝光历史、label输入权限、各seed真实optimizer updates、所有排除及未评项。Sync-C显示3位小数，内部保留全精度。导出预定12句的PNG→各生成视频+统一N音轨对比及各自音轨对比，播放器标清驱动/评分音轨；可播放MP4为展示副本，不替换FFV1评分源。不伪造人工评分。

最终通过BM更新本spec实施状态并创建或更新单一Experiments结论笔记，保留阴性/阻断结果，链接run与本spec。无需改CONTEXT/HANDOFF或旧历史结论文件。

## 7. Expected Change Surface

### Must change

- 新增P包：__init__.py、config.py、assets.py、support.py、conditioning.py、model.py、train.py、infer.py、mfa.py、quality.py、phone_eval.py、render.py、render_worker.py、sync_support.py、score_worker.py、analyze.py、check.py、report.py、run.py。
- 新增scripts/configs/phone_gain_static_tfg_mfa_v1.yaml。
- 新增tests/experiments/phone_gain_static_tfg_mfa/，覆盖第8节实际合同。
- 新run产物；本Research spec状态及一个Experiments结果实体。

### May change

确有接口阻断时最小修补phone_separability_enhancement公共helper并增回归测试；优先新adapter。静态worker无法流式复用时在新包实现worker并做原前向parity，保持旧调用者行为。MFA/ASR依赖下载与环境补齐属于实现步骤，记录版本，不全局升级用户环境。

### Should not change

原始数据、LRS3视频、旧run/配置、旧checkpoint、Wav2Lip/SyncNet权重、TTS生成流程、核心00–05脚本、AGENTS、CONTEXT、HANDOFF、用户其他dirty changes；不训练TFG，不追加新TTS模型，不解封项目sealed/test。本轮不做DAC/其他未完成父机制实验，也不将其写成已完成。

## 8. Validation Plan

1. 静态：python -m compileall -q scripts/experiments/phone_gain_static_tfg_mfa；对新包/测试定向ruff（环境已有时）与git diff --check。只检查本轮改动，不修无关dirty状态。
2. conditioning测试：STFT第0帧、共享边界、尾端、gap/SIL/UNK、Unicode phone规范化、OOV、重叠/NaN区间拒绝；B换身份条件张量不变，C真实变化；A换全部内部phone区间而固定mask时输出不变。置换条件不能改变evaluation labels/mask。
3. 模型/训练测试：初始zero-gain PCM identity；真实多步更新改变权重；冻结teacher参数grad=None、模型与输入梯度非零；C embedding在非零输出头后能获梯度；4 microbatches只做1 optimizer update；step0/25候选真实存在；完整resume与不中断运行在确定性toy上相同。
4. 权限测试：推理过程中构造teacher/optimizer/backward抛错仍能成功；前后模型hash相同；额外TTS/video字段拒绝；A/B/C错误conditioning schema拒绝；修改E/XLSR/SyncNet分数不影响DEV selection_lock。
5. support/评分：N主support不受T缺失影响；T自身时轴；legacy历史重放；argmax并列反例、错误自信margin为负；候选缺token不涨主分；ABX A≠X；多肖像不能增加独立n；等效/非劣/优势/不确定边界有可手算测试。
6. 视觉与媒体：PNG后缀伪装MP4、动态frame manifest、原视频路径、symlink错误绑定被拒；ROI外像素identity；source_frame_indices全0；相同PNG张量重复；音频长度控制frame数；target video不可读时静态推理成功。
7. SyncNet：真前端起止/末端mel chunk/±200ms支持fixture；W冻结后NaN必须失败；曲线已知最小值验证C/D/lag符号及anchor；四cell交互恒等式；原视频或展示转码不能混入评分。
8. 质量：−32768/32767饱和、新增clip计入FAIL、identity SNR有限序列化；保序停顿匹配、重复phone歧义/插入不zip错位；MFA缓存hash变化拒绝；真实PCM移位控制必须调用实际MFA，不用假TextGrid充当正式校准。
9. 回归：pytest -q tests/experiments/phone_separability_enhancement tests/experiments/phone_separability_mechanism tests/experiments/static_image_natural_to_tts_bridge/test_frontal_probe.py；有共享helper修改再补相关调用方测试。
10. CPU端到端：至少3labels/4groups/2split、fake teacher/TFG用于状态机与失败合同，显式test-only，覆盖完整/阴性/缺失/资源阻断。不能以fake end-to-end代替真实实验。
11. 真数据smoke只用FIT：最长合格句资源profile、A/B/C各至少一次实际更新、静态Wav2Lip及SyncNet forward、MFA repeat/shift、ASR一条、PCM与媒体重放。不得用E smoke挑配置。
12. 独立checker不导入runner的score/aggregate/decide/support生成函数：从冻结slots与token vectors重算phone、从V/A embeddings重算d矩阵、W/lag/C/D/anchor/组聚合/bootstrap、从PCM/视频像素核验约束/身份。共享只读I/O/hash可用。保存核验覆盖清单。
13. 独立真前向抽样固定hash选主肖像至少3句，覆盖N/D/C/T；官方SyncNet同crop/PCM矩阵最大误差≤1e−5，phone pooled vectors相对L2≤1e−4。全部输出做hash/元数据检查，不能把抽查写成全量新前向。
14. 篡改测试必须能发现：summary数字、矩阵、support、label、PCM保护区、checkpoint hash、portrait hash、评分音轨、视频frame绑定。checker禁止写回被验证summary，失败必须非零退出。
15. 文档验收：spec的每个必需分支对应实际产物或明确阻断原因；训练不是SKIPPED、TFG不是空manifest；人工未评显式保留；历史源曝光、预训练重合未知、静态参考适用范围全部可见。

## 9. Risks and Edge Cases

- 读MFA本身不构成泄漏，但同标签驱动优化与评分会偏向代理；XLSR/ASR/TFG/人工分别约束，不把模型评分等同人耳清晰度。
- TFG checkpoint可能预训练见过相关来源，目前未知；本协议解决目标视频推理时泄漏，并不自动证明训练数据完全独立。
- 39个历史phone有效组与40句TFG不是同一分母；新natural-only支持可能改变数值，必须分别标口径，不能篡改使其符合旧表。
- TTS时长差异与静态肖像发音偏置都会影响native评分。TTS phone等效与TFG等效是不同问题。
- 低维gain提高HuBERT而TFG不提高是有效阴性，不能追加SyncNet优化后仍声称独立下游验证。
- 当前PCM helper有int16极值/饱和判定陷阱，当前MFA runner在v2未接线；旧checker PASS覆盖不了这些新维度。
- legacy render_worker缓存全段帧与summary巨大JSON会造成内存/硬盘峰值，需流式媒体与紧凑产物；资源不足暂停可恢复，不能删数据。
- A/B/C实际超时导致训练预算不齐时单列同更新数敏感性（仅用预先保存checkpoint及DEV规则，不能按E选），未完成预算不能当充分阴性。
- 全部条件都依赖known transcript/MFA停顿mask；若未来部署只有音频，需要另做ASR→MFA端到端实验。本轮不替换成视觉转写。
- 没有足够MFA局部控制或人工评分时，仍可回答探针/TFG指标问题，但不能宣称发音时序/听感已完整保持。

## 10. Assumptions / Unknowns

- VERIFIED：P2已有40句D、固定checkpoint与E/XLSR分数，TFG未跑；历史N/T/D对应39组2256共同条目。
- VERIFIED：现LEARNED已用MFA标签监督loss但未输入身份条件；新C是不同输入假设。
- VERIFIED：现单PNG worker无需原视频；data/data/image/{3,6,9}.png本地存在；相关静态前端/固定支持与测试可以复用。
- VERIFIED：P2主support实际依赖N/T匹配；旧训练max_steps计的是microbatches，不能当optimizer更新数；当前时序/内容分支未完整实现。
- LIKELY：V100 16GiB可执行逐句HuBERT训练与静态Wav2Lip/SyncNet串行任务，必须以最长FIT profile核验。
- LIKELY：MFA条件能缓解固定增强器自行识别目标phone的负担；没有证据保证一定提高。
- UNKNOWN：GPU/磁盘当前占用、MFA/ASR缓存完整性、三PNG身份来源、TFG预训练数据重合、真正speaker跨split重合。
- UNKNOWN：D是否有TFG收益、C是否优于A/B、提升是否保持实际边界/听感；不能在实现前填写阳性。

## 11. Handoff Contract

按第3节锚点新增独立包，保持第5节所有输入权限和不变量；仿静态worker与修正后的前端支持模式，独立实现checker，不复制历史验收缺陷。先资源/资产审计与FIT校准，再真实训练/锁定/评价；DIRECT下游分支不等待学习阳性，条件学习也不因DIRECT下游阴性被取消。

最小完整交付：冻结manifest和防视频泄漏证据；D重放及A/B/C两seed真实训练/固定推理；N/T/D/A/B/C同口径phone对比；真实MFA/ASR状态与盲听包；静态肖像四cell TFG矩阵、native TTS参照、固定窗口C/D/anchor/offset；独立复算与可播放对比；BM结果。结果可以阴性或不确定，未执行不能包装为科学阴性或完整执行。

若仓库证据与核心合同冲突，保存具体复现证据、暂停受影响分支并继续独立分支；不得私改数据/模型/阈值。无关重构、原视频辅助推理、E/SyncNet选模型、删除用户资产均不在本任务范围内。

### Relations

- extends [[自然时钟受约束音素增强与机制对照 v2 Implementation Spec]]
- follows [[自然时钟受约束音素增强 v2 工程实现与校准验收 2026-09-21]]
- relates_to [[静态图片 Natural-to-TTS bridge 实验]]
- relates_to [[TTS 时间辨识与同文本实例交叉实验 2026-09-17 结果]]
