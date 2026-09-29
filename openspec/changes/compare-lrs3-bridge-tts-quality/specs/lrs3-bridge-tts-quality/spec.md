## Purpose

定义在固定 LRS3 自然音轨上，比较本地与云端 Qwen TTS 目标所产生 bridge replacement 效果的可复算实验，并将 TTS 来源差异、绝对 replacement 增益与独立音质证据分开解释。design.md 是本规范的数值、算法、接口和矩阵契约组成部分。

## ADDED Requirements

### Requirement: Fixed paired cohort and auditable TTS provenance

实验 SHALL 使用绑定确认队列的全部22条/22 source groups及原顺序；SHALL 校验 input-bindings.json 的文件SHA-256、ordered-ID hash与每条自然音频、文本、face video身份。两种来源 SHALL 分别固定为 `faster_qwen3 / Qwen/Qwen3-TTS-12Hz-0.6B-Base` 和 `dashscope_vc / qwen3-tts-vc-2026-01-22`。实验 MUST 用metadata与音频hash回溯来源，不以模型大小、路径名或历史Sync-C推断音质。本次 SHALL 标记fit-only、历史样本已见。

#### Scenario: Same utterance has both correctly identified providers

- **WHEN** 两侧TTS具有同一sample_id、文本、自然reference且各自模型/backend身份可验证
- **THEN** 可组成配对记录，并记录历史bridge目标为云端来源的证据

#### Scenario: An apparently comparable file belongs to another dataset or reference

- **WHEN** 本地文件来自AISHELL/RAMC、另一条句子、另一自然参考，或无法验证模型来源
- **THEN** 不接受该配对，不读取其分数替代LRS3结果

### Requirement: Frozen setup precedes bounded synthesis and complete input lock

实验 SHALL 在新合成前冻结design第2节setup，按身份复用合法音频，只为缺失的本地项生成一次注册输出，最多22次成功本地合成、新云端调用为零。SHALL 记录实际backend、权重、decoding kwargs、逐条seed、reference和输出hash，统一canonical规则。MUST NOT 根据听感、长度质量、对齐失败或得分重新抽取成功输出。所有新SyncNet评分之前 SHALL 冻结analysis lock。

#### Scenario: Missing local synthesis is prepared

- **WHEN** 某条无符合setup的既有本地输出
- **THEN** 使用固定文本、English、配对N及确定seed生成，保留原始与16k canonical波形和完整provenance

#### Scenario: A successful synthesis is poor or a cloud asset is absent

- **WHEN** 本地成功输出不满足输入门槛，或云端产物缺失/身份错误
- **THEN** 记录INPUT_BLOCKED与原始失败，禁止重抽、换云端模型、删样本或只取成功交集宣布完成

### Requirement: Common alignment and vocoder isolate the provider comparison

实验 SHALL 对双方应用design第3节同一冻结MFA3.4.1/english_mfa、音素映射、WavLM layer6、prematched HiFi-GAN流程，重新生成两侧M且长度等于N。N的对齐 SHALL 共用一次；每侧 SHALL 保留自己的TTS音素边界、完整mapping trace、fallback/尾部处理与权重代码hash。MUST NOT 一侧沿用不同历史算法、复制另一侧音素边界、以DTW或全局波形拉伸替代、把未知speech当静音。

#### Scenario: TTS durations differ

- **WHEN** 两来源TTS时长或停顿不同但均满足输入与音素映射契约
- **THEN** 使用各自音素内相对位置映射到同一自然frame grid，经过同一vocoder和有界尾部处理，保留差异诊断

#### Scenario: One source cannot align speech

- **WHEN** 任一必需记录出现unknown phone、未匹配speech或缺失静音fallback
- **THEN** 保留22条denominator，输出INPUT_BLOCKED，不在主分析里单边删除或补零修复speech

### Requirement: Deterministic natural-phase bridge and reconstruction control

实验 SHALL 生成N/B0/B_LOCAL/B_CLOUD四臂。两个候选 SHALL 使用design第4节的共同STFT、alpha=0.75、natural phase、一次RMS匹配、至多一次peak衰减和一次PCM量化；B0 SHALL 通过同算法alpha=0。N SHALL 保留原PCM。SHALL 区分file hash与decoded PCM hash，冻结从落盘波形计算的官方Wav2Lip mel movement等诊断。

#### Scenario: Providers exchange places with identical targets

- **WHEN** 测试将M_LOCAL与M_CLOUD设为同一PCM
- **THEN** 两个bridge输出及mel相同；仅改变provider标签不能改变音频

#### Scenario: A candidate moves less than expected

- **WHEN** 某侧不满足至少20/22 progress>=0.15及均值95%CI下界>0.15，或目标退化
- **THEN** 保留固定候选并报告BRIDGE_MOVEMENT_FAILED，不调alpha、响度或选择更好目标补救

### Requirement: Paired rendering and exact natural-audio scoring matrix

实验 SHALL 按design第5节四臂乘两次独立渲染生成176个视频，评分恰好308个cell；两次repeat SHALL 有独立推理进程和工作目录。SHALL 共享记录内face、预定几何、25fps调度、共同前F帧/前F*640采样支持及冻结模型。所有replacement cell MUST 配对应记录的原始N精确前缀。SHALL 验证mux PCM、视频码流、track/窗口支持和官方前向parity。

#### Scenario: A bridge has a higher own-audio score

- **WHEN** V_B_CLOUD/B_CLOUD优于其自然音轨cell
- **THEN** 只报告自身配对诊断；主端点仍使用V_B_CLOUD/N对V_B_LOCAL/N

#### Scenario: Repeat is copied or score support changes by arm

- **WHEN** 第二个repeat只是视频复制、不同臂选择不同时间/track，或音轨被隐式重编码改变PCM
- **THEN** 独立验证失败，输出ENGINEERING_BLOCKED而非科学结论

### Requirement: Controls calibrate only their registered scope

实验 SHALL 依design第8节验证四臂重复性、B0相对N等价范围及V_N的N/N_REV错误音轨响应；任一失败 SHALL 为CONTROL_FAILED，候选结果只作描述。SHALL 使用r平均后的记录为统计单位。MUST NOT 将这些控制解释为已证明局部时间传递、speaker identity或普遍SyncNet有效性；MUST NOT 直接继承历史LOCAL_SWAP失败作为新实验结果。

#### Scenario: Wrong audio lowers confidence but local timing transfer remains unknown

- **WHEN** N_REV损伤门槛通过
- **THEN** 只记录对严重错误音轨的端点敏感性，并继续明确局部时间传递未在本实验检验

### Requirement: Source effect uses a direct paired contrast

实验 SHALL 用design第7节deltaC公式作为唯一主检验，条内先平均两次repeat，再按22个source groups配对bootstrap10000次、PCG64(20260913)、percentile95%CI。SHALL 单独报告各来源相对N的gC/gD和Bonferroni两项C次要检验、offset兼容性。MUST NOT 用独立均值相减、不同队列、不同N基线、单侧显著性或更有利repeat替代主比较。

#### Scenario: Cloud is less harmful than local

- **WHEN** deltaC的CI下界>0，但云端相对N的gC未通过绝对增益门槛
- **THEN** 可在控制通过后报告CLOUD_BRIDGE_STRONGER，同时明确云端未建立正replacement增益

#### Scenario: One arm is significant and the other is not

- **WHEN** 单臂gC显著性不同但deltaC的CI跨零
- **THEN** source_comparison为SOURCE_DIFFERENCE_UNRESOLVED，不声称两来源有已建立差异

### Requirement: Quality is measured independently before attributing the source effect

实验 SHALL 生成88个匿名刺激及design第6节问卷、blind mapping、CSV模板，分别评原始T与M，要求每pair至少3名共同独立人工评审者。听评复制品的响度处理 MUST NOT 影响实验驱动或评分音频。SHALL 配对计算q_raw/q_target及记录和评审者双重bootstrap；逐条Spearman只作预注册探索关联。MUST NOT 由Sync-C、模型大小、provider标签或代理模型生成假MOS。

#### Scenario: Ratings are unavailable

- **WHEN** 人工评分未导入或任一pair共同评审者不足3人
- **THEN** 自动比较可完成，缺失阶段及质量总体明确QUALITY_NOT_ASSESSED，不凭预期将云端标为高质量

#### Scenario: Raw cloud quality improves but aligned target advantage is unestablished

- **WHEN** q_raw的CI下界>0而q_target未建立正优势
- **THEN** 明确进入bridge前的质量优势尚未得到验证，不将provider差直接解释为更高输入音质的作用

#### Scenario: Quality advantage and bridge advantage are both observed

- **WHEN** raw与target独立质量优势成立且主deltaC为正、控制和movement均通过
- **THEN** 只报告本队列中与质量关联假设一致的来源差异；音质因果效应和泛化确认标志仍为false

### Requirement: Missingness and immutable outputs remain independently reviewable

实验 SHALL 提供design第9节CLI、阶段清单、每条配对数据、原始日志、统计与多维终态，并由独立validator从输入/音轨/日志重新计算构造、矩阵、主差和判定，不能只读取producer的passed标志。SHALL 保留失败与缺失denominator。补充听评 SHALL 新建带父hash的analysis版本，不能改写已冻结评分或旧final。历史run MUST 保持不变。

#### Scenario: An interruption resumes

- **WHEN** 成功cell的protocol、输入、运行配置与输出hash均一致
- **THEN** 复用该cell，禁止按得分重跑；不一致必须报告具体绑定错误

#### Scenario: Complete negative or inconclusive result

- **WHEN** 完整矩阵、控制与movement有效但主CI跨零
- **THEN** 完成报告SOURCE_DIFFERENCE_UNRESOLVED与CI宽度，不扩大样本、换指标、调参或标为工程失败
