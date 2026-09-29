## Purpose

对已观察到的 LRS3 英文 TTS 原生 Sync-C 优势，执行可复算的分数组成分析和有界完整距离曲线诊断。design.md 与 input-bindings.json 构成本规范的数值和输入契约。

## ADDED Requirements

### Requirement: Preserve evidence and endpoint distinctions

实验 SHALL 明确 LRS3 的原生优势证据，不以“英文无优势”作为前提；SHALL 区分原生、strict replacement 和相对 NAT_ONLY 的 conditioning 对比。MUST NOT 将历史不同模型、数据集、原始TTS、MFA-linear和bridge结果混成单一总体增益。

#### Scenario: English positive and negative studies coexist

- **WHEN** LRS3 两模型原生对比为正，而其他英语语料或replacement对比为负
- **THEN** 分别保留条件、样本量与端点，解释为条件异质性而非语言整体有效/无效

### Requirement: Fixed auditable cohort and source groups

实验 SHALL 读取固定文件hash，以原manifest序号关联50条LRS3及200个历史评分cell；SHALL 使用真实源视频父目录得到45组。曲线队列 SHALL 固定12来源、ID151–162与24个LeapTalk视频，MUST NOT 依分数或缓存可用性换样本。

#### Scenario: Generic speaker key is encountered

- **WHEN** manifest的speaker_key对所有LRS3都等于lrs3
- **THEN** 从绑定video_local_path解析source_group并与stem核对，不将全部样本合成一组或将50条误当50独立来源

#### Scenario: A bound video or score is absent

- **WHEN** 必需输入缺失、hash改变或配对重复
- **THEN** 为受影响阶段记录具体阻断，保留原denominator；不补零或挑另一成功样本，其他独立CPU阶段可交付

### Requirement: Exact score decomposition with honest precision

实验 SHALL 按design第3节恢复B_hat=C_saved+D_saved，计算gain_match、gain_background并验证gain_C的加法恒等式；SHALL 区分record等权历史检查和group等权推断。MUST NOT 从两个标量伪造完整距离曲线或将分解称作声学因果归因。

#### Scenario: Confidence rises while minimum distance is unchanged

- **WHEN** TTS的C提高而D基本不变
- **THEN** 报告对应背景项变化及不确定性，不据此单独宣称真实同步精度改善或评估器作弊

### Requirement: Bounded native-video curve extraction

实验 SHALL 只复用绑定的LeapTalk原生媒体，使用官方预处理和固定模型配置；SHALL 在读新评分前冻结基于几何/时长的track选择，保存[T,31]矩阵和来源。MUST NOT 直接评分未裁剪全脸、搜索最佳track或新生成TFG视频。

#### Scenario: Multiple face tracks exist

- **WHEN** 官方检测产生多个有效track
- **THEN** 按实际帧数降序、起点升序、文件名字典序选择，并保留全部选择依据，不按C/D选择

### Requirement: Correct lag and support calculations

实验 SHALL 先按支持对时间行求均值，再对31个lag取median/min；SHALL 分开FULL、INTERIOR、EQUAL_COUNT、C_5与D0。列j的官方offset SHALL 为15−j。MUST NOT 将不同原生时钟的等窗口数量视为音素对齐。

#### Scenario: Padding changes the result

- **WHEN** FULL和INTERIOR增益不同
- **THEN** 同时报告两者及配对变化，不删除不利口径或把历史FULL值冒充无padding值

#### Scenario: A flat or truncated trough is measured

- **WHEN** C≤1e−6或半深度连续谷接触搜索边界
- **THEN** 分别标记flat/null或censored下界，不输出伪精确完整谷宽

### Requirement: Fixed controls and limited claims

实验 SHALL 完成ID151两臂的两次独立重复cell与两次200ms延迟cell；SHALL 使用同一冻结crop和固定共同支持。新SyncNet cell预算 SHALL 不超过28。控制失败 MUST 保留CONTROL_FAILED和原数据，不重选样本或追加得分重试。

#### Scenario: Delayed audio retains high searched confidence

- **WHEN** 音频右移200ms后最佳offset约移动−5帧但搜索后的C仍高
- **THEN** 区分搜索补偿与零延迟匹配，不把高C当作不存在延迟

### Requirement: Matched statistical estimands and multiplicity

实验 SHALL 使用design第5节同一group等权点估计与bootstrap分布，固定seed与10000次共同抽样；SHALL 报告预定校正区间与描述性区间，重复/控制不增加独立样本数。MUST NOT 用不显著证明等效或汇总完曲线再计算“平均C”。

#### Scenario: Multiple clips share a source

- **WHEN** 某源视频在50条中包含多条记录
- **THEN** 先组内平均配对差，再对45组等权估计与重采样，另列record等权历史均值

### Requirement: Resource-aware staged completion

实验 SHALL 先交付CPU分解，再在资源检查通过后串行持有GPU lease完成曲线；SHALL 保存断点身份并限制临时清理到本run。零新TTS、零新TFG、零训练与零云调用 SHALL 保持。

#### Scenario: GPU or disk is unavailable

- **WHEN** 其他任务占GPU，或空间不足以容纳峰值并留1GiB
- **THEN** 曲线阶段返回RESOURCE_WAIT，CPU结果仍可生成report；不终止其他进程、不删除历史数据、不宣称曲线已完成

### Requirement: Independent validation and perception boundary

实验 SHALL 从绑定原始评分和矩阵独立重算验收；SHALL 生成固定12对的匿名人工评估包。没有真实完整共同评审时感知结论 SHALL 为NOT_ASSESSED或PARTIAL，自动分析可单独完成。

#### Scenario: Automated analysis finishes before ratings

- **WHEN** 全部自动产物和验收完成但人工CSV尚空
- **THEN** 自动结果完整交付，报告说明没有感知证据，不生成或推测人类评分
