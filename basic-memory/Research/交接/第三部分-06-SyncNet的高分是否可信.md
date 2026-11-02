---
title: 第三部分 模块6：SyncNet的高分是否可信
type: note
permalink: tts-exp/research/交接/第三部分-06-sync-net-的高分是否可信
tags:
- research-handoff
- readable-guide
---

# 第三部分 模块6：SyncNet的高分是否可信

**本模块的问题：SyncNet是否对TTS有偏好？其他评分是否同意？处理错误能解释多少，独立的画面或人工判断又支持什么？**

这里按研究问题合并介绍主要实验，补算、故障恢复和后续修正不另算独立复现。原报告标题可能保留技术用语，正文解释以本页及后续修正为准。没有独立结论的资料仍列在后面的逐条来源中。

## 6.1 能区分自然与TTS的特征能否解释分数差

我们用既有中文主队列74对、13位说话人的声音特征，按说话人分组训练简单预测器。它容易区分自然与TTS，却不能可靠预测逐句 Sync-C 差。旧压缩处理也会改变评分构成。LibriSpeech因来源重合风险只保留技术诊断；分类准确不能当作评分偏置或原因已经找到。

资料：[43-processing-chain-predictive-audit](../../docs/experiments/43-processing-chain-predictive-audit.md)。

## 6.2 固定视频后，评分声音音量会影响多少

我们复用报告50的九组 May 视频，只改变评分声音音量，形成训练、驱动和评分音量的27种组合。SyncNet并非严格不受评分声音音量影响，但本批平均变化较小且方向依赖视频条件。没有新增训练或视频，不能把这一批结果推广成评分器永远对音量稳健。

资料：[51-synctalk-may-eval-gain](../../docs/experiments/51-synctalk-may-eval-gain.md)。

## 6.3 先用两种其他评分方法复查中文视频

我们复用中文20个来源、20自然和60次云TTS对应的 Wav2Lip 视频，用 MTDVocaLiST 和 Synchformer 检查。MTD旧的连续分数未通过已知移音测试；Synchformer只有6个来源符合固定5秒长度要求。旧MTD分数不能作为官方判断，后续报告69、70明确修正；长度子集也不能代表全20来源。

资料：[68 — 云TTS与自然原生视频的MTDVocaLiST及Synchformer评分](../../docs/experiments/68-cloud-mtd-synchformer-native-preference.md)。

## 6.4 为什么旧MTD分数不能识别移音

我们对同20条自然视频做真实声音平移，并核对 MTDVocaLiST 官方读取和计算。发现未通过的是项目自定义的平均连续分数，官方检索方式能够恢复已知时差。修正的是测量定义，不是调整参数直到结果变好；旧结果保留，随后正式比较改用报告70的官方方法。

资料：[69 — MTDVocaLiST官方调用与时移控制审计](../../docs/experiments/69-mtd-official-parity-and-shift-control-audit.md)。

## 6.5 改用官方MTD方法重新比较

我们按 MTDVocaLiST 官方检索计算复评同80个 Wav2Lip 视频，预先固定主要比较方式。TTS相对自然平均低0.566个百分点，60配对中21对较高，与 Sync-C 的方向多有不同。它是另一种评分判断，不是人类同步真值；各来源三次TTS仍只算重复测量。

资料：[70 — MTDVocaLiST官方检索指标重评云TTS原生视频](../../docs/experiments/70-mtd-official-native-cloud-preference.md)。

## 6.6 使用原版VocaLiST复查

我们取得原版 VocaLiST 官方权重，按作者方法复评同20来源、80个 Wav2Lip 视频。主要检索正确比例中，TTS平均低0.937个百分点，18/60对较高。它与 Sync-C 存在明显分歧，但分歧本身不能裁定哪个更准确，更不能直接证明 SyncNet 偏向TTS。

资料：[71 原版VocaLiST官方方法：云TTS与自然音频驱动视频的原生偏好](../../docs/experiments/71-vocalist-official-native-cloud-preference.md)。

## 6.7 检查SparseSync能否区分原视频

我们对同中文队列中长度合格的28个视频、6个完整来源运行 SparseSync，使用官方固定5秒输入并检查移音响应。它能检测大幅已知错位，但原视频的零偏移概率大量接近上限。因此不适合据此可靠排列细小差异，也不能将接近满分解释成两组实际质量相同。

资料：[72-sparsesync-official-native-cloud-preference](../../docs/experiments/72-sparsesync-official-native-cloud-preference.md)。

## 6.8 让几种评分方法查看相同时间范围

我们复用中文20来源、80个 Wav2Lip 视频，对 SyncNet、AVS_u、CoGenAV 在共同内部时间窗上拆解评分。Sync-C 的提高涉及最佳匹配和错位背景两项；AVS_u在共同内部片段与整段的判断不同。结果提示时间支持和评分构成重要，没有确定某项解释全部提高或背景项必然占主导。

资料：[73-common-window-cloud-curve-decomposition](../../docs/experiments/73-common-window-cloud-curve-decomposition.md)。

## 6.9 AVS_u首尾高分来自静音还是上下文

我们固定同80个视频的目标内容，改变它在整段输入中的边界位置和周围上下文，再分开检查静音与闭嘴。当前AVS_u实现有明显边界和上下文敏感性，单靠静音或闭嘴不足以解释。它限定本项目的具体实现，没有定位全部内部原因，也不能推广到所有同名评分程序。

资料：[74-avs-boundary-context-vs-silence-mouth-state](../../docs/experiments/74-avs-boundary-context-vs-silence-mouth-state.md)。

## 6.10 Echo改用官方配置会怎样

我们在6个中文来源上，以 EchoMimicV3 的公开配置组合复查，涉及响度处理、随机种子和输出长度等联合变化。Sync-C 正差约1.021。它说明配置差异值得注意，但只是小样本联合配置比较，不能代替20来源结果，也不能将分数变化单独归因于某个参数。

资料：[78 — EchoMimicV3-Flash 官方配置与音量/推理参数对照](../../docs/experiments/78-echomimic-v3-official-control.md)。

## 6.11 汇总四种生成模型的九项判断

我们只复算已有中文 Wav2Lip、FlashHead、EchoMimicV3 和 TalkLikeYou 的结果，并单列 EARS 英文。新三个模型的主要12来源没有同一指标在多个模型上重复通过九项校正。旧中文 Wav2Lip 的统一校正结果缺失，写作未知而非不显著；音量和版本差异也限制直接比较。

资料：[80 — TalkLikeYou、FlashHead、Echo与Wav2Lip九评分器比较](../../docs/experiments/80-cross-tfg-nine-scorer-comparison.md)。

## 6.12 PC-Talk的Sync-C为什么低于公开结果

我们先用中文20来源、固定人物图和云Qwen声音运行 PC-Talk，比较自有评分流程与可获得的公开信息。本批 Sync-C 自然约7.266、TTS约7.756。公开分数来自不同语言、数据和生成设置，不能把差值当成复现误差；完整作者评分程序也未公开。内部音频转换可能引入延迟，但尚未做去除该步骤的单因素比较。

资料：[81 — PC-Talk 中文20来源Sync-C与官方I2V分数对照](../../docs/experiments/81-pctalk-chinese20-syncnet-official-comparison.md)。

## 6.13 固定TalkLikeYou视频检查音轨与轨迹门槛

我们对中文20来源、41个唯一 TalkLikeYou 视频固定画面，分别改变评分音轨入口和最短人脸轨迹门槛。共同轨迹上两门槛分数相同，两种音轨的TTS分差差异也未确认。中间音频包含约69毫秒处理延迟，但没有无该转换的消融，不能据此说该延迟已解释分差。HDTF暂停实验没有恢复。

资料：[82 — 中文TalkLikeYou固定视频的SyncNet门槛与音轨对照](../../docs/experiments/82-talklikeyou-syncnet-protocol-controls.md)。

## 6.14 只看声音能否预测哪一组分数更高

我们用20来源、140条自然与双规模TTS声音及已有分数，按说话人留出训练简单预测器。加入 HuBERT 信息没有改善样本外误差，预定推进要求未通过。它限制的是这个声音预测办法；原视频也随生成输入变化，因此不能把预测成功或失败直接解释成纯评分偏置。

资料：[自然音频与双规模TTS的SyncNet偏好探针 2026-10-24](../../Experiments/%E8%87%AA%E7%84%B6%E9%9F%B3%E9%A2%91%E4%B8%8E%E5%8F%8C%E8%A7%84%E6%A8%A1TTS%E7%9A%84SyncNet%E5%81%8F%E5%A5%BD%E6%8E%A2%E9%92%88%202026-10-24.md)。

## 6.15 用局部声音差异解释评分的小试

我们复用4人28媒体的局部 SyncNet 信息，训练一个压缩差异的小模型及偏好预测头。重建信息有所改善，但仅一名留出人的8对预测为5对正确，低于多数类参照6对。它验证了技术链能运行，同时出现过拟合；不是完整解释、因果验证或全200人结果，完成后停止扩大训练。

资料：[WIMHF音频解释流程的4人试训练 2026-10-25](../../Experiments/WIMHF%E9%9F%B3%E9%A2%91%E8%A7%A3%E9%87%8A%E6%B5%81%E7%A8%8B%E7%9A%844%E4%BA%BA%E8%AF%95%E8%AE%AD%E7%BB%83%202026-10-25.md)。

## 6.16 用官方完整流程重评四十人视频

我们固定同批200个视频的画面及规范声音，完整运行官方 SyncNet 人脸跟踪与评分。新增30人TTS减自然平均−0.108，全40为−0.176，区间均跨零，没有恢复该批明显优势。官方处理中存在共同音频延迟，但没有单因素证据证明延迟导致差异，更不能因此认定分数准确反映人类同步判断。

资料：[完整 Qwen TTS 的40人官方流程复评分](../../Experiments/%E5%AE%8C%E6%95%B4%20Qwen%20TTS%20%E7%9A%8440%E4%BA%BA%E5%AE%98%E6%96%B9%E6%B5%81%E7%A8%8B%E5%A4%8D%E8%AF%84%E5%88%86%202026-10-23.md)。

## 6.17 为什么旧结果强、新结果弱

我们核对历史 AISHELL、一组英文多TTS和新40人流程，检查音量、图片、评分裁剪、声音转换及时间范围。多因素确有差异，官方与自定义链会改变分差，但单凭跨批比较无法分配各因素的原因占比。后续官方完整复评分已补齐，不能继续把这份早期审计的待办当成未执行。

资料：[历史TTS优势与40人NRT结果的协议差异核查](../../Experiments/%E5%8E%86%E5%8F%B2TTS%E4%BC%98%E5%8A%BF%E4%B8%8E40%E4%BA%BANRT%E7%BB%93%E6%9E%9C%E7%9A%84%E5%8D%8F%E8%AE%AE%E5%B7%AE%E5%BC%82%E6%A0%B8%E6%9F%A5%202026-10-22.md)。

## 6.18 其他评分器也需要先检查

我们对既有中文云TTS和自然视频使用 PEAVS、SyncReward，并检查作者推理、批量处理及真实时间移动。发现部分控制响应不足，PEAVS还有批量相关问题。计算完成并不代表评分可以直接当真值；这组限制须随之后各模型的九项结果一起交接，不能只因它们与SyncNet不同就相信它们。

资料：[PEAVS与SyncReward官方推理审计及批量掩码问题 2026-10-27](../../Experiments/PEAVS%E4%B8%8ESyncReward%E5%AE%98%E6%96%B9%E6%8E%A8%E7%90%86%E5%AE%A1%E8%AE%A1%E5%8F%8A%E6%89%B9%E9%87%8F%E6%8E%A9%E7%A0%81%E9%97%AE%E9%A2%98%202026-10-27.md)。

## 6.19 只看画面能否识别发音内容

我们尝试用只看嘴部画面的语音识别办法，检查自然与TTS对应视频中的内容信息，并在 Ditto 子集中比较其变化与 Sync-C。视觉识别正确与声音配合准确是不同问题，工具本身也受画面和数据条件影响。该系列不构成人类盲评，不能把某项识别结果直接当成整体同步质量。

资料：[VSR 视觉内容辨识与 TTS 配对验证结果 2026-09-18](../../Experiments/VSR%20%E8%A7%86%E8%A7%89%E5%86%85%E5%AE%B9%E8%BE%A8%E8%AF%86%E4%B8%8E%20TTS%20%E9%85%8D%E5%AF%B9%E9%AA%8C%E8%AF%81%E7%BB%93%E6%9E%9C%202026-09-18.md)。

## 6.20 真实视频能否校准嘴部动作测量

我们用 GRID 等真实音视频，先检查嘴部开度、轨迹形状及已知时间变化，再在独立标签子集复查。部分早期几何质量检查失败，后续16条评价只支持更窄的轨迹相关指标。已有测量不能因此被升级成真人口型准确度真值，也没有完成全项目人工盲评。

资料：[GRID16eval形状相关独立协议结果审计 2026-09-27](../../Experiments/GRID16eval%E5%BD%A2%E7%8A%B6%E7%9B%B8%E5%85%B3%E7%8B%AC%E7%AB%8B%E5%8D%8F%E8%AE%AE%E7%BB%93%E6%9E%9C%E5%AE%A1%E8%AE%A1%202026-09-27.md)。

## 本模块的补充实验、方案与修正记录

另外28条来源记录列在[本模块的逐条补充来源](#supplement)，包含子实验、方案、任务、复核与汇总。它们全部保留链接和状态说明，条目数不是独立实验数。

<a id="supplement"></a>

## 逐条补充来源

下列28条是尚未在主体单独展开的来源记录，包括具体子实验、方案、任务、复核和汇总。同一实验可能有多条记录，不能把条目数当实验数。这里翻译原状态字段以帮助查阅；它不是重新验收，更不能用旧任务状态覆盖主体已核对的最新状态。文中提到某模型或数据，不自动表示实际使用。归类仅用于导航，真正关系以原文为准。


- [LRS3 ASR 错误与局部同步低分相关性实验](../../docs/experiments/LRS3%20ASR%20%E9%94%99%E8%AF%AF%E4%B8%8E%E5%B1%80%E9%83%A8%E5%90%8C%E6%AD%A5%E4%BD%8E%E5%88%86%E7%9B%B8%E5%85%B3%E6%80%A7%E5%AE%9E%E9%AA%8C.md)。原记录登记该项工作完成；完成不表示假设获得支持。
- [scorer-negative-sampling-and-score-mechanisms-20261029](../../docs/experiments/scorer-negative-sampling-and-score-mechanisms-20261029.md)。原记录登记该项工作完成；完成不表示假设获得支持。
- [Ditto-50 VSR 与 Sync-C 增益关联验证结果 2026-09-18](../../Experiments/Ditto-50%20VSR%20%E4%B8%8E%20Sync-C%20%E5%A2%9E%E7%9B%8A%E5%85%B3%E8%81%94%E9%AA%8C%E8%AF%81%E7%BB%93%E6%9E%9C%202026-09-18.md)。原记录登记该项工作完成；完成不表示假设获得支持。
- [FlashHead Pro 中文20人可听音量九评分器配对实验](../../Experiments/FlashHead%20Pro%20%E4%B8%AD%E6%96%8720%E4%BA%BA%E5%8F%AF%E5%90%AC%E9%9F%B3%E9%87%8F%E4%B9%9D%E8%AF%84%E5%88%86%E5%99%A8%E9%85%8D%E5%AF%B9%E5%AE%9E%E9%AA%8C.md)。原记录登记该项工作完成；完成不表示假设获得支持。
- [GRID geometry8cal独立协议实现审计 2026-09-27](../../Experiments/GRID%20geometry8cal%E7%8B%AC%E7%AB%8B%E5%8D%8F%E8%AE%AE%E5%AE%9E%E7%8E%B0%E5%AE%A1%E8%AE%A1%202026-09-27.md)。原记录登记该项工作完成；完成不表示假设获得支持。
- [PEAVS与SyncReward原生自然TTS评分分歧及同步控制失败 2026-10-27](../../Experiments/PEAVS%E4%B8%8ESyncReward%E5%8E%9F%E7%94%9F%E8%87%AA%E7%84%B6TTS%E8%AF%84%E5%88%86%E5%88%86%E6%AD%A7%E5%8F%8A%E5%90%8C%E6%AD%A5%E6%8E%A7%E5%88%B6%E5%A4%B1%E8%B4%A5%202026-10-27.md)。计算有记录，但必要的评分检查未通过。
- [TTS 原生增益音轨审计与时间表交叉实验 2026-09-26](../../Experiments/TTS%20%E5%8E%9F%E7%94%9F%E5%A2%9E%E7%9B%8A%E9%9F%B3%E8%BD%A8%E5%AE%A1%E8%AE%A1%E4%B8%8E%E6%97%B6%E9%97%B4%E8%A1%A8%E4%BA%A4%E5%8F%89%E5%AE%9E%E9%AA%8C%202026-09-26.md)。原记录登记该项工作完成；完成不表示假设获得支持。
- [WIMHF式局部视听差异与SyncNet偏好解释方案 2026-10-24](../../Experiments/WIMHF%E5%BC%8F%E5%B1%80%E9%83%A8%E8%A7%86%E5%90%AC%E5%B7%AE%E5%BC%82%E4%B8%8ESyncNet%E5%81%8F%E5%A5%BD%E8%A7%A3%E9%87%8A%E6%96%B9%E6%A1%88%202026-10-24.md)。属于方案或候选阶段，不能当作已完成实验。
- [Wav2Lip historical shift rescore 2026-09-08](../../Experiments/Wav2Lip%20historical%20shift%20rescore%202026-09-08.md)。原记录登记该项工作完成；完成不表示假设获得支持。
- [固定视频 SyncNet 监督 Gate B 结果](../../Experiments/syncnet-supervision/%E5%9B%BA%E5%AE%9A%E8%A7%86%E9%A2%91%20SyncNet%20%E7%9B%91%E7%9D%A3%20Gate%20B%20%E7%BB%93%E6%9E%9C.md)。当前具体做法停止推进，范围以原记录说明为准。
- [Wav2Lip proxy-to-official parity gate](../../Experiments/wav2lip/Wav2Lip%20proxy-to-official%20parity%20gate.md)。原记录登记该项工作完成；完成不表示假设获得支持。
- [云端Qwen Audio3.1Plus的20来源原生四指标分歧 2026-10-27](../../Experiments/%E4%BA%91%E7%AB%AFQwen%20Audio3.1Plus%E7%9A%8420%E6%9D%A5%E6%BA%90%E5%8E%9F%E7%94%9F%E5%9B%9B%E6%8C%87%E6%A0%87%E5%88%86%E6%AD%A7%202026-10-27.md)。计算有记录，但必要的评分检查未通过。
- [受控评分器训练与同步专家交叉干预实验路线 2026-10-29](../../Experiments/%E5%8F%97%E6%8E%A7%E8%AF%84%E5%88%86%E5%99%A8%E8%AE%AD%E7%BB%83%E4%B8%8E%E5%90%8C%E6%AD%A5%E4%B8%93%E5%AE%B6%E4%BA%A4%E5%8F%89%E5%B9%B2%E9%A2%84%E5%AE%9E%E9%AA%8C%E8%B7%AF%E7%BA%BF%202026-10-29.md)。原状态字段不能单独确定实验是否完成，请按原文及后续修正判断。
- [完整 Qwen TTS 补组的 N/R/T 评分验证](../../Experiments/%E5%AE%8C%E6%95%B4%20Qwen%20TTS%20%E8%A1%A5%E7%BB%84%E7%9A%84%20NRT%20%E8%AF%84%E5%88%86%E9%AA%8C%E8%AF%81%202026-10-22.md)。原记录登记该项工作完成；完成不表示假设获得支持。
- [无人工条件下的独立纯视觉内容与时间校准审计 2026-09-26](../../Experiments/%E6%97%A0%E4%BA%BA%E5%B7%A5%E6%9D%A1%E4%BB%B6%E4%B8%8B%E7%9A%84%E7%8B%AC%E7%AB%8B%E7%BA%AF%E8%A7%86%E8%A7%89%E5%86%85%E5%AE%B9%E4%B8%8E%E6%97%B6%E9%97%B4%E6%A0%A1%E5%87%86%E5%AE%A1%E8%AE%A1%202026-09-26.md)。原记录登记该项工作完成；完成不表示假设获得支持。
- [独立40人同长交叉与评分边界探索（2026-09-29）](../../Experiments/%E7%8B%AC%E7%AB%8B40%E4%BA%BA%E5%90%8C%E9%95%BF%E4%BA%A4%E5%8F%89%E4%B8%8E%E8%AF%84%E5%88%86%E8%BE%B9%E7%95%8C%E6%8E%A2%E7%B4%A2%202026-09-29.md)。原记录登记该项工作完成；完成不表示假设获得支持。
- [Ditto-50 VSR 与 Sync-C 增益关联验证 Spec](../Ditto-50%20VSR%20%E4%B8%8E%20Sync-C%20%E5%A2%9E%E7%9B%8A%E5%85%B3%E8%81%94%E9%AA%8C%E8%AF%81%20Spec.md)。实现有记录，是否已有合格实验结果须看正文。
- [Ralph Step 2 - Phone Target Geometry Screening](../Ralph%20Step%202%20-%20Phone%20Target%20Geometry%20Screening.md)。原状态字段不能单独确定实验是否完成，请按原文及后续修正判断。
- [VSR 视觉内容辨识与 TTS 配对验证实施部署 Spec](../VSR%20%E8%A7%86%E8%A7%89%E5%86%85%E5%AE%B9%E8%BE%A8%E8%AF%86%E4%B8%8E%20TTS%20%E9%85%8D%E5%AF%B9%E9%AA%8C%E8%AF%81%E5%AE%9E%E6%96%BD%E9%83%A8%E7%BD%B2%20Spec.md)。实现有记录，是否已有合格实验结果须看正文。
- [A1 Wav2Lip–SyncNet proxy parity 与 TTS 条件归因](../../Tasks/A1%20Wav2Lip%E2%80%93SyncNet%20proxy%20parity%20%E4%B8%8E%20TTS%20%E6%9D%A1%E4%BB%B6%E5%BD%92%E5%9B%A0.md)。原记录登记该项工作完成；完成不表示假设获得支持。
- [A1 hard-coordinate endpoint objective v3](../../Tasks/A1%20hard-coordinate%20endpoint%20objective%20v3.md)。原状态字段不能单独确定实验是否完成，请按原文及后续修正判断。
- [H-only historical endpoint sensitivity assay](../../Tasks/H-only%20historical%20endpoint%20sensitivity%20assay.md)。原记录登记该项工作完成；完成不表示假设获得支持。
- [Ubuntu同步评分器部署与EARS40云TTS 2026-10-28](../../Tasks/Ubuntu%E5%90%8C%E6%AD%A5%E8%AF%84%E5%88%86%E5%99%A8%E9%83%A8%E7%BD%B2%E4%B8%8EEARS40%E4%BA%91TTS%202026-10-28.md)。原记录登记该项工作完成；完成不表示假设获得支持。
- [云TTS与自然原生视频的MTDVocaLiST和Synchformer评分 2026-10-28](../../Tasks/%E4%BA%91TTS%E4%B8%8E%E8%87%AA%E7%84%B6%E5%8E%9F%E7%94%9F%E8%A7%86%E9%A2%91%E7%9A%84MTDVocaLiST%E5%92%8CSynchformer%E8%AF%84%E5%88%86%202026-10-28.md)。原记录登记该项工作完成；完成不表示假设获得支持。
- [对齐 SyncNet loss 与官方评分](../../Tasks/%E5%AF%B9%E9%BD%90%20SyncNet%20loss%20%E4%B8%8E%E5%AE%98%E6%96%B9%E8%AF%84%E5%88%86.md)。原状态字段不能单独确定实验是否完成，请按原文及后续修正判断。
- [PC-Talk Chinese-LiPS21配对九评分器2026-10-31](../../codex/tts-exp/PC-Talk%20Chinese-LiPS21%E9%85%8D%E5%AF%B9%E4%B9%9D%E8%AF%84%E5%88%86%E5%99%A82026-10-31.md)。原记录登记该项工作完成；完成不表示假设获得支持。
- [PC-Talk中文20来源其余八评分器2026-10-31](../../codex/tts-exp/PC-Talk%E4%B8%AD%E6%96%8720%E6%9D%A5%E6%BA%90%E5%85%B6%E4%BD%99%E5%85%AB%E8%AF%84%E5%88%86%E5%99%A82026-10-31.md)。原记录登记该项工作完成；完成不表示假设获得支持。
- [TalkLikeYou中文20来源九评分器实验2026-10-30](../../codex/tts-exp/TalkLikeYou%E4%B8%AD%E6%96%8720%E6%9D%A5%E6%BA%90%E4%B9%9D%E8%AF%84%E5%88%86%E5%99%A8%E5%AE%9E%E9%AA%8C2026-10-30.md)。原记录登记该项工作完成；完成不表示假设获得支持。

返回[研究交接总览](%E7%A0%94%E7%A9%B6%E4%BA%A4%E6%8E%A5%E6%80%BB%E8%A7%88.md)。

## Relations

- part_of [[研究交接总览]]
