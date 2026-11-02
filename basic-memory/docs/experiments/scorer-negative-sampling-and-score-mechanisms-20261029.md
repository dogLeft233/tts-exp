---
title: scorer-negative-sampling-and-score-mechanisms-20261029
type: research
permalink: tts-exp/docs/experiments/scorer-negative-sampling-and-score-mechanisms-20261029
status: concluded
date: '2026-10-29'
scientific_status: METHOD_AUDIT_AND_EXPLORATORY_HYPOTHESES
training: false
new_network_forwards: 0
tags:
- syncnet
- synchformer
- negative-sampling
- score-definition
- cloud-tts
---

# scorer-negative-sampling-and-score-mechanisms-20261029

✅ 一手论文、官方训练源码、当前部署配置与既有评分产物核对完成。最具体的共同线索是“正确时间位置的可区分性/置信度”，但没有发现足以单独解释SyncNet与Synchformer偏TTS的统一负采样机制。训练范围、推理候选范围与最终标量不能混用。

| 修改日期 | 修改 | 授权 |
|---|---|---|
| 2026-10-29 | 首次训练负例/时移/取分对照；已存logits的CPU置信度诊断 | 用户要求仔细比较并寻找TTS偏好机制线索 |

## 范围与方法

本次是有明确范围的方法核查和探索性CPU重算，不是系统综述、训练复现或新的感知实验。检查SyncNet、Synchformer、SparseSync、VocaLiST、MTDVocaLiST、CoGenAV、AVS_u底层AV-HuBERT和SyncReward；PEAVS按此前暂停决定不作为主要机制比较对象。

academic-research-suite quick工作流与research后台研究：两个子代理分别核查SyncNet/Voca系列、CoGenAV/AVS_u/SyncReward；ROOT独立读Synchformer/SparseSync论文和官方源码、SSH只读当前checkpoint配置，并复算已有28媒体logits。没有新模型前向、GPU作业、TTS/TFG生成、训练、媒体移位或正式评分覆写。后台核查稿：
- .codex/scratch/sync_negative_training_audit_20261029.md
- .codex/scratch/representation_negative_training_audit_20261029.md

仓库E:/Documents/tts-exp；分支codex/cross-tfg-audio-head-20261003，当前本地SHA `2b71639c3b555293375d248350999377168d515d`，未构造未验证的远程commit链接。新研究材料未提交。

## 训练负样本、时移与当前取分

| 模型/版本 | 训练负例及损失 | 训练时间范围/粒度 | 本项目实际端点 |
|---|---|---|---|
| SyncNet原方法 / 当前V2 | 原论文同clip错位音频；margin欧式对比损失 | 原论文最多2秒，最小排除区未披露；不能确定V2完整训练配方 | V2未输出L2归一化；Sync-C=median_k d(k)−min_k d(k)。本项目常用推理±15帧=±600ms/40ms，非训练范围 |
| Synchformer LRS3 23-12-23T18-33-57 | S1同时间A/V为正；同视频其他段及batch其他视频段为负，InfoNCE。S2冻结骨干，时移交叉熵分类 | S1段长0.64秒、14段无重叠起点；S2类中心−2到+2秒/0.2秒，训练音频起点另有最多±0.05秒jitter，标签不随jitter改变 | 原生中心5秒输入、14个半重叠段，21类softmax中P0；不是max概率，也不是官方数据集Acc@1 |
| SparseSync LRS3 NoFaceCrop 22-07-13T22-25-49 | 同视频随机错位，端到端21类时移分类，骨干有各自预训练 | 与Synchformer相同±2秒/0.2秒类中心；保存配置也有±0.05秒训练jitter | 与Synchformer相同中心5秒及P0标量，次端点负期望绝对类别时移 |
| 原版VocaLiST | 同视频另选音频起点，正负等量，BCE | 源码从全有效起点另抽，仅排除同起点；最小1帧40ms，最大随clip长度而变。±15是测试候选范围 | 官方K5：逐5帧窗检索31候选，argmax落在零±1帧的窗口比例；K7–15先时间平均候选分数 |
| MTDVocaLiST | 同上同步BCE，加教师注意力/value关系蒸馏 | 论文声称±15帧；公开及部署训练代码实际全clip另抽，存在未闭合差异 | 官方K5与教师同类，不能再引用旧自定义mean-logit结果替代 |
| CoGenAV Base | 跨说话人+同说话人错位负对；逐帧ReLU cosine的时间均值做BCE，加ASR文本生成目标 | 原文§2.2/§3及公开loss helper未给负时移幅度、最小间隔或混合比例；训练sampler未公开 | 主分为对应时间A/V逐帧ReLU cosine均值；搜索/contrast为项目另定义派生端点 |
| AVS_u复现 | 底层AV-HuBERT masked cluster prediction，再唇读微调；不是另训时移分类器 | 不适用“负时移范围”；mask替换不是同步负标签 | 单模态分别整段编码，逐帧cosine均值；本项目checkpoint/归约为明确复现选择 |
| SyncReward Stage2 | S1同clip其他时间段+跨clip语义负例InfoNCE；S2生成视频人评分数回归+排序 | 论文名义S1为18×0.5秒/stride0.25；发布S1实际全clip均匀铺18段，间隔片长依赖，不能确定checkpoint训练时使用哪版 | 冻结编码器+Transformer/head输出2 sigmoid，预测0–2人评reward；当前推理中心4.75秒，不是P0 |

### 一手定位与复现边界

SyncNet：原论文§2.3、§3.1 p7、§4.1及§4.2；训练说明取自[作者上传的全文](https://www.researchgate.net/profile/Joon-Son-Chung/publication/315311266_Out_of_Time_Automated_Lip_Sync_in_the_Wild/links/5aeee477aca2727bc005e50d/Out-of-Time-Automated-Lip-Sync-in-the-Wild.pdf)，[Oxford入口](https://www.robots.ox.ac.uk/~vgg/publications/2016/Chung16a/chung16a.pdf)本次访问超时。原论文256维，当前[V2代码默认1024维且输出无L2归一化](https://github.com/joonson/syncnet_python/blob/master/SyncNetModel.py)；公开demo没有V2完整训练manifest/采样器，不能将原方法的训练范围冒称当前权重的已核验履历。原论文推理±1秒与本项目±600ms另列。原论文以正配对距离阈值过滤训练数据后再训，但本次没有证据证明该做法造成TTS偏好。

Synchformer：[原文§3.2及Appendix A](https://arxiv.org/html/2401.16423v1#S3.SS2)、[发布训练配置](https://github.com/v-iashin/Synchformer/blob/b66668a1521d7567cc760e5544b2b5b53179b687/configs/sync.yaml)、[采样器及apply_a_jitter](https://github.com/v-iashin/Synchformer/blob/b66668a1521d7567cc760e5544b2b5b53179b687/dataset/transforms.py#L241-L388)、[S1 loss](https://github.com/v-iashin/Synchformer/blob/b66668a1521d7567cc760e5544b2b5b53179b687/model/modules/feat_extractors/train_clip_src/open_clip/loss.py)。
S1将B×S展平，除了同clip其他时间外也含跨clip负例；不能简化为“只有时间负例”。骨干先从AST AudioSet和Motionformer SSv2初始化，再做LRS3同步预训练；两个同步训练阶段均LRS3不等于全部初始化从未见过其他数据。S2权重冻结特征提取器。0.64秒是原始分段长度，不代表最后分类器内部只保留0.64秒时间分辨率；它保留段内时间特征。
ROOT读取Ubuntu实际S2配置并另存本地；官方S1配置通过README实际链接新下载7123 bytes，未下载模型：
- .codex/scratch/scorer_training_audit_20261029/cfg-23-12-22T16-04-18.yaml，SHA256 `3d284e8c769fb5332882cac734bfa3699f9c73fd363bd046d92781471a19157e`
- cfg-23-12-23T18-33-57.yaml，SHA256 `db5e6ce4fe6383be65e9e02eb7d3e821184843e52b109bd7c40a9a6fa8bd1241`
保存配置S1/S2的audio_jitter_sec均0.05，S1 p_audio_aug=0.0、14段/step1；S2 step0.5、21类/±2秒/5秒。jitter在可行边界会截短，因此是“最多±50ms”。类中心200ms间隔与训练jitter都不等于经过校准的人类感知阈值。

SparseSync：[原文§3](https://arxiv.org/pdf/2210.07055)、[当前部署版本采样器](https://github.com/v-iashin/SparseSync/blob/a5bee8a047c0ebeec66b18d4ddf4c5f0ef098a4f/dataset/transforms.py#L261-L366)。ROOT读取实际checkpoint配置，确认LRS3、5秒、21类、max_off_sec2、audio_jitter_sec0.05、p_audio_aug0、非冻结音视频骨干。该配置本地早已保存于run72/source_snapshot。两模型的负时移网格相同但结果不同，是反对“范围决定一切”的直接约束。

VocaLiST：[论文§3.2/§4.1](https://arxiv.org/html/2204.02090v1)、[部署版本训练采样](https://github.com/vskadandale/vocalist/blob/c265b59bcf7cd8559a265aaf454a9333ec24cb0f/train_vocalist_lrs2.py#L97-L148)、[官方测试K5](https://github.com/vskadandale/vocalist/blob/c265b59bcf7cd8559a265aaf454a9333ec24cb0f/test_lrs2.py#L123-L174)。
低能量正位置RMS≤0.01时，源码倾向抽RMS>0.01负音频并有限重试；精确全零的rms函数又返回1，不能描述成完善静音过滤。它能鼓励“发声/不发声状态不匹配”线索，但本次没有因果证据说明它导致自然偏好。训练BGR/255与MTD RGB0–255保持各自官方路径。

MTD：[论文§4.1](https://arxiv.org/html/2210.15563v3#S4.SS1)写±15 visual frames，括号1.2s是总宽度；25fps下单侧为0.6秒。相比之下[公开及部署train_MTD.py](https://github.com/xjchenGit/MTDVocaLiST/blob/5795416f9277094d3ae6215596a436a0a01f79ad/train_MTD.py#L242-L291)没有±15限制。两份事实均保留，不选择更符合本项目结果的一侧代替权重训练履历。

CoGenAV：[论文§2.2.1及Eq1–2](https://arxiv.org/html/2505.03186v2#S2.SS2.SSS1)、[作者loss helper](https://github.com/HumanMLLM/CoGenAV/blob/dbd3f835a2cc859766c421aaf3918cda2246008d/models/cogenav.py#L178-L191)。不能称它只看语义；同说话人时间负例明确存在。ASR生成约束是有依据的任务差异，但不能据此断言它更公平或天然抑制TTS优势。

AVS_u：[原论文§3.4–3.5/§4.1](https://arxiv.org/html/2405.04327v1)。原文采用lip-reading finetuned AV-HuBERT，整段上下文；本项目VSR Base LRS3 433h不是作者确切checkpoint已闭合的官方完整评分软件。

SyncReward：[官方说明](https://github.com/WeChatCV/SyncReward)、[Stage1实际调用spread_segments](https://github.com/WeChatCV/SyncReward/blob/f29fbdc2bb2dd3d59644ffd3c60d207dd3e62615/train_stage1.py#L159-L167)、[实际分段定义](https://github.com/WeChatCV/SyncReward/blob/f29fbdc2bb2dd3d59644ffd3c60d207dd3e62615/syncreward/data.py#L69-L116)、[loss定义](https://github.com/WeChatCV/SyncReward/blob/f29fbdc2bb2dd3d59644ffd3c60d207dd3e62615/syncreward/losses.py)。
S1铺段代码间隔约(片长−0.5)/17且有帧取整；0.25–4.25秒只适用于论文名义18段网格，不能宣称当前发布训练代码必然如此。S2 SmoothL1 beta0.1+0.1排序，后者只用人评分差≥0.75配对。已有中文BF16±400ms控制14/40下降，仍不能把其偏好当真值。

## 更具体的数学线索：相对可区分性不是原始匹配

令d(k)是SyncNet跨窗口均距，z_k是Synchformer对偏移类别k的logit。

Sync-C = median_k d(k) − min_k d(k)

P0 = exp(z0)/sum_k exp(zk) = 1/[1+sum_(k≠0) exp(zk−z0)]

二者都含相对比较：Sync-C受错位背景距离影响；P0受其他类别相对零类的竞争影响。P0没有把原输入实际平移后取max，也不直接计算SyncNet式背景中位数；不能把分类logit当成对实际PCM平移后的物理匹配曲线，更不能将两个公式宣布为同一机制。

若TTS在模型表征里有更清晰、时间上更可辨认的变化，即使所有片段已经被判作同步，也可能因竞争候选更易排除而提分。这是可检验推断，可能包含真实动作改善，也可能包含声学/表征偏好，当前不能拆成因果百分比。对logits整体加常数不改变P0，因此不从原始z0的绝对变化单独推断“正匹配改善”或“背景下降”。

## 本次新增CPU诊断：相同28媒体 / 18对 / 6来源

使用两模型已有21类logits，输入ID集合严格相同。Synchformer源run68、SparseSync源run72；自然10、TTS18，形成18对/6来源。保持历史FP16/FP32前向产物；只以FP64重算概率与自定义敏感性，不假称重新全精度网络前向。

| 端点 | Synchformer | SparseSync |
|---|---|---|
| 28条原生最佳类别 | 28/28为零偏移 | 28/28为零偏移 |
| 若只比较最佳类别，18对胜负 | 全18对平局 | 全18对平局 |
| 正式P0 TTS胜 | 15/18，83.3% | 6/18，33.3% |
| 正式P0均差，百分点 | +1.293670 | −0.009311 |
| 最强非零竞争类别 | 全28条为±200ms | 16/28为±200ms，其余分布在更远类 |
| FP64零类减最强非零类margin | T−N +1.553928；18/18正 | −0.461235；6/18正 |
| 去均值后除本条21 logits的RMS，再算margin（自定义） | T−N +0.243177；15/18正 | −0.062068；12/18正，均差仍负且区间跨0 |

正式P0均差取冻结旧FP32汇总；FP64复算Synchformer +1.293662、SparseSync −0.009310个百分点，差异是标量浮点归约，18/18配对方向分别与旧结果一致。Synchformer“最佳类别全零”不代表每一局部嘴形无误，只说明该分类粒度上没有发现总体偏移。

源等权bootstrap10000、seed20261029，均为探索性描述：
- Synchformer raw margin均差CI[+0.926215,+2.120985]；RMS标准化margin[+0.100423,+0.388213]。
- 自定义RMS标准化P0仍15/18偏T，但均差CI[−0.007646,+0.042518]跨0。不能只挑这个跨0结果宣布“尺度解释完了”；标准化margin仍正。
- 因此纯粹“每条logit统一放大”不足以解释全部Synchformer现象；类间相对形状/零类突出程度仍有差别。归一化改变评分定义，没有独立温度校准，不替代正式分数。
- 最近±200ms两类占非零概率质量的均值，N0.916960/T0.944842；这是先每clip取占比、再按配对均值。并非物理时间误差分布的已校准估计。

代码 .codex/scratch/scorer_training_audit_20261029/confidence_diagnostic.py，执行SHA256 `bd4229ae61bbedbd339f93d15af9f9f20df07224c3d7461c297a8f5e0f4cf708`。confidence_diagnostic.json保存输入SHA、全部端点/全部媒体、配对及统计；independent_check.json用独立stdlib math/statistics而非导入producer，复算56组softmax/RMS/margin，最大差8.88e−16。另有平移logits常数、正比例缩放后的自定义标准化端点不变性检查。独立检查未另做所有bootstrap重采样，不能冒称整套统计经第二实现重算。本轮未运行pytest，未修改生产代码。

## 与既有实验合并后，可支持的线索排序

1. **SyncNet距离尺度和错位背景：有本队列直接证据。** 报告73共同窗口Δcontrast+0.551，分解最佳项+0.247、背景贡献+0.304；两项直接差的区间跨0，不能声称背景主导。特征单位化后背景贡献不稳定，提示几何尺度；但最佳项仍正。原始零时移−d0也有T−N+0.184、42/60 T胜，说明最佳搜索与背景不是全部原因。单位化不是音量归一化，不能把这些结果自动归因为物理响度。

2. **Synchformer的TTS差别主要表现为分类margin/置信度：已由保存输出确认。** 全部原生类别都为0，P0把这些均判同步的片段细排。S1要求同clip不同0.64秒段可区分，是与SparseSync端到端训练不同的具体表征压力；S2又冻结该骨干。候选机制是TTS的时间证据更易辨认，而非更多视频落入正确类别。但还没有把这项差异因果定位到S1、音频或视频分支。

3. **Voca系列的反向并非同一量尺的干净反证。** 官方K5是±40ms局部检索正确率；本队列原版与MTD存在约−3帧内部峰位和低分/边缘依赖。原版K5 N4.001%/T3.063%，MTD N3.771%/T3.206%，不能把这几个百分点的反向直接当自然更精确。当前没有移动媒体或按样本校准来改善分数。

4. **AVS_u的反向也并不稳定。** 报告73全片均值T−N−0.007529，共同内部窗零移+0.009466；报告74同内容改变编码边界/上下文会大幅改分。因而“只有SyncNet和Synchformer偏TTS”已经不是跨协议事实。

5. **相同英语真人域、同视频时间负例、粗时移分类都不是充分解释。** Voca系列/CoGenAV/SyncReward也有时间负例；SparseSync与Synchformer同步训练数据域、5秒支持、21类网格和jitter相同，P0偏好却不同。CoGenAV LRS2、AVS_u LRS3也不是中文训练，无法靠语言域标签将两派分开。

历史2026-09-26共同音素负池实验在另一队列上未稳健支持“纯负池phone组成”或“纯范数”解释，并有旧动态校准外推至静态eval不匹配限制；不能将本次候选机制包装成此前从未研究过、或已排除所有竞争解释。该历史结果不直接外推本次6/20来源。

## 最有区分力的后续验证（仅建议，未运行）

- 在预先固定同一真实窗口和原时钟下，对N/T分别施加±40/80/120/160/200/400ms物理时移，报告原始零移匹配、相对下降和峰宽。只看±400ms通过不足以验证40–120ms细微误差；不将逐条峰值用于修正主端点。
- 加入短时局部错位/嘴部冻结且其余片段不动，与同长度无操作处理对照；避免拼接/编码伪影成为唯一线索。若整体P0仍高但局部错误漏检，支持“整体定位置信度不能直接当局部口型质量”的解释。
- Synchformer可用冻结S1段相似度矩阵，比较正确段与同片其他段的margin；同时报告标准化/原始特征，检验S1时间辨识压力是否与P0优势相关。关联不等于因果，不能仅凭相关宣布机制完成。
- 保留明确可用的评分器资格与全部端点，不挑自然或TTS胜率更符合预期的定义作为真值；不以未经验证的统一40ms补偿掩盖生成输出真实延迟。

## Observations

- [finding] 训练负例范围与推理搜索范围必须分开；Voca原版源码训练全clip另抽，MTD论文与公开实现存在未闭合差异。
- [finding] Synchformer和SparseSync保存配置同为±2秒/200ms类中心及最多±50ms训练jitter；范围本身不足以解释两者偏好差异。
- [finding] 两模型原生28/28 argmax均零，连续P0偏好是置信度排序；Synchformer归一化margin仍偏T，纯整体logit尺度解释不足。
- [limitation] CoGenAV负时移幅度/比例、SyncNet V2完整训练配方、MTD与SyncReward发布权重的准确采样履历仍有明确证据缺口。
- [limitation] 本次机制综合未建立感知口型优劣，也未证明某种负采样因果导致TTS偏好。
- [artifact] .codex/scratch/scorer_training_audit_20261029/保存两阶段官方配置、CPU代码、全量诊断及独立算术核对。

## Relations

- relates_to [[73-common-window-cloud-curve-decomposition]]
- relates_to [[74-avs-boundary-context-vs-silence-mouth-state]]
- relates_to [[72-sparsesync-official-native-cloud-preference]]
- relates_to [[71-vocalist-official-native-cloud-preference]]
- relates_to [[70-mtd-official-native-cloud-preference]]
- relates_to [[PEAVS与SyncReward官方推理审计及批量掩码问题 2026-10-27]]
- relates_to [[原生 TTS 优势的共同音素负样本池干预 2026-09-26]]
