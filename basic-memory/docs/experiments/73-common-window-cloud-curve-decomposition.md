---
title: 73-common-window-cloud-curve-decomposition
type: experiment
permalink: tts-exp/docs/experiments/73-common-window-cloud-curve-decomposition
status: concluded
scientific_status: SCORER_DECOMPOSITION_COMMON_SUPPORT
date: '2026-10-28'
run_id: common_window_curves_cloud_20261028_attempt01
training: false
tags:
- experiment
- syncnet
- avs-u
- cogenav
- cloud-tts
- common-window
---

# 73-common-window-cloud-curve-decomposition

✅ 既有云端TTS与自然音频驱动视频的共同窗口曲线、物理时移控制、独立重算及本地结果回传全部完成。SyncNet增益含最佳匹配与背景两项，背景项较稳定但不能证明占主导；AVS_u在共同内部窗口下也支持TTS，主要来自最佳匹配；CoGenAV尚不确定。

| 修改日期 | 修改 | 授权 |
|---|---|---|
| 2026-10-28 | 冻结共同支持设计，80媒体计算、独立审计、支持敏感性及最终报告 | 用户明确要求三模型共同窗口曲线分解 |

## Material Passport
- ID：73 / common_window_curves_cloud_20261028_attempt01
- 类型：code experiment，探索性评分器机制分解；非人评、非感知真值验证
- 状态：CONCLUDED；科学状态 SCORER_DECOMPOSITION_COMMON_SUPPORT
- 产物：runs/common_window_curves_cloud_20261028_attempt01（Windows/Ubuntu同相对路径）
- 执行：academic-research-suite experiment-agent code runner 子代理实现运行；ROOT不导入执行器，独立从原始特征重建全部矩阵、端点、配对及bootstrap。

## 问题与冻结设计
使用之前中文云Qwen qwen-audio-3.1-tts-plus队列：20自然视频及60云TTS驱动视频，20独立来源、60配对。生成模型是Wav2Lip GAN、固定图3，保留既有RMS归一化（0.0376838172675746）和原生各自音轨。没有重新TTS/生成视频、交叉音轨、DTW、时间拉伸或训练。严格同句克隆参考12来源/36配对、扩展参考8来源/24配对分开保留。三次draw共享来源，不能当60独立样本。

protocol.json与source_frozen.py在任何本轮曲线前冻结；历史评分已知，因此是探索性实验，不能冒称全新盲法确认或事前注册。

时移k为−15至+15帧、40ms步长，即±600ms。正k表示选更晚音频特征。所有lag使用相同视频query，避免零填充或合成尾帧。每clip取真实视频、音频和有效特征长度共同支持，排除两端15个query；每来源N+3T取最少可用Q数，各clip中央截相同Q。80条全部可测，Q为24–148；最短五帧窗口集合总支持为28帧=1.12秒，不能说每条都评5秒。跨自然/TTS只匹配窗口数量与中央选段规则，没有匹配相同音素/词语事件。

SyncNet使用5帧视频窗口；AVS_u/CoGenAV先计算每帧对应相似度，再平均同q..q+4五帧。这里共同的是名义query时间支持，实际感受野并不相同：SyncNet音频20个MFCC约215ms，AVS_u/CoGenAV含全片上下文及不同口部ROI。

## 实现与时间基准
SyncNet复用官方SyncNet V2网络及权重、MFCC默认int16尺度、batch20。检查发现旧自然AVI评分音轨相对canonical PCM有+1105样本（约69.06ms）延迟，不能与另外两种源PCM特征直接拼接。因此全部80音频从canonical PCM样本0重算MFCC和音频特征；20N视频特征复用此前已逐官方重放验证的video1024，60C视频从已有官方face-crop AVI通过ffmpeg JPEG解码后新forward。核验track从源帧0连续，排除AVI额外尾帧。新的共同窗分数与旧fullclip Sync-C不是完全相同输入及支持，不能将两次差异全归于窗口或全归于69ms。

AVS_u使用此前论文定义复现版AV-HuBERT VSR编码缓存，CoGenAV使用官方Base编码缓存；80/80上游audio/video path及SHA绑定验证PASS，AVS缓存自身也有input SHA。CoGenAV保留论文ReLU cosine，raw cosine另作敏感性；本批二者相同。AVS_u不是作者发布的评分软件，checkpoint/归约仍是本项目明确复现选择。

主实验是在full-context编码之后平移特征索引计算31个匹配，不是31次移PCM后重新编码。保存了实际PCM时移后重新提取的既有AVS/Co控制表示，并对3N补算SyncNet真实PCM时移控制。不能把feature-lag曲线描述成完整波形干预的响应曲线。模型仍为英语数据训练/中文应用。

## 分解定义
统一为越大越匹配：SyncNet S=-Euclidean（官方pairwise_distance eps=1e−6），AVS_u为cosine，CoGenAV为ReLU cosine。

先对共同query求均值得到S(k)，再取：
- best=max S(k)，background=median S(k)，zero=S(0)；
- contrast=best−background，zero_contrast=zero−background，search_uplift=best−zero。
- 每对Δcontrast = Δbest + (−Δbackground)。最佳匹配贡献为Δbest；背景贡献为−Δbackground。

这三个贡献以各模型自身单位报告，不横比不同模型的绝对大小。SyncNet是官方公式形式的共同窗重算，AVS/Co的contrast为自定义派生端点，不替代作者benchmark。报告没有使用不稳定的百分比中介贡献。

## 主结果：20来源 / 60配对
先每来源平均3draw差，再等权来源。区间为10,000次来源bootstrap、seed20261028的边际95%描述区间，所有组件使用相同来源重采样；非多重终点确认检验。

| 模型 | 最佳匹配贡献Δbest [95%CI] | 背景贡献−Δbackground [95%CI] | 总Δcontrast [95%CI] | contrast TTS胜 |
|---|---|---|---|---|
| SyncNet | +0.247 [−0.008,+0.499] | +0.304 [+0.059,+0.506] | +0.551 [+0.124,+0.966] | 44/60=73.3% |
| AVS_u | +0.018887 [+0.009460,+0.028552] | −0.003040 [−0.010375,+0.003897] | +0.015847 [+0.009118,+0.023079] | 47/60=78.3% |
| CoGenAV | +0.006490 [−0.011439,+0.024722] | +0.001396 [−0.009740,+0.012780] | +0.007886 [−0.002610,+0.019399] | 34/60=56.7% |

SyncNet最佳项T胜35/60，背景贡献正50/60；AVS最佳项47/60，Co最佳项29/60。不得因一个CI跨0另一个不跨0，就断言两项有显著差异。事后直接比较SyncNet“背景贡献−最佳贡献”=+0.057486，95%[−0.163286,+0.284949]，跨0。可说背景有可见且较稳定贡献、点估计略大；不能断言背景贡献严格大于最佳项，也不能说全部增益都是背景。

零时移ΔS0：
- SyncNet +0.184，95%[+0.003,+0.382]，42/60 T胜；
- AVS_u +0.009466，[+0.000650,+0.018906]，33/60；
- CoGenAV −0.003230，[−0.018867,+0.012265]，29/60。

模型原生峰位：SyncNet N18/20、T54/60为+1帧，其余0；AVS N19/20与全部T为+2帧，另N一条+3；Co N17/20与T59/60为+2，余+3。模型有固定峰位差，不把这些估计直接宣布为物理延迟真值。

## 严格参考层
12来源/36配对的best / background贡献 / contrast：
- SyncNet +0.288 / +0.391 / +0.679，contrast95%[+0.353,+1.038]，T胜32/36。
- AVS_u +0.018179 / −0.001028 / +0.017151，contrast[+0.007521,+0.027519]，26/36。
- Co +0.007834 / +0.001096 / +0.008930，contrast[−0.000325,+0.018926]，22/36。
严格层SyncNet最佳项CI也为正[+0.049,+0.540]，因此“只有背景改善”并不符合这层结果。扩展8来源完整值在summary.json，不能替代主层选择。

## 单位范数敏感性（冻结设计内）
用相同SyncNet特征先各自单位范数化，再计算距离，无新增GPUforward：
- best贡献 +0.034326 [ +0.009014,+0.060304 ]；
- background贡献 −0.007765 [−0.025650,+0.007838]；
- contrast +0.026560 [−0.008925,+0.061099]。
原始距离下的正背景贡献在此视图不再稳定，提示特征幅值/几何尺度值得研究；单位化也改变距离定义，不能将raw与unit数值直接相减、算消除比例或宣布声学电平已被因果定位。单位化后最佳项仍有正差，也不支持“只有范数、完全没有匹配差异”。

## 事后支持检查：为什么AVS方向变了
因本次AVS_u零时移转正，ROOT明确追加CPU posthoc_support_diagnostic.json，未改正式端点、无新GPU：

| AVS_u零时移汇总 | 平均T−N | 95%描述CI |
|---|---:|---|
| 原缓存全片逐帧均值（复现旧结果） | −0.007529 | [−0.016286,+0.001546] |
| 本次相同query中央单帧、不做五帧平均 | +0.009857 | [+0.000991,+0.019390] |
| 本次相同query五帧平均（正式） | +0.009466 | [+0.000650,+0.018906] |
| 各clip全部内部合法query、数量不再跨臂相同 | +0.012125 | [+0.004157,+0.020778] |

因此AVS偏好变化已经发生在不搜索lag、不减背景的评分里；主要关联时间支持/聚合选择，并非仅靠套用Sync-C公式。中央单帧与五帧均值接近，不能把转向简单归于五帧平滑。不同时间支持评估的发音内容权重也不同，不能据此选择一个更符合预期的结果作真值。

各clip全部内部query时，Sync最佳/背景贡献/contrast约+0.272/+0.212/+0.484；其背景贡献CI跨0。AVS约+0.023161/−0.004433/+0.018728，Co约+0.010198/−0.000897/+0.009301（Co总差仍跨0）。说明“Sync背景贡献是否稳定”本身也依赖窗口选择，不可概括所有协议。

保持正式query不变，仅将lag范围收窄±400ms：Sync contrast+0.693、AVS+0.015542、Co+0.010054；Co区间仍跨0。背景中位数依赖预先定义的lag网格，数值不是绝对物理量。

## 验证、产物及可复现性
- SyncNet 3自然fixture使用原官方保存MFCC重新前向，audio1024最大差全部0；新canonical主输入另行标注。
- 真实PCM时移的相对峰移动：SyncNet6/6（±160ms），AVS40/40（±400ms），Co160/160（±160ms）全通过，容差1帧。控制只验证这些扰动的定位，不证明局部细微口型/感知真值。
- ROOT独立脚本不导入生产代码，从80NPZ全量重算5种矩阵、206控制、所有端点、60配对、三层来源bootstrap。最大矩阵误差5.56e−16，配对3.34e−16，bootstrap1.05e−16，分解闭合1.78e−15。Windows与Ubuntu独立重算均PASS。
- 执行器Ubuntu/Windows各5项真实artifact测试PASS；ROOT在Windows实际执行生产5项+独立审计1项，共6项PASS。源码compile检查通过。
- 执行无模型崩溃、无科学端点修复。Co上游manifest绑定及环境源码pin是完成后的补充审计receipt，明确post-forward，不伪装成事前验证；原协议/评分源码快照不回写。
- 新SyncNet模型计算约50秒完成，原始特征/矩阵/控制/JSON约250.7MB已回传本地。没有全组重新生成媒体或下载权重。自有GPU进程已退出，外来5652MiB进程未动。

代码：scripts/experiments/common_window_curves_20261028.py；独立审计与图 scripts/experiments/common_window_curves_independent_audit_20261028.py；对应tests同名。冻结生产源码source_frozen.py，ROOT最终审计源码应另存parent_audit_source.py，不覆盖评分快照。

关键文件：protocol.json、inputs.json、items/*.npz及.json、items.json、controls.json、parity.json、pairs.json、summary.json、provenance_binding.json、environment_and_source_receipt.json、parent_independent_audit.json、posthoc_support_diagnostic.json、chain_status.json。
图：common_curve_decomposition.png/.pdf。各曲线按来源平衡的N20与T60等权3draw展示，面板单位不同；图上平均曲线的max不替代逐clip先max后配对统计。

## 可以支持与不能支持
支持：本次共同支持下SyncNet正差确有背景项贡献，同时不能排除最佳匹配改善；AVS_u的TTS最佳匹配改善与背景机制不同；Co尚无稳定差异；时间支持与距离尺度都能影响解释。

不支持：SyncNet已被证明存在TTS偏置、TTS人眼口型一定更好、全部优势由背景造成，或其他模型都不支持TTS。AVS共同内部窗也偏TTS，应修正简单的指标投票叙述。下一项有价值的机制问题是固定时间支持后进一步控制SyncNet特征范数/局部声学条件；本轮未启动新的干预实验。

## Relations
- relates_to [[云端Qwen Audio3.1Plus的20来源原生四指标分歧 2026-10-27]]
- relates_to [[AVS_u、AVS_m、AVS_v原论文与本项目适用性核对 2026-10-27]]
- relates_to [[LRS3 TTS 原生优势的分数分解与曲线诊断]]
- relates_to [[72-sparsesync-official-native-cloud-preference]]
