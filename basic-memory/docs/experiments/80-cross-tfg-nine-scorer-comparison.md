---
title: 80-cross-tfg-nine-scorer-comparison
type: experiment
permalink: tts-exp/docs/experiments/80-cross-tfg-nine-scorer-comparison
status: concluded
verification_status: ANALYZED
scope: CPU-only synthesis; legacy Chinese Wav2Lip Holm9 unavailable
inputs: Reports75–79 and historical Chinese source reports
---

# 80 — TalkLikeYou、FlashHead、Echo与Wav2Lip九评分器比较

## Material Passport

Verification Status: ANALYZED。2026-10-31汇总已冻结结果，仅新增CPU统计与静态图；没有TTS、TFG、评分器模型重跑。新模型54格统计逐项保留原summary；旧中文Wav2Lip数值来自正式报告，缺失的统一Holm9不推造；英语Wav2Lip另列。原报告75–79及历史凭据不回写。

## 结论

目前没有某一个端点的效应，在至少两个同条件新TFG上都重复通过Holm9。主要同句参考12来源中，TalkLikeYou、FlashHead Pro、EchoMimicV3-Flash的九项均未过Holm9；全20来源的补充层只有FlashHead的PEAVS负差（q=.04131）和TalkLikeYou的VocaLiST K5正差（q=.04635）过阈值。两项均接近.05，既不是相同评分器跨模型复现，也不代表主要12来源确认。

**PEAVS对TTS的负分差，是四个中文模型中最一致的方向性现象。** 全20来源：TalkLikeYou−0.384344、FlashHead−0.452971、Echo−0.493687、旧Wav2Lip官方batch64版本−0.408085；主要12来源也全部为负。但新模型中仅FlashHead全20通过Holm9，旧Wav2Lip无统一Holm9；PEAVS的既有时移响应不足和批次掩码依赖没有消失。因此可称均值方向重复，不能称四模型确认或自然语音驱动的感知同步更好。

Sync-C在全20中呈TLY+0.495、Flash+0.212、Echo−0.191、旧W2L约+0.745，方向未普遍复现。尤其Flash主要12是−0.065、扩展8是+0.628，不能以合计正差替代主要分层。TLY主要12+0.509、Echo−0.226，这一效应差在本次完整事后Holm27下仍通过：+0.735、95%CI[+0.408,+1.110]、q=.02727；其余53个“模型对×端点×分层”格未过各自27格家族。未检出差异不是模型效应等价。

VocaLiST/MTD的TTS正差在TLY与Flash方向一致，但Echo及旧W2L方向相反；CoGenAV、AVS_u、SyncReward和两个P0端点也没有四模型统一方向。旧Wav2Lip中文的两个分类器还含一条自然基线的40ms历史取窗失配，不能把其六来源均值当无保留的严格复制目标。

## 如何读表

Δ＝每来源三TTS均分−该来源自然分，再对完整来源等权平均；同一来源三draw不是三个独立人。中文四模型原计划均20来源、20N+60T。K5与P0四行统一显示**百分点（pp）**，其他行保留原生分数单位；不横比不同指标的差值大小。

每格标Δ、独立来源n、双侧sign-flip的Holm9校正q。粗体表示q<.05。NA只表示旧报告没有提供/本次无法重建统一Holm9，**不是不显著**。旧W2L数值按原报告舍入精度，不能把表格补零当作额外有效位数。置信区间与完整机器可读数值在comparison.json/CSV和图中；边际bootstrap CI不能代替校正检验。

三个新模型的Holm9分别在各模型、各分层内校正，不是把所有模型合并为36项的全局校正；英文W2L也是其独立九项家族。所有汇总判定应保留此检验家族范围。本次新成对效应差则对三个同输入模型×九端点共27项、每分层分别Holm；不是此前以TLY为焦点两基线的Holm18，也不替代旧冻结结果。

## same_reference12

每格为 Δ；独立来源n；Holm9 q。K5与P0四行单位为百分点。NA＝未计算，不能判不显著。

|评分器|TalkLikeYou|FlashHead Pro|EchoMimicV3-Flash|Wav2Lip中文旧批次|
|---|---|---|---|---|
|SyncNet|+0.509; n=12; q=0.15345|-0.065; n=12; q=1.00000|-0.226; n=12; q=1.00000|+0.740; n=12; q=NA|
|CoGenAV|+0.004162; n=12; q=1.00000|-0.009954; n=12; q=1.00000|-0.016753; n=12; q=1.00000|+0.001689; n=12; q=NA|
|PEAVS|-0.437580; n=12; q=0.29898|-0.550276; n=12; q=0.23454|-0.536595; n=12; q=0.61118|-0.474832; n=12; q=NA|
|SyncReward|+0.008898; n=12; q=1.00000|+0.019314; n=12; q=1.00000|-0.017144; n=12; q=1.00000|-0.031487; n=12; q=NA|
|AVS_u|-0.009593; n=12; q=1.00000|+0.010678; n=12; q=1.00000|+0.015173; n=12; q=1.00000|-0.006639; n=12; q=NA|
|MTDVocaLiST (pp)|+5.936782; n=12; q=0.15744|+8.433822; n=12; q=0.94702|-2.884269; n=12; q=1.00000|-0.716549; n=12; q=NA|
|VocaLiST (pp)|+5.868505; n=12; q=0.15744|+8.177553; n=12; q=0.90719|-2.757486; n=12; q=1.00000|-0.777; n=12; q=NA|
|Synchformer (pp)|-4.436960; n=6; q=1.00000|-6.355936; n=7; q=1.00000|+28.303426; n=7; q=1.00000|+1.293700; n=6; q=NA|
|SparseSync (pp)|+27.414902; n=6; q=1.00000|+0.523809; n=7; q=1.00000|-3.567533; n=7; q=1.00000|-0.009311; n=6; q=NA|

## all20

每格为 Δ；独立来源n；Holm9 q。K5与P0四行单位为百分点。NA＝未计算，不能判不显著。

|评分器|TalkLikeYou|FlashHead Pro|EchoMimicV3-Flash|Wav2Lip中文旧批次|
|---|---|---|---|---|
|SyncNet|+0.495; n=20; q=0.37247|+0.212; n=20; q=1.00000|-0.191; n=20; q=1.00000|+0.745; n=20; q=NA|
|CoGenAV|+0.007878; n=20; q=1.00000|-0.009681; n=20; q=1.00000|+0.008056; n=20; q=1.00000|-0.000836; n=20; q=NA|
|PEAVS|-0.384344; n=20; q=0.06248|**-0.452971; n=20; q=0.04131**|-0.493687; n=20; q=0.06921|-0.408085; n=20; q=NA|
|SyncReward|+0.005143; n=20; q=1.00000|+0.017839; n=20; q=1.00000|-0.018490; n=20; q=1.00000|-0.019522; n=20; q=NA|
|AVS_u|-0.009759; n=20; q=1.00000|+0.007638; n=20; q=1.00000|+0.001934; n=20; q=1.00000|-0.007529; n=20; q=NA|
|MTDVocaLiST (pp)|+4.792599; n=20; q=0.44298|+6.088155; n=20; q=1.00000|-1.981485; n=20; q=1.00000|-0.565642; n=20; q=NA|
|VocaLiST (pp)|**+6.416272; n=20; q=0.04635**|+6.581145; n=20; q=1.00000|-2.718419; n=20; q=1.00000|-0.937; n=20; q=NA|
|Synchformer (pp)|-4.436960; n=6; q=1.00000|-6.355936; n=7; q=1.00000|+28.303426; n=7; q=1.00000|+1.293700; n=6; q=NA|
|SparseSync (pp)|+27.414902; n=6; q=1.00000|+0.523809; n=7; q=1.00000|-3.567533; n=7; q=1.00000|-0.009311; n=6; q=NA|

## Wav2Lip英语EARS40：跨语料参考

|评分器|来源n|Δ|Holm9 q|
|---|---:|---:|---:|
|SyncNet|40|+0.859|0.000090|
|CoGenAV|40|+0.010686|0.103499|
|PEAVS|40|+0.085555|0.728843|
|SyncReward|40|+0.101172|0.000090|
|AVS_u|40|-0.006360|0.728843|
|MTDVocaLiST (pp)|40|-0.725827|0.483995|
|VocaLiST (pp)|40|-0.278419|0.728843|
|Synchformer (pp)|40|+0.759434|0.048510|
|SparseSync (pp)|40|+0.200744|0.483995|

## 同输入三模型效应差：本次事后Holm27

每格差为(A的T−N)−(B的T−N)。不包含Wav2Lip，不替代旧报告Holm18。

|分层|A−B|指标|共同n|效应差|边际95%CI|Holm27 q|
|---|---|---|---:|---:|---|---:|
|same_reference12|TalkLikeYou−Echo|SyncNet|12|+0.735|[+0.408,+1.110]|0.027270|

## 协议差异与可比资格

|项目|TalkLikeYou|FlashHead Pro|EchoMimicV3-Flash（报告77）|旧Wav2Lip中文|Wav2Lip英文EARS40|
|---|---|---|---|---|---|
|原计划来源/媒体|20来源/80逻辑媒体|同左|同左|同20来源/80媒体|40英语说话人/80媒体|
|独立TTS重复|三draw先平均，含重复字节；41唯一生成|同左|同左|同批既有云TTS，60draw仅21种唯一波形|每人1个既有云TTS draw|
|参考组|主要同句12、扩展参考8|相同来源分层|相同来源分层|相同来源分层|独立参考句，全40共用目标文本/参考文本|
|输入电平|RMS−24dBFS|同一80WAV逐SHA相同|同一80WAV逐SHA相同|RMS约−28.477dBFS|RMS约−28.477dBFS|
|图片与时钟|同3.png、512²/25fps；无移音/拉伸|同左|同左|相同固定图3、25fps|相同固定图3、25fps，但语料不同|
|生成设置|habit192、stage0+neutral、lip/pose各一步、seed42；FM唇部+Ditto姿态/渲染|Pro/stream、seed42；每片重置|Flash完整checkpoint、8步、seed42、BF16/offload/TeaCache|Wav2Lip GAN、nosmooth、batch4|相同W2L GAN recipe|
|音频前处理|官方mel与Ditto姿态处理保留|官方W2V前端保留|受控RMS入口，显式绕过作者−23LUFS；中文W2V do_normalize=False|W2L原生mel前端|W2L原生mel前端|
|原生支持|57–193帧，尾部少194–237ms画面|官方末chunk长度，按实际完整帧支持|61–197帧、4k+1量化，尾部少至多154ms|原生尾部可短数帧；按各评分器实际支持|原生尾部可短数帧，最长音频18.8秒|
|两5秒分类器|28媒体，6完整来源|33媒体，7完整来源|33媒体，7完整来源|28媒体，6完整来源；S0031自然旧视频起点少一帧|80媒体，40完整来源；整数窗修复已实施|
|SyncReward|BF16，官方内部短片补齐；完整4.75秒7来源另报|BF16；完整4.75秒8来源另报|BF16；完整4.75秒7来源另报|完整原生比较仍是FP32；后续BF16只审计控制，未重评全云TTS|BF16、全部原生长度足够|
|PEAVS|固定80顺序64+16；另4×20头部重放|同左|同左|主表用审计后的官方64，非旧batch1；批次依赖仍保留|80原生64+16；控制三元组分63/57，另batch1敏感性|
|运行时/来源验证|192/A6000；26共享源码及5环境与另两模型匹配|同左|同左|旧Ubuntu/V100、Torch2.5系列；不能视同新环境|旧Ubuntu/V100，各自实际环境receipt|
|统计|来源bootstrap10k、signflip100k、seed20261030、各分层Holm9|相同|相同|旧来源bootstrap分散在多个run；未有统一Holm9|speaker bootstrap10k、signflip100k、seed20261029、Holm9；百万次扩展不改判定|

新三模型在输入字节、人像和评分代码/环境上可对齐，**生成器原生时间支持与内部前端仍不同**。旧W2L比新三模型低约4.477dB；英文W2L还同时改变语料、语言、文本/参考设计和draw数。它们只能提供条件性或外部队列证据，不能把差值变化单归因生成器或语言。

Echo一律取报告77的受控RMS完整队列，不将报告78的官方LUFS/配置短片敏感性混成另一个主结果；此选择由用户本轮比较对象和同输入资格确定，不按分数选最优配置。

## 逐评分器的复现判断

|评分器|中文均值方向（TLY/Flash/Echo/W2L）|当前能支持什么|
|---|---|---|
|Sync-C|+ / + / − / +|部分模型方向相同；三个新模型均未过Holm9，主要层TLY−Echo效应有差异，不支持通用正效应|
|CoGenAV Eq1|+ / − / + / −（旧接近0）|混合方向，未有校正后跨模型复现|
|PEAVS|− / − / − / −|最强的方向重复；只有Flash全20过Holm9，且控制不足，不能称感知劣化|
|SyncReward|+ / + / − / −|混合方向，旧FP32/新BF16不同；无跨模型确认|
|AVS_u|− / + / + / −|混合方向，无跨模型确认|
|MTD K5|+ / + / − / −|TLY/Flash正向但均未过Holm9；旧W2L低值/边界效应限制比较|
|VocaLiST K5|+ / + / − / −|仅TLY全20过Holm9；Flash方向同但未确认，主要12仍无确认|
|Synchformer P0|− / − / + / +|少量长片支持、旧起点失配与P0高区；没有稳定跨模型方向|
|SparseSync P0|+ / + / − / −|少量支持、旧W2L严重天花板，不能把置信度微差当精细时差|

英语EARS40的W2L在Sync-C（+0.859、q=.000090）、SyncReward（+.101172、q=.000090）、Synchformer（+.759434pp、q=.048510）有校正后正分差，但没有任何一个对应端点在本轮三个中文新TFG中重复通过Holm9，故不能作为严格跨模型复制。英语Synchformer百万次扩展q=.046473仍同方向；冻结主表仍用100k。

## 控制效度

新三模型六项控制通过（CoGenAV峰位，AVS_u、两个K5、两个P0的已知时移降分）。新三模型的SyncNet/PEAVS/SyncReward本队列未重测，旧控制响应不足边界保留。

旧W2L中文：AVS_u32/40降分到达门，CoGenAV波形时移峰位可追踪；PEAVS默认64控制25/40下降、SyncReward BF16控制14/40下降，均低于32/40门。两个官方K5存在−3帧附近的内部峰与低正确率，真实移位恢复通过不能替代零点评分校准；不能再引用报告68已被报告70替代的mean-logit主分。

英语W2L：Sync-C降分15/80但相对位移80/80恢复；PEAVS41/80、SyncReward33/80降分均未到64/80门。SparseSync79/80 P0>.9995，效应受单来源影响。这些分别限制“评分高”解释为“嘴型更准”，不删除已冻结分数，也不改端点使门通过。

## 数据缺口与统计验证

旧中文W2L完整逐来源九项数组没有在当前macOS工作区找到，旧Ubuntu只读SSH本次关闭连接。历史报告主要给来源bootstrap而非统一九项sign-flip/Holm；不能由均值/CI/胜率恢复正确的Holm9，也不能把60次TTS比较当60独立来源算p。本汇总对旧中文显著性留NA，不补造或把旧CI“显著”当作多重比较通过。PEAVS默认64的CI未发布于现有审计摘要，也不借用batch1的CI填入。

旧分类器受影响的S0031来源应在后续正式历史修复中恢复精确窗口后重评，或另做明确五来源敏感性；本次没有完整逐来源数组，未伪称已经完成五来源复算。历史六来源数值保留legacy标签。

三个artifact检查PASS：54条新模型冻结统计/百分点转换一致；18条旧中文条目的缺失q不被误标为未显著；新54个模型效应对比的Holm27逐秩复算，并与已有TLY−两基线的36格未校正差值/p逐值对照。脚本不导入模型，不访问权重，不更改旧产物。81条对照行、54条新模型对比与来源SHA在comparison.json，平表在comparison.csv，图已实际查看。

## 统计谬误检查（11/11）

|检查|处理与限制|
|---|---|
|分层/Simpson风险|主要12与全20都保留；Flash严格层与扩展层Sync-C符号不同，不用总体替代分层|
|生态推断|统计单位为来源/说话人，不把60draw作60人，不外推单个说话人或所有模型|
|筛选/Berkson风险|中文20是固定队列，非总体随机抽样；长片资格受音频长度影响|
|碰撞/条件选择|双臂完整5秒支持可能造成选择偏差，6/7来源只解释相应支持集|
|基率忽略|P0是类别置信度而非校准PPV；不转译真实同步概率|
|均值回归|未按本轮极端分数选来源，无前后改进设计；不构造均值回归因果结论|
|幸存者偏差|明确全部计划80媒体及各指标实际n，不将不可测当零分或科学阴性|
|多重搜索|原每模型/分层Holm9范围明确；本次成对差异另用事后Holm27，不按星号选报告|
|分叉路径|版本/精度/批次、分层和旧分类器问题全披露，未选最好版本合并|
|相关与因果|原生比较同时改变生成与评分输入，电平/语料有混杂；无感知真值|
|反向因果|没有将统计预测或指标间一致性解释为单一方向机制，也未开展新机制实验|

## 可复现文件

- [完整机器可读对照](../../../results/cross_tfg/nine_scorer_comparison_20261031/comparison.json)与[CSV](../../../results/cross_tfg/nine_scorer_comparison_20261031/comparison.csv)。
- [四模型配对差图PNG](../../../results/cross_tfg/nine_scorer_comparison_20261031/paired_effects.png)／[PDF](../../../results/cross_tfg/nine_scorer_comparison_20261031/paired_effects.pdf)。空心旧W2L表示协议有差异且q未知；旧PEAVS64无现成CI只画点。
- 分析源码scripts/analysis/compare_tfg_nine_scorers_20261031.py；artifact测试tests/test_compare_tfg_nine_scorers_20261031.py（3 PASS）；仅CPU，原始分数没有更新。
- 正式来源：[[76-flashhead-chinese20-nine-scorers]]、[[77-echomimic-v3-chinese20-nine-scorers]]、[[79-talklikeyou-chinese20-nine-scorers]]；旧中文四指标正式专题、AVS Task、[[70-mtd-official-native-cloud-preference]]、[[71-vocalist-official-native-cloud-preference]]、[[72-sparsesync-official-native-cloud-preference]]及[[052-integer-frame-native-window-preselection]]；英语[[75-ears40-english-wav2lip-nine-scorers]]。

## Observations
- [status] concluded
- [conclusion] 中文四模型PEAVS负差方向一致，但没有同条件端点在至少两个新TFG上重复通过Holm9；旧中文W2L的统一Holm9未知，不能视为不显著。
- [result] 三新模型全部事后27项家族中，仅主要12的TLY−Echo Sync-C效应差+0.735、q=.02727通过；其他差异未确认不是等效证明。
- [limitation] 旧W2L电平较低、SyncReward精度和旧分类器取窗不同；英文40另有语料/语言/参考设计差异。

## Relations
- synthesizes [[79-talklikeyou-chinese20-nine-scorers]]
- synthesizes [[76-flashhead-chinese20-nine-scorers]]
- synthesizes [[77-echomimic-v3-chinese20-nine-scorers]]
- qualifies [[75-ears40-english-wav2lip-nine-scorers]]
- qualifies [[72-sparsesync-official-native-cloud-preference]]
