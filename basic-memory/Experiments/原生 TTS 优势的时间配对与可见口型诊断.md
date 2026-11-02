---
title: 原生 TTS 优势的时间配对与可见口型诊断
type: note
permalink: tts-exp/experiments/原生-tts-优势的时间配对与可见口型诊断
status: implemented
started: 2026-10-01
repository: dogLeft233/tts-exp
---

# 原生 TTS 优势的时间配对与可见口型诊断

> 路径说明：`${USER_HOME}`代指原记录对应的个人主目录，不表示接收者已有该文件。定位归档时继续核对资产映射与原运行凭据；示例中的相对仓库路径从仓库根解释。

## 问题与背景

主08完整80/40 N来源四门未通过，但自然句级RMS生成G已正向：主脸raw C=+0.192、二脸+0.135，unit同正。原生T−N的guard20可测39speakers差仍+0.421，固定RMS双端后+0.213。全局CAL mean T−N方向不是音素动态的替代，也不是原生N/T视频对照。需要直接解读原生分差的representation-time pairing与独立可见phonetic shape。

本诊断在root已读main/sham/source-axis视觉结果后冻结，明确为informed follow-up，不称原预注册。权威119行spec openspec/changes/diagnose-native-geometry-visual-20261001/spec.md，rawSHAb11c48d185beb30a3ae3f0036ca7ba02eb4d54ef24fdc13f5f96dfd3d71810bd，本地c5ff67196374e34bd1def791870a052c095fbe95，未push。

## 方法与数据

缓存分数：复用retry3封存raw/unit full V×A matrices与independent08完整2560row验证。N/T各自RAW/FIXED生成×评分四格，原共同recipient支持、k3/col18；B/D/C/anchor和正确search D0−D（col15−min）。每clip100 PCG64 full-A permutations，seed20260930+orderedindex，within source共用π于RAW/FIXED和双脸；N/T长度不同用same-seed独立domain、各自记录完整πhash，不虚构frame correspondence。比较原生/processed T−N observed vsmean-null，以及各source G/E/I，clip→两clip→speaker，20k seed20260930 99CI。

视觉：只新增primary face N-RAW/N-FIXED/T-RAW/T-FIXED×80=320 unique MP4的CPU轨迹；q01重复q00视频不重复计数。使用原N/T MFA各自时钟和原资格，不应用physical-only S0064 overlay。各source RAW/FIXED共同valid mask，跨N/T只比较合格scalar、不warp视频；先封存phone/count/valid availability，再读aperture effect。沿用CAL v2宽类、latency=-0.085秒/count/MediaPipe参数，separation/AUC、zero-latency sensitivity。四个固定contrast：Traw−Nraw、Nfixed−Nraw、Tfixed−Traw、FIXED后native gap改变。全80/40意向分母和完整双clipspeaker支持明确区分。

旧runtime无per-forward记录保留；不将旧输入源推断为false。每个视频/矩阵/embedding原SHA及新包绑定、framecounts/FPS独立检查。核心transfer不含MP4/JPEG，需独立320视频包。实现不导入producer scoring/ops，不再生成TTS/模型视频，不改08或旧视觉轨迹。

## 判断边界与实施
Observed-minus-null阳性表明额外contrast依赖SyncNet representation时间配对，不能直接定声学因果或感知。视觉使用原N/T各自MFA时钟，不做物理overlay；CAL event-rematching失败，时序保持探索。

## 实际执行与视觉结果

Native package local ${USER_HOME}/.codex/tts-exp-artifacts/20261001/native_geometry_visual_20261001_attempt01/；manifest raw SHA73c2f0d9993b9af5f070259ffb869efdf8105030e06b9be4f81e8098330ad0c8，仓库mapping与verification e1cb6ca10fab520367f9986ef99421a42a59da6d23c9bfad8d15926e3884e6e0。320 unique MP4与160原MFA grids完整核验，不使用physical-only overlay。

CPU extraction320/320、零失败；extraction manifest3238c1cc446cb936ff6a3c4d6121aeee7cd33ae6300eaff6bd66d991afeef0ec。主报告 runs/recovered_20260930/independent_validation/native_geometry_visual_20261001_attempt02/analysis_v2/native_4cell_visual_analysis.json，SHA72e7f68df2920a062f8703855bdc6d994f59223851f266e5a94713356e9ac341。root直接读取并核SHA。v2 consumer7311aecb3ef0d6e622b40254f595003d60d7964dee17a4d207b7012d0403e4b2；availability freeze160b5500e0829f64e37be9044648491a1e8b8ccfbbd35a1117f0e5dbaa54274c在任何aperture effect读取前冻结。v1的clock map raw/baseline键错误在静态review发现，原v1/freeze保留，未读effect。

80/40意向分母不变，原MFA79/80。各contrast只有两类phone且两个clip都可用的speaker进入等权bootstrap：native gap52clips/16speakers，N fixed52/16，T fixed53/17，gap change52/16。以下CAL latency=-0.085s的99%CI全部跨零：

| contrast | separation Δ [99%CI] | token-AUC Δ [99%CI] |
|---|---|---|
| T_RAW−N_RAW | -0.00215 [-0.01810,+0.00973] | +0.01081 [-0.12483,+0.13255] |
| N_FIXED−N_RAW | +0.00143 [-0.00422,+0.00620] | +0.03325 [-0.02448,+0.09948] |
| T_FIXED−T_RAW | -0.00112 [-0.00439,+0.00213] | -0.00049 [-0.03480,+0.02745] |
| FIXED后T−N gap改变 | -0.00260 [-0.00805,+0.00286] | -0.03377 [-0.11658,+0.03377] |

zero-latency所有对应CI也跨零。当前预设phone端点未达到改善标准，不能升级为完整cohort阴性或主观同步结论。全轨迹mean覆盖80/40：N aperture .052159→.055058、absolute velocity .309818→.332406；T aperture .062983→.061729、velocity .384550→.377449。全80/40 geometry配对差的独立补充已完成并由root读数，文件analysis_v2/native_geometry_contrast_supplement.json，SHA b713a1f0bfc5d38ce8684d52b4dccd70f9153757283e37cb3c00ba48d7923c39；script ec7638d06be59da91066b5c532914c1216544ab1364e3fc73a5a951cac546451。只读封存v2 per-clip means，20k PCG64 speaker bootstrap/seed20260930/99CI，无feature reopening。

| contrast | aperture Δ [99%CI] | abs velocity/s Δ [99%CI] |
|---|---|---|
| T_RAW−N_RAW | +0.010824 [+0.008752,+0.012922] | +0.074732 [+0.063392,+0.086525] |
| N_FIXED−N_RAW | +0.002899 [+0.001844,+0.003940] | +0.022588 [+0.014555,+0.030767] |
| T_FIXED−T_RAW | -0.001254 [-0.002108,-0.000396] | -0.007101 [-0.013010,-0.001405] |
| FIXED后gap改变 | -0.004153 [-0.005565,-0.002760] | -0.029689 [-0.041260,-0.018409] |

mouth width native差-0.005461[-0.007064,-0.003896]、N fixed差-0.001821[-0.002664,-0.001100]、gap改变+0.002067[+0.000824,+0.003494]；T fixed区间跨0。native pitch-0.001585[-0.002068,-0.001109]；其余pose变化区间跨0。句级level干预造成可见mouth amplitude/motion响应，方向与现有生成端SyncNet结果一致；不能把它改称phonetic shape或人类同步改善。

v2 median字段null确认为aggregator要median而_summary_values只输出mean/sd/p10/p90，属于未测字段；v2原报告不修改，本补充不回填median。上述mean指标全部80clips/40完整speaker可用。

整体状态IMPLEMENTED：视觉部分已独立分析、root读回；geometry paired差已完成；cached temporal null/full边界padding诊断仍在实施中，原08/v2均不改。
## Observations
- [status] implemented
- [result] native四格320轨迹完整提取；phone可用52clips/16完整speaker（T fixed53/17），所有separation/AUC和zero-latency对应99%CI跨零。
- [result] 全80clips/40speaker的独立配对geometry支持句级音量改变实际运动：N fixed开合+0.002899、速度+0.022588；T fixed开合-0.001254、速度-0.007101，对应99%CI均不跨零。supplement b713a1f0…923c39由root直接核验。
- [verification] v2报告72e7f68d…9ac341和原MFA79/80不改；paired补充只读封存per-clip means。median因键缺失属未测字段，原报告保留。
- [decision] 原生来源、静态轴、句级音量分别判断；运动幅度不等于音素判别或感知同步，full/valid/guard15/guard20同时报告。
- [next_step] 完成100 full-A cached temporal-null和padding/边界统计，独立重建并读取结果。
## Relations

- follows [[TTS 来源方向与声学机制确认]]
- follows [[TTS 来源干预的独立可见口型复核]]
- complements [[WORLD 声源谱形与局部能量的双向移植]]
- advances [[解释 TTS 的 TFG 与 SyncNet 优势]]
Native cached-null 独立CPU消费仍未完成：attempt01错误读取top-level anchor，真实protocol.method.global_anchor_column=18；source9f228...b0dc/new02已修该字段并实际运行，随后覆盖检查失败。静态审阅找到face_source_groups计数在4cell循环内部、contexts在2source循环内部，实际检查分母应在各自外层；保留两个旧attempt，出新attempt03。不改变fullV×fullA、共享100π、四支持域或统计设计。该工程失败不影响已独立完成的视觉v2及full80/40几何supplement。

## 配对边界的独立时间置换结果

cached-null attempt05已实际CPU完成exit0。源cd1ab039a7a0794adb10509c6b8da859755c02190cd955a97fb8a5664c8e6d5a；完整报告`runs/recovered_20260930/independent_validation/native_cached_null_attempt05/independent_native_cached_null.json`66,613,857B/SHA7b4c8e96155e103bb93a149c55bc97d2bdeb7fa99986bd8d5a36467a38ddd8f9，gzip12,374,468B/SHA79543171060f1f2ac3996d4ac9cf38e04e11d9872ec2bf5f3b2bed29efe9fae7。两个下载分片、gzip、解压JSON全SHA核验通过。完整2560 plan/results匹配，消费native1280 rows/80contexts/160sample-face/160sample-source/320face-source，所有context/V/score哈希通过；raw矩阵最大误差3.815e−6/unit2.384e−7。k=3/col18、100 full-A PCG64 permutations、20k99% equal-speaker bootstrap。

两套分析预先匹配观测与null音频边界：legacy query_n与full-A敏感性，支持查询保持原common V prefix。短T S0099W0123的guard20 unavailable保留，native gap40intended/39complete，不能写完整cohort达标。guard15两边界一致；主脸原生T−N raw C+.416101[+.268742,+.555063]、mean-null+.181129、time-excess+.234971[+.054424,+.399414]；unit原生+.059268[+.037393,+.078847]、excess+.036781[+.011949,+.057999]。同支持fixed自身gap的excess raw+.030718[−.117866,+.178871]、unit+.009397[−.008808,+.025941]。第二脸guard15原生raw与unit observed正，但time-excess CI跨0，不概括为两脸原生时间优势成立。

FULL主脸原生raw legacy+.041311[−.125208,+.199203]、time-excess−.194526[−.390728,−.015016]；full-A raw+.116234[−.053138,+.274097]、excess−.142133[−.339130,+.037237]。FULLunit legacy excess+.018699[−.002087,+.037633]，full-A excess+.022971[+.002849,+.041522]。原生优势的边缘/几何/边界依赖明确报告，不能只选guard。

更稳定的正向路径是自然音频RAW→FIXED的生成G：评分音频仍同一RAW-N，FULL/VALID/guard15/guard20、两脸、raw/unit全部time-excess99%下界正。full-A guard20主rawG+.192073[+.111269,+.276076]，mean-null+.007188，excess+.184885[+.107570,+.265645]；unitG+.026961、excess+.025240[+.014847,+.035569]。第二脸rawexcess+.132618[+.067367,+.204720]、unit+.015297[+.007368,+.024051]。FULL主rawexcess+.181402[+.111225,+.252483]、unit+.024258[+.014516,+.034291]。这定位了音量经生成器产生的、保留正确时间配对的评价响应，与独立嘴部开合/速度正向结果共同构成实际机制链。

工程与范围：native01–04错误保留；05修短T None，未改变数学/阈值。新padding annex与配对gapchange/time-excess统计addendum仍需独立补齐，不修改此immutable报告。SyncNet时间特异性不等于逐音素可见形状或人类感知同步的结果。

## 原生差距缩小的配对独立补充

新addendum源e5495abf038df2ff307e39fc4943d9867c40b3a7eb53f31b763f14a3f1df5b4c已实际离线CPU运行；完整JSON`runs/recovered_20260930/independent_validation/native_q00_q11_gap_addendum_20261001_attempt01.json`2535532B/SHAe8e1491866d5478e4826e5c5e6764bc9dcd7cc360ac7b3dec68aaa5739460c62由root读取并核hash。COMPLETE160cells覆盖2边界×2脸×raw/unit×4支持×5metric；q00/q11共享100π，20k speaker paired bootstrap，代数误差≤2.23e−16。estimand=(T−N q00)−(T−N q11)，不通过两个独立CI相减推断。

full-A guard15主脸raw gap缩小+.215440[+.125922,+.311430]，time-excess缩小+.204253[+.118742,+.295291]；unit observed+.029505[+.015899,+.043529]，excess+.027385[+.014619,+.040366]。第二脸raw observed+.172298[+.077946,+.275994]、excess+.153777[+.064025,+.253055]；unit observed+.019360[+.008560,+.030713]、excess+.017144[+.006871,+.028001]。所有为99CI/40paired speakers。full-A FULL主raw缩小+.221424[+.133576,+.319005]、excess+.189113[+.112032,+.275318]；secondary raw+.173733/excess+.149961[+.064274,+.240349]；unit两脸FULL也正。

因此，两侧自然/TTS音量标准化会缩小原生自评分的T−N差距，且缩小含正确时间配对的部分；它与同RAW-N评分的生成G及独立嘴部幅度/速度反应一致。不能把差距缩小写成固定后每域完全消失，也不能外推所有模型/语言。原生T−N边界/支持依赖和短T guard20 39/40 ineligible继续保留。原parent报告不改；padding annex仍待封存。