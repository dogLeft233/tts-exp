---
title: 79-talklikeyou-chinese20-nine-scorers
type: experiment
permalink: tts-exp/docs/experiments/79-talklikeyou-chinese20-nine-scorers
status: concluded
run_id: talklikeyou_chinese20_rms24_20261030_attempt01
model: TalkLikeYou
cohort: Chinese20
n_sources: 20
science_scope: exploratory evaluator evidence
---

# 79 — TalkLikeYou 中文20来源九评分器与同输入模型效应比较

✅ CONCLUDED（探索性评价器证据），2026-10-31完成。用户确认测试192已部署的TalkLikeYou，沿用此前中文20来源协议；本轮没有部署LiveTalk，没有重启已停止的InfiniteTalk/MultiTalk。

## 结论

TalkLikeYou没有呈现跨九评分器一致的TTS优势。主要同句克隆参考12来源中，九项端点均未通过各分层Holm9校正：Sync-C自然4.937、TTS5.446、Δ+0.509，边际95%CI[+0.170,+0.796]，10/12提高，但Holm9 p=.15345。全20来源Sync-C自然4.895、TTS5.390、Δ+0.495，CI[+0.014,+0.926]、15/20提高，Holm9 p=.37247。边际bootstrap区间与sign-flip校正检验回答不同问题，不据区间跨零与否替代预定多重比较判定。

全20来源仅VocaLiST K5正差通过Holm9：自然49.833%、TTS56.249%，差+6.416个百分点，95%CI[+2.493,+10.633]个百分点，17/20提高，Holm9 p=.04635。它属于补充分层，主要12来源的同项差+5.869个百分点、Holm9 p=.15744；不升级为主要分层确认或人类同步质量结论。PEAVS全20平均差−0.384344，边际CI负向，但Holm9 p=.06248，未过.05。

与同输入FlashHead/Echo的来源级效应差比较中，主要12来源的TalkLikeYou−Echo ΔSync-C效应差+0.735，95%CI[+0.408,+1.110]，11/12来源的效应更高，Holm18 p=.01818。这个结果直接比较两模型各自的TTS−自然，而非仅比较各自显著性。对FlashHead的相应效应差+0.574，Holm18 p=1.000，尚未确认模型差异；不能解释为效应等价。全20来源对Echo的效应差+0.686，CI[+0.244,+1.096]，Holm18 p=.12835，仍未确认。

因此不能笼统说新模型与老模型效应一致，也不能据此声称所有新模型都没有TTS效应。当前证据支持模型、评分器和支持集条件依赖。历史Wav2Lip中文20来源Sync-C约+0.745、VocaLiST K5约−0.937个百分点，但它的电平、运行时及部分旧分类器取窗不同；只作历史背景，不作为本轮隔离生成器因素的比较。

## 冻结输入与生成

与报告76/77逐文件相同的20N+60既有云Qwen TTS规范WAV，共20说话人、41种唯一波形；每人三TTS先平均，20来源为统计单位。同句参考12为主要、扩展参考8和全20为补充。重复draw按明确alias引用同一生成产物，不作为独立合成重复。无新TTS/ASR云调用或训练。

80条输入保持mono/16kHz/PCM16与RMS−24dBFS，实测RMS范围[−24.000007666,−23.999993547]dBFS，最大峰值0.857666，无削波。没有二次调增益。固定3.png，SHA f425463ba279fea88ac807d02bf048fd2f891d4efae1c8903eae3afc22d216c9。

官方源码commit d1088c49071148853c76d8a8f0c1fb97d895758b，权重revision652b744bbf86510c370ff6f302d046bdd63632e1；17项资产和129项源码文件在执行前fresh SHA通过。TalkLikeYou是单步唇部Flow Matching加Ditto音频姿态/渲染的组合系统。沿推荐habit ID192、stage0+neutral、motion1.0、CFG1.3、唇部/姿态各1步、audio pose、平滑、overlap5、crop2.3；固定seed42，每个唯一WAV在新进程完整运行官方main，避免跨片缓存。包装器仅设置种子、记录原始特征/姿态以及ffmpeg -nostdin，没有修改官方推理体。

实际80逻辑视频/41唯一推理，512²/25fps，57–193帧，全部解码、非静态、规则PTS与PCM逐样点校验通过。原生16列mel窗口使画面尾部少3106–3792采样点，即194.125–237.000ms；不补帧、不补输入静音、不拉伸、不同步移音、不根据分数校正offset。完整原始驱动PCM留在ALAC评分媒体，AAC原片用于播放。所有官方唇部mel与Ditto姿态音频前处理保留。

## 评分、支持与统计

共享26个评分入口/辅助源码SHA及五个评分环境pip-freeze集合均与报告76/77匹配。SyncNet官方FP32/min_track25与PCM16 MKV；CoGenAV Eq1、AVS_u帧cosine；MTD/VocaLiST官方K5；Synchformer/SparseSync整数中心5秒21类P0；SyncReward BF16/encode_batch1；PEAVS固定字典序全80、batch64+16。输入与评价定义延续旧流程，不混合旧低电平N分数。

七项覆盖80/80媒体、20/20来源。两分类器各28/80媒体、6/20完整来源，全部6来源属于主要12；其他条目标UNMEASURABLE，不插补。资格清单在评分前按冻结窗口长度另存predeclared_support.json，最终与实际解码支持一致。不能把5秒音频直接当作5秒画面。

每分层10000次来源bootstrap、100000次sign-flip敏感性、seed20261030、Holm9；同一来源三draw先平均。区间为边际区间，sign-flip依赖对称/可交换假设，并非随机分配TTS的因果检验。模型间对比在共同可评分来源上计算(TLY T−N)−(基线 T−N)，每分层两基线×九指标Holm18；无等效性界值，未检出差异不等于相同。

## same_reference12

|评分器|配对人数|自然均值|TTS均值|差值|95% CI|TTS较高|Holm9 p|
|---|---:|---:|---:|---:|---|---:|---:|
|SyncNet|12|4.937|5.446|+0.509|[+0.170, +0.796]|10/12|0.15345|
|CoGenAV|12|0.411044|0.415206|+0.004162|[-0.010697, +0.021126]|7/12|1.00000|
|PEAVS|12|2.468162|2.030581|-0.437580|[-0.795598, -0.066430]|4/12|0.29898|
|SyncReward|12|1.453125|1.462023|+0.008898|[-0.054036, +0.062934]|8/12|1.00000|
|AVS_u|12|0.164325|0.154732|-0.009593|[-0.031337, +0.008697]|6/12|1.00000|
|MTDVocaLiST|12|0.553955|0.613323|+0.059368|[+0.017331, +0.101121]|9/12|0.15744|
|VocaLiST|12|0.504656|0.563342|+0.058685|[+0.016176, +0.096471]|11/12|0.15744|
|Synchformer|6|0.367238|0.322868|-0.0443696|[-0.199332, +0.127744]|2/6|1.00000|
|SparseSync|6|0.204447|0.478596|+0.274149|[+0.0355916, +0.545168]|3/6|1.00000|

## all20

|评分器|配对人数|自然均值|TTS均值|差值|95% CI|TTS较高|Holm9 p|
|---|---:|---:|---:|---:|---|---:|---:|
|SyncNet|20|4.895|5.390|+0.495|[+0.014, +0.926]|15/20|0.37247|
|CoGenAV|20|0.412933|0.420811|+0.007878|[-0.006353, +0.023244]|12/20|1.00000|
|PEAVS|20|2.260098|1.875753|-0.384344|[-0.628356, -0.140191]|5/20|0.06248|
|SyncReward|20|1.449219|1.454362|+0.005143|[-0.053971, +0.055405]|13/20|1.00000|
|AVS_u|20|0.174202|0.164442|-0.009759|[-0.026855, +0.006012]|9/20|1.00000|
|MTDVocaLiST|20|0.565468|0.613394|+0.047926|[-0.002693, +0.095428]|14/20|0.44298|
|VocaLiST|20|0.498326|0.562489|+0.064163|[+0.024930, +0.106330]|17/20|0.04635|
|Synchformer|6|0.367238|0.322868|-0.0443696|[-0.199332, +0.127744]|2/6|1.00000|
|SparseSync|6|0.204447|0.478596|+0.274149|[+0.0355916, +0.545168]|3/6|1.00000|


扩展8来源及完整逐项表见交付根report.md；所有分数单位保持评分器原定义，不能把不同端点混成总分。

## 控制与敏感性

AVS_u、MTD、VocaLiST真实±400ms控制各82/82降分；Synchformer/SparseSync各34/34降分。CoGenAV真实±160ms的82/82峰位移动均在一帧内，median peak-excess为正。六项均过既定80%诊断门。

SyncNet本队列未重做时移控制；PEAVS与SyncReward本队列也未重测，保留此前中文控制响应不足的边界。不能把 native 分差直接称为真实嘴型准确度或感知同步。首对抽帧工程QA与配对差图已看，没有人工盲评。

SyncReward保留官方window_segments内部短片补齐，不延长输入媒体。完整原生4.75秒支持的7来源补充子集Δ+0.063244，95%CI[+0.013021,+0.106027]，6/7提高；这是不同支持集、未并入主要Holm9家族的敏感性结果，不替代主要12或全20端点。

PEAVS用已保留特征另做4×20官方metric重放，没有新增特征提取。全20差−0.384344，同句12差−0.437580、扩展8差−0.304490，与主64+16在显示精度相同；主结果不改。

## 原始复核与精度修复

80音频/视频SHA、RMS/峰值、无损PCM、帧数和条件绑定通过；七类原始矩阵/embedding/logit归约复核PASS，SyncNet独立FP64误差最大1.928e−6。PEAVS/SyncReward核对实际权重前向、输出身份、覆盖与有限值，没有独立重实现其网络。

原audit.py把FP64 softmax与CPU FP32保存概率要求绝对差<1e−7，SparseSync三个零点概率出现1.09–1.27e−7差而退出。原源码、失败log、finish_failure.json和非零stage凭据均保留。独立诊断102组原始logits后，新增audit_precision_replay.py要求CPU FP32完整概率向量逐值精确重放，再用FP64独立核对；所有组通过，全部概率位最大FP64差1.598e−7。精度修复没有新TFG/评分模型前向，170个科学文档在修复前后SHA完全相同。不能把此复核器精度问题当模型测量失败或科学阴性。

服务器针对性测试4项PASS，覆盖原生长度、来源配对、Holm家族与真实精度回归/一ULP篡改拒绝。本机3项PASS/1项因缺Torch跳过；精度测试已在目标服务器实际通过。源码compileall通过。

## 资源与终局

按server-192 skill读取当次飞书1048手册后运行。代码、独立环境、权重、cache、临时文件、输入和输出均在/home/data2/wangjiajun个人区；复用既有环境，无共享base/驱动修改，不操作他人任务。nohup、单A6000 UUID GPU-3464c5ee-e2ed-81e4-e275-b612a929bfe1、CPU affinity0–3/线程4、torch显存60%；生成与各评分GPU阶段顺序执行。

41次唯一生成的冷进程/封装/验证累计829.509秒；本人PID NVIDIA每2秒采样峰5856MiB，torch峰allocated2322007552B/reserved2602565632B；不把采样峰视为瞬时绝对上界或实时性能指标。终局terminal_delivery_receipt.json确认41生成退出0、九评分器阶段完成、后处理完成、170科学文档未变和本人GPU计算进程为空。原失败仅是保留的旧复核器stage。

## 产物与复现

- 远端run：/home/data2/wangjiajun/av-evaluation/jobs/talklikeyou_chinese20_rms24_20261030_attempt01。
- 本地交付：[results/talklikeyou/chinese20_rms24_20261030_attempt01](../../../results/talklikeyou/chinese20_rms24_20261030_attempt01/)。summary.json、report.md、per_video_scores.csv、paired_scores.csv和video_index.json可直接复核。
- [模型间效应比较](../../../results/talklikeyou/chinese20_rms24_20261030_attempt01/cross_model_comparison.md)；[九项分差图](../../../results/talklikeyou/chinese20_rms24_20261030_attempt01/paired_differences.png)。
- 标准播放媒体在results/talklikeyou/{natural_raw,tts_raw}/<source>/；完整无损评分媒体、精确WAV、原始数组/特征/控制、日志、源码/环境快照在evidence/。
- evidence.tar.gz：260554096字节，SHA256 87e25057f9443c3428f751bc777f0bf32215e1bd6821a0e4c7a885da7fbad58f；2785成员含manifest，本机2784内容文件全SHA读回与80视频导出PASS。未声明网盘备份。
- 本机local_statistics_audit.json独立复算25非空指标分层；local_comparison_audit.json独立复算50非空效应对比，均包含bootstrap/sign-flip/Holm，最大数值差0。旧基线pairs文件哈希与远端绑定一致。
- 打包完成时的终局凭据单列terminal_delivery_receipt.json。档内打包时RUNNING及旧失败状态是历史快照，不能据此重新launch；final_delivery_receipt.json为本机最终验收入口。
- 源码scripts/experiments/talklikeyou_chinese20_20261030/，测试tests/test_talklikeyou_chinese20_20261030.py；新增未提交，继承dirty未清理，无PR。

## Observations
- [status] concluded
- [conclusion] 主要12来源九项Holm9均未确认TTS优势；全20仅VocaLiST K5正差过校正，Sync-C+0.495未过Holm9。
- [result] 同输入主要12来源TalkLikeYou相对Echo的Sync-C效应差+0.735、Holm18 p=.01818；对FlashHead未确认差异且非等效证明。
- [verification] 80逻辑媒体/41唯一生成、九评分器、原始归约、2784文件与25/50统计复算全部PASS，原失败复核器与170科学SHA不变凭据保留。

## Relations
- compares_with [[76-flashhead-chinese20-nine-scorers]]
- compares_with [[77-echomimic-v3-chinese20-nine-scorers]]
- relates_to [[71-vocalist-official-native-cloud-preference]]
- uses [[server192-nine-scorers]]
- completes [[TalkLikeYou中文20来源九评分器实验2026-10-30]]
