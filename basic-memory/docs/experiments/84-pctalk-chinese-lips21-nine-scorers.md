---
title: 84-pctalk-chinese-lips21-nine-scorers
type: experiment
permalink: tts-exp/docs/experiments/84-pctalk-chinese-lips21-nine-scorers
status: concluded
date: '2026-10-31'
---

# 84 — PC-Talk Chinese-LiPS21 配对九评分器试跑

✅ CONCLUDED（探索性评价器证据），2026-10-31。用户要求用新准备的数据跑部分PC-Talk推理及评分；复用另一会话已有TTS，不重新合成或调用ASR。

**主要17人：仅Sync-C通过Holm9，ΔC=+0.569，95%[+0.320,+0.795]，p=0.00675。** 其余八项未确认平均差，不能解释为等效。补充全21人Sync-C仍正且通过校正（+0.547，p=0.00081），PEAVS负差通过（−0.410676，p=0.01464）；PEAVS保留控制不足和批次依赖边界。Synchformer本批未复现平均TTS优势。

## 问题与样本

问题：在新的多说话人中文音频及多张人像上，PC-Talk是否仍出现自然／TTS的原生评分差？

数据为Chinese-LiPS官方test子集21人×5个目标，另每人独立ID参考。另一会话“查找九评分器实验中的模型差异”已用qwen-audio-3.1-tts-plus取得104/105条克隆；缺024_29_F_TY_139，多次明确Engine411失败保留，未造补项。使用既有转写，未逐句听辨/ASR核验。参考ID与目标不同，但原录制绝对区间不重叠未核实。

本轮仅取每人成功目标ID字典序首条：21人、21个自然／TTS配对、42唯一视频；每人一个既有成功TTS draw，不是全104对，不按模型分数或TTS好坏选样。本轮0云ASR/TTS调用、0训练；42新TFG及九项共378个原生媒体端点值，另保存六类共504条±400ms移音／峰移诊断。

主要strict_quality17在生成前预定：自然源PCM零饱和且TTS无provider峰值触顶标记。024、027、074、141四人不入主要分层但全部生成、评分并留在all21；其中027/074/141自然源已饱和，四人TTS触顶。该分层只是已知标记筛查，不等于人工质量认证；降音量不能修复原始饱和。preparation/selection及原始协议已有17/21定义，后述模板元数据更正未改变选样或分层。

## 固定生成与评分协议

- 自然音频8.384–11.863秒、均10.140；TTS6.320–12.560秒、均9.962。两臂为mono16k PCM16，另存标量RMS−24派生，原件不改；入选42条均能满足−1dB峰值余量，保留增益和原件/派生SHA。本机从42原件逐样点重建归一化通过。
- 每名音频说话人使用一张固定、按F/M标签匹配并固定seed分配的HDTF异人人像，N/T同图，21张不同命名人像。采用hdtf_gender_matched_20261031，不用旧20图/21人的历史分配；HDTF人物的作者训练/测试归属未核，不能声称全组件无泄露。
- PC-Talk官方源码commit a0392744cbac857d96db1b99ceab70cf7b5f6c83，12权重revision e6220816118bd8714ff57dc83e5906122b4e9168；100 tracked源码及12权重前后SHA一致。preset192、lip1、无EMC/情绪、crop2.3、seed42，每唯一输入独立进程。先按原生left-right检测顺序找最大bbox的索引，21张均为index0，原图及overlay已核看；不直接继承旧3.png的index1。
- 沿用官方pasteback及音频窗口：25fps，输出随输入人像为454–904平方像素，未统一到512；153–309帧，视觉尾部少3082–3789样点（约0.193–0.237秒）。完整PCM留在ALAC评分媒体，视频/音轨零起点，未外部变速、补帧或移音。原始官方AAC成片也保留。
- SyncNet官方joonson源码907c0b579c2e2d83f0eae1b2ac9e720cde4e5623，v2/S3FD权重及18文件SHA同旧实验；FP32，min_track25、vshift15、crop_scale .4。42条实际唯一主脸轨迹155–311帧，都超过100帧；这不是另一次min_track100前向对照。轨迹均值bbox与输入主脸最小IoU .907610；原矩阵复算/官方stdout三位小数全匹配，FP64最大C误差2.251327e−6。
- 官方SyncNet pipeline内部仍转MPEG4/MP3，42条最优offset均−2帧。本轮不补偿该偏移，不是去MP3/延迟补偿消融；入口PCM无损不意味着全链无损，也未复现完整作者论文评分协议。
- CoGenAV Eq1、AVS_u帧cosine；MTDVocaLiST/VocaLiST官方K5；Synchformer/SparseSync原生整数中心125帧、21类P0；SyncReward BF16/encode_batch1及官方window_segments；PEAVS按42条文件名字典序、metric_batch_size64、实际单batch42。两分类器42条都满足原生5秒窗口，SyncReward42条都覆盖4.75秒；不靠外部补齐取得资格。
- 26共享评分源码与五环境前后均匹配Flash76/Echo77冻结基线；PC-Talk环境也与部署freeze一致。原生95%脸支持门槛不放宽，CoGenAV/AVS_u均42/42通过；所有九端点均42媒体、21完整来源，无缺测、插补或重复媒体alias。

统计单位为说话人（每人仅一对），TTS−自然。每分层10k配对bootstrap、100k双侧sign-flip，seed20261031；九端点共同Holm9，主要17与补充21分别校正。CI为边际95%，不按CI替代校正显著性。sign-flip依赖对称/可交换假设，不是随机分配TTS的因果检验；未通过不代表等效。机器表K5/P0为0–1，下表转为百分比及百分点pp。

## 主要：无已知饱和／触顶标记17人

|评分器|配对人数|自然均值|TTS均值|TTS−自然|95% CI（边际）|TTS较高|Holm9 p|
|---|---:|---:|---:|---:|---|---:|---:|
|SyncNet|17|7.458|8.027|+0.569|[+0.320, +0.795]|14/17|0.00675|
|CoGenAV|17|0.343020|0.358061|+0.015040|[-0.002228, +0.035421]|11/17|0.71039|
|PEAVS|17|2.692397|2.336246|-0.356151|[-0.602259, -0.124984]|5/17|0.09032|
|SyncReward|17|1.646599|1.671875|+0.025276|[-0.011489, +0.059283]|12/17|0.71039|
|AVS_u|17|0.135947|0.144666|+0.008719|[-0.002385, +0.019086]|11/17|0.71039|
|MTDVocaLiST K5|17|6.861%|5.134%|-1.727 pp|[-2.880, -0.601] pp|6/17|0.09032|
|VocaLiST K5|17|6.981%|6.206%|-0.775 pp|[-1.944, +0.393] pp|6/17|0.71039|
|Synchformer P0|17|97.571%|97.419%|-0.153 pp|[-3.375, +3.000] pp|12/17|0.95884|
|SparseSync P0|17|96.938%|91.085%|-5.853 pp|[-17.032, -0.002] pp|6/17|0.69875|

## 补充：全部21人

|评分器|配对人数|自然均值|TTS均值|TTS−自然|95% CI（边际）|TTS较高|Holm9 p|
|---|---:|---:|---:|---:|---|---:|---:|
|SyncNet|21|7.489|8.036|+0.547|[+0.351, +0.742]|18/21|0.00081|
|CoGenAV|21|0.345829|0.361198|+0.015369|[-0.001547, +0.034124]|13/21|0.62679|
|PEAVS|21|2.712836|2.302160|-0.410676|[-0.637984, -0.193308]|5/21|0.01464|
|SyncReward|21|1.645833|1.667411|+0.021577|[-0.007813, +0.048372]|14/21|0.62679|
|AVS_u|21|0.135579|0.146160|+0.010581|[+0.000364, +0.020702]|14/21|0.36720|
|MTDVocaLiST K5|21|6.553%|5.264%|-1.289 pp|[-2.300, -0.265] pp|8/21|0.17990|
|VocaLiST K5|21|7.065%|6.313%|-0.752 pp|[-1.700, +0.213] pp|7/21|0.62679|
|Synchformer P0|21|97.833%|97.119%|-0.714 pp|[-3.837, +2.305] pp|15/21|0.76919|
|SparseSync P0|21|97.426%|92.781%|-4.645 pp|[-13.676, +0.126] pp|10/21|0.76919|

## 分布与解释边界

Sync-C：全21有18人提高（85.7%），主要17有14人提高（82.4%），均值和校正检验方向一致。本轮支持该队列的Sync-C原生正向差，不把它升级为人类嘴型更准确、纯生成端改善或训练泄露已排除。

Synchformer全21虽15人TTS分数较高，但均差为−0.714个百分点，Holm9 p=.76919；主要17均差−0.153个百分点、p=.95884。胜数与均值不同，不能据胜数宣称平均优势。SparseSync全21均差−4.645个百分点，受131_29_F_LY_009的较大负差影响（自然P0 .999960、TTS .082365）；该样本保留，不据分数删样，其余差异也未确认。原始逐源结果与视频供检查，未对这些个案作盲评或原因归因。

PEAVS仅补充全21负差通过校正，主要17未过；本队列未重测该项移音响应，且其原生输出受批次/长度填充影响，不能据此推断感知同步更差。

## 移音与峰移诊断

|端点|实际诊断|结果|
|---|---|---|
|CoGenAV|84/84峰移在1帧内，median peak excess正|PASS|
|AVS_u|83/84真实±400ms降分|PASS|
|MTDVocaLiST|68/84降分；80%门需要68|PASS（接近门槛）|
|VocaLiST|77/84降分|PASS|
|Synchformer|84/84降分|PASS|
|SparseSync|84/84降分|PASS|
|SyncNet / PEAVS / SyncReward|本队列未重测该类控制|NOT_RETESTED_THIS_COHORT；保留既有控制限制|

通过上述操作诊断不等于人类同步真值。MTD旧中文20续评为65/82未过，本批68/84通过，分别保留，不回写旧结果。

## 与旧PC-Talk中文20的差异

旧[[83-pctalk-chinese20-nine-scorers]]全20 ΔC=+0.490、Holm9未过；本轮全21+0.547，主要17+0.569。方向一致，本轮校正后确认；不能把显著性差异当作已证明效应增强或模型升级。两批用相同PC-Talk版本，但音频/文本、参考方案、人像、长度、分层及有效支持同时变化，未做这些因素的因果拆分。

|条件|旧中文20 / 报告83|本轮Chinese-LiPS21|
|---|---|---|
|自然来源|AISHELL中文20来源|Chinese-LiPS官方test，21人各首个成功ID|
|TTS设计|每来源3逻辑draw，许多音频相同|每人1个既有成功draw，42媒体全唯一|
|参考/主要分层|12同句参考主要，8扩展参考补充|全体独立ID参考；主要17按已有质量标记|
|人像|固定3.png一张|21张HDTF异人人像，按F/M标签匹配|
|原生输出|512平方，57–193帧|454–904平方，153–309帧|
|5秒P0支持|6完整来源|21完整来源|
|PEAVS实际批次|64+16|42，均使用上限64|
|统计|旧分层及seed20261030|主要17/补充21，seed20261031；同bootstrap/signflip/Holm9方法|

## 修复、完整性与服务器使用

使用192前已读当次飞书1048手册；个人data2区、单GPU UUID GPU-7272933b-1219-b5f8-a9dd-ceade9bbde37、4线程/affinity0–3、nohup顺序任务。生成Torch45%，评分Torch60% allocator界限；不是管理员配额或ONNX硬上限。生成有效PID NVIDIA采样峰值2348MiB，不是连续硬件峰值。没有修改共享base/驱动、他人任务或新增云调用；HDTF20 TalkLikeYou的用户暂停未恢复。

三项修复均保留历史及SHA，不通过重抽结果解决：
1. 首N官方生成454平方，旧512尺寸验收断言失败。worker原本成功；新私有校验按官方原图尺寸验收并复用成片，两MP4 SHA不变，无重复该TFG前向。
2. 完成27段后，下一条131_29_F_LY_009_T加载前资源断言失败，无该条process/worker_result。将原16GiB空余且util≤85的断言改45秒等待循环，相同GPU/参数续跑缺项。原执行33文件SHA、27完成项388科学文件SHA不变。
3. 克隆模板遗留了chinese20/20人/80条、旧父输入、旧PEAVS批次及strict_reference描述。生成后、续评期间显式修正非计算元数据，原protocol/inputs及新旧SHA保留；567媒体科学文件未变。实际21对选择、主要17分层、模型/音量/端点/支持/种子均未变，42次TFG与42次SyncNet未重跑。

服务器原始数组审计：七类原始端点重算通过，两分类器完整CPU FP32 softmax精确重放并独立FP64验证（最大差1.737391e−7）；PEAVS/SyncReward直接核验原生输出/身份/有限值，不是独立重实现网络。本机完整SHA、42媒体/图片绑定及归一化逐样点重建通过，九端点来源配对重建通过；18个指标×分层的均值、bootstrap、signflip、Holm9和胜数独立复算最大误差0。无接近.05的校正判定，不追加选择性模型重跑或额外精确枚举。

42成片各首/中/尾截图已核看，无空白或错选背景脸；不是连续视听盲评。九面板PNG/PDF已视觉核看，45个最终HTML链接可用。2个目标pytest及语法检查通过；没有提交Git或创建PR。

terminal_receipt确认47个记录PID均退出、无绑定本run的本人命令；全局GPU当时1279MiB/15%可能为其他任务，不称整卡空闲。旧RUNNING/资源失败/校验失败快照为历史，不能据此重启。

## 可复现产物

本机根：`/Users/wangjiajun/tts-exp/results/pctalk/lips21_rms24_nine_20261031_attempt01/`。
服务器run：`/home/data2/wangjiajun/av-evaluation/jobs/pctalk_lips21_rms24_nine_20261031_attempt01`。

- [最终视频浏览页](/Users/wangjiajun/tts-exp/results/pctalk/lips21_rms24_nine_20261031_attempt01/index.html)
- [逐源九项CSV](/Users/wangjiajun/tts-exp/results/pctalk/lips21_rms24_nine_20261031_attempt01/paired_scores.csv)
- [原始精度JSON](/Users/wangjiajun/tts-exp/results/pctalk/lips21_rms24_nine_20261031_attempt01/summary.json)
- [九项差值图PNG](/Users/wangjiajun/tts-exp/results/pctalk/lips21_rms24_nine_20261031_attempt01/paired_differences.png)
- [本机统计复核](/Users/wangjiajun/tts-exp/results/pctalk/lips21_rms24_nine_20261031_attempt01/local_statistics_audit.json)
- [本机原始端点复核](/Users/wangjiajun/tts-exp/results/pctalk/lips21_rms24_nine_20261031_attempt01/local_raw_audit.json)

evidence.tar.gz为531226799字节，SHA256 `5f461c6d9e58f0e7a897fc0b0f22c23ff277f6041a173c0f148fd3b9196571d8`；2925内容文件+manifest共2926成员全部读回校验。包括本轮归一化前16k输入、42归一化音轨/42视频、原始AAC、矩阵、嵌入、logits、控制、轨迹/裁剪和代码/环境锁；重建缓存及模型权重不打包，权重另有固定revision/SHA。terminal和本机复核/图/浏览页为封存包外的补充验收，local_source及manifest保存本机执行源码。模型MP4在results/pctalk/{natural_raw,tts_raw}/各来源目录，未声明网盘备份。

## Relations
- follows [[83-pctalk-chinese20-nine-scorers]]
- relates_to [[Chinese-LiPS 官方测试集 21人×5句]]
- relates_to [[PC-Talk Chinese-LiPS21配对九评分器2026-10-31]]
- relates_to [[SYNC-C-SCORES]]
