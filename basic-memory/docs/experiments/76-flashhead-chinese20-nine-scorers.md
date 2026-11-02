---
title: 76-flashhead-chinese20-nine-scorers
type: experiment
permalink: tts-exp/docs/experiments/76-flashhead-chinese20-nine-scorers
status: CONCLUDED
run_id: flashhead_chinese20_rms24_20261030_attempt01
tags:
- FlashHead
- paired-comparison
- nine-scorers
---

# 76 — FlashHead Pro 中文20人自然/TTS九评分器配对实验

✅ CONCLUDED（评价器分数的描述性比较），2026-10-30。用户要求完整中文20人实验，并明确将音量归一化到可听水平。

## 主要结论

本条件下没有确认TTS带来稳定的跨评分器提升。主要同句克隆参考12人中，九项均未通过Holm9校正；Sync-C差值−0.065，95% CI [−0.725,+0.567]。合计20人Sync-C自然6.143、TTS6.355、差值+0.212，95% CI [−0.399,+0.873]，10/20来源提高，Holm9 p=1.000。

合计20人中仅PEAVS在Holm9后偏向自然：TTS−自然−0.452971，95% CI [−0.724357,−0.182919]，Holm9 p=0.04131；其batch20敏感性差值−0.452971，与主64+16批次接近。PEAVS和SyncReward本队列未重做移音控制，旧中文控制失败边界保留，因此这些均为评价器证据。六项实际控制通过，不能据此把所有指标等同感知真值。

两分类器受原生5秒支持限制，均测33/80条、形成7人完整配对；其余七项均覆盖80条、20人完整配对。SyncReward保留官方模型内部短片片段补齐，另报四条原音频均完整覆盖4.75秒的8人子集，差值+0.031901、CI跨零。

## 固定输入与实施边界

20N+60既有云端TTS，来源20人；同人三T先平均，扩展参考8人与同句参考12人分层。41种唯一规范派生WAV；相同音频、图片和seed/settings的重复条目复制同一视频并保留alias，41次真实生成、80条逻辑媒体，不将重复draw当独立人。

统一RMS−24dBFS，相对旧规范电平增加约4.477dB；所有80条实测RMS [−24.000007666,−23.999993547]dBFS，最大峰值0.857666，无削波。原件保留，新N/T均重生成重评分，不拼接旧低增益自然分数。固定3.png、Pro/stream、seed42、512×512、25fps。原生生成的末chunk处理保持官方实现；评分不额外补时长或校正偏移。评分ALAC音轨与归一化驱动PCM逐样点相同，播放副本为原生成AAC。

SyncNet沿用旧中文PCM16 MKV、min_track25、官方FP32原始矩阵归约；MTD/VocaLiST按原始音频与视频共同完整帧支持。新Torch2.7/A6000、SyncReward BF16与旧Torch2.5/V100/FP32不宣称逐位一致，本次N/T在同一新运行时配对。

## 完整配对表与敏感性分析

状态：完整运行结束；以下是评价器分数比较，不是感知同步或嘴型准确度结论。
固定图片3.png、Pro/stream、seed42。全部输入单声道16k PCM16、RMS −24dBFS；评分ALAC音轨逐样点等于生成驱动音频。
每人3次既有TTS先取均值，再与该人自然语音配对；完全相同音频/视频的重复项保留明确复用凭据。英文未纳入。
12人同句克隆参考为主要分层，8人扩展参考单列；短输入不补静音，各指标实际配对人数在表中。差值为TTS−自然，正值只表示该评分器给TTS更高分。

## same_reference12

|评分器|配对人数|自然均值|TTS均值|差值|95% CI|TTS较高|Holm9 p|
|---|---:|---:|---:|---:|---|---:|---:|
|SyncNet|12|5.883|5.818|-0.065|[-0.725, +0.567]|6/12|1.00000|
|CoGenAV|12|0.457462|0.447508|-0.009954|[-0.053601, +0.037265]|3/12|1.00000|
|PEAVS|12|2.791043|2.240767|-0.550276|[-0.934022, -0.154259]|1/12|0.23454|
|SyncReward|12|1.571615|1.590929|+0.019314|[-0.028646, +0.071398]|7/12|1.00000|
|AVS_u|12|0.129406|0.140083|+0.010678|[-0.010178, +0.030741]|8/12|1.00000|
|MTDVocaLiST|12|0.573185|0.657524|+0.084338|[-0.014165, +0.177895]|8/12|0.94702|
|VocaLiST|12|0.574553|0.656328|+0.081776|[-0.005891, +0.168109]|7/12|0.90719|
|Synchformer|7|0.131429|0.0678701|-0.0635594|[-0.276055, +0.0758442]|4/7|1.00000|
|SparseSync|7|0.00276146|0.00799956|+0.00523809|[-0.00480825, +0.0215398]|5/7|1.00000|

## extended_reference8

|评分器|配对人数|自然均值|TTS均值|差值|95% CI|TTS较高|Holm9 p|
|---|---:|---:|---:|---:|---|---:|---:|
|SyncNet|8|6.533|7.161|+0.628|[-0.541, +1.849]|4/8|1.00000|
|CoGenAV|8|0.477065|0.467792|-0.009272|[-0.085130, +0.077307]|4/8|1.00000|
|PEAVS|8|2.303699|1.996685|-0.307013|[-0.602569, -0.040481]|3/8|0.84779|
|SyncReward|8|1.570312|1.585938|+0.015625|[-0.051758, +0.075195]|6/8|1.00000|
|AVS_u|8|0.142790|0.145869|+0.003078|[-0.025149, +0.027818]|6/8|1.00000|
|MTDVocaLiST|8|0.697389|0.723086|+0.025697|[-0.100681, +0.155199]|4/8|1.00000|
|VocaLiST|8|0.691326|0.733192|+0.041865|[-0.115742, +0.193319]|4/8|1.00000|
|Synchformer|0|—|—|—|不可测|—|—|
|SparseSync|0|—|—|—|不可测|—|—|

## all20

|评分器|配对人数|自然均值|TTS均值|差值|95% CI|TTS较高|Holm9 p|
|---|---:|---:|---:|---:|---|---:|---:|
|SyncNet|20|6.143|6.355|+0.212|[-0.399, +0.873]|10/20|1.00000|
|CoGenAV|20|0.465303|0.455622|-0.009681|[-0.050193, +0.034510]|7/20|1.00000|
|PEAVS|20|2.596106|2.143135|-0.452971|[-0.724357, -0.182919]|4/20|0.04131|
|SyncReward|20|1.571094|1.588932|+0.017839|[-0.021097, +0.056257]|13/20|1.00000|
|AVS_u|20|0.134759|0.142397|+0.007638|[-0.009104, +0.023725]|14/20|1.00000|
|MTDVocaLiST|20|0.622867|0.683748|+0.060882|[-0.019356, +0.140711]|12/20|1.00000|
|VocaLiST|20|0.621262|0.687074|+0.065811|[-0.017099, +0.147635]|11/20|1.00000|
|Synchformer|7|0.131429|0.0678701|-0.0635594|[-0.276055, +0.0758442]|4/7|1.00000|
|SparseSync|7|0.00276146|0.00799956|+0.00523809|[-0.00480825, +0.0215398]|5/7|1.00000|

## 适用边界

- Chinese cloning references differ: primary12 same utterance, secondary8 extended references.
- Some TTS draws are identical, not independent synthesis replicates.
- Native short inputs are unmeasurable for the two fixed5s classifiers; complete pairs vary by metric.
- SyncReward uses unchanged official window_segments including internal short-clip tail repetition/audio padding; no input media padding. Native4.75s sensitivity is separate.
- RMS -24dBFS is a new gain condition; do not splice historical lower-gain N scores.
- Scores are evaluator evidence, not perceptual or visual ground truth; no combined total or vote.
- Sign-flip sensitivity assumes symmetry/exchangeability, not randomized TTS allocation. CI are marginal; Holm9 p is separate.

## 实际测量与控制

- SyncNet：80/80媒体，20/20完整配对；控制状态 NOT_RETESTED_THIS_COHORT。
- CoGenAV：80/80媒体，20/20完整配对；控制状态 PASS。
- PEAVS：80/80媒体，20/20完整配对；控制状态 NOT_RETESTED_THIS_COHORT。
- SyncReward：80/80媒体，20/20完整配对；控制状态 NOT_RETESTED_THIS_COHORT。
- AVS_u：80/80媒体，20/20完整配对；控制状态 PASS。
- MTDVocaLiST：80/80媒体，20/20完整配对；控制状态 PASS。
- VocaLiST：80/80媒体，20/20完整配对；控制状态 PASS。
- Synchformer：33/80媒体，7/20完整配对；控制状态 PASS。
- SparseSync：33/80媒体，7/20完整配对；控制状态 PASS。

## SyncReward 原生完整4.75秒子集

主表沿用官方短片内部补齐。下列只保留四条原始音频均覆盖4.75秒的完整来源，队列与主表不同，属于补充敏感性分析。

配对人数 8；自然均值 1.552734，TTS均值 1.584635，差值 +0.031901，95% CI [-0.038745, +0.104175]。

## PEAVS 批次敏感性

主分析固定官方batch64+16；旧中文脚本曾按20条分批。使用同一批已保留的特征，仅重放官方metric为batch20的4批，没有新增特征前向，原主分数不改。

|分层|人数|主64+16差值|补充20×4差值|补充95% CI|
|---|---:|---:|---:|---|
|all20|20|-0.452971|-0.452971|[-0.724357, -0.182919]|
|same_reference12|12|-0.550276|-0.550276|[-0.934022, -0.154259]|
|extended_reference8|8|-0.307013|-0.307013|[-0.602570, -0.040481]|

## 可复现性与验证

远端根：/home/data2/wangjiajun/av-evaluation/jobs/flashhead_chinese20_rms24_20261030_attempt01。
本地交付：results/soulx_flashhead_pro/chinese20_rms24_20261030_attempt01/。
原始评分、矩阵/embeddings/logits、PEAVS特征、音频、80条视频、代码/环境快照及日志在evidence/。逐视频per_video_scores.csv、逐人paired_scores.csv、summary.json、report.md和video_index.json分别提供数值与标准视频位置。图为paired_differences.png/.pdf，已实际查看。

证据包309272893字节，SHA256 `7cf9916be529e8f0f4099bd346a4cae168bb590bd254af9266cfee213ee7a700`；包含2235归档成员，本地逐一核对2234内容文件+manifest。80条视频导出到results/soulx_flashhead_pro/{natural_raw,tts_raw}/<source>/，文件名rms24_20261030_<id>.mp4。local_readback.json=PASS。

independent_audit.json=PASS：80条规范原件/派生/视频SHA、RMS和PCM，以及七种原始归约独立核验；SyncNet FP64交叉检查最大绝对差2.4485e−6。PEAVS/SyncReward保留实际模型输出并核对身份、完整性和有限值，不宣称独立重实现。local_statistics_audit.json=PASS：25个非空指标分层的来源均值、bootstrap10000、signflip100000、Holm9和胜负计数全部复算，最大差0。4个针对归一化安全、时序、三draw来源配对与Holm的测试通过。

源码为scripts/experiments/flashhead_chinese20_20261030/（11个Python文件+README），对应tests/test_flashhead_chinese20_20261030.py。新增文件未提交；其他1353项工作区状态包含既有修改，不清理或回滚。未创建PR。

服务器按server-192 skill当次读取1048手册，个人目录存储、nohup、单GPU UUID、torch显存60%、CPU4线程、顺序GPU阶段；没有修改共享base、驱动或他人任务。生成峰值6.146GiB。最终本人推理/后处理进程已退出，GPU任务清单无本人PID。全局192 skill参考补充SyncNet CLI差异、官方短片处理、确定输入复用和派生音量凭据，quick_validate=PASS。

## Observations
- [result] FlashHead中文20人80媒体与九评分器全部完成；两个5秒分类器有效配对7人，其余20人，原始证据与本地复算PASS。
- [result] 主要同句参考12人未确认九项校正后效应；全20人Sync-C+0.212，PEAVS偏自然的Holm9 p=0.04131。
- [decision] 报告来源级配对与原生支持，保留参考差异、重复TTS、内部padding及批次敏感性边界，不做总分或感知结论。
- [next_step] 复核时从本报告、summary.json与不可变证据包进入；无需重新部署或重跑完整队列。

## Relations
- uses [[server192-nine-scorers]]
- documents [[FlashHead Pro 中文20人可听音量九评分器配对实验]]

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| Complete Chinese20 FlashHead audible-input experiment, nine scorers and independent readback | October 30, 2026 | user |
