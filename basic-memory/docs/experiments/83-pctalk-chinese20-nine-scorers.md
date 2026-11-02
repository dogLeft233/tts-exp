---
title: 83-pctalk-chinese20-nine-scorers
type: experiment
permalink: tts-exp/docs/experiments/83-pctalk-chinese20-nine-scorers
status: concluded
date: '2026-10-31'
---

# 83 — PC-Talk中文20来源九评分器续评

✅ CONCLUDED（探索性评价器证据），2026-10-31。用户在报告81的Sync-C检查点之后授权其余八项；本轮0新TFG/0新SyncNet/0云TTS/ASR/训练，父视频与SyncNet字节复用。HDTF TalkLikeYou仍保持用户暂停。

主要同句参考12：九端点均未通过Holm9 .05门限；未检出不等于零效应或等效。

补充全20：通过九项校正的是CoGenAV（Δ-0.0208175，Holm9 p=0.027930）、PEAVS（Δ-0.286525，Holm9 p=0.049200）、MTDVocaLiST（Δ-0.0249899，Holm9 p=0.010080）、VocaLiST（Δ-0.0215031，Holm9 p=0.014640）。

事后CPU敏感性：对所有非空端点分层枚举全部sign patterns并重算Holm9，25项精确p另用独立meet-in-the-middle加二分计数逐值复核通过。主要12仍无通过项；全20仍为CoGenAV/PEAVS/MTD/Voca四项负差通过，判定全部不变。PEAVS精确Holm9 p=.046143（主100k Monte Carlo .049200）；冻结主结果不替换。文件exact_signflip_sensitivity.json与exact_independent_audit.json在封存包外，属于事后CPU敏感性，不是新网络评分。

## 固定流程与支持

20来源，20自然+60既有云Qwen TTS逻辑媒体，41唯一音视频；三TTS在来源内平均再与该来源自然配对。主要12为同句克隆参考，另8扩展参考单列。固定3.png、RMS−24dBFS、mono16kPCM16、原生25fps512²、PC-Talk preset192/lip1/无EMC/face-index1/seed42。原生画面57–193帧，少约194–237ms视觉尾部，原始驱动PCM不人为补齐/移音/变速。无按分调参。

父run `pctalk_chinese20_rms24_20261031_attempt01` 与报告81保留不变。新run `pctalk_chinese20_rms24_nine_20261031_attempt01` 在执行前逐SHA复制80WAV/媒体、生成凭据和41SyncNet分数/距离矩阵；reuse_receipt.json记录0新生成/0SyncNet前向，另拷贝父已完成track/crop作为证据，无重新检测。

CoGenAV Eq1、AVS_u帧cosine；MTD/VocaLiST官方K5；Synchformer/SparseSync原生整数中心5秒21类P0；SyncReward BF16/encode_batch1，保留官方window_segments短片内部补齐；PEAVS全部80字典序batch64+16。其余逐片评分器对完全相同媒体使用显式alias，PEAVS保持原定批次完整组成。26共享源码与5个环境执行前后与Flash76/Echo77冻结记录匹配；评分体/权重/门限未为PC-Talk调节。

CoGenAV/AVS_u沿用原95%脸检门限，预先配置仅将该门限不足或无检测记UNMEASURABLE，其余异常为FAILED，不重抽或放宽门槛。两分类器执行前按实际原片帧数/音频样点冻结28媒体/6完整来源（都在主要12），不按得分选择。各端点来源需N+三T全部可测，无插补。

每分层10k来源bootstrap、100k双侧sign-flip（seed20261030），九端点含复用SyncNet共同Holm9；sign-flip依赖对称/可交换假设，不是随机分配TTS的因果检验。边际CI与校正p不同，不根据CI判九项显著。K5/P0机器表原生0–1，百分比显示时差为百分点。

## same_reference12

|评分器|配对人数|自然均值|TTS均值|差值|95% CI|TTS较高|Holm9 p|
|---|---:|---:|---:|---:|---|---:|---:|
|SyncNet|12|7.182|7.639|+0.457|[+0.020, +0.855]|9/12|0.38185|
|CoGenAV|12|0.344022|0.328896|-0.015126|[-0.028510, -0.001958]|4/12|0.38185|
|PEAVS|12|2.142062|1.881253|-0.260809|[-0.519014, -0.007675]|4/12|0.38185|
|SyncReward|12|1.659505|1.654080|-0.005425|[-0.030165, +0.020182]|4/12|1.00000|
|AVS_u|12|0.142806|0.136831|-0.005975|[-0.016706, +0.004817]|4/12|0.96365|
|MTDVocaLiST|12|0.073883|0.055251|-0.018632|[-0.032072, -0.004405]|3/12|0.27522|
|VocaLiST|12|0.080659|0.066686|-0.013973|[-0.027160, -0.001941]|4/12|0.38185|
|Synchformer|6|0.982475|0.996883|+0.0144076|[+0.00192764, +0.0294419]|6/6|0.27522|
|SparseSync|6|0.998793|0.991516|-0.00727692|[-0.0240187, +0.00153981]|3/6|1.00000|

## extended_reference8

|评分器|配对人数|自然均值|TTS均值|差值|95% CI|TTS较高|Holm9 p|
|---|---:|---:|---:|---:|---|---:|---:|
|SyncNet|8|7.393|7.932|+0.539|[-0.313, +1.316]|5/8|1.00000|
|CoGenAV|8|0.372188|0.342833|-0.029355|[-0.047713, -0.007311]|2/8|0.33187|
|PEAVS|8|1.943652|1.618554|-0.325098|[-0.578810, -0.091996]|2/8|0.33187|
|SyncReward|8|1.632812|1.677734|+0.044922|[+0.007812, +0.083008]|4/8|0.62819|
|AVS_u|8|0.154415|0.151910|-0.002505|[-0.022987, +0.016075]|4/8|1.00000|
|MTDVocaLiST|8|0.076393|0.041867|-0.034526|[-0.052627, -0.012710]|1/8|0.28332|
|VocaLiST|8|0.082032|0.049234|-0.032798|[-0.053581, -0.012924]|1/8|0.28332|
|Synchformer|0|—|—|—|不可测|—|—|
|SparseSync|0|—|—|—|不可测|—|—|

## all20

|评分器|配对人数|自然均值|TTS均值|差值|95% CI|TTS较高|Holm9 p|
|---|---:|---:|---:|---:|---|---:|---:|
|SyncNet|20|7.266|7.756|+0.490|[+0.074, +0.884]|14/20|0.15475|
|CoGenAV|20|0.355288|0.334471|-0.020818|[-0.032254, -0.008664]|6/20|0.02793|
|PEAVS|20|2.062698|1.776173|-0.286525|[-0.471130, -0.104405]|6/20|0.04920|
|SyncReward|20|1.648828|1.663542|+0.014714|[-0.008854, +0.039062]|8/20|0.80420|
|AVS_u|20|0.147449|0.142863|-0.004587|[-0.015037, +0.005492]|8/20|0.80455|
|MTDVocaLiST|20|0.074887|0.049897|-0.024990|[-0.036647, -0.012349]|4/20|0.01008|
|VocaLiST|20|0.081208|0.059705|-0.021503|[-0.033479, -0.009613]|5/20|0.01464|
|Synchformer|6|0.982475|0.996883|+0.0144076|[+0.00192764, +0.0294419]|6/6|0.15475|
|SparseSync|6|0.998793|0.991516|-0.00727692|[-0.0240187, +0.00153981]|3/6|0.93740|

## 适用边界

- Chinese cloning references differ: primary12 same utterance, secondary8 extended references.
- Some TTS draws are identical, not independent synthesis replicates.
- PC-Talk native mel windows omit approximately0.2s of final visual support; full audio unchanged. Official PC-Talk mel/encoder/motion processing retained; no time shifts, external padding or posthoc offset correction.
- SyncNet scores reused byte-for-byte from completed parent checkpoint; official internal MP3/69ms first-pair diagnostic retained, not modified.
- Native face support below unchanged95percent is UNMEASURABLE for CoGenAV/AVS_u; complete source requires all four media.
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
- MTDVocaLiST：80/80媒体，20/20完整配对；控制状态 CONTROL_FAILED。
- VocaLiST：80/80媒体，20/20完整配对；控制状态 PASS。
- Synchformer：28/80媒体，6/20完整配对；控制状态 PASS。
- SparseSync：28/80媒体，6/20完整配对；控制状态 PASS。

## 控制解释的限制

MTD的±400ms控制只有65/82降分（79.27%），低于冻结80%门限（至少66），标CONTROL_FAILED，不能把其负差升级为同步质量退化。VocaLiST67/82、AVS_u79/82、Synchformer/SparseSync各34/34降分通过；CoGenAV峰位移动82/82在一帧内且median peak-excess为正。控制通过仍不是人类感知真值。

SyncNet、PEAVS和SyncReward本队列未重测移音控制；PEAVS/SyncReward沿用既往中文控制响应不足边界。PEAVS还依赖输入长度/填充和批次组成，本轮只解释固定80媒体/64+16结果，不能外推其他批次。四个补充全20负差都是原生评分器证据，不是四票证明TTS嘴型更差。

## SyncNet与官方基准边界

报告81自然7.266/TTS7.756/Δ+.490全部字节复用；本报告新增的是完整九项Holm，不能回写父检查点。官方PC-Talk I2V HDTF9.370/MEAD8.190，本轮中文/固定脸/片长/RMS与风格配置不同，不能当同测试集复现误差。官方SyncNet pipeline内部MP3，首对+69.0625ms、首N145→147帧、41最佳offset−2的边界保留；无本队列去MP3对照。报告82对TLY中文的100/25和AAC/PCM控制是另模型固定视频，且两分支都经历MP3路径，不能当PC-Talk已测的协议控制或无MP3消融。

## 独立核验与交付

服务器independent_audit.json对80媒体和原PCM绑定、SyncNet距离、CoGenAV/AVS_u原embedding、MTD/Voca距离与K5、两分类器完整logit/CPU FP32 softmax逐值重放和独立FP64归约通过。PEAVS/SyncReward核对原生网络输出身份/覆盖/有限值，不声称独立重实现网络。

本机2339证据内容文件逐大小/SHA通过，九原生端点从数组/原输出解析后独立映射80媒体、重组来源配对，复用SyncNet文件SHA与父完全相同；25非空端点分层的均值/10000bootstrap/100000sign-flip/Holm9及胜数独立重算PASS，最大差0.0。父80视频仍可从报告81的HTML索引播放，本轮生成0新视频。4项针对性测试PASS、compileall。

新交付 `results/pctalk/chinese20_rms24_nine_20261031_attempt01/`：summary.json、report.md、per_video_scores.csv、paired_scores.csv、paired_differences.png/PDF、local_raw_audit.json、local_statistics_audit.json与terminal_receipt.json。证据包243894259B，SHA256 `38645c29d26ba5c373ce64c53e7d730f569308a6a90ebdd10020f8023b7e3816`；内容manifest含源码/环境快照、原始embedding/距离/logit/控制/PEAVS特征与80输入媒体。完成的SyncNet父裁剪/轨迹从父拷贝，未新增网络前向。没有网盘归档声明。

本轮先读取当次飞书1048手册，代码/数据/cache/tmp均个人data2；nohup、单A6000 UUID727、CPU线程4/affinity0–3、Torch60%上限，GPU阶段顺序；未改共享base、驱动或他人任务。模型加载GPU资格记录见各stage原生凭据，MediaPipe辅助EGL上下文不等同主Torch推理设备。所有本实验controller/finish PID退出，终局GPU状态：2 MiB, 0 %。归档内RUNNING类为打包时快照，terminal_receipt权威，不自动重跑。

SyncReward首次包装启动因缺GPU_UUID环境变量，在Python/模型加载前即退出（0新前向），旧失败JSON/stage/log/PID完整保留failed_launch01。新增私有controller_launch02.py只在子进程环境加入GPU_UUID=GPU，保持已完成七项；715科学文件含输入绑定在恢复前后SHA未变（resume02_preservation_audit.json）。原controller与共享评分器源码未改，未重跑已有科学单元。

源码`scripts/experiments/pctalk_chinese20_nine_20261031`，测试`tests/test_pctalk_chinese20_nine_20261031.py`；新增未提交，继承dirty未动。

## Relations
- follows_up [[81-pctalk-chinese20-syncnet-official-comparison]]
- completes [[PC-Talk中文20来源其余八评分器2026-10-31]]
- uses [[PC-Talk 192服务器部署]]
- compares_with [[79-talklikeyou-chinese20-nine-scorers]]
- relates_to [[82-talklikeyou-syncnet-protocol-controls]]
