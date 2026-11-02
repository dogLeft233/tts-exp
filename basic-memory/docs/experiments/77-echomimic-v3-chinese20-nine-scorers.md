---
title: 77-echomimic-v3-chinese20-nine-scorers
type: experiment
permalink: tts-exp/docs/experiments/77-echomimic-v3-chinese20-nine-scorers
status: CONCLUDED
model: EchoMimicV3-Flash
cohort: Chinese20
n_sources: 20
logical_videos: 80
unique_audio_inputs: 41
normalization: RMS -24dBFS
science_scope: exploratory evaluator evidence; not perceptual ground truth
tags:
- tfg
- chinese20
- nine-scorers
- rms24
---

# 77 — EchoMimicV3-Flash 中文20人九评分器配对比较

## 结论与状态

✅ CONCLUDED：80条逻辑视频、九评分器、原始数值归约、归档内容与本地统计复算全部完成。在本次RMS−24dBFS受控输入条件下，没有确认稳定的TTS评分优势；同句参考12人这一主要分层、扩展参考8人以及全20人，九项端点均未通过各自Holm9校正。这不是两条件等价或感知同步相同的证明。

主要12人Sync-C：自然4.803、TTS4.577，差−0.226，95% CI[−0.681,+0.202]；全20人自然4.945、TTS4.755，差−0.191，95% CI[−0.600,+0.218]。全20人PEAVS偏自然，差−0.493687、边际CI不跨0，但sign-flip p=.007690、Holm9 p=.069209，未达到.05家族阈值；不能只根据边际CI判显著。

## 问题与冻结设计

比较同一批20位说话人的自然语音与既有云端TTS驱动EchoMimicV3-Flash后的九种评分。20N+60TTS，每来源三TTS先平均再与自然配对；20来源对应20位不同说话人。完全相同音频的draw不视作独立合成重复。保留80条逻辑媒体，41组唯一音频真正生成；输入WAV、图片、seed/设置相同的条目保留明确成片alias与SHA。

直接复制报告76已冻结的80条RMS−24dBFS PCM16/mono/16kHz WAV，逐文件SHA相同；没有重新合成、二次调增益、压缩或移动音轨。固定同一3.png（SHA f425463ba279fea88ac807d02bf048fd2f891d4efae1c8903eae3afc22d216c9）、512²、25fps、seed42。Flash使用实际发布的echomimicv3-flash-pro完整检查点，8步、Flow_Unipc、CFG6、AudioCFG3、shift5、BF16、model CPU offload、TeaCache0.1/skip5，每片重置generator/TeaCache。固定文本“A person is speaking.”，negative prompt为空，无额外人脸mask或参考视频。

这是受控RMS入口：绕过官方infer_flash中的−23LUFS步骤；实际中文wav2vec2配置do_normalize=False，特征提取器对传入float32前缀逐值不变。保留网络内部原生处理，不声称内部LN/GN不存在。该处理不同于作者默认LUFS入口，也不同于2025预览版英文wav2vec2配置；本结果不能称为作者默认配置的SOTA排名。

帧数沿原生floor(samples/640)后向下取4k+1，移除演示81帧上限，实际61–197帧。原始归一化音轨完整保留，生成条件使用有效前缀；未支持尾部最多0.154125秒，N平均0.078169秒、TTS平均0.064秒，并非全部尾部都静音。各评分器沿其既定原生长度处理；共同整帧/中心窗按实际视频支持锁定，未人为补帧、补静音或拉伸来凑五秒。

统计单位是来源：同句参考12为主要、扩展参考8及全20单列；每指标complete pairs，无插补。10000次来源bootstrap，100000次sign-flip敏感性，seed20261030；每分层九指标Holm校正。CI为边际区间，sign-flip依赖对称/可交换假设，不是随机分配TTS条件的因果检验。

## 同句参考12人（主要分层）

|评分器|配对人数|自然均值|TTS均值|TTS−自然|95%配对bootstrap CI|Holm9 p|
|---|---:|---:|---:|---:|---|---:|
|SyncNet|12|4.803|4.577|-0.226|[-0.681, +0.202]|1.00000|
|CoGenAV|12|0.413204|0.396451|-0.016753|[-0.059370, +0.023282]|1.00000|
|PEAVS|12|2.840677|2.304081|-0.536595|[-1.018064, -0.047248]|0.61118|
|SyncReward|12|1.525391|1.508247|-0.017144|[-0.058382, +0.027344]|1.00000|
|AVS_u|12|0.151624|0.166797|+0.015173|[-0.013677, +0.046129]|1.00000|
|MTDVocaLiST|12|0.459292|0.430450|-0.028843|[-0.075494, +0.015580]|1.00000|
|VocaLiST|12|0.459434|0.431859|-0.027575|[-0.073111, +0.013214]|1.00000|
|Synchformer|7|0.315559|0.598593|+0.283034|[-0.142178, +0.643397]|1.00000|
|SparseSync|7|0.172430|0.136755|-0.035675|[-0.154269, +0.081387]|1.00000|

## 全20人（补充分层）

|评分器|配对人数|自然均值|TTS均值|TTS−自然|95%配对bootstrap CI|Holm9 p|
|---|---:|---:|---:|---:|---|---:|
|SyncNet|20|4.945|4.755|-0.191|[-0.600, +0.218]|1.00000|
|CoGenAV|20|0.415552|0.423608|+0.008056|[-0.032512, +0.050160]|1.00000|
|PEAVS|20|2.682277|2.188589|-0.493687|[-0.809330, -0.175708]|0.06921|
|SyncReward|20|1.514453|1.495964|-0.018490|[-0.059766, +0.019141]|1.00000|
|AVS_u|20|0.163351|0.165286|+0.001934|[-0.016779, +0.022479]|1.00000|
|MTDVocaLiST|20|0.460591|0.440776|-0.019815|[-0.067391, +0.026525]|1.00000|
|VocaLiST|20|0.463782|0.436597|-0.027184|[-0.071422, +0.017157]|1.00000|
|Synchformer|7|0.315559|0.598593|+0.283034|[-0.142178, +0.643397]|1.00000|
|SparseSync|7|0.172430|0.136755|-0.035675|[-0.154269, +0.081387]|1.00000|

扩展参考8人Sync-C：自然5.159、TTS5.020、差−0.138，95% CI[−0.922,+0.669]。两固定五秒分类器在此分层无完整来源对；其余端点Holm9均不显著。完整三分层逐项表见[原始分析报告](../../../results/echomimic_v3_flash/chinese20_rms24_20261030_attempt01/report.md)。

## 支持集、控制与敏感性

七项各80/80逻辑媒体、20/20来源配对。Synchformer/SparseSync各33/80媒体、7/20完整来源对；20个唯一视频可测、21个唯一视频原生不足125完整帧。缺失不插补，也不合成九指标总分或投票。

AVS_u、MTDVocaLiST、VocaLiST真实±400ms平移各82/82下降；Synchformer40/40、SparseSync39/40下降，均通过既定80%诊断门槛。CoGenAV真实±160ms峰位移动82/82在一帧内，median peak-excess为正。SyncNet本队列未重测偏移控制；PEAVS/SyncReward也未在此队列重测，保留此前中文控制失败的边界。分数变化不能直接解释为嘴型或感知同步变化。

SyncReward保留官方window_segments内部短片尾帧/音频补齐，输入媒体未外部延长；原生视频与音频均完整覆盖4.75秒的完整配对子集为7人，差−0.031622、95% CI[−0.094494,+0.040551]，只作不同支持集的补充。

PEAVS主分析固定全80按文件名字典序一次batch64+16；用同一保留特征重放4×20，零新增特征前向。全20差−0.49368718（主−0.49368715），12/8分层方向与数值也一致；不能把未经校正p值升级为确认性结论。

## 工程验收与资源

当次读取飞书1048手册，使用wangjiajun个人/home/data2/wangjiajun；代码、环境、权重、缓存、输入与输出均在个人区。下载前有限检查公共模型目录；共享Anaconda仅用于创建个人独立克隆环境。单A6000 UUID GPU-de7ff5ff-3510-739f-917c-cb98892de759，CPU affinity4逻辑核、Torch/OMP4，GPU阶段顺序执行，nohup运行。五个原评分环境pip-freeze集合与报告76冻结记录完全一致，没有改共享base、驱动或他人任务。

19项资产按官方HF revision绑定；大文件核对SHA256，小文件核对Git blob。Flash严格全键加载；不下载随后会被全部覆盖的基础DiT。上游T5低内存加载与Diffusers0.35.2不兼容，会回退FP32；初始诊断中止并保留旧worker、日志/显存及修复记录，未采用其未完成产物。个人包装器改为CPU加载后显式BF16并断言四核心模块dtype，再完成全部41次有效生成。

有效生成NVIDIA驻留显存5秒采样峰值15214MiB（约14.86GiB），PyTorch allocated峰值14238921216B、reserved15575547904B；22GiB为本次allocator上限。实际硬件是A6000，未声称在物理24GB卡上另跑一遍，也不保证任意长度/分辨率。41唯一输入生成计时累计2601.69秒（不含权重加载）。80条PCM逐样点一致、512²/25fps/帧数全部通过，最小平均帧差1.08039；首对N/T九帧联系表已人工查看，属于工程QA，不是盲评。

服务器七种原始矩阵/embedding/logit归约复核通过，SyncNet独立FP64最大差1.8521e−6。PEAVS/SyncReward是实际权重前向与输出身份/覆盖/有限值核查，不冒称独立模型重实现。3项服务器测试通过；本机普通Python缺Torch的那项测试未在本机通过，目标服务器三项全通过。12个本地实验脚本编译通过。PNG图已实际查看。

本地287333571字节证据包、2376内容文件、80标准播放视频哈希读回PASS；25个非空分层的来源均值、10000bootstrap、100000sign-flip、Holm9及胜负数独立复算最大差0。实验相关GPU进程已退出，环境与权重保留供复用。

## 产物与复现入口

- [本地结果目录](../../../results/echomimic_v3_flash/chinese20_rms24_20261030_attempt01/)：report.md、summary.json、逐视频/来源配对CSV、video_index.json、PNG/PDF及evidence/。
- [九项配对差图](../../../results/echomimic_v3_flash/chinese20_rms24_20261030_attempt01/paired_differences.png)；区间为边际95%bootstrap。
- [逐来源配对数据](../../../results/echomimic_v3_flash/chinese20_rms24_20261030_attempt01/paired_scores.csv)；[逐视频数据](../../../results/echomimic_v3_flash/chinese20_rms24_20261030_attempt01/per_video_scores.csv)。
- 远端run：/home/data2/wangjiajun/av-evaluation/jobs/echomimic_v3_flash_chinese20_rms24_20261030_attempt01。
- [代码](../../../scripts/experiments/echomimic_v3_chinese20_20261030/README.md)，远端副本在av-evaluation/experiments/同名目录。已完成run保持只读；新实验使用新attempt目录，不直接重跑原orchestrator覆盖记录。
- 标准AAC播放媒体位于results/echomimic_v3_flash/{natural_raw,tts_raw}/<source>/；无损ALAC评分媒体和确切WAV保存在evidence/videos与evidence/audio。
- evidence.tar.gz SHA256：ca8cd17cb6974016e69e3470c9cd01fb9927bb2d9d0b8eefca2f51ff3f09380b；2377成员含manifest，manifest SHA256 a43c66ab44dc716ec421cbaff3067dbaf2765826ede04d45d6f9a7c92a2f5b05。
- [官方代码pin](https://github.com/antgroup/echomimic_v3/tree/7e89489ca51c0d008fc1963ec6c03fc5bd0b9397)，全部权重revision/字节/哈希在download_manifest及receipt中；评分来源和环境快照随包保留。

## Observations

- [result] 同一静态图和RMS−24dBFS中文20来源，80逻辑视频/41唯一生成与九评分器已完成；25统计分层本地复算差0。
- [result] 主要12人和全20人的九端点Holm9均不显著，未确认本条件下TTS评分优势。
- [scope] 受控入口绕过作者−23LUFS；中文do_normalize=False，原生4k+1留下最多154ms未支持尾部；不推广为默认模型或感知同步结论。
- [verification] 有效推理驻留显存采样峰值约14.86GiB；全部输入/媒体/原始归约/归档与本地统计读回通过。

## Relations

- compares_with [[76-flashhead-chinese20-nine-scorers]]
- relates_to [[2025年TFG的响度与输入尺度处理]]

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| Complete controlled-RMS EchoMimicV3-Flash Chinese20 nine-scorer experiment, immutable evidence and independent local readback | October 30, 2026 | user |
