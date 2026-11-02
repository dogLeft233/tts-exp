---
title: 72-sparsesync-official-native-cloud-preference
type: experiment
permalink: tts-exp/docs/experiments/72-sparsesync-official-native-cloud-preference
status: concluded
scientific_status: SCORER_PREFERENCE_ONLY_CEILING
date: 2026-10-28
model: SparseSync
cohort: Mandarin cloud native 20 sources; official 5s subset 18 pairs / 6 sources
tags:
- experiment
- SparseSync
- cloud-tts
- syncnet
- official-parity
---

# 72-sparsesync-official-native-cloud-preference

SparseSync ✅ 已部署，官方示例一致性、物理时移控制、原生评分和独立审计已完成。主要发现是严重的零时移概率饱和：它能检测本实验的±400ms扰动，但不能据此可靠排序原生视频之间的细小同步差异。

| 修改日期 | 修改 | 授权 |
|---|---|---|
| 2026-10-28 | 初次报告：官方5秒子集、原生偏好、时移控制、饱和及精度诊断 | 用户要求部署评分，评分前确认官方一致 |

## 问题与冻结设计

研究TTS驱动视频的Sync-C优势是否也被不同评分器认可。使用此前中文云TTS/自然视频原生音轨，沿用已有RMS处理，不重新合成、不交叉音轨、不拉伸、不搜索或校正最佳时移。自然视频配自然音频，云TTS视频配各自TTS音频。

官方LRS3 NoFaceCrop 21类权重22-07-13T22-25-49，时移类别−2至+2秒，步长200ms。保持官方test transform的5秒输入。原80媒体中28足够5秒（自然10、TTS18）；52短媒体排除。完整自然/TTS配对18，独立来源6；每个来源的3次TTS作为重复测量，不能当18个独立来源。此子集全部属于此前strict12参考组，因此本run strict12统计与主统计相同。视频取与此前Synchformer相同的帧网格中心5秒。

主端点为官方softmax输出的零时移概率P0，越高越好；次端点为负概率加权绝对类别时移。两者是项目预先固定的连续比较端点，**并非作者数据集级分类accuracy**。保留全部21类logits和概率。冲突复用相同视频ID的此前整段Sync-C，因此评价时间支持不同：SparseSync只看中心5秒，Sync-C为原整段。

协议在任何模型结果产生前写入protocol.json。统计为严格符号，平局单列；配对T−N差，来源聚类bootstrap 10000次、seed20261028。控制预先设定P0在至少80%的±400ms物理扰动下降，失败不调端点。

## 官方调用一致性

源码官方commit [a5bee8a047c0ebeec66b18d4ddf4c5f0ef098a4f](https://github.com/v-iashin/SparseSync/tree/a5bee8a047c0ebeec66b18d4ddf4c5f0ef098a4f)，31个源码/配置文件按GitHub tree逐个验证git blob。无官方源码改动。完整同步checkpoint653037857字节，strict=True加载54,348,652参数。仅把两个特征提取器初始化ckpt_path设None，随后由完整checkpoint覆盖全部权重，避免额外下载初始化模型。

28媒体全部使用作者get_video_and_audio、get_transforms(cfg)['test']和prepare_inputs。官方读取器从此前float PCM MOV读出的16k音频与原WAV逐样本完全相同；视频RGB、中心空间224裁剪、ImageNet归一化，音频n_fft512/hop128对数谱及逐频率标准化均由作者代码执行。输入video[1,125,3,224,224]、audio[1,1,257,626]。

**评分前真实执行作者scripts/example.py main**：仅省略输入重建可视化，保留原Recorder、预处理、strict权重重载、forward和decode函数，捕获结果与批量直接调用对照。第一条自然样本输入音频/视频最大差均0，全部logits最大差0，全部概率最大差5.55e−17。该门控先于控制与原生评分。FP32遵循example.py；当前PyTorch2.5.1/PyAV12.3兼容运行环境，不宣称2022锁版本逐位重现。

## 物理时移控制

10条可用自然固定视频，各自全PCM零填充平移−6400/+6400样本，即±400ms，视频不动，仍用同一5秒裁剪。20/20的P0低于原生，通过预设门槛；20/20预测相对时移准确追踪±0.4秒，最大浮点误差约5.96e−9秒，没有利用偏好结果调校。

这是大时移检测验证，不能自动验证原生饱和区间的细粒度评分质量；边缘零填充也应视作本控制的一部分。

## 原生结果

以下自然均值按18个配对重复对应自然基线，而非10条自然视频的无配对均值。

| 端点 | 自然均值 | TTS均值 | 平均T−N | TTS胜 | 自然胜 | 平局 | 与Sync-C冲突 |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0（概率） | 0.999995212 | 0.999902099 | −0.0000931124 | 6/18=33.3% | 12/18=66.7% | 0 | 12/18=66.7% |
| 负期望绝对类别时移（秒） | −0.00000180388 | −0.00008807308 | −0.00008626921 | 6/18=33.3% | 12/18=66.7% | 0 | 12/18=66.7% |
| 既有Sync-C（整段） | — | — | +1.028 | 18/18=100% | 0 | 0 | — |

P0换成百分比：自然99.9995212%，TTS99.9902099%，平均T−N=**−0.00931124个百分点**。来源bootstrap 95%区间为[−0.02314945,+0.00018742]个百分点，跨0。12个冲突全部为Sync-C认为TTS更高、SparseSync P0认为自然更高，反方向0。

次端点区间[−0.0002246172,−0.0000006596]秒。它是概率分布加权的类别距离，**不能解释成真实嘴型时移只差0.086ms**，200ms分类网格并不提供亚毫秒时间定位能力。

| 来源 | P0平均T−N（百分点） | 平均ΔSync-C |
|---|---:|---:|
| BAC009S0003W0121 | +0.00087023 | +1.538 |
| BAC009S0015W0121 | −0.00012318 | +0.650 |
| BAC009S0016W0121 | −0.00048876 | +0.826 |
| BAC009S0023W0121 | −0.04132390 | +1.550 |
| BAC009S0025W0122 | −0.01492103 | +0.844 |
| BAC009S0031W0121 | +0.00011921 | +0.758 |

平均负差主要来自后两条较低概率来源，所有来源的绝对P0仍接近1。

## 饱和与证据边界

这是本次最重要的限制：自然10/10、TTS18/18的argmax都为零时移；**全部28个P0均大于99.95%**。若仅比较作者输出的最佳时移类别，18对完全相同，无法区分自然与TTS。33.3%胜率和66.7%冲突来自接近1的连续置信度细分，不能夸大为12对存在明显视觉差异。

事后用同一保存logits独立float64 softmax复算，18/18配对方向与正式FP32 P0一致，最小绝对FP32配对差1.07288e−6，因此本次方向不是简单的FP32舍入翻转。不过精度稳定不等于感知有效，置信度饱和、仅6来源、英语LRS3训练/中文应用和不同评价窗口仍限制解释。此处未新增GPU前向，未修改正式端点。

因此SparseSync提供一条线索：**粗粒度同步分类在这批原生视频上几乎全部通过，而Sync-C仍能拉开明显差距**。这支持继续研究评分任务和表征差异；不能据此判定SyncNet存在偏置，也不能断言自然口型更好。SparseSync与Synchformer同一研究路线，不视作完全独立真值。

## 复现与审计

Ubuntu根 /home/wjj/tts-audio/tts-exp；Windows根 E:/Documents/tts-exp。正式run：runs/sparsesync_official_cloud_rescore_20261028_attempt01。代码scripts/experiments/sparsesync_official_cloud_rescore_20261028.py，测试tests/test_sparsesync_official_cloud_rescore_20261028.py。

关键文件：protocol.json、inputs.json、sync_pairs.json、model_load.json、preprocessing_validation.json、official_parity.json、official_example_stdout.txt、known_shift_controls.json、scores.json、paired_scores.json、summary.json、independent_audit.json、saturation_precision_diagnostic.json、source_snapshot/及analysis_source_snapshot.py。评分源码快照与事后分析源码分别保留。

独立NumPy复算28组softmax和两个端点、36配对端点，最大概率差1.09011e−7；CPU复算的summary与原结果完全一致。Ubuntu和Windows各4项真实artifact集成测试通过。初次评分链在pytest未安装处停止，模型评分和审计此前已完成；仅给独立venv安装pytest后补测通过，未重复GPU评分。chain_status最终COMPLETE_OFFICIAL_PARITY_SCORES_AUDIT_TESTS。权重按HTTP范围/总长核验，未额外做全文件SHA读回。

## Relations

- relates_to [[68-cloud-mtd-synchformer-native-preference]]
- relates_to [[71-vocalist-official-native-cloud-preference]]