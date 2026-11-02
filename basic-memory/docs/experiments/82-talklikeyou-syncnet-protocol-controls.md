---
title: 82-talklikeyou-syncnet-protocol-controls
type: experiment
permalink: tts-exp/docs/experiments/82-talklikeyou-syncnet-protocol-controls
status: concluded
scientific_status: POSITIVE_SCORE_DIRECTION_RETAINED_NO_CONFIRMED_PROTOCOL_EFFECT
date: '2026-10-31'
inputs: Report79 fixed Chinese TalkLikeYou80 logical /41 unique existing videos
outputs: runs/talklikeyou_syncnet_protocol_controls_20261031_attempt01
---

# 82 — 中文TalkLikeYou固定视频的SyncNet门槛与音轨对照

✅ CONCLUDED。固定报告79已完成的中文视频，完成两种音轨 × 两档门槛对照；未生成新TTS或视频，未恢复HDTF暂停实验。

**门槛100/25在共同合格支持上没有改变裁剪或分数；原MP4音轨与规范PCM没有改变本批TTS−自然分差的正向量级。** 全20来源的音轨效应为−0.002，95%区间[−0.098,+0.086]。未校正单指标p跨过.05，不能把“量级稳定”写成“每种统计判定都相同”。

## 问题与冻结设计

输入父run为 `talklikeyou_chinese20_rms24_20261030_attempt01`，20个中文来源、每来源1条自然和3条既有云端TTS draw，共80逻辑视频／41唯一生成。12来源同句克隆参考为原主要分层，另8来源扩展参考单列。固定3.png、TalkLikeYou已生成画面与原RMS−24dBFS驱动音频；重复draw沿用明确别名，不能当独立样本。

新run：`runs/talklikeyou_syncnet_protocol_controls_20261031_attempt01/`。执行机为192服务器个人数据区，沿用同一官方SyncNet V2及S3FD环境、权重与参数。SyncNet权重SHA为 `961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442`，S3FD为 `d54a87c2b7543b64729c9a25eafd188da15fd3f6e02f0ecec76ae1b30d86c491`。

四格为：
- 原生成 `generation/<id>/generated_aac.mp4` 内嵌AAC音轨，min_track25／100。
- 同一视频流重封装规范PCM16、16kHz、mono的MKV，min_track25／100。

所有原MP4、规范WAV和别名原AAC的SHA在前向前冻结。没有增益调整、补帧、补静音、裁句、重定时或偏移补偿。官方内部转码保留不变。详见[前轮官方源码核查](../reference/wav2lip-official-evaluation-parity.md)。

先在每个来源内平均三个逻辑TTS draw，再减自然；来源等权。门槛对照主支持要求同一来源全部四条逻辑视频在全部四格可测；音轨另报共同25支持的全20来源。采用来源bootstrap10000、seed20261031及双侧精确sign-flip。三项协议效应（两种音轨的100−25、100门槛下原MP4−PCM）每分层Holm3。边际区间不等于同时区间，sign-flip依赖对称／可交换假设。

## 覆盖与门槛结果

| 条件 | 可测唯一视频 | 可测逻辑视频 | 完整配对来源 |
|---|---:|---:|---:|
| 原MP4／25 | 41 | 80 | 20 |
| 规范PCM／25 | 41 | 80 | 20 |
| 原MP4／100 | 27 | 52 | 13 |
| 规范PCM／100 | 27 | 52 | 13 |

共同13来源包括原主要12个同句参考来源和1个扩展参考来源。被排除的7来源全部属于原扩展参考8人；每组四条视频都未达到100检测门槛，共28逻辑视频。它们保留为不可测，没有赋零分或改变长度。

共同支持上的100／25轨迹、裁剪AVI完全相同，C、D、D0和offset差均为0。54个100格裁剪逐字节等于相应25格，按冻结规则复用相同裁剪的网络前向，而非声称重复做了54次新网络评分。两档官方人脸处理均实际执行；完整科学调用为136次pipeline、82次scorer和41次PCM remux，259个阶段全部exit0。

## TTS−自然的Sync-C

`ΔC = mean(TTS三draw) − 自然`。下表95%区间均为单项来源bootstrap区间。

| 支持 | 音轨 | 门槛 | 自然C | TTS C | ΔC | ΔC的95%区间 | T较高 |
|---|---|---|---:|---:|---:|---|---:|
| 共同13 | 原MP4 | 25或100，同分 | 4.858 | 5.415 | +0.557 | [+0.255,+0.813] | 11/13 |
| 共同13 | 规范PCM | 25或100，同分 | 4.918 | 5.429 | +0.511 | [+0.198,+0.767] | 11/13 |
| 主要12 | 原MP4 | 25或100，同分 | 4.882 | 5.434 | +0.552 | [+0.220,+0.830] | 10/12 |
| 主要12 | 规范PCM | 25或100，同分 | 4.937 | 5.446 | +0.509 | [+0.159,+0.787] | 10/12 |
| 全20，补充 | 原MP4 | 25 | 4.887 | 5.380 | +0.493 | [+0.033,+0.871] | 15/20 |
| 全20，补充 | 规范PCM | 25 | 4.895 | 5.390 | +0.495 | [+0.005,+0.924] | 15/20 |

规范PCM／25本轮C与报告79逐值相同，最大差0；CI较原报告微小差别来自本轮预定bootstrap seed不同，不是重算分数变动。共同13的Sync-D配对差原MP4为−0.155、规范PCM−0.079，均按D越低越好理解；完整C/D/D0/B/offset保存在summary.json。

## 协议效应：比较差值本身

音轨效应定义 `ΔΔC = ΔC原MP4 − ΔC规范PCM`，不是比较两组p值是否跨阈值。

| 支持 | ΔΔC | 95%区间 | 精确p | Holm3 p |
|---|---:|---|---:|---:|
| 共同13，100门槛 | +0.046 | [−0.044,+0.134] | .351074 | 1.000000 |
| 主要12，100门槛 | +0.043 | [−0.054,+0.140] | .410645 | 1.000000 |
| 全20，共同25支持，补充 | −0.002 | [−0.098,+0.086] | .962824 | 未纳入三项主家族 |

两项门槛效应为0，区间[0,0]、p=1，来自完全相同的实际轨迹和裁剪。未确认音轨效应不等于已证明两协议等效；没有预设等效界值或等效检验。

共同支持中仅有1个扩展参考来源。机器summary／run report保留n=1算法输出，其退化bootstrap点区间不能用于总体不确定性推断；本报告不以它作独立科学结论。

## 统计判定边界

全20单指标精确p：规范PCM .052282，原MP4 .035086。分差几乎相同仍可能因逐来源差异分布而跨.05；不得按显著／不显著二分声称存在协议交互。

另作明确的事后CPU敏感性：保持原报告另外8项冻结Monte Carlo p和各自支持不变，仅将SyncNet p替换成本轮精确p，再计算Holm9。原MP4的SyncNet主要12 q=.101074、全20 q=.245600，仍未通过.05。规范PCM对应q=.153809／.365976。该混合精确／冻结MC的敏感性不是重跑九评分器，也不替代原冻结报告；结果在 `holm9_syncnet_input_sensitivity.json`。

因此可保留“本批SyncNet原生分差正向”的描述；本次不能将原报告九指标校正后未确认的SyncNet优势升级为已确认，更不能解释为人类感知同步或嘴型准确度。

## 官方中间AVI：实际编码、波形和延迟

82个音轨条件（41唯一视频×2）的中间 `video.avi` 全部为 **MPEG-4视频＋MP3音频（16kHz）**，即使输入是PCM MKV也会重编码。测得：

- 两种输入的输入→AVI波形延迟均为 **+1105样点＝+69.0625ms**，82/82完全一致；互相关搜索±1秒，无边界命中。正号表示AVI音频相对输入延后。
- 原AAC输入独立解码相对规范WAV的延迟为0，41/41；但0/41逐样点相同，波形余弦.999711–.999974，中位.999928，末端样点数可多0–944。AAC与PCM不能称相同波形。
- PCM输入→AVI在诊断时间对齐后波形余弦.965775–.997424，中位.990599；最小二乘增益.896347–.948532，中位.941969；增益拟合残差SNR中位17.749dB。原MP4输入分支对应余弦中位.990550、增益.940734、SNR17.708dB。说明编码转换确实改变波形，不能用“最终提取成PCM”推断无损。
- 独立解码AVI音频与官方pipeline实际提取的audio.wav逐样点相同，82/82；评分器最终读入的裁剪PCM也与独立crop解码逐样点相同，82/82。
- 两种音轨条件的中间视频像素、S3FD faces、scene、tracks与最终裁剪视频像素均41/41一致。故本轮音轨效应没有混入画面／脸框／追踪变化。

每片ffprobe、waveform诊断、PTS、原始WAV和所有官方中间媒体均保留。互相关仅用于测量，未用于评分校正。本轮两分支都经历了共同MP3／延迟路径，未执行“无MP3中间编码”或“补偿69ms”消融；不据此断言共同编码路径对绝对分数或分差没有影响，也不将69ms单独归因为某一个codec参数。

## 验证、恢复与归档

初始producer11201完成27个唯一条目后，在第28条scorer前因共享GPU利用率99%的资源门退出；没有科学子阶段失败。revision02将资源检查改为等待，并校验复用已完成pipeline阶段；resume producer60457完成剩余条目。另一次辅助重复启动12360在fresh-run断言处拒绝，零新前向。旧代码、日志、failure、原finisher停止与resume凭据全部保留。163份续跑前阶段凭据在最终产物中SHA不变。

独立审计：41个唯一媒体的136格PASS矩阵FP64归约误差最高2.570e−6；12个单格均值与9个协议效应的bootstrap／精确检验独立复算通过；本机另核全20及主要12的6组均值、bootstrap和精确p通过。来源均值独立误差0，5项针对性测试通过。

完整tar：`runs/talklikeyou_syncnet_protocol_controls_20261031_attempt01.tar.gz`，1,011,531,246字节，SHA256 `16be485cdf65601fdbf2af0b97933d03fe527ff7ec2280dd07c78a7521784228`。本机全部31,052文件／1,037,499,167字节逐SHA读回通过，41对原MP4/WAV绑定通过；源码权重body外置但完整SHA冻结。额外本机统计与Holm敏感性文件在run目录，独立于原封存tar。没有声称云端备份。

证据入口：
- `protocol.json`、`summary.json`、`paired_common_four_cells.csv`
- `independent_audit.json`、`local_readback_receipt.json`、`local_full20_statistical_audit.json`
- `items/<id>/<cell>/transport.json`、`distances.npy`及完整官方work
- 源码 `scripts/experiments/talklikeyou_eval_protocol_controls_20261031/`，测试 `tests/test_talklikeyou_eval_protocol_controls.py`

## Observations

- [status] CONCLUDED：固定视频门槛与音轨对照、独立审计、完整本机读回完成。
- [conclusion] 主要12全部通过100门槛，100/25同裁剪同分；全20原MP4与PCM的ΔC为+0.493/+0.495，差−0.002，未确认音轨协议效应。
- [measurement] 官方中间AVI对两分支均为MP3音轨并引入1105样点波形延迟；所有脸框和视频像素在音轨分支间一致。
- [limitation] 单指标p跨.05但Holm9敏感性仍未通过；两协议未证明等效，且没有感知真值或无MP3延迟消融。

## Relations

- follows_up [[79-talklikeyou-chinese20-nine-scorers]]
- relates_to [[Wav2Lip官方评分路径与当前项目流程对比]]
- relates_to [[研究主线与证据地图]]
