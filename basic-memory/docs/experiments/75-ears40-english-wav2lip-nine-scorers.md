---
title: 75-ears40-english-wav2lip-nine-scorers
type: experiment
permalink: tts-exp/docs/experiments/75-ears40-english-wav2lip-nine-scorers
status: concluded
scientific_status: SCORER_SPECIFIC_ENGLISH_TTS_SCORE_ADVANTAGE
date: '2026-10-29'
run_id: ears40_w2l_nine_scorers_20261029_attempt01
training: false
verification_status: ANALYZED
tags:
- experiment
- ears
- english
- wav2lip
- tts
- nine-scorers
---

# 75-ears40-english-wav2lip-nine-scorers

✅ CONCLUDED：40 位 EARS 英语说话人、80 条 Wav2Lip 视频、九评分器全部原生评分与720条真实移音控制完成，ROOT 独立核对原始输出、配对统计及交付哈希。当前条件下存在 Sync-C 的 TTS 优势；九指标的证据有差异，不能据此认定人类感知嘴型更准确。

| 修改日期 | 修改 | 授权 |
|---|---|---|
| 2026-10-29 | 首次：EARS40 自然/TTS、Wav2Lip生成、九评分器、移音控制及独立分析 | 用户要求沿用既有流程检验英文TTS优势 |

## 问题与结论

在同一40位英语说话人、同一句目标文本、同一固定头像与 Wav2Lip GAN 流程中，既有云 Qwen Audio 3.1 Plus TTS 驱动的视频，是否比自然语音驱动的视频得到更高同步评分？每臂用自己的音轨评分，因此这是“生成输入＋评分输入”联合改变的原生比较。

Sync-C 自然 7.327、TTS 8.186，配对平均 T−N **+0.859**，边际95%配对bootstrap区间 [+0.624, +1.092]，TTS 胜 34/40。九主终点 Holm 校正后，TTS 分数较高的指标为 **SyncNet、SyncReward、Synchformer**；自然分数较高为 **无**；其余 **CoGenAV、PEAVS、AVS_u、MTDVocaLiST、VocaLiST、SparseSync** 未确认平均差。未确认不是等效或“没有效应”的证明。

这三项也有不同边界：SyncReward的移音降分控制不通过；Sync-C虽然降分控制不通过，但相对偏移80/80恢复；Synchformer通过控制，Holm p接近.05，百万次精度扩展判定不变。因此确认的是本协议的评分差，不能把“三项显著”当作三项已验证的人类同步质量优势。

这个结论只适用于本轮 EARS／云 Qwen3.1／Wav2Lip／统一RMS条件。历史报告04的本地0.6B／Ditto／其它英语队列不能作为本轮预期，也不能用新旧不同队列作语言交互的因果证据。研究地图中“英文不能概括为不成立”的边界在本轮得到新的条件性证据；训练来源重合未完成系统排除。

## Material Passport 与冻结方法

- Verification Status：ANALYZED。全部真实前向和独立数值/身份审计已完成；未额外重复整套九模型前向，不标记为完整reproducibility rerun VERIFIED。
- Run：`runs/ears40_w2l_nine_scorers_20261029_attempt01`；输入复用 `runs/ears40_cloud_qwen31_20261028_attempt01`，没有新TTS API调用或训练。
- 执行方式：academic-research-suite / experiment-agent run；executor 顺序真实GPU前向，ROOT用独立CPU脚本复核，未导入producer统计实现。
- 单位：40 位说话人，20女/20男，全部保留。每人一句 `sentences_16_regular` 自然目标及一个既有 `qwen-audio-3.1-tts-plus` TTS draw；同人独立 `sentences_15_regular` 作为克隆参考。全40人共享同一句目标、同一句参考文本；不是40种文本。
- 原始自然总时长503.975秒（8.436–18.823秒），TTS438.320秒（8.320–14.320秒）。80条各自时钟保留，无裁尾、拉伸、降噪或强制时间对齐。原始双臂电平差中位数约+6.062dB，生成前统一RMS为0.0376838172675746，mono16k PCM16、样点数不变、无削波。
- Provider 流式WAV的40条 RIFF/data 长度是占位头；实际PCM可解码，canonical16头合法，原始文件不改。输入manifest SHA256：`1958309cb175a6c161f67ddf36e7dca417b60d0267d57e992614f576b619fcef`。
- 生成：Wav2Lip GAN、固定 `data/data/image/3.png`，25fps，`--nosmooth --face_det_batch_size 4 --wav2lip_batch_size 4`；权重SHA `ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8`。Wav2Lip原生视频可比音频短数帧，保留该行为，支持长度按实际decoded frame与PCM共同限制，不重定时。
- 原始视频流不变；多数模型使用ALAC MP4，SyncNet完整作者管线使用PCM MKV输入，后续作者AVI/解码前端有自己的处理延迟。因此各模型不是同一PCM前端或完全相同时间支持。
- 冻结主端点：SyncNet官方完整管线FP32 Sync-C；CoGenAV论文Eq1全片ReLU余弦；PEAVS作者batch64原生分数；SyncReward作者Stage2 BF16、中心4.75秒、低显存encode batch1；AVS_u论文定义的项目复现、全片帧余弦；两个VocaLiST官方K5；Synchformer与SparseSync官方21类中心5秒网络的项目P0标量。AVS_u不是作者发布的现成评分器，两个P0也是项目选定的连续端点。
- PEAVS原生80条固定排序为64+16两批；移音控制按21/19个自然源的三元组形成63/57两批，与各自**同批**自然基线比较。另保存仅评分batch1敏感性。padding/批构成属于该实现的分数条件。
- Synchformer按作者cfg启用FP16 autocast、logits转FP32 softmax；SparseSync为FP32。SyncReward本轮BF16，不与历史中文FP32分数直接作语言效应量比较。环境清单是实际forward之后补采的版本证据，不能称预注册环境快照。
- 统计：40个说话人等权配对差T−N，10000次说话人bootstrap、seed20261029、边际95%区间；双侧100000次Monte Carlo sign-flip（+1修正），九主端点一起Holm9。配对sign-flip依赖交换性/差值对称性，未随机分配TTS处理。性别分组、K7–15、负期望类别距离、batch1、CENTRAL125均为描述性次分析，不加作独立评委。

冻结协议见 [protocol.json](../../../runs/ears40_w2l_nine_scorers_20261029_attempt01/protocol.json)。其中generation旧recipe的`conditions=[N,R]`和`max_outputs=20`未被新driver消费；实际循环audio_inputs全部N/T各40，实际80命令及说明见`generation_execution_receipt.json`。不以改写冻结协议消除这个遗留元数据差异。

## 九个冻结主端点

SyncNet 为 Sync-C；CoGenAV/AVS_u 为各自原生余弦分数；PEAVS/SyncReward 为模型原生分数。两个 VocaLiST 的均值显示 K5 百分比、两个时移分类器显示 P0 百分比，这四行的差和区间为百分点。各行单位不同，不横比绝对差大小。

| 模型 | 自然均值 | TTS均值 | T−N | 95%配对bootstrap区间 | T胜/N胜/平局 | MC p | Holm9 p | 本轮主端点 |
|---|---:|---:|---:|---|---|---:|---:|---|
| SyncNet | 7.327 | 8.186 | +0.859 | [+0.624, +1.092] | 34/6/0 | 0.000010 | 0.000090 | TTS分数较高 |
| CoGenAV | 0.366571 | 0.377257 | +0.010686 | [+0.002524, +0.019217] | 20/20/0 | 0.017250 | 0.103499 | 未确认平均差 |
| PEAVS | 2.697459 | 2.783014 | +0.085555 | [-0.055042, +0.225240] | 22/18/0 | 0.242948 | 0.728843 | 未确认平均差 |
| SyncReward | 1.527148 | 1.628320 | +0.101172 | [+0.071484, +0.133008] | 34/3/3 | 0.000010 | 0.000090 | TTS分数较高 |
| AVS_u | 0.105184 | 0.098824 | -0.006360 | [-0.018050, +0.003555] | 15/25/0 | 0.277487 | 0.728843 | 未确认平均差 |
| MTDVocaLiST | 6.826142 | 6.100315 | -0.725827 | [-1.556127, +0.096469] | 14/26/0 | 0.096799 | 0.483995 | 未确认平均差 |
| VocaLiST | 6.757159 | 6.478740 | -0.278419 | [-1.102891, +0.496835] | 19/21/0 | 0.512595 | 0.728843 | 未确认平均差 |
| Synchformer | 98.789891 | 99.549325 | +0.759434 | [+0.178475, +1.576140] | 32/8/0 | 0.006930 | 0.048510 | TTS分数较高 |
| SparseSync | 99.798108 | 99.998852 | +0.200744 | [+0.000042, +0.600372] | 23/17/0 | 0.105889 | 0.483995 | 未确认平均差 |

区间是边际描述性区间；确认口径使用冻结的双侧Monte Carlo配对sign-flip及Holm9。两种方法不同，单看某条边际区间是否跨0不能替代九项主检验。评分偏好不等于人类感知同步真值。

## 真实音频时移控制

| 模型 | 主分降分/80 | 冻结64/80降分门 | 相对偏移恢复/80 |
|---|---:|---|---:|
| SyncNet | 15/80 | CONTROL_FAILED | 80 |
| CoGenAV | 80/80 | PASS | 80 |
| PEAVS | 41/80 | CONTROL_FAILED | 未测该端点 |
| SyncReward | 33/80 | CONTROL_FAILED | 未测该端点 |
| AVS_u | 79/80 | PASS | 未测该端点 |
| MTDVocaLiST | 79/80 | PASS | 80 |
| VocaLiST | 76/80 | PASS | 80 |
| Synchformer | 80/80 | PASS | 80 |
| SparseSync | 80/80 | PASS | 78 |

40条自然音频分别对各评分链实际输入的完整PCM施加±400ms非循环时移，再按该模型规则取窗；SyncNet使用实际作者管线评分crop解码PCM，保留其原处理延迟。降分和相对峰位恢复分别解释；Sync-C会搜索最佳lag，不能把降分门失败直接解释成不能定位时移。零填充以及各模型不同的编码上下文都是控制定义的一部分。

## 中央125 query敏感性

125 nominal 25Hz query centers; receptive fields and lag candidate context can extend beyond these centers. Native full-context embeddings/official scoring crops retained, no new five-second media benchmark or matched phonetic events. No Holm9 claims from this secondary view.

| 模型 | T−N | 95%区间 | T胜/N胜/平局 |
|---|---:|---|---|
| CoGenAV | +0.014277 | [+0.001851, +0.027239] | 24/16/0 |
| AVS_u | -0.009317 | [-0.025819, +0.004491] | 16/24/0 |
| SyncNet | +1.199 | [+0.836, +1.577] | 34/6/0 |
| MTDVocaLiST | -1.160000 | [-2.380000, +0.040000] | 13/20/7 |
| VocaLiST | -0.660000 | [-1.820500, +0.460000] | 16/19/5 |

## 其他冻结次端点

以下端点均保留，属于描述性敏感性分析，不作为九个主检验之外的独立评委或额外确认结论。K7–15是同一网络的相关时间聚合。负期望类别距离以秒表示概率分布的加权量，不是亚毫秒物理定位误差。

| 次端点 | T−N | 95%区间 | T胜/N胜/平局 |
|---|---:|---|---|
| MTDVocaLiST_K7 | -0.934804 pp | [-1.767013, -0.09937224] pp | 16/24/0 |
| MTDVocaLiST_K9 | -0.7040473 pp | [-1.570825, +0.1720946] pp | 13/27/0 |
| MTDVocaLiST_K11 | -0.6137142 pp | [-1.434004, +0.2124811] pp | 16/24/0 |
| MTDVocaLiST_K13 | -0.6896948 pp | [-1.544634, +0.1549784] pp | 19/21/0 |
| MTDVocaLiST_K15 | -0.6782679 pp | [-1.440373, +0.08802168] pp | 17/23/0 |
| VocaLiST_K7 | -0.3610186 pp | [-1.21698, +0.4885849] pp | 18/22/0 |
| VocaLiST_K9 | -0.2454008 pp | [-1.149525, +0.643639] pp | 19/21/0 |
| VocaLiST_K11 | -0.2536835 pp | [-1.149068, +0.637442] pp | 19/21/0 |
| VocaLiST_K13 | -0.4222242 pp | [-1.331219, +0.4647387] pp | 19/21/0 |
| VocaLiST_K15 | -0.6491587 pp | [-1.496782, +0.1921245] pp | 18/21/1 |
| Synchformer_negative_expected_class_offset | +0.001646309 s | [+0.0003739052, +0.003395592] s | 32/8/0 |
| SparseSync_negative_expected_class_offset | +0.0004042864 s | [+9.257585e-07, +0.001204821] s | 21/19/0 |
| PEAVS_batch1_sensitivity | +0.06502165 | [-0.08249169, +0.2119515] | 21/19/0 |

## Monte Carlo数值精度敏感性（事后，不替代主结果）

在最后回归器结果齐全前，因Synchformer的100000次p接近可能的Holm边界，记录统一对全部九指标扩展至1000000次的计划。前100000次符号序列与冻结主分析逐项完全一致；端点、样本、统计量不变。此处只减少Monte Carlo计算噪声，不增加独立样本，不改写主表。alpha=.05判定改变的指标：无。

| 指标 | 冻结100k Holm9 p | 扩展1m Holm9 p |
|---|---:|---:|
| SyncNet | 0.000090 | 0.000009 |
| CoGenAV | 0.103499 | 0.103170 |
| PEAVS | 0.728843 | 0.728831 |
| SyncReward | 0.000090 | 0.000009 |
| AVS_u | 0.728843 | 0.728831 |
| MTDVocaLiST | 0.483995 | 0.483965 |
| VocaLiST | 0.728843 | 0.728831 |
| Synchformer | 0.048510 | 0.046473 |
| SparseSync | 0.483995 | 0.483965 |

若某行处于阈值附近，应按连续证据强弱解释，不能将极小阈值穿越当成稳定发现。simulation-only Wilson区间见mc_precision_sensitivity_1m.json，不是效应区间。

## 诊断与解释边界

Sync-C控制降分15/80而相对时移恢复80/80，零时移重放40/40矩阵逐元素一致。Sync-C搜索最佳lag，保留高对比度与正确定位时移可以同时发生；冻结降分门标为CONTROL_FAILED，不删除主结果，也不把它写成软件失败。其它模型的降分/恢复结果见上表，降分控制合格也不能证明人类效度。

PEAVS在严格同批基线下只有41/80控制降分，也为CONTROL_FAILED。原生与batch1都未确认平均差，只能陈述本轮模型分数结果；响应门失败限制其作为本条件“感知同步改善”测量的解释，不等于真实口型没有差异，也不通过改批次或更换端点挽救。

SyncReward虽有明确的原生TTS高分，真实移音后仅33/80控制降分，同样CONTROL_FAILED。其分数提升与当前错位响应不足必须同时保留；不能把这一显著分差直接称为已被本轮控制验证的同步质量优势。Synchformer主表Holm p接近.05，百万次扩展仍低于.05，但证据强度低于Sync-C/SyncReward，且P0处于高分区。

SparseSync有79/80原生P0>0.9995、80/80最佳类别为零；Synchformer对应8/80、80/80。SparseSync均值几乎由p105自然P0约0.920254、TTS约0.999960的一个配对贡献，其余多在天花板。保留全部40人，不按结果剔样。保存logits改用FP64 softmax时两个分类器全部40对方向均不变，天花板不只是FP32正负翻转问题。P0的极小差和负期望类别距离也不是亚毫秒物理定位精度。

两个VocaLiST的K5无原生精确零值，整体分数较低；原生全局峰位MTD为−2帧21条/0帧59条，VocaLiST为−2帧49条/0帧31条，不能预设历史中文的固定偏差。本轮对应错位控制分别恢复80/80。不同K是同网络相关时间聚合，不能把六个K当六个独立评分器。

全片与CENTRAL125比较保留原完整编码上下文，感受野和lag候选可越过中心边界；全片长度、停顿和不同发声的音素时间并未配平。没有构造统一5秒重编码基准、交叉轨或DTW对齐。本轮只能回答本协议原生评分偏好，不定位生成端/评分端各占多少，也不证明评分器“偏爱TTS”或排除普通发声实例匹配效应。

EARS在本轮提供自然音频，不提供对应真实说话人同步口部视频真值。所有输出使用同一固定头像；未做合格盲评或独立口部运动准确性测量，九指标共识或分歧都不能直接当作感知同步真值。

## 执行修复与不可变凭据

Synchformer作者`EqualifyFromRight`先截10秒，最长p046_N的冻结中心5秒窗却结束于11.84秒，首次执行在已保存24条时停止。另`int((i/25)*25)`在4/80媒体会少一视频帧。修复为按冻结整数`i`先取video[i:i+125]与完整移音后PCM[i*640:(i+125)*640]，再交原作者变换、局部起点0；不改端点/样本/精度。

旧24条保存为修复目录及SHA；48个旧新native/control输入中44个完全相同、4个仅已知视频帧差，音频全相同。22条缓存逐值保留，两媒体所需4次前向重算；首媒体旧新logits误差0。SparseSync启动前采用相同整数窗口，作者主路径与直接batch张量/logits一致。作者transform源码不改，冻结协议和原始source_snapshot不覆盖。详见 [ADR052](../adr/052-integer-frame-native-window-preselection.md) 与 run/execution_repairs、synchformer/repairs。

历史中文28合格媒体只读核查另发现`BAC009S0031W0121_N_0`名义29帧旧路落28帧，影响那一源三配对的两个分类器历史比较资格；本轮不重算、不重写历史冻结结果。跨中英差异不能忽略这一修正。

另两项启动诊断保留：预检同名example.py快照冲突在科学执行前改用完整路径保存；AVS CPU预检的python-c触发上游DBG分支，真实CLI上下文strict-load通过。二者均未改作者公式或删除样本。

## 验证与复现入口

最终远端科学文件3585项、675359638bytes在本地由executor及ROOT分别逐SHA/大小核验通过；清单SHA为`6483624c09ceb31e333bc65efba3991e05c0a38e1aec0871413fa8860f3ae0d8`。ROOT额外核对最终SyncReward160条的前80条与既有只读partial及原log前缀完全一致。本轮GPU科学进程已结束，未恢复或改动其它暂停任务。

ROOT独立核验输入SHA与归一化（最大误差小于1 PCM16 LSB）、80视频交付SHA、真实PCM/帧绑定、各模型80原生与80唯一±6400样点控制，以及原始数组/logits/作者raw输出与汇总标量。PEAVS/SyncReward的CPU核查是保留作者raw输出的身份/数值绑定，不冒称独立重新实现其神经网络。不同精度数值容差与各模型最大误差保存在raw_endpoint_audit.json及raw_control_audit.json。

修复后ROOT亲自运行六个针对pytest文件：30 passed，0.54秒；receipt为tests_postrepair_root.json。没有因为低影响文档格式重复跑全仓测试。权重身份以model_asset_binding、asset_receipts和执行pin为准：部分既有大权重使用已存SHA配合本次size/mtime身份，不声称全部九模型本轮均重新完整读取hash。新增脚本当前未提交；基线Git HEAD `2b71639c3b555293375d248350999377168d515d`不包含新增工作，实际源码SHA与source_snapshot才是本次代码身份。继承脏工作区不纳入本次结论。

独立复算命令（仅需NumPy）：

```bash
python scripts/experiments/ears40_results_20261029.py --run runs/ears40_w2l_nine_scorers_20261029_attempt01 --raw-audit --central
```

- [九项完整统计、边界与诊断](../../../runs/ears40_w2l_nine_scorers_20261029_attempt01/parent_analysis/summary.json)
- [360行逐说话人配对分数](../../../runs/ears40_w2l_nine_scorers_20261029_attempt01/parent_analysis/paired_scores.csv)
- [原始主端点数值复算](../../../runs/ears40_w2l_nine_scorers_20261029_attempt01/parent_analysis/raw_endpoint_audit.json)、[720条移音控制复算](../../../runs/ears40_w2l_nine_scorers_20261029_attempt01/parent_analysis/raw_control_audit.json)
- [中央query敏感性](../../../runs/ears40_w2l_nine_scorers_20261029_attempt01/parent_analysis/central_support_sensitivity.json)、[全部次端点](../../../runs/ears40_w2l_nine_scorers_20261029_attempt01/parent_analysis/secondary_metrics.json)
- [80视频独立副本manifest](../../../results/wav2lip/ears40_20261029_manifest.json)，媒体在`results/wav2lip/{natural_raw,tts_raw}/<speaker>/ears40_20261029.mp4`。
- [独立分析源码](../../../scripts/experiments/ears40_results_20261029.py)，执行、回传及环境凭据在run根与parent_analysis；Material Passport和哈希链ledger随parent_analysis保存。

## Observations

- [result] EARS40英语当前云Qwen3.1/Wav2Lip/RMS原生比较，TTS Sync-C +0.859，34/40说话人提高，九项Holm后支持。
- [result] 九评分器方向与统计支持异质，不等同九个独立人类评委，未证明感知同步优势。
- [decision] 全40对、失败控制、天花板与修复旧缓存全部保留；不改冻结历史结果、不据本轮自动启动训练或新机制搜索。
- [limitation] 单文本/固定脸/单draw/特定模型与不同原生前端；跨队列语言交互、数据来源重合和人类真值尚未建立。

## Relations

- relates_to [[研究主线与证据地图]]
- qualifies [[04-english-replication]]
- qualified_by [[052-integer-frame-native-window-preselection]]