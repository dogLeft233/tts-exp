---
title: 78-echomimic-v3-official-control
type: experiment
permalink: tts-exp/docs/experiments/78-echomimic-v3-official-control
status: CONCLUDED
scientific_scope: exploratory_six_source_C1_common_prefix
date: '2026-10-30'
tags:
- echomimic
- syncnet
- official-control
- volume
- inference
---

# 78 — EchoMimicV3-Flash 官方配置与音量/推理参数对照

✅ 已完成；科学状态 CONCLUDED，仅为6来源探索性诊断。原20来源九评分器实验 [[77-echomimic-v3-chinese20-nine-scorers]] 及其归档保持不变。

## 问题与主要发现

检验原RMS−24、长片、seed42配置是否使Sync-C偏低。预先按长度选取6个来源后，官方配置组合使TTS−自然由−0.332变为+1.021，差上差+1.353，描述性来源bootstrap 95%区间[0.446,2.444]。这说明配对评分对设置敏感，但不证明原部署错误，也不构成全20来源或感知同步结论。

驱动端仅恢复LUFS，自然变化−0.069、TTS+0.054；仅81帧上限，TTS+0.939；仅seed43，自然−0.992。此次主要敏感因素是生成帧数及seed，音量在已测范围内较小。官方组合自然分数反而下降0.759，TTS上升0.594，不能概括为整体绝对分普遍提高。

## 冻结设计与支持集

在原同句12来源和扩展8来源中，分别按自然语音长度排序，固定取最短、下中位、最长来源，选择不读取分数。每来源自然与既有C1 TTS各一条，6对/12条，不取三TTS平均。

同句：BAC009S0011W0121、BAC009S0014W0121、BAC009S0023W0121。扩展：BAC009S0010W0121、BAC009S0006W0121、BAC009S0008W0121。同一3.png、已验证Flash完整检查点、中文W2V、BF16、CPU offload及长度mask语义SDPA。

旧配置：RMS−24dBFS、完整原生4k+1帧、seed42。单变量组分别仅恢复官方整段−23LUFS、81帧上限、seed43；官方组合同时恢复这三项。8步、Flow_Unipc、CFG6/audioCFG3、shift5、TeaCache0.1/skip5原已一致。官方768是面积上限，512图实际仍为512；没有放大输入。

每来源全部条件统一评价前min(81,N原生帧数,T原生帧数)：S0010为65帧，其余81帧。驱动LUFS在整段波形上计算后截取特征前缀；生成对照始终使用相同RMS24评分PCM。另设scorer_gain_only，只改变旧视频的评分音量。官方入口的LUFS只作用于特征，不能把成片音轨增益混成生成器效应。工程容器保留原完整音轨，真正评分输入严格裁为F×640样点。

## 配对结果

Sync-C越高越好；Δ=TTS−自然。每来源C1与自然配对，以下是6来源均值，不能与报告77的20来源完整时长/三TTS均值直接混排。

|条件|自然|TTS|Δ|自然相对旧配置变化|TTS相对旧配置变化|
|---|---:|---:|---:|---:|---:|
|旧受控配置|4.887|4.556|-0.332|+0.000|+0.000|
|仅恢复−23LUFS|4.818|4.610|-0.208|-0.069|+0.054|
|仅81帧上限|4.753|5.494|+0.741|-0.134|+0.939|
|仅seed43|3.895|4.622|+0.727|-0.992|+0.066|
|官方配置组合|4.128|5.149|+1.021|-0.759|+0.594|
|仅评分音轨−23LUFS|5.075|4.571|-0.504|+0.188|+0.016|

官方组合5/6来源TTS高于自然；其Δ的描述性95%区间[0.263,1.740]。仅81帧组Δ=+0.741，区间[0.256,1.212]；相对旧配置的TTS增益区间[0.402,1.476]。这些是小样本探索性区间，不报确认性p值，不作多端点显著性宣称。

帧数实际上发生变化的4对，TTS的Sync-C均提高；原生帧数已≤81的另两对构成4条无操作控制，其前缀像素与分数逐项相同。仅seed43就足以改变配对方向，因此后续稳定性研究需要独立设计多个seed；不能从两个seed推断整个随机分布。单变量和联合组未覆盖完整交互设计。

## 工程验证与修复来源

50次真实生成：两条旧配置重放、四组各12条。两条重放全部解码画面与旧视频逐帧一致，且共同前缀分数复现。50条媒体帧数/尺寸/非静态及ALAC音轨逐点一致通过。74项真实SyncNet评分完成；31列FP32原始距离矩阵全部保存，本地FP64独立归约最大误差2.963202e-6。

对首对自然/TTS分别做真实±400ms循环位移，四项拟合offset准确反向移动±10帧，全部通过。Sync-C搜索时延，可恢复的位移不一定降低confidence，故控制检验offset响应。控制不进入配对均值。

首次评分在前缀PTS预检查失败，尚无任何SyncNet分数：H264流复制截81帧包含3.24秒而遗漏3.20秒。保留失败树与原control.py，用独立score_repair.py绑定原计划/源码/失败产物哈希，采用解码后截F帧、x264 CRF0/r25/bf0，并逐片证明前缀解码像素完全相同与PTS=i/25，再继续原权重和FP32归约。试验02使用trim/passthrough仅得到80帧，保留；试验03的正确81帧与像素哈希通过。启动器attempt02的Windows CRLF失败在评分前发生，LF-only attempt03继续评分；原视频和旧实验没有重跑或改写。

本地独立核验1280内容文件、74个矩阵、全部来源均值和bootstrap区间通过；四条帧上限无操作控制通过；原报告77证据包SHA保持不变。SyncNet环境pip-freeze集合与旧实验相同。3项本地测试和5个Python编译通过；server-192 skill经临时PyYAML运行quick_validate通过。

## 资源与规则

当次通过Edge读取飞书1048手册，重新核验账号、HOME、磁盘及GPU。全部新产物位于个人数据区，单A6000 UUID GPU-de7ff5ff-3510-739f-917c-cb98892de759、CPU affinity0–3/4线程、生成allocator22GiB上限；GPU生成和评分顺序执行，不改共享环境、驱动或他人任务。未安装新服务器依赖。

50次生成累计43.20分钟，不含加载；PyTorch allocated峰值13.260GiB、reserved 14.467GiB，5秒NVIDIA采样峰值15174MiB。是A6000实测，不是物理24GB卡测试。结束回查本人实验进程为空，GPU驻留4MiB、空闲48666MiB、利用率0%。

SSH曾间歇性握手/认证/传输中断；经用户确认后有界重试，核验主机指纹，文件分块传输、哈希核对及原子更名。未更改代理或SSH服务设置，未重复启动生成。

## 复现与证据

代码：scripts/experiments/echomimic_v3_official_control_20261030/；测试：tests/test_echomimic_v3_official_control_20261030.py。已执行的固定run包装器不可直接重跑。

远端：/home/data2/wangjiajun/av-evaluation/jobs/echomimic_v3_official_control_20261030_attempt01。
本地：results/echomimic_v3_flash/official_control_20261030_attempt01/，包含paired_comparison.csv/json、scores.json、comparison.png/pdf、contact_sheet.png、local_readback.json及evidence/。

证据包76292497字节、1281成员，SHA256 9c87ccf21c39b60a5f231fbc9859a3c6d9b0c45402e6e5d5831243d027b23588；manifest SHA256 697c21c15d70ae9a6ccc4ec77b04c233c6dbf5c1c62aaa392e55fce23093a10f。保留模型输出、基线视频、PCM、原始矩阵、像素审计、源码/环境及失败修复证据；排除可无损重建的评分前缀容器及解码缓存，并保留其哈希清单。归档不重新打包覆盖。

[官方Flash配置](https://github.com/antgroup/echomimic_v3/blob/7e89489ca51c0d008fc1963ec6c03fc5bd0b9397/run_flash.sh)，[官方LUFS前端](https://github.com/antgroup/echomimic_v3/blob/7e89489ca51c0d008fc1963ec6c03fc5bd0b9397/infer_flash.py)。

## 限制

仅6来源/C1/65或81帧前缀，不是完整20来源、三TTS或九评分器重测。81帧上限同时改变总生成长度、音频特征上下文及latent时序形状，本试验未继续拆解这些长度相关机制。原报告77仍仅代表其RMS24/完整帧/seed42受控配置。保留SDPA、BF16加载兼容修复与CPU offload，不宣称字节级无改动官方CLI、FlashAttention端到端对照或论文benchmark复现。Sync-C是评价器证据，不能推出感知唇形质量。

## Observations

- [result] 官方组合6对ΔSync-C=+1.021；旧受控配置同前缀为−0.332；配对方向对设置敏感。
- [result] 单独驱动LUFS变化较小，帧上限和seed的影响更大；官方组合未普遍提高绝对分。
- [verification] 50生成、74评分、4时移控制、1280文件及全部bootstrap读回通过，旧归档不变。
- [decision] 旧20人结果和本6对探索性对照分别冻结，不能混排或替换原实验结论。

## Relations

- investigates [[77-echomimic-v3-chinese20-nine-scorers]]
- relates_to [[echomimic-v3-flash]]
- follows [[003-immutable-execution-repair-provenance]]
