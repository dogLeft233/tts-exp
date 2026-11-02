---
title: WORLD 谱形干预的独立嘴部运动复核
type: experiment
permalink: tts-exp/experiments/world-谱形干预的独立嘴部运动复核
status: concluded
project: tts-exp
started: 2026-10-01
tags:
- world
- visual
- phonetic-shape
- independent-validation
---

# WORLD 谱形干预的独立嘴部运动复核

## 问题与冻结设计

谱形、声源和局部能量移植引起的 SyncNet 响应是否伴随可见嘴部运动变化？本实验读取旧 WORLD v6 的完整1,600视频，独立计算嘴部几何，不读取 SyncNet 分数或模型效应数组。双向移植各80句、40位男性说话人、每人2句；10生成臂为 FIXED、RT、EX、SP、BOTH、F0、AP、EX_EM、SP_EM、E。

CAL 事先固定 bilabial/open_vowel 两类和−85ms音频到视频延迟，主端点为开元音减双唇音的开口分离。零延迟分离、AUC及7项整体几何为次级端点。以受体 RT 为干预基线，RT−FIXED 为独立重合成桥。availability09 在读取几何之前按身份、grid、标签、时间戳和共同validmask冻结；不依据几何效果选样本或阈值。

## 实际运行与支持集

tracker07 实际全部1,600轨迹 complete、0失败，消费器05/analysis09退出0，审计PASS/SEALED。完整覆盖160方向context、80样本、40说话人，3202个track成员哈希绑定。原始MFA79合格/1不合格保留；S0064 overlay两方向提供物理时钟，但不恢复原排除样本的主端点资格。

整体几何两方向均使用完整80句/40说话人。主音素端点T→N有52 eligible clips、16完整双句说话人，实际32贡献句；N→T为53/17/34。计划分母仍40说话人，不能将52/53写成完整配对样本数。每个说话人先平均两句，再等权；固定PCG64种子20260930、共享20,000×40重采样、点态99% percentile区间。稀疏主端点的bootstrap只使用完整双句说话人；非联合家族区间。

## 合格结论

| 方向/端点 | 对比 | 均值差 | 点态99%区间 | 完整说话人 |
|---|---|---:|---|---:|
| T→N 整体开口 | SP−RT | +0.001501779 | [+0.000089710,+0.002943500] | 40 |
| T→N 整体开口 | SP_EM−RT | +0.001479994 | [+0.000182840,+0.002813177] | 40 |
| N→T 整体开口 | SP−RT | +0.000636843 | [−0.000924526,+0.002122712] | 40 |
| N→T 整体开口 | SP_EM−RT | +0.000659432 | [−0.001040714,+0.002283348] | 40 |
| T→N −85ms音素分离 | RT−FIXED桥 | −0.013358849 | [−0.022822175,−0.005386650] | 16 |
| N→T −85ms音素分离 | RT−FIXED桥 | −0.008541399 | [−0.014780813,−0.002283738] | 17 |

T→N谱形和局部能量匹配谱形带来小幅整体开口增加，SP速度−0.002709387 [−0.013108589,+0.007244682]跨零；N→T两项开口区间跨零。全部8干预臂相对RT的−85ms音素分离区间在两方向均跨零。RT合成桥对主音素分离有负变化，N→T桥AUC −0.135679272 [−0.313901611,−0.020587710]亦负；桥对整体开口与速度的区间均跨零。

这些数据确认谱形操作可改变生成的整体开口，当前预设phone-shape测量没有稳定正向提升；桥损伤单独记账。它支持运动响应与评分响应有不同端点，不能将开口变大等同更准确的音素形状、人工感知同步或自然/TTS原生优势的中介比例。这里的局部类别只是两类CAL预设端点，并非对所有音素的视觉结论。

## 独立复核与产物

权威完整目录 `runs/recovered_20260930/remote/world_visual_analysis_attempt09/`。

- `world_visual_analysis.json` raw SHA256 `5d0b8d18ec4565572292d5dbbdf43d25b92e72fbb3a0120eac0731536f31f98a`（12,351,271B）；availability freeze raw `67131b9bd381dd2e393ab739ae51408e5edf15594cd3e9b67b844c5857621ccf`，封存前无几何读取。
- root实际运行独立NumPy2.3.5复算脚本 `scripts/experiments/portable_source_axis_20260930/world_visual_report_independent_recheck_20261002.py`：418 summary objects、5016字段、0 mismatch；均值、完整双句mask、PCG64索引和20k区间全部一致。`world_visual_independent_reconstruction_attempt01.json` raw `d7153b8fb64075ed5d6252348c37968664c8cb8f0de7084c03142bbe7285c83e`，完整端点表raw `b0a8973965d89390319715f0d6c489ad28e94b5b763ec968601a3b9041681d1d`。此外root复核15,840逐context对比定义，最大差0。未独立重新提取几何。
- metadata04实际退出0；root读完整脚本并实际复跑 `world_visual_metadata_bundle_independent_readback_20261002.py`，全1606 tar成员大小/哈希及1600sidecar对冻结grid/video/clock/array-descriptor/phone支持集join PASS。报告raw `afb6994005369ff9e3141d6c9da625829f8147285235946c3d027ac892ab65ee`；tar raw `9ee31fb942c300f5c82217016c48d0d623b58a04d63eb42c7492ac2ead6e7346`。metadata阶段NPZ保持opaque，其完整备份进入新归档。
- 首次12MB raw下载因8MiB上限在写入前被拒，使用fresh exactgzip读回并验证解压size/rawSHA；原失败记录保留。失败tracker/consumer/helper候选的工程状态均保留，未作为科学结果。

## Observations

- [result] 完整1600视觉轨迹、1606元数据归档成员、418统计summary已实际独立复核PASS。
- [result] T→N SP与SP_EM整体开口小幅增加；全部干预的预设−85ms音素分离区间跨零，RT桥降低该端点。
- [decision] 主端点16/17完整说话人与整体40说话人的支持集分开呈现，点态99%区间不升级成联合家族证据。

## Relations

- complements [[WORLD 声源谱形与局部能量的双向移植]]
- relates_to [[WORLD 音素局部响应与声学剂量]]
- relates_to [[原生 TTS 优势的时间配对与可见口型诊断]]
- relates_to [[解释 TTS 的 TFG 与 SyncNet 优势]]