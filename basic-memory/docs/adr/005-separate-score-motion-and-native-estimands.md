---
title: 005-separate-score-motion-and-native-estimands
type: decision
permalink: tts-exp/docs/adr/005-separate-score-motion-and-native-estimands
status: open
alternative:
- 以原生Sync-C差概括自然音轨迁移收益或感知同步
- 按单独显著端点筛结果或比较mu与rho强弱
- 以平均效应比估计中介百分比
project: tts-exp
decision: 分别报告原生配对差、固定评价音的生成响应、局部G贡献和独立视觉几何；各自完整预定统计族及支持集保持。
decided: '2026-10-02T15:02:48.4431658+00:00'
repository: dogLeft233/tts-exp
rationale: 这些端点改变的变量、分母和统计校正不同；独立口型开口增加与固定音频Sync-C下降在同一轨迹干预中并存，单一质量指标会混淆动作和对应。
consequence:
- 正式报告保留全部预定行、支持差异和INCOMPLETE状态
- 11776评分族和54几何族分别使用其共享maxT临界值
- pointwise旧结果与simultaneous新结果分别标注
- 工程控制失败保留出处，不作为科学阴性
tags:
- decision
- tts
- tfg
- evaluation
---

# 005-separate-score-motion-and-native-estimands

## Context and choice

本项目研究TTS增强TFG的原因。原生TTS/自然自评分同时改变生成输入、评价输入和时长；固定A生成干预只改变生成条件；局部G按预定窗口分母描述固定anchor距离缩减；独立嘴部几何测量动作。选择对这四种问题分别报告，不将它们合并成单一质量变化。

男/女增益路径已独立复制。新音素mu/rho五格的11776评分端点已完整复算；54几何端点也已actual独立07 PASS。N→T内部轨迹−SELF_GRID使开口增加，同时固定音频Cfixed降低，明确说明更大开口与更强对应并非同一估计。该结果按原冻结方向/支持/全族CI解释。

## Rationale and alternatives

按完整管线问题保存原生配对gap，按固定评价音问题保存G/评分响应，按输入覆盖保存局部G，按独立测量保存动作。各自分母、固定量、支持和CI直接展示，读者能知道每个因果对比改变了什么。

不采用原生Sync-C直接替代自然音轨迁移收益或感知同步；不按显著数量对mu/rho排序；不从组间均值比写出中介百分比。直接mu−rho、交互与几何各端点都保留数值/区间，不把跨零区间改写成严格无效应。

## Consequences and affected evidence

最终报告25完整保留11776评分端点和54几何端点，分别使用其共同20k speaker-bootstrap同时99%区间。FULL/VALID重复值不称独立复制；短Tg20的1472评分端点保持INCOMPLETE，phone几何16/17完整speaker与整体40分开。旧WORLD/visual逐点99%区间保留历史出处，不以新校正冒充旧结果。

影响 `openspec/changes/factor-world-phone-shape-mean-dynamics-20261001/`、最终 `basic-memory/docs/experiments/25-tts-tfg-mechanism.md`、完整CSV/PDF与source registry。独立07 full receipt在 `.codex/scratch/world_shape_visual_geometry_independent03_receipt_readback_attempt01/independent_reconstruction_receipt.json`，raw95a82b8941f6c66c947d5cddba3390a64c35ee89cb59d3abaa860bee8adb4c15。数值规范仍按ADR004。

## Observations

- [decision] 原生配对差、固定A生成响应、局部G与独立几何分别解释并展示完整预定支持和统计族。
- [rationale] 同一局部轨迹干预的开口与固定音频对应可向不同方向变化，动作幅度不等同同步或感知质量。
- [alternative] 不用显著数量排名、不用均值比宣称中介百分比、不用原生评分替代自然音轨迁移收益。
- [consequence] 正式报告同时标注pointwise与simultaneous区间，设计不完整和工程失败各按原出处保留。

## Relations

- affects [[音素平均谱形与内部轨迹的因果分解]]
- affects [[解释 TTS 的 TFG 与 SyncNet 优势]]
- relates_to [[004-canonical-float32-score-field]]