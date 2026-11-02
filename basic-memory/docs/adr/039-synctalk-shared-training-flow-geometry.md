---
title: 039-synctalk-shared-training-flow-geometry
type: decision
permalink: tts-exp/docs/adr/039-synctalk-shared-training-flow-geometry
status: open
project: tts-exp
decided: 2026-10-17
---

# 039-synctalk-shared-training-flow-geometry

## Context
四新身份官方完整训练电平实验的独立录制测试需训练专属几何。ADR038/v1尝试用train BFM初始与官方BA细化相机拟合单一相似变换；Brown实际旋转p95 6.675°、平移14.896mm，超过2°/5mm门。其三split landmark/flow单项通过不能替代共坐标资格。官方联合BA自由更新每帧相机和3D点，细化并非只改变一个全局坐标系，因此v1相机配对残差混合了真实细化与坐标自由度。v1 failed JSON/log/几何保留，不能改写通过。

## Decision and rationale
原版训练BFM、原版500+8000联合BA及最终训练相机完全保留。只由训练BA相机与原版训练NeuralNRT flow tracks三角化训练参考帧2的固定3D关键点；heldout沿用同一训练参考图/alpha mask/选点，原NeuralNRT对heldout每帧建立对应。固定这些训练3D点不更新，用原版flow重投影loss、Adam1e-4、8000步仅拟合heldout Euler/trans，直接得到训练BA坐标的相机。heldout不再拟合自己的点/形状，也不在最后套全局相似变换。原train-only BFM/BA相似变换仅作优化初值，旧残差仍报告。

将坐标验收改为实际训练内相机恢复：固定train前4800帧三角化点，用train3600:4800和4800:6000两个固定控制块重求相机，与原train BA相机比较。旋转p95仍≤2°、平移仍≤5mm；landmark median/p95≤5/10px、flow≤2/5px与1e6条件数限值保留。三角化条件数、点SHA不变、原六帧dense flow重算、全heldout投影误差必须通过。训练id/focal/background、lazy-loader等价门、全部训练/eval预算不变。

v2在任何TFG新训练/生成前冻结于scripts/experiments/synctalk_multispeaker_full_20261017/preprocessing_contract_v2.json，SHA37f0714cecd41cca0f531bb062b6029bad2e96fe5bc19bd3ad0827d61f3dd005。冻结前做了几何诊断：Brown前4800校准点在最后1200帧恢复原BA，旋转p95 .150257°、平移.185434mm；两段1200控制均通过旧相机误差数量门。跨录制val/test各20均匀抽样frame用共享点，flow p95 3.044/1.956px；40帧对应图已人工看过，点落在脸部，未删帧。明确这些是方法诊断、使用了heldout图像作姿态资格而没有使用SyncNet处理效应分数；不能将抽样成功写为全量通过。

## Alternatives and consequences
放宽v1阈值会掩盖错误坐标假设，拒绝。用每录制自由id/BA拟合后直接拼接会令生成器测试坐标失配，拒绝。只用BFM姿态放弃官方训练BA会改变训练预处理，拒绝。利用测试目标更新训练形状/点/背景或共同BA会使训练状态包含测试信息，拒绝。

共享训练参考点保留训练官方流程、为独立录制提供直接共坐标，但官方没有这项默认跨录制流程，须称“heldout conditioning实现适配”，不能冒充原样官方预处理。相同人跨录制外观变化会使NeuralNRT匹配失败，全部帧必须过门，失败就不训练，不按SyncNet挑人物/clip。点三角化不是原脚本末尾未保存的pts，需相机恢复控制证明其对原BA姿态的兼容。初次诊断Euler符号误用已由roundtrip断言检测，attempt01诊断保留标无效；v2采用官方Rx(theta)Ry(phi)Rz(-psi)并测试，不影响原v1矩阵残差。

## Observations
- [decision] 新heldout姿态直接观测冻结的训练3D关键点，训练原版BA不变；相似映射只作初值。
- [decision] 显式版本化修复坐标模型，保留v1失败与旧阈值数量，使用训练内pose恢复作实际坐标验收。
- [result] Brown训练内两控制及跨录制40帧方法诊断通过，但全量v2资格未完成，训练数仍0。

## Relations
- refines [[038-synctalk-training-owned-geometry-and-lazy-loading]]
- implements [[SyncTalk 四新身份官方完整训练电平实验 Spec]]