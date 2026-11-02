---
title: 038-synctalk-training-owned-geometry-and-lazy-loading
type: decision
permalink: tts-exp/docs/adr/038-synctalk-training-owned-geometry-and-lazy-loading
status: open
project: tts-exp
decided: 2026-10-17 10:00:55.713641+00:00
---

# 038-synctalk-training-owned-geometry-and-lazy-loading

## Context
四身份完整SyncTalk电平实验要求同人不同录制测试、训练几何/焦距不利用测试监督，以及官方完整训练预算。独立拟合每场身份再直接合并会引入任意几何坐标差异。官方bundle adjustment联合自由3D点与相机也有相似变换自由度。6000个512平方训练帧的RGB+RGBA float32预加载至少44040192000字节，超过31GiB主机物理内存，np.stack还会产生额外峰值。

## Decision and rationale
训练录制单独运行固定官方BFM tracker与原版联合bundle adjustment；验证/测试固定训练身份和焦距，仅拟合表达/姿态。验证/测试bundle先按官方500步固定相机拟合3D点，再冻结这些点并用同一原版光流重投影损失细化8000步姿态，防止评价录制另起任意点/相机坐标自由度。只由训练BFM初始和训练BA相机配对估计一个正尺度相似变换，统一评价相机到训练坐标，检查正交性、训练残差及测试投影恒等式。此评价端约束是同人不同录制留出的实现适配，明确区别于官方默认每段自由重拟合；生成器网络、训练loss和两阶段预算保持官方代码。

使用官方--preload 0解决内存，必须先在共同小支持上与preload1逐像素和ray输入核验等价；不缩rays或改精度/epochs。官方portrait彩色mask通过255-G形成三同值alpha通道才交给光流，不能直接复制RGB可视化。光流保留原模型/crop/12iters、原Laplacian选点块，存首次2500帧刚性像素与全部关键点轨迹，并验证稠密子集一致以避免几十GB稠密输出。

## Alternatives and consequences
不采用独立测试身份重拟合后直接拼pose、不让测试决定训练内参/背景、不绕过flow/BA冒充完整预处理。原版全量float32预加载在当前主机不可行；降低65536 rays或沿用May4096/FP32会改变实验预算。按需读取仅改变I/O，仍须实际输入等价和完整预算预检。

preprocessing_contract_v1.json在任何新身份/焦距/光流全量/BA或生成器训练前冻结具体约束与数据质量阈值。所有阈值独立于SyncNet排名。合格仍须四人全部人脸/音画/坐标门、原版完整阶段、相同初始化与实际optimizer更新记账；约束未通过不准入训练。7单元测试与3帧原NNRT光流子集已经通过，但不替代全量数据门。

## Observations
- [decision] 训练拥有身份、焦距、背景与坐标映射；评价只条件化自己的姿态/表情/眼部，不回写训练状态。
- [rationale] 固定训练几何和由训练估计的正相似变换使不同录制pose可比较，冻结评价BA点避免额外任意gauge。
- [alternative] 未采用独立身份重拟合、跳过姿态细化、缩rays或改precision作为内存补救。
- [consequence] 使用官方按需读取前必须证明相同像素/ray输入；完整共同初始化、采样和成功更新预算另验。

## Relations
- affects [[SyncTalk 四新身份官方完整训练电平实验 Spec]]
- affects [[准备并核验四个新身份的 SyncTalk 完整训练数据]]