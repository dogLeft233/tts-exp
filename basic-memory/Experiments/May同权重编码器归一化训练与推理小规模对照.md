---
title: May同权重编码器归一化训练与推理小规模对照
type: experiment
permalink: tts-exp/experiments/may-同权重编码器归一化训练与推理小规模对照
status: concluded
execution_status: SUCCEEDED
tags:
- may
- encoder
- normalization
- pilot
scientific_status: ENGINEERING_PILOT_ONLY
---

# May同权重编码器归一化训练与推理小规模对照

✅ 小规模实验完成：四臂各12000实际成功FP32更新、0跳步，24格/48矩阵/12000真实视频帧及原PCM/ROI/像素/距离独立QA通过。hubert_base匹配on/on−off/off：ΔC+0.029、ΔD0-0.049，mouth MSE相对+1.38%；wav2vec2_base匹配on/on−off/off：ΔC-0.027、ΔD0+0.069，mouth MSE相对+1.68%。这是训练/推理联合流程差异，条件内训练、推理和失配拆分见完整表。 单May/单seed/短预算，不作总体或感知结论；Patrick仍暂停。

正式报告[[54-may-encoder-normalization-pilot]]；F证据F:\Datasets\tts-exp\may_encoder_normalization_pilot_20261019。

## 冻结设计（执行前记录）
用户授权用May小规模检验主流编码器归一化在训练与推理中的影响，并强调更换编码器不能排除其他因素。因此因果主比较限定同一checkpoint内仅切换输入归一化，原AVE仅历史参照，Patrick保持coarse7暂停。

冻结HuBERT-base与wav2vec2-base两个既有固定revision/权重SHA，各on/off两次独立训练，seed42共享全随机768输入生成器initial；每臂12000成功FP32更新/4096rays，原首阶段LR的前12000步，不加速调度，不用训练权重warmstart。内部norm和encoder权重不改。归一化使用实际HF处理器原公式，完整split后再20秒分块，不把RMS0.1冒充原生zscore。

沿用报告50的4150/550/1100真实监督帧及原时钟。每模型交叉测试推理norm on/off与−6/0/+6gain，两编码器共24cells；只预定test_00和test_03两10秒clip，固定最终EMA12、原PCM16、原GT共享ROI、SyncNet FP32 TF32off，guard20主/full次，报告C/Dmin/D0/B/offset和全图/口部框MSE。旧60k结果仅描述参照；不同编码器、维数或预算的分差不归因归一化。单身份单seed不作总体或感知推断。

工程门：实际模型权重/SHA和processor开关、完整共同初始化、frames/rays/RNG/LR一致；重复encoder与像素读回；每步finite state/gradient，闭合epoch全state/trace F持久。首真实数值或job失败封存并停止新任务，不盲重试。固定预算不按结果扩训。该段为执行前冻结设计；结果以正式报告和真实产物为准。

## 产物
`F:\Datasets\tts-exp\may_encoder_normalization_pilot_20261019`；protocol.json为冻结机器协议，初始时尚无结果。


## Observations
- [status] concluded
- [result] hubert_base匹配on/on−off/off：ΔC+0.029、ΔD0-0.049，mouth MSE相对+1.38%；wav2vec2_base匹配on/on−off/off：ΔC-0.027、ΔD0+0.069，mouth MSE相对+1.68%。这是训练/推理联合流程差异，条件内训练、推理和失配拆分见完整表。
- [limitation] 只单May/单seed/12000更新，不外推RMS0.1/FP16 Patrick数值失败，不作感知结论。

## Relations
- concluded_in [[54-may-encoder-normalization-pilot]]
- follows [[53-encoder-input-normalization-ablation]]
- relates_to [[50-synctalk-may-gain-training-pilot]]
