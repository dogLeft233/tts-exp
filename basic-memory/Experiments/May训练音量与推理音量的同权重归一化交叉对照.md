---
title: May训练音量与推理音量的同权重归一化交叉对照
type: experiment
permalink: tts-exp/experiments/may-训练音量与推理音量的同权重归一化交叉对照
status: concluded
execution_status: SUCCEEDED
inputs: same May split, exact report54 initial/model/source pins
outputs: F:\Datasets\tts-exp\may_encoder_gain_norm_factorial_20261019
model: frozen HuBERT-base and wav2vec2-base, native input normalization on/off
code_paths:
- scripts/experiments/may_encoder_gain_norm_factorial_20261019
tags:
- may
- gain
- normalization
- factorial
scientific_status: ENGINEERING_PILOT_ONLY
---


# May训练音量与推理音量的同权重归一化交叉对照

已验证读数：固定生成器内，输入归一化开启时±6dB推理音量的C极差不超过约2×10⁻⁵，关闭时HuBERT最高约0.004、wav2vec2约0.031。训练音量没有跨encoder一致单调规律；固定推理0dB的六组匹配流程on/on−off/off点估计中，5/6嘴部MSE上升1.21%–3.42%，1/6下降0.84%，C/D0方向混合。稳健性不等质量收益；单May/单seed/12000步不能判普遍有害或最佳音量。

✅ 已完成：八新模型各12000成功FP32更新、0skip，复用四个已资格训练0dB模型；72条件/144矩阵，新增24000帧与96矩阵全数值/像素/PCM/ROI/时钟独立QA及实际图检通过。固定同encoder权重切norm，单May/单seed/短预算，不作总体或感知结论；Patrick仍暂停。

正式报告[[55-may-gain-normalization-factorial]]；证据根F:\Datasets\tts-exp\may_encoder_gain_norm_factorial_20261019。

## 执行前冻结设计

用户要求在不同输入音量训练和推理，比较无归一化与有归一化。固定同一HuBERT-base或wav2vec2-base预训练权重，只切native输入z-score，不换模型、不改内部norm；不是RMS0.1。May原划分、seed42完整随机initial、实际frame/ray/LR序列、FP32/4096rays/maxsteps16、原AdamW和60k LR前缀、每组12000成功更新/final EMA12保持。

训练gain −6/0/+6 ×训练norm on/off ×推理norm on/off ×驱动gain −6/0/+6 ×2encoder，共72条件/144十秒clip与矩阵。0dB四模型/24格/48矩阵只在完整来源/断点/数组SHA核验后复用报告54，不重复训练或渲染；新增8模型共96000 formal updates，入口8×2更新不进入正式训练初始。原自然PCM16评分、source-owned原GT ROI、25fps/原时钟，SyncNet FP32 TF32off，guard20主/full次。训练/推理条件效应、gain交互、错误指标与C分别解释；单身份单seed短预算，不作总体/感知或旧RMS数值因果结论。

首scientific nonfinite/job failure须保存完整state/trace/job到F，独立复核并停止后续新job；无自动replay、精度/scaler/clip调整或换gain。全部新fullcp保留。F主根和E独立media namespace分摊空间；旧报告50云整包及643成员fresh独立Linux读回后，才允许按固定名单迁移其old renders的9mkv/9selectedfloat到既有云包，保留locator、SHA、JSON/PNG/score/最终cp。未经完整proof不删本地，CloudDrive配置/共享venv/其他科学资产不动。Patrick三档coarse7及旧队列持续暂停。

证据根：F:\Datasets\tts-exp\may_encoder_gain_norm_factorial_20261019；protocol SHA4fdf3fa68595be8032839e1a5cfff1c5839b51c650440e619113055755e51518。


## Observations
- [status] concluded
- [result] 72格/144矩阵完整，输入norm压低推理gain响应至约2e−5；匹配流程嘴部MSE有5/6略升，质量方向混合，详见报告55。
- [limitation] 不外推RMS0.1或完整训练/感知。

## Relations
- concluded_in [[55-may-gain-normalization-factorial]]
- follows [[54-may-encoder-normalization-pilot]]
