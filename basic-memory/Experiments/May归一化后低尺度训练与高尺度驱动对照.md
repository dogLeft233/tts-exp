---
title: May归一化后低尺度训练与高尺度驱动对照
type: experiment
permalink: tts-exp/experiments/may-归一化后低尺度训练与高尺度驱动对照
status: concluded
execution_status: SUCCEEDED
inputs: report54 same initial/data/encoder pins; post-zscore scales
outputs: F:\Datasets\tts-exp\may_postnorm_scale_20261020
model: frozen HuBERT-base/wav2vec2-base
code_paths:
- scripts/experiments/may_postnorm_scale_20261020
tags:
- may
- postnorm
- input-scale
scientific_status: ENGINEERING_PILOT_ONLY
---

# May归一化后低尺度训练与高尺度驱动对照

✅ 已完成：两个低尺度模型各12000成功FP32更新/0skip，两个标准训练模型原件复用；12条件/24矩阵，新增20视频/5000帧及151900距离独立QA、实际三图检查与断电后freshSHA复核通过。wav2vec2低训练/高推理相对标准C+0.093、D0−0.051、嘴部MSE−1.66%；固定低训练后再调高驱动额外仅约+0.001。HuBERT C近不变且D0恶化，单May/seed不判普遍或感知收益。

正式报告[[56-may-postnorm-scale]]；证据根F:\Datasets\tts-exp\may_postnorm_scale_20261020。科学作业断电前已全部自然rc0，断电只清空易失SHM，F结果完整；未重训/重评分或新GPU调用。

## 执行前冻结设计

用户授权检验低于归一化电平训练、高于归一化电平推理。固定HuBERT-base和wav2vec2-base各自原权重，完整split的float32 native z-score后再乘0.5/1/2，随后直接进入编码器；无第二次归一化、clip、limiter、投影或内部norm变更。目标近似RMS0.5/1/2，实际均值/RMS/峰值及样本SHA逐条件记录，1e−7导致目标仅近似。此处倍数相对z-score输出，绝非原录音−6/0/+6dB标签。

新训练每encoder仅scale0.5一组，各12000成功FP32更新、0skip，总24000formal；scale1标准训练直接核验并复用报告54同initial/采样/预算/EMA12的完整断点，不重训。每个固定模型交叉推理scale0.5/1/2，共12条件/24十秒clip，两个scale1训练×scale1推理的标准cell精确导入，新增10cell/20clip。所有新尺度特征实际编码，不能复用旧原录音gain或归一化前gain结果冒充。新模型从同SHA完整随机initial、seed42起，不warmstart旧训练权重；原AdamW/60k LR前缀/4096rays/maxsteps16、原4150/550/1100帧、固定test_00/test_03不变。

原自然PCM16仅评分/换轨，source-owned历史GT ROI，25fps原时钟；SyncNet FP32 TF32off，guard20主/full次；C、D0、背景、嘴部像素MSE/L1和AV offset分开。主要比较low/high−standard/standard、固定推理下训练scale差、固定模型下推理scale差及交互，不按测试选模型/尺度/预算。单May/seed短预算，非感知/总体有益证明。首真实非有限/lost/job故障封存完整state/trace/jobs并停止后续，不自动replay/restart/改precision或scaler。Patrick/旧队列持续暂停。

持久根F:/Datasets/tts-exp/may_postnorm_scale_20261020；独立新remote /dev/shm/may-postnorm-scale-20261020，旧封存F/E科学包保持；只读复用SHM父数据/官方源，不改共享venv/CloudDrive，不抢活动GPU锁。4CPU测试与4真实入口更新/fullstate资格已通过，资格单独计数且未warmstart正式训练；完整科学结果和断电读回见正式报告。


## Observations
- [status] concluded
- [result] 两新24000成功更新/0skip，12格/24矩阵完整；高推理额外C效应很小，详细条件差与交互见报告56。
- [conclusion] 当前不支持低训练高驱动的跨encoder一致优势；wav2vec2单次训练组改善仍需独立训练重复。
- [limitation] 不外推旧RMS0.1/FP16、总体或感知。

## Relations
- concluded_in [[56-may-postnorm-scale]]
- follows [[55-may-gain-normalization-factorial]]
