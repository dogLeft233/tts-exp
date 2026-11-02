---
title: 运行May归一化后输入尺度训练推理对照
type: task
permalink: tts-exp/tasks/运行-may-归一化后输入尺度训练推理对照
status: done
priority: high
current_step: 3
tags:
- may
- postnorm
- input-scale
completed: '2026-10-20T11:01:30.326693+08:00'
---

# 运行May归一化后输入尺度训练推理对照

✅ 已完成：两个低尺度模型各12000成功FP32更新/0skip，两个标准训练模型原件复用；12条件/24矩阵，新增20视频/5000帧及151900距离独立QA、实际三图检查与断电后freshSHA复核通过。wav2vec2低训练/高推理相对标准C+0.093、D0−0.051、嘴部MSE−1.66%；固定低训练后再调高驱动额外仅约+0.001。HuBERT C近不变且D0恶化，单May/seed不判普遍或感知收益。

逐项证据：14缓存公式/SHA与4真实入口fullstate见feature/qualification review；两新完整模型见training_independent_review；12格/24矩阵与实际图检见analysis/qa及visual_review，scientific_manifest读回通过。断电后239成员fresh检查、父758成员全SHA及四finalcp/trace核验、151900新距离重建通过，0科学重放。

## Requirements
### Requirement: SHALL 核验归一化后尺度、共享初始化和基线来源
#### Scenario: 实际输入只在z-score之后缩放
- **WHEN** 原模型/数据/initial与标准基线freshSHA通过，14条件缓存及4真实入口更新/fullstate roundtrip通过
- **THEN** 保存公式顺序/实际RMS/样本特征SHA和独立资格review；标准训练不重放，新训练不warmstart。
### Requirement: SHALL 完成两个新等预算低尺度模型或封存首故障停止
#### Scenario: 每模型12000成功FP32更新
- **WHEN** 新模型从完整random initial执行固定预算
- **THEN** 全部fullcp/trace/job在F逐成员核验，实际frame/ray/LR与原标准组逐行一致；首真实故障后不发新job。
### Requirement: SHALL 完成12条件交叉评测并独立交付
#### Scenario: 固定final EMA12与原自然PCM
- **WHEN** 两新模型和两原标准模型资格通过
- **THEN** 24clip矩阵（20新增4导入）及像素/PCM/ROI/时钟/距离QA通过，输出尺度响应/交互与报告55参照，真实图检、BM发布、最终manifest读回；未满足不标done。

## Observations
- [status] done
- [result] 全部Requirements场景已验证；正式报告56和最终交付manifest绑定结果与发布。
- [close_date] October 20, 2026
- [limitation] 此Task完成不代表旧Patrick/原24full完成，旧队列仍暂停。

## Relations
- project [[May归一化后低尺度训练与高尺度驱动对照]]
- concluded_in [[56-may-postnorm-scale]]
- follows [[运行May训练推理音量与归一化交叉对照]]
