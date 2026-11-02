---
title: 运行May训练推理音量与归一化交叉对照
type: task
permalink: tts-exp/tasks/运行-may-训练推理音量与归一化交叉对照
status: done
priority: high
current_step: 3
tags:
- may
- gain
- normalization
completed: '2026-10-20T01:47:12.459335+08:00'
---

# 运行May训练推理音量与归一化交叉对照

✅ 已完成：八新模型各12000成功FP32更新、0skip，复用四个已资格训练0dB模型；72条件/144矩阵，新增24000帧与96矩阵全数值/像素/PCM/ROI/时钟独立QA及实际图检通过。固定同encoder权重切norm，单May/单seed/短预算，不作总体或感知结论；Patrick仍暂停。

逐项证据：来源/缓存/入口见parent_full_member_import_review和feature/qualification独立review；八新模型见training_independent_review；完整72格见analysis/qa、visual_review和正式报告[[55-may-gain-normalization-factorial]]。

## Requirements
### Requirement: SHALL 冻结并核验完整因素、复用来源和正式准入
#### Scenario: 旧0gain与新±6gain同源
- **WHEN** 两encoder±6训练缓存、共同完整random initial、frame/ray和16次入口更新通过
- **THEN** 保存protocol/source pins、旧四组SHA导入review和独立资格review，无旧训练warmstart或科学重放。
### Requirement: SHALL 完成八个新等预算模型并保留全部完整断点
#### Scenario: 每组12000次成功更新
- **WHEN** 所有新模型按固定初始和采样执行
- **THEN** 每组fullstate/trace与F全成员SHA通过，或首真实失败封存并停止全部新job、Task blocked。
### Requirement: SHALL 完成72格交叉评测和独立交付
#### Scenario: 新旧同合同矩阵合并
- **WHEN** 八新模型和四已核验旧0gain模型固定final EMA12
- **THEN** 144clip/matrix完整、原PCM/时钟/像素/ROI/距离QA，输出训练×推理音量及归一化差异表/图/报告，实际view_image，BM读回和跨卷最终manifest通过；旧资产迁移只有fresh云全包成员proof后执行。

## Observations
- [status] done
- [result] 全部固定Scenario完成，跨卷全成员读回、独立QA与实际图检通过。
- [close_date] October 20, 2026
- [limitation] 此Task完成不代表旧Patrick或原24full完成。

## Relations
- project [[May训练音量与推理音量的同权重归一化交叉对照]]
- follows [[运行May同权重编码器归一化训练推理对照]]
