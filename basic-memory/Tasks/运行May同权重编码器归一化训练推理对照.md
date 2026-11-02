---
title: 运行May同权重编码器归一化训练推理对照
type: task
permalink: tts-exp/tasks/运行-may-同权重编码器归一化训练推理对照
status: done
current_step: 3
tags:
- may
- encoder
- normalization
- pilot
completed: October 19, 2026
---

# 运行May同权重编码器归一化训练推理对照

✅ 小规模实验完成：四臂各12000实际成功FP32更新、0跳步，24格/48矩阵/12000真实视频帧及原PCM/ROI/像素/距离独立QA通过。hubert_base匹配on/on−off/off：ΔC+0.029、ΔD0-0.049，mouth MSE相对+1.38%；wav2vec2_base匹配on/on−off/off：ΔC-0.027、ΔD0+0.069，mouth MSE相对+1.68%。这是训练/推理联合流程差异，条件内训练、推理和失配拆分见完整表。 单May/单seed/短预算，不作总体或感知结论；Patrick仍暂停。

Requirements逐条验证：准入见qualification_independent_review.json；四臂全state/trace见training_independent_review.json；24格/两clip和RGB/PCM/ROI/matrix见analysis/qa.json及visual_review.json；最终报告[[54-may-encoder-normalization-pilot]]。

## Requirements
### Requirement: SHALL 冻结同checkpoint单开关的准入
#### Scenario: 新科学执行前
- **WHEN** 准备encoder特征与生成器训练
- **THEN** 固定原May划分/processor/权重/源SHA，on/off只变归一化，真实provider时钟/输入/共同initial和采样资格通过。
### Requirement: SHALL 执行固定小预算配对训练和交叉推理
#### Scenario: 资格通过
- **WHEN** 四臂从共同随机initial开始
- **THEN** 各12000 FP32实际成功更新、同frames/rays/LR，全state与trace F鲜验；最终固定EMA交叉24格/两clip，首失败停新job并封存。
### Requirement: SHALL 独立核验并与历史结果保留条件比较
#### Scenario: 训练与推理结束
- **WHEN** 真实矩阵/像素/PCM和完整持久证据通过
- **THEN** 报告曲线、同encoder归一化主对比及失配效应、历史同clip参照和限制，实际看图并发布结果，不推感知或总体因果。

## Observations
- [status] done
- [result] 固定四臂/48000实际更新与24格/48矩阵完成，F全成员与独立QA、实际看图和报告发布通过。
- [close_date] October 19, 2026
- [limitation] 此任务完成不代表Patrick coarse10/new27或原24full完成。

## Relations
- project [[May同权重编码器归一化训练与推理小规模对照]]
