# Reference conditioning interaction probe

## Why

上一轮只在第一帧静态参考图上检验音频增量。固定干预可能受参考外观条件影响；在不训练、不改音频干预的情况下，用同一人的第二张固定参考图做2×2交互检查，可以决定今后是否必须把图像条件纳入控制与评估。

## What Changes

- 音频driver N/CORRECT × 静态参考F0/F46；复用F0两cell，只新增F46。
- 主问题为音频效应的配对差中差，不以换图后的绝对分数作增益证据。
- 对新参考单独执行natural repeat和已知音频延迟控制。

## Impact

独立包 `wav2lip_reference_conditioning_interaction` 及其测试/run，依照公共并行契约。此实验诊断reference dependence，不证明嘴部姿态因果、可用生成头或跨模型收益，不改变旧残差终态。
