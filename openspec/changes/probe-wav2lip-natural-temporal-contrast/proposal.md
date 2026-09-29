# Natural temporal contrast probe

## Why

旧TTS谱迁移与内容残差未建立replacement收益。另一种可低成本证伪的假设是：natural的局部时间变化强度不适合Wav2Lip，而非其音色需要靠近TTS。以固定、对称、无时间搬移的算子测试，不训练、不搜索最优参数。

## What Changes

- 新增N/SMOOTH/SHARP三臂seen-record探索；N为完整natural mel，不是重建基线。
- 固定两方向和幅度，共32个候选视频，父控制与跨会话复验后执行。
- 保留原N音轨和固定natural anchor，最多只给独立确认建议。

## Impact

独立包 `scripts/experiments/wav2lip_natural_temporal_contrast/`，相应测试和run；遵循 `../../parallel-replacement-probes-20260909.md`。不改共享模型代码。不覆盖旧natural-policy阴性；不声称mel结果已能生成可用波形或跨TFG泛化。
