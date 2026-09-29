## Why

历史 SHIFT_200 重评分保留了自由 offset 下的正向分数观察，但共享 natural anchor 未改善，评分入口的配对差异也未确认。它既没有确认 replacement，也没有定位出评分程序错误；偏移分支已收口。

另一条历史线索是 masked reconstruction 的 TTS 辅助收益，但其参照 NAT_ONLY 本身缺失了目标自然音频，不能代表完整 natural baseline。下一步值得做一次有限的桥接检验：已有模型学到的辅助增量，能否改善未遮挡的 natural driver？

## What Changes

- 新增一次无需训练、无需新 TTS 的 direct-mel 探索实验：固定16 records/8 groups，复用三个已训练模型的缓存，先平均辅助增量再生成。
- 四臂：完整未遮挡的自然 mel、自然 mel + 正确配对增量、自然 mel + 同音素错实例增量、自然 mel + 时间打乱增量。
- 固定静态 face、相同自然 PCM、共同内部窗口与 natural offset；先通过接口/重复/评分敏感性检查，再运行候选。
- 下游提供小实验包、独立验收、结果报告及同一 BM 笔记状态更新。

## Capabilities

### New Capabilities
- `wav2lip-natural-content-residual`: 检验已有自然重建模型的内容辅助增量能否超越未遮挡自然基线。

### Modified Capabilities
- 无。

## Impact

最多66次短视频生成、84个CPU评分cell；新增训练为0。新增代码建议放 `scripts/experiments/wav2lip_natural_content_residual/`，结果放独立 runs 目录。既有模型、数据、run及旧门禁保持不可变。

## Non-goals

本轮不是新模型训练或独立确认，不验证整段长音频、可听波形或跨TFG泛化。这里的“内容”指语音内容条件，不等同于句子语义理解。正结果最多支持另写独立确认 spec；负结果只停止该缓存增量构造。
