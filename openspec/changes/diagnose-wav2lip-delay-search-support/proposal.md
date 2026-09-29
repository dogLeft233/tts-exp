## Why

上一轮 natural-content-residual 在 Stage A 得到 CONTROL_FAILED：parity、重复性和固定 natural anchor 错时损伤均通过，offset 仅13/16通过，候选尚未运行。只读复核发现三个失败样本的自然最优列为27、27、28；已知音频延迟5帧后预期列为32、32、33，全部超出原0–30列。需要验证有限搜索范围是否解释失败。

历史 MAG_075 的小幅正向观察未获确认；后续谱结构实验 MAG-N ΔC=-0.890。历史shift重评分的自由ΔC=+0.341，但固定anchor收益=-0.099且CI跨零。当前值得保留的是一次有明确停止条件的自然基底辅助增量探索；本轮先解决它的控制可观测性。

## What Changes

- 新增CPU诊断，复用上一轮16条记录的N/A_DELAY embeddings和媒体，独立复现旧门禁。
- 对全部记录统一使用由已知延迟推导的配对搜索域，保留原U、natural anchor和14/16门槛；区分“修订控制通过”与“三条异常均得到解释”。
- 提供小型runner、独立validator、逐条证据及BM状态更新。此change只实现和运行诊断，下一阶段候选执行需显式协议修订。

## Capabilities

### New Capabilities
- `wav2lip-delay-search-support`: 诊断已知音频延迟下的有限offset搜索域截断。

### Modified Capabilities
- 无。父spec、代码与run均只读；新诊断不回写父CONTROL_FAILED。

## Impact

新增代码建议位于 `scripts/experiments/wav2lip_delay_search_support/`；新run位于 `runs/wav2lip_delay_search_support_<id>/`。使用32个缓存评分cell；新增TFG生成、神经网络forward、训练、TTS和GPU调用均为0。

## Non-goals

本轮不验证replacement收益、历史shift有效性或跨模型泛化，不调整生成头、候选幅度、样本、U或评分模型。诊断成功不代表内容残差有效。

## 下游入口

依次阅读本文件、design.md、specs/wav2lip-delay-search-support/spec.md、tasks.md。按tasks实现、执行、独立验收并更新同一BM笔记；不要调用父runner的all/candidates阶段。设计时尚未计算修订搜索域的结果。
