## Why

历史 SHIFT_200（natural 延迟 200 ms 后驱动 Wav2Lip，换回 untouched natural 音轨）在 23 条记录上出现 ΔSync-C=+0.298，95% CI=[+0.131,+0.471]；D benefit=+0.164，CI 跨零，而且最佳 offset 只改变 0–1 帧。这是待解释的分数信号，尚非真实口型改善的确认。后续局部 warp、谱迁移和 endpoint 对账都没有直接复验这个全局偏移干预。

在研究新的语义辅助生成头前，做一次有预算上限的诊断：该信号在共同内部时间窗口上是否保留？它是否依赖动态人脸输入？生成口部是否跟随音频偏移？

## What Changes

- CPU 复算历史 N、SHIFT_200 的 69 个缓存 cell，保留原评分与内部窗口结果。
- 从已有 22 条 seen-fit ROI cohort 按固定顺序取 12 个 source groups，每组一条。
- 新生成动态／静态两种人脸设置，每种仅 N、N_REPEAT、DELAY_200、ADVANCE_200 四个 driver；最多 96 个生成视频、192 个评分 cell。
- 全部 replacement 使用 untouched N；own-audio 与同视频错配 cell 只用于诊断。报告完整曲线、固定自然基线 offset 的距离、自动搜索 offset 的 C/D、简单像素时序响应。
- 输出可复核结果和 BM 记录，不训练模型。本轮结束即收口，语义辅助方向另立 spec。

## Capabilities

### New Capabilities

- `wav2lip-global-shift-response`: 冻结全局偏移、人脸设置、评分支持与诊断解释，禁止把分数上升直接写成真实同步或泛化收益。

### Modified Capabilities

None.

## Impact

新增小包 `scripts/experiments/wav2lip_global_shift_response/` 及对应 tests，复用现有 ROI generation worker、PCM mux、SyncNet distance-matrix scorer。结果写新 `runs/wav2lip_global_shift_response_<id>/`，历史资产只读。

## Non-goals

不复活谱迁移构造，不训练语义／波形头，不扫描偏移幅度，不换模型，不访问 sealed splits，不证明纯 mouth leakage 或感知质量改善，不重写历史 CONTROL_FAILED。
