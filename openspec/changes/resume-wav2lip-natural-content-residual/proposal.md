## Why

历史直接 phone-aligned TTS mel 迁移和后续 MAG/ENV 实验未建立 natural replacement 收益；200ms shift 保留了自由 offset 分数改善，但没有共享 natural anchor 改善。masked 重建的辅助收益仍值得一次有限检验：同模型的正确条件增量加回完整 natural 后，是否有绝对收益？

原 `probe-wav2lip-natural-content-residual` 已完成 Stage A，候选尚未运行。后续 CPU 诊断证明三个异常峰超出旧搜索域，配对域恢复16/16，具备显式修订后续跑条件。本 change 完成这个尚未回答的问题。

## What Changes

- 新建版本化续跑协议，只修订已知 A_DELAY 的搜索坐标；原四臂、16条/8组、幅度、主窗口、候选评分与统计规则继承原设计。
- 只读复用父资产，重新验证修订控制；增加两个当前环境 N 复跑和两个 scorer parity，随后运行48个原定候选。
- 所有入口在候选生成前强制独立验收；新 run 输出真实科学结论、自审与 BM 更新。

## Capabilities

### New Capabilities
- `wav2lip-natural-content-residual-continuation`: 在版本化已知延迟控制修订下，完成完整自然基底辅助增量检验。

### Modified Capabilities
- 无。以新 capability 引用冻结父协议，不改写父 run 或旧 spec。

## Impact

小型 continuation runner 和独立 validator，复用现有 driver、generation、scoring、analysis；新增上限50个TFG短视频、52个评分cell，无训练/TTS/vocoder。旧 runner 的 `all` 路径未显式在候选前调用独立 validator，新入口必须落实这一约束。

## Non-goals

不重开 shift/频谱扫参，不确认句义机制、可听波形或跨模型泛化。本轮仍为已观察记录、前3.84s特征支持的探索；阳性才建议独立源组确认，阴性停止这个固定增量构造。
