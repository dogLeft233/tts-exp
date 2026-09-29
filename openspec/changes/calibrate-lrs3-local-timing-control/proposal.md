## Why

上一轮 LRS3 bridge confirmation 的 `LOCAL_SWAP` 自身配对 ΔSync-C 为 −2.900，换回 natural 反而更好，导致 `CONTROL_FAILED`。在继续音频头路线前，需要先排除输入/缓存/评分错误，再验证一个可用的局部时间对照；bridge 的特征移动不能代替下游收益证据。

## What Changes

- 对上一轮产物进行只读审计，给出逐项证据和唯一分支选择。
- 若发现可复现的工程错误，修复后仅重跑原来的三个控制臂；若审计完整且未发现错误，使用一个固定 ±120 ms 平滑时间扰动替代 `LOCAL_SWAP`。
- 两个分支互斥，均使用原来的 22 条 fit-only records、三个驱动臂、66 个新视频和 88 个评分 cell；不搜索强度。
- 固定自身配对有效性、重复性与换音频损伤门槛。输出 `CONTROL_CALIBRATED`、`CONTROL_FAILED` 或 `BLOCKED`，以及可独立复核的产物。
- 本 change 只校准对照，不重测 bridge、不训练、不访问 sealed validation/test。校准通过后，下一步仍需单独设计 bridge 收益确认实验。

## Capabilities

### New Capabilities

- `lrs3-local-timing-control-calibration`: 审计现有对照、按证据选择修复或平滑扰动分支，并执行固定矩阵的校准实验。

### Modified Capabilities

无。上一轮 confirmation 的规范和历史结论保持原样。

## Impact

- 新实现及测试分别放在 `scripts/experiments/lrs3_local_timing_control_calibration/` 与对应 `tests/experiments/` 目录。
- 复用 confirmation 的 PCM/mux、Wav2Lip、SyncNet、bootstrap 基础函数；有证据的公共工程缺陷可做最小修复并补回归测试。
- 每次执行使用新的 `runs/lrs3_local_timing_control_calibration_<run_id>/`。使用现有 NumPy 和模型环境，无新增模型依赖。
- 下游 agent 从本 change 的 `tasks.md` 开始；数值与判定契约以 delta spec 为准。
