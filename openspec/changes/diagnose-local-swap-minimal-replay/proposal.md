## Why

历史 LOCAL_SWAP 视频配自身音频的平均 Sync-C 为 2.543，配自然音频却为 5.331。这个反向现象已见于历史分数，但尚不能据此认定 SyncNet 错误或生成器失效。需要一个能直接观看、复算、定位问题的小实验。

## What Changes

- 全量只读核对历史 22 条记录的两个 LOCAL_SWAP cell；新评分仅用 cohort 原顺序的前三条。
- 用原生成视频复跑历史评分，并将同一份固定视频裁剪分别配两种音频直接评分，检查封装、缓存和裁剪链路。
- 用真实视频构造“原视频/同步交换视频 × 原音频/同步交换音频”的 2×2 对照。正确同步关系由帧和 PCM 的排列确定，不依赖 Wav2Lip 是否成功生成。
- 共 3 条、24 个新评分 cell，零次 Wav2Lip 推理；输出逐条分数、局部距离曲线、可播放对照和独立结论。

## Capabilities

### New Capabilities

- `local-swap-minimal-replay`: 对历史 LOCAL_SWAP 反向评分进行最小重放及真实音视频交换诊断。

### Modified Capabilities

无。

## Impact

下游实现放在 `scripts/experiments/local_swap_minimal_replay/`，对应测试放在 `tests/experiments/local_swap_minimal_replay/`，新产物使用独立 `runs/local_swap_minimal_replay_<run_id>/`。复用官方 SyncNet 权重及前向计算，记录实际执行代码 hash。

已有 `calibrate-lrs3-local-timing-control` 和 `diagnose-lrs3-real-video-local-timing` 涉及更大规模的校准与平滑扰动。本方案只复核原始四分段交换现象，用真实配对交换提供直观对照；可复用经检查的实现，不自动重跑这些旧方案。

## Non-goals

不训练、不生成 TTS、不重新对齐、不重测 bridge、不搜索扰动强度、不改写旧 run。此三样本诊断不能确认 replacement 收益、普遍否定 SyncNet 或授权 audio head。
