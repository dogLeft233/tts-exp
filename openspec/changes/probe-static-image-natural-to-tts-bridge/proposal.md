## Why

历史 bridge 探索轮 MAG_075 在 18/23 样本上提升 Sync-C（均值 +0.078），确认轮 BRIDGE_075 仅 9/22 提升（均值 +0.031，95% CI 跨零）。这两轮使用动态说话视频；LOCAL_SWAP 重放说明现象可复现，却未唯一定位原因。需要直接检验：固定单张参考图后，同一 bridge 波形能否比 natural 驱动产生更匹配原自然音频的画面。

## What Changes

- 冻结确认轮全部 22 条 seen-fit 记录，复用原 N、BRIDGE_075、LOCAL_SWAP PCM，不重新搜索强度。
- 每条只用源视频第 0 帧 PNG；固定真实人脸框、评分 crop 和时间支持，所有生成臂视觉输入相同。
- 设置 N、N_REPEAT、RT（仅 STFT 往返重建）、B（原 bridge）、S（原 LOCAL_SWAP）五个生成臂；B 必须同时比较 N 与 RT。
- 先用重复生成及同视频配延迟音频检验测量；LOCAL_SWAP 的局部双向配对偏好作为独立时序诊断，不要求 S 的自身分数与 N 等效。
- 最多 110 个新生成视频、198 个评分 cell；输出逐样本提升比例、效应量/区间、距离曲线、可播放对照和独立验收结果。

## Capabilities

### New Capabilities

- `static-image-natural-to-tts-bridge`: 静态单图输入下的冻结 bridge 收益实验、对照、可审计评分及结论边界。

### Modified Capabilities

无。历史协议及阴性结论保留。

## Impact

本次交付仅为 spec、design 和实施任务，另按用户要求更新 BM 结果。未来实现限定在 `scripts/experiments/static_image_bridge/`、对应测试及新的 `runs/static_image_bridge_<run_id>/`。复用现有 Wav2Lip/SyncNet 和本地音频资产；不训练、不调用 TTS/ASR/MFA、不下载模型、不访问封存集。

该队列已被观察，结果是机制探索。静态输入与旧动态实验同时涉及生成框和评分口径变化，不能用历史均值差直接证明原嘴型泄漏。
