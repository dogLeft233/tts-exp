## Why

先在 Wav2Lip 建立可复核的 replacement 效应，再决定是否推广到其他 TFG。当前尚不能直接训练生成头：历史 timing-transfer v8 的 A/B 为 22/22、21/22，C/O 为 0/22；bridge confirmation 的控制失败，replacement Sync-C 增益 CI 跨零。

代码检查发现历史生成端使用 `constant_full_frame_fallback`，把整帧作为人脸送入 96×96 的模型输入。这是一个需要控制的生成条件，不是已经证明的失败原因。本实验仅改变生成端人脸区域，保留音频候选、模型权重和评分规则。

## What Changes

- 增加一个两阶段、无训练、22 条 fit-only 的实验：A 先验证人脸 ROI 下的局部时序响应；只有 A 通过才执行 B 的固定 `BRIDGE_075` replacement 检验。
- 生成端从原视频预提取并冻结逐帧人脸框，各音频臂共用；评估端继续使用历史真实视频裁脸轨迹。
- 最多 88 个新 Wav2Lip 视频、242 次 SyncNet 评分；A 不通过即停，不搜索参数或换模型救结果。
- 通过仅记为 `WAV2LIP_REPLACEMENT_PILOT_PASS`，授权提出下一份跨模型验证 spec，不等于泛化成立或可训练。

## Capabilities

### New Capabilities

- `wav2lip-face-roi-replacement`: 人脸 ROI 下的 Wav2Lip 控制与 replacement 分阶段验证。

### Modified Capabilities

None. 历史实验契约及终态保持只读。

## Impact

- 后续实现位于 `scripts/experiments/wav2lip_face_roi_replacement/`，对应测试位于 `tests/experiments/wav2lip_face_roi_replacement/`。
- 复用已有音频构造、官方评分 worker、局部矩阵分析和媒体验证，仅增加 ROI 适配与阶段门控。
- 本次只交付 spec；不运行实验、不部署 Ditto/LeapTalk、不租卡、不训练、不访问 sealed 数据。
