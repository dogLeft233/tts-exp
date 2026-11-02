---
title: Wav2Lip ROI retiming oracle 2026-09-06
type: experiment
permalink: tts-exp/experiments/wav2-lip-roi-retiming-oracle-2026-09-06
tags:
- wav2lip
- roi
- retiming
- oracle
- control
- concluded
status: concluded
---

# Wav2Lip ROI retiming oracle 2026-09-06

本实验承接 [[Wav2Lip ROI local peak recheck 2026-09-06]] 与 [[Wav2Lip face-ROI replacement pilot 2026-09-06]]。上一轮独立 SyncNet forward 已复现 8/8 历史局部峰失败和 2/2 通过，但历史科学终态仍为 `CONTROL_FAILED`，own-audio 未重测，bridge、训练和泛化均未授权。

本轮按 OpenSpec `diagnose-wav2lip-roi-retiming-oracle`，固定父实验完整 22 条 seen-fit cohort。直接从现有自然音频驱动 ROI 视频 G_N 的无损像素帧构造恒等臂 V_ID 和已知像素重定时臂 V_ORACLE，再与 N/W 音频交叉评分。目标是判断当前局部 timing 判据在“已知视频重定时”参照上的有效性，不重新生成 Wav2Lip 视频，不把 oracle 结果解释为实际生成器因果证据。

## Observations
- [status] concluded
- [execution] 正式 run 为 `runs/wav2lip_roi_retiming_oracle_20260907_oracle_v4/`：固定 22 records / 22 source groups，构造 44 个无损像素视频流并完成 88 个新 SyncNet cells；没有重新生成 Wav2Lip 视频、TTS、训练或 bridge。
- [validation] runner engineering decision 为 `GO`；独立 validator 为 `valid=true`，`independent_difference_count=0`，说明媒体、PCM、像素、embedding/matrix 和统计复算一致。
- [result] 恒等臂复现通过：22/22，最大矩阵差 0.000072002，小于 0.001；baseline=22/22；B=22/22、C_oracle=21/22、O_oracle=22/22。
- [result] oracle own-audio gate 未通过：C 均值 -0.469851，95% CI [-0.633657, -0.318777]；D 均值 -0.360790，95% CI [-0.507712, -0.230130]；offset agreement=22/22，但 CI 下界未达到预注册的 -0.10 门槛。
- [result] V_ORACLE 相对 V_ID 的 damage 描述性指标为正：C 均值 4.650019，95% CI [4.216514, 5.081484]；D 均值 4.476869，95% CI [4.078104, 4.865844]；both-positive=22/22。但由于 own-audio gate 失败，不能把该结果解释为控制通过。
- [decision] 本轮诊断终态为 `ORACLE_OWN_AUDIO_UNRESOLVED`；历史 `CONTROL_FAILED` 保持不变，`reference_conditioned_audio_head_spec_eligible=false`，bridge、训练和泛化均未授权。
- [boundary] V_ORACLE 只是从 G_N 画面构造的已知像素重定时参照；本轮没有证明实际 Wav2Lip 生成器传递 timing，也没有证明 replacement effect。
- [fix] 为适配当前 ffmpeg 8，编码命令使用 `-colorspace rgb`，ffprobe 仍报告冻结契约要求的 `color_space=gbr`；音频产物按 protocol sample id 命名，validator 与 runner 已一致。
- [artifacts] 结果详见 `runs/wav2lip_roi_retiming_oracle_20260907_oracle_v4/result.md`、`final.json`、`validation.json`；OpenSpec strict validation 通过，14 个相关测试通过。
- [next_step] 不进入 training、bridge 或泛化；如需继续，必须另立最小 OpenSpec 专门解释/修复 oracle own-audio gate，且不能改写本轮负结论。
## Relations
- implements [[openspec/changes/diagnose-wav2lip-roi-retiming-oracle]]
- relates_to [[Wav2Lip ROI local peak recheck 2026-09-06]]
- relates_to [[Wav2Lip face-ROI replacement pilot 2026-09-06]]
- relates_to [[LRS3 真实视频局部时间敏感性诊断]]
- relates_to [[LRS3 Wav2Lip 局部时间传递诊断结果]]