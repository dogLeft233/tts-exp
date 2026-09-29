## Why

replacement 目标仍值得做一次低成本、可否证的检查，但通用生成头尚无可训练的成功证据。历史严格 MFA replacement 为 NO_GO；bridge confirmation 未证明有效收益且控制失败；ROI 的实际生成 C gate 为 14/22；独立评分复现失败。最近线性 oracle 的 timing 为 22/22/21，own C 均值 −0.202、CI [−0.369,−0.062]，仍未通过。

这组证据不能把问题唯一归因于 Wav2Lip。连续音频变速、像素插值、评估器的时间表示，以及比较时包含不同内容窗口，仍混在一起。再搜索插值核的价值低。本轮改用一个可直接验证的参照：两段内部都是原 PCM 和原像素的整数平移，所有评分窗口远离拼接点。

## What Changes

- 固定原 22 条 seen-fit 数据；唯一新控制 P 为前半段 +5 帧、后半段 −5 帧的源索引平移。
- Stage A：CPU 构造像素 oracle，66 fresh SyncNet cells；按源内容匹配的 baseline 验证平移等价性及局部错配检测。
- 只有 A 科学通过且独立验收有效，Stage B 才用 P 驱动冻结 Wav2Lip，最多 22 个 GPU 视频、44 个额外评分。
- 两阶段均预先冻结，失败即结束；输出支持继续评估、暂停当前生成链或暂停控制评估链的证据。

## Capabilities

### New Capabilities

- `wav2lip-integer-plateau-control`: 整数平台控制、源内容匹配的 oracle 验收及实际生成链比较。

### Modified Capabilities

无。新干预和新 estimand 独立登记，不修改旧 own gate 或历史 CONTROL_FAILED。

## Impact

新增小型实验包、对应测试和独立 run；复用现有媒体、评分和 GPU worker。CPU 优先，GPU 最多生成 22 条，不需新模型或租卡。当前交付仅为 spec，未生成新音频、视频或分数。

## Non-goals

本轮不运行 bridge、TTS、MFA/DTW、训练、跨模型或 sealed 数据。不声称 oracle 等价性、控制通过或非劣性就是 replacement 收益。
