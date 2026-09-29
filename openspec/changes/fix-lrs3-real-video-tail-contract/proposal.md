## Why

真实视频局部 timing 实验在 prepare 阻塞：固定 22 条中有 3 条音频尾部超过视频 1.2、1.4、1.4 帧，违反原来的一帧契约，尚无科学评分。需要显式修订输入契约，让完整原始媒体能进入同一个诊断；现有证据不足以将尾差归因为容器 padding。

## What Changes

- 为这组固定数据引入 `bounded_audio_tail_v2`：音频短尾仍最多 1 帧，长尾最多 2 帧，以整数采样数比较；核验同源起点和连续视频 PTS。
- 保留完整 PCM、视频帧及原科学判据；审计所有 22 条并保存尾差和有效支持范围，局部评分排除边缘。
- 新 run 显式记录协议修订及父实验；独立 validator 检查审计，旧终态保持原样。

## Capabilities

### New Capabilities

- `lrs3-real-video-tail-contract`: 固定真实视频诊断的有界音频尾差审计、协议修订与验证。

### Modified Capabilities

无。父能力仅存在于未归档 change `diagnose-lrs3-real-video-local-timing`，尚不在 `openspec/specs/`；本 change 通过明确覆盖条款与父 spec 组合使用。

## Impact

小范围修改现有 `scripts/experiments/lrs3_real_video_local_timing/` 的 config、protocol、prepare/runner、validator 及对应测试。复用原三臂媒体和评分流程，无新增依赖。实施验收包括一次新 run 和 BM 结果更新；本次交付仅为 spec。
