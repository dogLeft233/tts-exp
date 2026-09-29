## Why

真实视频的离散局部视频扰动已获得 22/22 的 SyncNet offset 恢复，但历史 `LOCAL_WARP_120` 的 Wav2Lip 自身配对与替换敏感性均失败。需要在相同裁脸、时间窗口和评分设置下，检查音频扰动能否被检测，以及生成画面是否呈现预期的局部时间响应。

## What Changes

- 固定已看过结果的 22 条 cohort，复用真实视频、已有 Wav2Lip N/W 视频和 N/W 音频；构建 3×2 交叉评分矩阵。
- 每条增加两次独立基线评分，合计 132 个交叉 cell + 44 个重复 cell；不新运行生成器。
- 以已知音频时间映射及其逆映射预测局部 offset，分别检验真实视频敏感性、生成域敏感性、生成响应与自身对齐。
- 写明生成视频较短时的共同支持范围，交付可复算终态和 BM 结果记录。

## Capabilities

### New Capabilities

- `lrs3-wav2lip-timing-transfer-diagnostic`: 用既有媒体定位局部 timing 控制失败的环节。

### Modified Capabilities

无。历史实验契约及终态保持原样。

## Impact

新增独立实验模块 `scripts/experiments/lrs3_wav2lip_timing_transfer/` 及对应测试；复用真实视频诊断的裁脸、官方 SyncNet worker 和矩阵基础设施。产物写入新 run，不引入新模型或依赖。

## Non-goals

本轮只做 fit-only 诊断，不证明纯声学损伤或 Wav2Lip 架构因果，不重测 bridge，不运行 TTS、MFA/DTW、训练或 held-out evaluation。通过也不自动授予音频头训练资格。
