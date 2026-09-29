## Why

natural→MFA-linear 谱迁移的 discovery ΔSync-C 为 +0.078，22 条确认轮为 +0.032、CI 跨零；最新 MAG 在局部 U 上为 −0.889，ENV 为 −0.042。当前谱迁移构造已经停止，训练通用 replacement 头的前提尚未建立。

这些数字并非同一实验口径：discovery 样本不同；确认轮使用整帧 Wav2Lip 生成框及 SyncNet 再检测裁剪，最新轮使用固定 ROI；整段评分还包含 offset 边界的零嵌入距离，U 只用内部窗口。需要一次小型、可验收的历史对账，说明哪些差异可由固定缓存直接解释，哪些仍混杂。

## What Changes

- 新增纯 CPU 的缓存重分析协议，固定确认轮与最新轮共同的 22 records/22 source groups。
- 复用旧 66 个距离矩阵与新 110 个距离矩阵，计算 FULL、COMMON_INTERIOR、U 三个固定汇总，共 528 个 endpoint rows；新增生成视频、音频、模型 forward 均为零。
- 检查 N/M/candidate PCM、生成框、裁剪、帧时钟及评分来源；先复现历史已报告数字，再做配对分解。
- 交付独立 validator、简短结果表和 BM 同一笔记的状态更新。只有诊断结论，没有新的科学 GO。

## Non-goals

不重跑 TTS、MFA、Wav2Lip、SyncNet 或人脸检测，不修复旧控制门禁，不搜索 α/窗口/样本，不训练，不访问其他 fit 或 sealed 媒体，不改写历史结果。本轮不证明某个生成框、裁剪或音素机制是原因。

## Impact

- 新 capability：`wav2lip-replacement-endpoint-reconciliation`。
- 下游新增小包：`scripts/experiments/wav2lip_replacement_reconciliation/` 及对应 tests；父实验代码只读。
- 本 change 只设计实验，执行任务见 tasks.md；已有父 change 的未勾任务不能作为重新执行父 runner 的理由。
