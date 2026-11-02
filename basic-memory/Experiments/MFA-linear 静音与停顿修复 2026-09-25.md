---
title: MFA-linear 静音与停顿修复 2026-09-25
type: experiment
permalink: tts-exp/experiments/mfa-linear-静音与停顿修复-2026-09-25
status: concluded
date: '2026-09-25'
tags:
- mfa-linear
- pause
- silence
- syncnet
---

# MFA-linear 静音与停顿修复 2026-09-25

## 状态

concluded。用户要求修复 MFA-linear 中缺失自然静音和停顿被 TTS 内容填充的问题；实现已接入正式生成路径并完成同批自然音轨视频评分。

## 方法与结果

- `mfa_linear_target` 新增 `silence_fallback="tts_silence"`：未匹配自然静音使用最长 TTS 静音段的 WavLM 特征；未知语音 spn 不当作静音。默认旧策略保留供历史脚本复现。
- `pilot_generate_mfa_linear.py` 默认启用修复；未匹配的 ≥100 ms 自然静音段在声码器输出中置零，两端最多 20 ms 淡入淡出；`--legacy-global-silence` 可重现旧行为。元数据记录替换帧数和波形静音区间。
- 同批 n=10 自然音轨 Sync-C：旧 M 5.229，修复 F 5.477，Δ=+0.248；7/7 条实际改变音频的样本改善。排除旧 M 无法精确重算的 a1_057 后，n=9 的 M/F 为 5.438/5.695，Δ=+0.257，改变的 6/6 上升。
- 7 段已知缺失停顿的核心区修复后 RMS=0；其余 9 条在非停顿位置与先前单因素 S 音频逐采样一致。全部新视频官方裁剪帧数匹配 M，最佳偏移均不在边界。
- 修复后的 5.477 仍低于自然视频基线 5.920；本批沿用发现故障的样本，不能视为独立泛化验证。

## Observations

- [status] concluded
- [result] 正式修复版对自然音轨的 Sync-C 同批提升 +0.248/10；排除旧基线异常样本后 +0.257/9。#syncnet
- [result] 7 段缺失停顿的输出核心区归零，非停顿语音区域与已验证的 S 臂一致。#pause
- [conclusion] 缺失自然停顿的 TTS 内容填充问题已在正式 MFA-linear 生成路径修复；仍需新样本检验泛化。
- [limit] 缺少独立测试集；a1_057 的旧 M 重算异常仍在。#reproducibility
- [report] 完整协议、逐样本分数和复现路径见 [[34-mfa-linear-silence-pause-repair]]。

## Relations

- extends [[MFA-linear 内容停顿韵律与映射消融 2026-09-25]]
- relates_to [[33-mfa-linear-content-pause-prosody-mapping]]
- relates_to [[34-mfa-linear-silence-pause-repair]]

## Changelog

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 接入正式生成路径并完成同批自然音轨验证 | September 25, 2026 | user（修复请求）；agent（执行） |
