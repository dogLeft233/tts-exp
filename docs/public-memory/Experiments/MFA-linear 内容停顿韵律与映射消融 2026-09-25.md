---
title: MFA-linear 内容停顿韵律与映射消融 2026-09-25
type: experiment
permalink: tts-exp/experiments/mfa-linear-内容停顿韵律与映射消融-2026-09-25
status: concluded
date: '2026-09-25'
inputs: runs/mfa_linear_vocoder_wav2lip_split_20260925; runs/aishell1_qwen_mfa_linear_n100_20260816
outputs: runs/mfa_linear_content_pause_prosody_mapping_20260925
tags:
- mfa-linear
- wav2lip
- syncnet
- mechanism
result: Missing-pause P +0.188 Sync-C (5/5 affected improved), same-duration speech
  control C -0.149; silence-feature S +0.194 (6/6 improved); boundary B -0.003; F0
  correction F-W ~0.000, energy E-M -0.042.
conclusion: Unmatched natural silence with TTS-feature fallback is a confirmed local
  transfer failure; F0/energy correction and phone-interpolation clamp do not recover
  the remaining gap in this n=10 protocol.
report: docs/experiments/33-mfa-linear-content-pause-prosody-mapping.md
---

# MFA-linear 内容、停顿、韵律与映射消融

继声码器/Wav2Lip 拆分，固定 10 条 AISHELL-1 Qwen 云端 TTS 的自然时间轴与自然音轨，审核文本及韵律，并比较停顿修复和映射变体对生成视频自然音轨 SyncNet 的影响。

## Observations
- [status] concluded
- [scope] 10 条 AISHELL-1 云端 Qwen TTS 同样本、同脸、同自然音轨与官方 SyncNet V2 协议完成内容筛查、停顿/映射消融、发声位置对照及 F0/能量干预。完整报告 [[33-mfa-linear-content-pause-prosody-mapping]]；运行产物 `runs/mfa_linear_content_pause_prosody_mapping_20260925/`。
- [result] M 基线 Sync-C 5.229；缺失停顿原 M 波形静音 P 5.416（+0.188；受影响 5/5 上升），等时长自然发声位置对照 C 5.080（−0.149）；静音特征回退 S 5.422（+0.194；实际改变 6/6 上升），匹配音素帧内限制 B 5.225（−0.003）。P/S 的 Sync-D 同时降低。
- [evidence] 缺失停顿窗口 Wav2Lip mel 对自然误差 M/P/C/S=1.191/0.536/1.191/0.543。排除 a1_057 后，其余 4 条受影响样本 P 平均 +0.381，C 平均 −0.249。
- [result] WORLD 原样重合成 W 对 M 为 −0.467 Sync-C；自然 F0 臂 F 对 W 约 0.000（5/10 上升）；自然能量臂 E 对 M −0.042（4/10 上升）。F0 对自然误差 W/F=1.156/0.090 半音、能量 M/E=4.444/1.659 dB，干预确实生效，未带来平均 Sync-C 增益。
- [content] 10/10 TTS 输入文本与自然侧 MFA 文本相同；自然/TTS 强制对齐 465/466 音素标签匹配。独立 faster-whisper small 平均 CER N/TTS/M=0.111/0.053/0.116；数字、同音字和 ASR 错误限制解释，未确认值得重生成的原始 TTS 文本错误。
- [limit] a1_057 的旧 M 波形不能被当前冻结代码精确重现（max abs 0.07897），因此 B/S 在该条不运行。n=10 为已知失败位置的机制诊断，P/S 增益不可相加，WORLD 自身伪影大；剩余差距未唯一归因。
- [conclusion] 优先修复未匹配自然静音回退并在独立样本验证；目前没有证据支持单靠音素帧限制、F0 或能量匹配恢复自然音轨同步。

## Relations

- extends [[MFA-linear 声码器与 Wav2Lip 响应拆分 2026-09-25]]
- relates_to [[MFA-linear 逐音素口型与缺失停顿诊断 2026-09-25]]

## Changelog

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 用户要求执行内容、停顿、韵律、映射实验，建立实验笔记 | September 25, 2026 | user（实验请求）；agent（执行） |
| 完成 10 条停顿/发声对照、映射、F0/能量及独立 ASR 审核，结论写入编号报告 33 | September 25, 2026 | user（实验请求）；agent（执行与核验） |
