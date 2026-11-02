---
title: MFA-linear 声码器与 Wav2Lip 响应拆分 2026-09-25
type: experiment
permalink: tts-exp/experiments/mfa-linear-声码器与-wav2-lip-响应拆分-2026-09-25
status: concluded
date: '2026-09-25'
hypothesis: 自然特征直接声码器重建的R视频若接近N，M视觉差距主要来自TTS特征和MFA映射；若R也偏离，声码器链路本身足以改变口型。
inputs: runs/aishell1_qwen_mfa_linear_n100_20260816; runs/mfa_linear_nas_short_expanded_20260924
outputs: runs/mfa_linear_vocoder_wav2lip_split_20260925
code_paths:
- scripts/experiments/mfa_linear_vocoder_wav2lip_split.py
tags:
- mfa-linear
- wav2lip
- vocoder
- syncnet
- experiment
result: N/R/M video with natural audio Sync-C 5.920/5.843/5.229; M self 6.504
conclusion: Natural direct WavLM+HiFi-GAN reconstruction does not reproduce most of
  MFA-linear transfer deficit; TTS/MFA input properties warrant separate ablation.
report: docs/experiments/32-mfa-linear-vocoder-wav2lip-split.md
---

# MFA-linear 声码器与 Wav2Lip 响应拆分

复用 2026-09-24 的 10 位说话人 N/M 原视频，新增自然语音直接经同一冻结 WavLM-L6 + prematched HiFi-GAN 重建的 R 音频与 Wav2Lip 视频。三路视频 V_N/V_R/V_M 对三路音频 A_N/A_R/A_M 做同样时钟的 3×3 SyncNet V2 交叉评分，按自然音轨替换、同音轨评分和 Wav2Lip mel/口型局部差异解释。验证已有 N/M 官方分数公式和同视频帧评分一致性。

## Observations

- [status] concluded；完成 10 位说话人 × 3 视频 × 3 音轨交叉评分与 mel、口型、MFA 时间核验。完整报告 [[32-mfa-linear-vocoder-wav2lip-split]]；原始产物 `runs/mfa_linear_vocoder_wav2lip_split_20260925/`。
- [result] 固定自然音轨，N/R/M 视频平均 Sync-C=5.920/5.843/5.229；R 相对 N 仅低 0.076，M 相对 R 低 0.614（9/10 条）。M 自轨比 M 视频配自然音轨高 1.276（10/10 条）。
- [evidence] Wav2Lip mel MAE R−N/M−N=0.395/0.649；嘴唇开度相关 R/N 与 M/N=0.891/0.774；R 的自然音素匹配 464/466，时间中心中位误差 5 ms。无缺失长停顿的 5 条中，M−R 自然音轨 Sync-C 仍为 −0.364。
- [conclusion] WavLM+HiFi-GAN 直接重建自然音频不足以解释 M 的主要自然音轨失分；M 的 TTS/MFA-linear 输入与自然语音有较大声学及口型差异，具体组件归因仍需单独消融。
- [baseline] 旧自轨 M 视频+M 音频均值 6.635，旧尾部规范化 M 视频+自然音轨均值 5.243；本次同帧同音轨处理链重测对应值为 6.504、5.229，N 自然基线为 5.920。后续使用本次同协议数值比较。
- [question] 声码器对自然语音的重建本身是否足以改变 Wav2Lip 口型？若 R 保持自然口型而 M 偏离，差异更可能来自 TTS 特征和 MFA 映射。

## Relations

- relates_to [[MFA-linear 逐音素口型与缺失停顿诊断 2026-09-25]]
- relates_to [[NAS 短窗口 10 说话人扩大样本试验 2026-09-24]]

## Changelog

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 用户要求执行声码器/口型响应拆分，建立实验并开始生成自然重建音频 | September 25, 2026 | user（实验与 BM 请求）；agent（执行） |
| 完成统一时钟的 3×3 评分、官方抽查、声学/口型/MFA 分析，结论转存编号报告 32 | September 25, 2026 | user（实验与 BM 请求）；agent（执行） |
