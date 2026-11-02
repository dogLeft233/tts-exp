---
title: RAMC + AliMeeting 多样域中文试点（50 条）
type: task
permalink: tts-exp/tasks/ramc-ali-meeting-多样域中文试点-50-条
assigned_to: claude
current_step: 7
priority: high
started: '2026-08-13'
status: done
steps:
- 流式部分下载 RAMC 25 条 + AliMeeting 25 条
- 建试点 manifest（排序/说话人/转录/泄漏状态）
- 生成配对 TTS（faster_qwen3 ICL）
- 中文 MFA 对齐（特殊 token 归一化）
- 三臂生成 natural_raw/raw_tts/mfa_linear
- Wav2Lip + SyncNet 下游评测（temp 隔离）
- 配对统计 + Basic Memory 实验笔记
tags:
- task
- mfa-linear
- ramc
- alimeeting
- chinese-diverse
- pilot
completed: '2026-08-13'
---

# RAMC + AliMeeting 多样域中文试点（50 条）

## Observations

- [description] 在非 AISHELL 中文自发/对话语域验证 MFA-linear 三臂协议：下载 RAMC + AliMeeting 共 50 条、建 manifest、配 TTS、MFA 对齐、三臂生成、Wav2Lip+SyncNet 下游评测
- [status] done
- [assigned_to] claude
- [current_step] 7
- [completed] 2026-08-13

## Steps

1. [x] 下载与切条完成（RAMC 25 条 + AliMeeting 25 条，各 25 说话人，金标转录）
2. [x] 试点 manifest（固定排序 + encoder-leakage 字段）
3. [x] faster_qwen3 ICL 配对 TTS：两域各 25/25
4. [x] 中文 MFA 对齐（conda mfa env 2.2.17 + mandarin_mfa）：两域各 25/25，0 失败
5. [x] mfa_linear 臂生成：两域各 25/25
6. [x] 三臂 Wav2Lip + SyncNet 评测：两域各 75/75 cell 完成
7. [x] 统计 + 实验笔记：`Experiments/多样域中文试点：MFA-linear 在自发对话域失败`（负结果已记录）

## 最终结论

mfa_linear 在两种自发中文语域统计显著变差：AliMeeting ΔSync-C −0.899 (p=0.040)、RAMC −1.223 (p=0.013)，与 AISHELL-1 朗读域 +0.964 方向相反。raw-TTS 优势本身也缩水到 +0.11–0.19（不显著），支持“TTS 唇同步红利主要来自规整朗读语域”。覆盖率（均值 0.91–0.94）与 ΔC 相关不显著。路线判断：mfa_linear 自发域 No-Go；朗读域 n=15 阳性需 50 条完整 cohort 复验。

## Context

- 泄漏判定：HuBERT-base（纯英文）与 WavLM-Large（英文+欧洲语）对中文数据集零泄漏；RAMC/AliMeeting 为自采录音零抓取重叠。XLS-R 时须避开 CommonVoice/MLS。
- 归档结构（实测）：RAMC 每 .wav = 一场 ~32 分钟双人对话，TXT/ 目录逐轮 `[start,end] 说话人 方言,普通话 文本`；AliMeeting Eval 近场 25 声道 + TextGrid 单 tier `c1`。
- MFA 2.2.17 本地安装依赖链：conda-forge + python 3.10 + joblib<1.4 + setuptools<81 + postgresql；调用时 PATH 需含 env bin；pip 路线不可行（kalpy 无 py3.12 wheel）。
- 许可：RAMC CC BY-NC-ND 4.0；AliMeeting CC BY-SA 4.0。产物路径见实验笔记。

## Relations

- relates_to [[多样域中文试点：MFA-linear 在自发对话域失败]]
- relates_to [[MFA-linear 冻结 VC 重合成与下游验证]]
- relates_to [[推进 TTS 节奏因素实验]]
