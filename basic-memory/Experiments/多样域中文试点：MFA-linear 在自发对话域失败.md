---
title: 多样域中文试点：MFA-linear 在自发对话域失败
type: experiment
permalink: tts-exp/experiments/多样域中文试点-mfa-linear-在自发对话域失败
date: '2026-08-13'
status: concluded
tags:
- mfa-linear
- ramc
- alimeeting
- spontaneous-speech
- negative-result
- domain-dependence
- syncnet
---

# 多样域中文试点：MFA-linear 在自发对话域失败

## 背景

AISHELL-1 n=15 结果显示 mfa_linear（自然时钟上的 TTS WavLM 特征 + 冻结 HiFi-GAN 重合成）在朗读域大幅改善 Wav2Lip+SyncNet（vs natural ΔSync-C +0.964）。但 AISHELL-1 是朗读式 ASR 语料、单说话人 valid、固定参考图协议，域单一。本试点在两种自发中文语域检验该方法与 raw-TTS 优势的可推广性。

## 数据

| 数据集 | 域 | n | 说话人 | 转录 | 许可 |
|---|---|---|---|---|---|
| MagicData-RAMC (SLR123) | 手机双人自由对话 | 25 | 25 | 金标（含时间戳/说话人/方言） | CC BY-NC-ND 4.0 |
| AliMeeting Eval (SLR119) | 真实会议近场单人声道 | 25 | 25 | 金标 TextGrid | CC BY-SA 4.0 |

固定取样：RAMC 跨步选 25 场对话（避免首个批次全四川的区域偏差）、每场切 1 条普通话轮次；AliMeeting 每声道切 1 条 8–20s utterance。均自采录音，与网络抓取训练集零重叠；对 HuBERT-base（纯英文）与 WavLM-Large（英文+欧洲语）零泄漏。

## 方法（与 AISHELL-1 n=15 完全一致）

- faster_qwen3 0.6B ICL 自克隆配对 TTS（同文本，canonical 16 kHz）
- MFA 2.2.17 + mandarin_mfa 对齐 natural/TTS 两侧（本地首次安装）
- 冻结 WavLM-Large L6 + `mfa_linear_target` + 冻结 prematched HiFi-GAN，16 kHz exact-N，无 loudnorm
- Wav2Lip + SyncNet V2，`min_track=50`，单固定参考脸协议，每 cell 独立 cwd/temp

## 结果（三臂，n=25 × 2 域）

| 域 | natural C | raw_tts C | mfa_linear C | raw_tts−natural ΔC (p) | mfa_linear−natural ΔC (CI, p, 正向) |
|---|---:|---:|---:|---|---|
| AISHELL-1 朗读（历史 n=15） | 5.758 | 6.049 | **6.722** | +0.291 | **+0.964** ([+0.650,+1.266], 4e-05, 13/15) |
| AliMeeting 会议 | 6.939 | 7.132 | 6.040 | +0.193 (0.282) | **−0.899** ([−1.780,−0.180], 0.040, 11/25) |
| RAMC 电话对话 | 6.262 | 6.387 | 5.039 | +0.125 (0.337) | **−1.223** ([−2.113,−0.398], 0.013, 11/25) |

Sync-D 方向：两自发域 mfa_linear 与 natural 差异均不显著（AliMeeting −0.162, p=0.26；RAMC −0.089, p=0.41）。mfa_linear 的 Sync-C 标准差翻倍（~2.0–2.3 vs ~0.8–1.1），逐样本极不稳定。

## 结论

1. **raw-TTS 优势存在域依赖**：朗读域 ~+1.0（92/100），两自发域仅 +0.11–0.19 且 n=25 不显著。支持“TTS 唇同步红利主要来自规整朗读语域”的判断。
2. **mfa_linear 不跨域推广**：在两种自发域中统计上显著地变差（−0.9 至 −1.2），且比 raw_tts 更差约 −1.1 至 −1.4。与朗读域的正效应方向相反、跨两个自发域一致。
3. **覆盖率不是主因**：MFA 两侧 phone 匹配覆盖率均值 0.91–0.94（最低 0.20–0.32），但覆盖率与 ΔSync-C 相关 r=−0.237（p=0.25）不显著；最低覆盖样本反而有正 ΔC。
4. 候选机制（未验证）：自发语音自然时钟含犹豫/修复/重叠停顿，把干净 TTS 特征硬映射到该时钟再重合成会制造伪影；朗读域的自然网格规整，映射代价小且 TTS 特征本身信息量大。

## 判断与下一步

- mfa_linear 在自发中文域的路线为 **No-Go**（试点规模）；朗读域的 n=15 阳性仍待更大固定 cohort 复核，但不应再默认它能推广。
- 若继续 TFG 方向：优先在规整/录音室语域验证；若要覆盖自发域，需要先解决 TTS 红利本身在该域缺失的问题（raw_tts 就几乎无红利），而不是改进时钟映射。
- 待办：AISHELL-1 朗读域 50 条完整 cohort 的 mfa_linear 复验（baseline 已存在）；试点中 mfa_linear 逐样本异常值（低分高方差）的音频听检。

## 产物位置

- 切分与 TTS：`data/ramc_alimeeting_pilot50/{alimeeting25,ramc25_utterances}/`
- MFA 对齐：`runs/ramc_alimeeting_pilot50/{alimeeting25_mfa,ramc25_mfa}/tokens.json`
- mfa_linear 音频：`runs/ramc_alimeeting_pilot50/{alimeeting25_mfa_linear,ramc25_mfa_linear}/`
- 两臂评测：`runs/ramc_alimeeting_pilot50/{alimeeting25_wav2lip,ramc25_wav2lip}/`
- 三臂评测：`runs/ramc_alimeeting_pilot50/{alimeeting25_wav2lip_3arm,ramc25_wav2lip_3arm}/`
- 脚本：`scripts/{pilot_partial_download,ramc_select_25,alimeeting_cut_25,ramc_cut_25,pilot_generate_tts,pilot_run_mfa,pilot_generate_mfa_linear,eval_pilot_wav2lip_syncnet}.py`

## Observations

- [result] 两自发中文域 raw-TTS 优势仅 +0.11–0.19，n=25 不显著；朗读域 ~+1.0。 #syncnet #domain
- [result] mfa_linear 在 AliMeeting 与 RAMC 均显著变差：ΔSync-C −0.899 (p=0.040) 与 −1.223 (p=0.013)，与朗读域 +0.964 方向相反。 #negative #mfa-linear
- [result] MFA 覆盖率与 ΔSync-C 相关不显著（r=−0.237），低覆盖不是退化的直接原因。 #alignment
- [decision] mfa_linear 自发域 No-Go（试点规模）；朗读域 n=15 阳性需 50 条完整 cohort 复验，不再默认可推广。 #decision
- [learning] MFA 2.2.17 本地安装依赖链：conda-forge + python 3.10 + joblib<1.4 + setuptools<81 + postgresql，调用时 PATH 需含 env bin。 #tooling
- [boundary] 单固定参考脸协议、n=25、探索性；无 heldout/generalization 声明。 #scope

## Relations

- relates_to [[MFA-linear 冻结 VC 重合成与下游验证]]
- relates_to [[RAMC + AliMeeting 多样域中文试点（50 条）]]
- relates_to [[TTS Feature Alignment and Duration Model Research]]
## Observations
- [status] concluded
- [result] 两自发域 mfa_linear 显著变差: AliMeeting ΔSync-C -0.899 (p=0.040), RAMC -1.223 (p=0.013); raw-TTS 优势域依赖 (朗读 +0.964, 自发域 +0.11~0.19 不显著)
- [conclusion] mfa_linear 在自发中文域为 No-Go; TTS 唇同步红利主要来自规整朗读语域
- [inputs] MagicData-RAMC SLR123 25 条 + AliMeeting Eval SLR119 25 条 (各 25 说话人); AISHELL-1 n=15 朗读域对照; paired TTS = faster_qwen3 0.6B ICL 自克隆
- [outputs] runs/ramc_alimeeting_pilot/ 系列: mfa_linear 音频 + Wav2Lip 视频 + SyncNet summary (逐项见笔记正文)
- [model] faster_qwen3 0.6B ICL; WavLM-Large L6 冻结 + 冻结 prematched HiFi-GAN; MFA 2.2.17 mandarin_mfa; Wav2Lip + SyncNet V2 (min_track=50)
- [code_paths] scripts/pilot_generate_mfa_linear.py, scripts/eval_pilot_wav2lip_syncnet.py (及 alimeeting_cut_25.py 等数据准备)
- [boundary] 试点规模 (n=25×2 域); 朗读域 n=15 阳性仍需更大固定 cohort 复核
- [hypothesis] mfa_linear 方法与 raw-TTS 唇同步优势能跨域推广到自发对话/会议语音
- [report] 无对应 docs/experiments 编号报告; 试点记录仅在本记忆笔记
