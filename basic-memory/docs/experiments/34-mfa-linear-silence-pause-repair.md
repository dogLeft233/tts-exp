---
title: 34-mfa-linear-silence-pause-repair
type: experiment
permalink: tts-exp/docs/experiments/34-mfa-linear-silence-pause-repair
status: concluded
date: '2026-09-25'
cohort_size: 10
artifact: runs/mfa_linear_silence_pause_fix_20260925
tags:
- mfa-linear
- silence
- pause
- wav2lip
- syncnet
---

# MFA-linear 静音与停顿修复

## 修复内容

`scripts/knn_vc_retrieval.py` 的 `mfa_linear_target` 新增 `silence_fallback="tts_silence"`：当自然侧静音 token 没有匹配的 TTS 静音 token 时，沿 TTS 最长的、确实有 WavLM 特征帧的静音段取帧，避免原先全句相对位置回退把语音填进停顿。未匹配的发声音素仍用原回退；TTS 完全没有静音特征帧时也保留原回退并记录帧数。MFA `spn` 未知语音标记不作为静音来源，也不被静音。

`scripts/pilot_generate_mfa_linear.py` 默认启用上述特征修复，并在声码器生成、精确等长之后，将未匹配的自然侧 ≥100 ms 静音 token 对应波形置零；两端各最多 20 ms 淡入淡出。这一步按自然音频采样时钟计算，不能改变时长。`--legacy-global-silence` 保留原方法用于历史结果复现。逐样本 `summary.json` 记录静音特征替换帧数和波形静音区间；总表记录是否开启修复。

## 验证协议

沿用 [[33-mfa-linear-content-pause-prosody-mapping]] 的 10 条 AISHELL-1、云端 Qwen VC TTS、MFA、WavLM-L6、prematched HiFi-GAN、同一张脸、Wav2Lip GAN `--nosmooth`、25 fps 视频规范化。修复版 F 由正式生成脚本重新产生；全部视频都用同一条自然音轨和官方 SyncNet V2 视频裁剪评分，与旧 M、单因素 P/S 分数配对。复用 33 号实验中的既定脸、时钟、评分代码；没有用自音轨替代自然音轨。

输入：`runs/aishell1_qwen_mfa_linear_n100_20260816/00_pairs/cohort.json` 中的这 10 条，原 TTS meta 和 `03_tokens_paired/tokens.json`。输出：`runs/mfa_linear_silence_pause_fix_20260925/` 中 `cohort.json`、`audio/summary.json`、`manifest.json`、`video/`、`pipeline/`、`scores.json`、`validation.json`。命令：

```bash
.venv/bin/python scripts/pilot_generate_mfa_linear.py \
  --manifest runs/mfa_linear_silence_pause_fix_20260925/cohort.json \
  --tts-meta runs/aishell1_qwen_mfa_linear_n100_20260816/01_tts_retry/tts_meta.json \
  --tokens runs/aishell1_qwen_mfa_linear_n100_20260816/03_tokens_paired/tokens.json \
  --outdir runs/mfa_linear_silence_pause_fix_20260925/audio --device cuda
```

随后用 `scripts/experiments/mfa_linear_content_pause_mapping.py` 的 `render`、`score` 函数，令 `OUT` 指向新 run、`ARMS=("F",)`，按本 run 的 `manifest.json` 复用相同渲染和评分流程。`scores.json` 保存每条官方 Sync-C、Sync-D、偏移和固定零延迟距离。

## 结果

10/10 音频生成成功，长度逐条与自然音频完全一致；7 段缺失停顿的淡入淡出核心区输出 RMS 全为 0，旧 M 的对应窗口 RMS 在 0.013–0.047。除 a1_057 外，F 的非停顿采样与上次 S 静音特征臂逐采样完全一致。a1_057 的旧 M 在上次实验中已无法精确重算，因此这条不能归因于单纯修复；其余 9 条的声码器重算已由 33 号实验校验。全部新视频的官方裁剪帧数与对应 M 一致；10/10 最佳偏移不在 ±15 帧边界。

| 条件 | 自然音轨 Sync-C 均值，n=10 | 对旧 M 差值 |
|---|---:|---:|
| 旧 M | 5.229 | 0.000 |
| P：仅缺失停顿静音 | 5.416 | +0.188 |
| S：仅静音特征替换 | 5.422 | +0.194 |
| **F：正式修复版，S+P** | **5.477** | **+0.248** |
| N：自然音频生成视频 | 5.920 | +0.691 |

7/7 条实际改变音频的样本分数上升。排除旧 M 重算异常的 a1_057 后，n=9 的 M/F 分别为 **5.438/5.695**，配对差 **+0.257**；其中 6 条音频改变，6/6 上升。相同 9 条的 S 为 5.653，F−S 为 +0.042；P 为 5.607。不能把 P 和 S 的单因素增益直接相加。

| 样本 | M | F | Δ |
|---|---:|---:|---:|
| a1_002 | 6.565 | 6.565 | +0.000 |
| a1_010 | 4.992 | 5.536 | +0.544 |
| a1_014 | 5.632 | 5.636 | +0.004 |
| a1_019 | 4.990 | 4.990 | +0.000 |
| a1_028 | 5.638 | 6.150 | +0.512 |
| a1_031 | 5.777 | 6.340 | +0.563 |
| a1_040 | 4.905 | 5.024 | +0.119 |
| a1_044 | 4.953 | 4.953 | +0.000 |
| a1_049 | 5.488 | 6.062 | +0.574 |
| a1_057 | 3.346 | 3.509 | +0.163* |

`*` 旧 M 重算异常，不能作为严格配对归因。

## 结论与限制

修复解决了当前 MFA-linear 生成路径中“缺失自然停顿被 TTS 语音填充”的可重复问题；同批自然音轨 Sync-C 提升。这个批次沿用了发现故障的 10 条样本，**不是独立测试集的泛化估计**。修复后平均 5.477 仍低于自然视频基线 5.920，剩余差距不由此次静音修复解释。若 TTS 完全没有静音特征帧，长停顿仍由波形静音处理，较短未匹配静音则记录回退情况；此边界需要在新样本上核查。

## Observations

- [status] concluded
- [result] 正式生成路径修复版自然音轨 Sync-C 5.477，旧 M 5.229，同批 Δ=+0.248；排除 a1_057 后 n=9 Δ=+0.257。#syncnet
- [result] 7 段 ≥100 ms 缺失停顿的输出核心区 RMS 为零；其余 9 条的非停顿音频与先前 S 臂完全相同。#pause
- [conclusion] MFA-linear 的缺失自然停顿被 TTS 内容填充的问题已在生成路径修复；仍需独立样本验证泛化。
- [limit] 同批验证不是独立泛化测试；a1_057 旧基线无法精确重算。#reproducibility

## Relations

- extends [[33-mfa-linear-content-pause-prosody-mapping]]
- relates_to [[32-mfa-linear-vocoder-wav2lip-split]]

## Changelog

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 完成静音特征回退与缺失停顿静音修复，并用原批次自然音轨评分验证 | September 25, 2026 | user（修复请求）；agent（执行） |
