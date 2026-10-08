---
title: 33-mfa-linear-content-pause-prosody-mapping
type: experiment
permalink: tts-exp/docs/experiments/33-mfa-linear-content-pause-prosody-mapping
status: concluded
date: '2026-09-25'
cohort_size: 10
artifact: runs/mfa_linear_content_pause_prosody_mapping_20260925/report.md
tags:
- mfa-linear
- wav2lip
- syncnet
- pause
- prosody
---

# MFA-linear 内容、停顿、韵律和特征映射拆分

## 问题和固定协议

沿用 [[32-mfa-linear-vocoder-wav2lip-split]] 的 10 位 AISHELL-1 说话人、Qwen 云端 VC TTS、MFA-linear 音频 M、同一张脸、Wav2Lip GAN `--nosmooth`、25 fps 尾部规范化和官方 SyncNet V2 人脸裁剪。所有新视频只用**同一条自然音轨**及上轮保存的同协议 AAC→AVI→16 kHz PCM 时钟评分；M 的自然音轨基线均值为 Sync-C **5.229**，N 自然视频/自然音轨为 **5.920**。所有变体音频等长，新的官方裁剪帧数逐样本与 M 一致，所有评分最佳偏移均未触及 ±15 帧边界。不能把变体自音轨高分当作自然音轨改善。

## 内容检查

10/10 条提交给 TTS 的文本与自然侧 MFA 对齐文本完全一致；自然/TTS 各自强制对齐的 466 个非静音音素标签中，顺序匹配 465 个，1 个在 a1_031 不匹配。另有 a1_057 的 unknown speech 标志 1 个。强制对齐使用给定文本，标签匹配**不能独立证明实际发音正确**。

另用冻结 faster-whisper **small**、普通话、beam 5、无文本提示分别识别 N、原始 TTS、M，繁简统一后的平均字编辑率为 **0.111 / 0.053 / 0.116**；M 比原始 TTS 高的样本 7/10。base 模型复核相应为 0.113 / 0.117 / 0.217，方向类似但模型间幅度不稳定。数字读法、同音字和 ASR 本身的错误使这些 CER 只能作为内容异常筛查。原始 TTS 没有得到可独立确认、足以单独重生成的文本错误，因此本实验未做“改文本后重生成 TTS”的因果臂。M 的识别退化也可能来自映射和声码器。

## 停顿与映射干预

先前逐音素审计确认：5 条样本有 **7 段 ≥100 ms 自然停顿**没有在 TTS 中保留。只在这些样本实施以下处理，其余样本音频/视频沿用 M 分数：

- **P**：原 M 波形的 7 段缺失停顿置静音，边缘 20 ms 淡入淡出；保留其它采样。
- **C**：同样的 7 段时长和淡入淡出，按预先固定的随机种子放在相应样本的自然发声位置，离真实缺失停顿至少 250 ms，作为“任意静音也会增分”的对照。
- **S**：固定同一份 TTS WavLM-L6 特征、自然/TTS MFA 和 prematched HiFi-GAN；仅将未匹配自然静音帧的全局 TTS 位置回退改为最长 TTS 静音段的对应位置。6 条样本的特征实际改变，含较短的未匹配静音。
- **B**：仅将已匹配音素的线性插值位置限制在属于该 TTS 音素的 WavLM 帧中心，避免取邻音素帧；9 条音频实际改变。

旧 M 波形在前 9 条用当前冻结代码**逐采样精确重现**；a1_057 重算与旧文件的最大绝对差为 0.07897，原因未查明，因此该条只参与直接以旧 M 波形为源的 P/C；B/S 在该条继承 M，不将它算作映射消融。排除它后，P/C 的受影响样本方向不变。

### 自然音轨视频评分

配对 n=10；分数为均值，越高越好。受影响 Δ 只计算实际改变音频的样本。

| 条件 | Sync-C | Δ 对 M | 变更样本中改善数 | 受影响 Δ | Sync-D ↓ |
|---|---:|---:|---:|---:|---:|
| M 原基线 | 5.229 | 0.000 | — | — | 8.365 |
| P 缺失停顿静音 | **5.416** | **+0.188** | **5/5** | +0.375 | **8.160** |
| C 发声位置静音对照 | 5.080 | −0.149 | 1/5 | −0.297 | 8.492 |
| S 静音特征回退 | **5.422** | **+0.194** | **6/6** | +0.323 | **8.183** |
| B 音素内帧限制 | 5.225 | −0.003 | 3/9 | −0.004 | 8.367 |

在 5 条有缺失停顿的样本，P 全部上升，P−C 逐条都为正，平均 +0.672；排除 a1_057 后其余 4 条 P 平均 +0.381，C 平均 −0.249。a1_049 的 M/P/C/S 分别为 5.488/5.862/5.213/5.976。P 和 S 是不同干预，增益不可相加；即使取 S，距离自然基线仍约 0.497 Sync-C。

Wav2Lip 实际 mel 的自然音频误差（10 条、每条先平均再平均）：所有视频帧 M/P/C/B/S 为 **0.659/0.633/0.685/0.659/0.630**；仅缺失停顿窗口为 **1.191/0.536/1.191/1.192/0.543**。7 段停顿的 M 输出 RMS 明显偏高，P/S 处理后降低；C 没有改变这些窗口。位置对照和口型评分一致支持“缺失停顿填入 TTS 声学内容”是一个可修复的局部机制。

## 韵律审核与单因素干预

对独立匹配的自然/TTS 音素，取音素内 20/40/60/80% 位置，用 WORLD DIO+StoneMask 的 F0 和 40 ms RMS 审核。10 条逐样本中位数再取中位数：自然与原始 TTS F0 绝对差 **0.709 半音**，自然与 M **1.189 半音**，自然与直接 WavLM+HiFi-GAN 重建 R **1.018 半音**。TTS/M/R 相对自然的能量差中位数分别约 **+4.963/+2.927/−2.863 dB**。这些是描述性声学差异，不能单独作因果归因。

为单独测试可控的韵律分量，另做 10 条三臂：

- **W**：用 WORLD 将 M 原样分析重合成，控制 WORLD 自身的伪影。
- **F**：同一 WORLD 谱包络和非周期性，仅在自然与 M 都有声的 20 ms 帧，把 M 的 F0 换成自然 F0，限制为原 M 的 0.7–1.4 倍；与 W 配对比较。
- **E**：直接在 M 波形上，对两者都有声的帧施加自然/M RMS 比值，限制为 0.5–2 倍，并平滑 5 帧；自然缺失停顿及其邻近区域保持原样；与 M 配对比较。

输出重新测得：F 对自然的平均逐条中位 F0 绝对差从 W 的 **1.156** 降到 **0.090 半音**（10/10 改善）；E 对自然的能量绝对差从 M 的 **4.444** 降到 **1.659 dB**（10/10 改善），干预确实生效。Wav2Lip mel 全帧对自然的误差 M/W/F/E 为 **0.659/0.712/0.687/0.590**。

| 条件 | 自然音轨 Sync-C | 合适的配对比较 | 改善条数 | Sync-D ↓ |
|---|---:|---:|---:|---:|
| M 原基线 | 5.229 | — | — | 8.365 |
| W WORLD 原样重合成 | 4.762 | W−M = −0.467 | 1/10 | 8.775 |
| F 自然 F0 | 4.761 | **F−W ≈ 0.000** | 5/10 | 8.749 |
| E 自然能量 | 5.187 | E−M = −0.042 | 4/10 | 8.337 |

F0 对齐未在 WORLD 内部带来平均 Sync-C 增益；WORLD 自身的较大失分使 F 与原 M 的直接差不能解释为纯 F0 效应。能量臂令 mel 更接近自然，但 Sync-C 未提升，虽然 Sync-D 有很小改善。不能据此排除其它韵律成分或更保真的 F0 编辑方法。

## 结论与边界

**优先修复 MFA-linear 对未匹配自然静音的回退**。缺失停顿处理在受影响的每条样本上提高自然音轨 Sync-C，等时长发声位置对照通常下降；静音特征替换给出同方向结果。限制音素内插值、只校正 F0 或只校正能量没有带来稳定的平均 Sync-C 提升。TTS 输入文本在本批没有可确认的系统性错误；M 的声学/可懂度和其它局部口型差异仍需后续检查。

本批 n=10，为对已知失败样本的机制诊断；P 位置由上一轮审计发现，增益不是独立测试集的泛化估计。S 使用全局最长 TTS 静音特征是诊断性替换，部署时应建立局部静音策略并在新样本验证。a1_057 映射重算异常、WORLD 伪影、ASR 错误和未测试的复杂韵律均限制解释。

## 复现产物

- 原始运行与逐样本分数：`runs/mfa_linear_content_pause_prosody_mapping_20260925/` 的 `manifest.json`、`scores.json`、`analysis.json`、`acoustic.json`、`report.md`。
- 韵律臂：同目录 `prosody/` 的 `manifest.json`、`verification.json`、`acoustic.json`、`scores.json`、`analysis.json`；内容审计 `content_prosody_audit.json`、`content_asr_audit.json`、`content_asr_audit_small.json`。
- 代码：`scripts/experiments/mfa_linear_content_pause_mapping.py`（依次 `audio control render score acoustic analyze`，最后 `finalize` 汇总）；`mfa_linear_prosody_intervention.py`（`audio render verify acoustic score analyze`）；`mfa_linear_content_prosody_audit.py`；`mfa_linear_content_asr_audit.py`。ASR 模型固定为本次运行目录里的 faster-whisper base/small 快照。

## Observations

- [result] 缺失停顿 P 自然音轨 Sync-C +0.188/10，受影响 5/5 上升；发声位置对照 C −0.149/10。#pause
- [result] 静音特征回退 S +0.194/10（实际改变 6/6 上升），音素内插值 B −0.003/10。#mfa-linear
- [result] F0 误差和能量误差确实降低，F−WORLD 的 Sync-C 约 0，E−M 为 −0.042；WORLD 原样重合成 −0.467。#prosody
- [limit] a1_057 的旧 M 不能用当前冻结代码精确重现，B/S 在该样本未运行；内容审计和本批干预均不构成跨样本泛化证明。#reproducibility

## Relations

- extends [[32-mfa-linear-vocoder-wav2lip-split]]
- relates_to [[MFA-linear 逐音素口型与缺失停顿诊断 2026-09-25]]
- relates_to [[MFA-linear 内容停顿韵律与映射消融 2026-09-25]]

## Changelog

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 完成同批内容审计、缺失停顿和映射消融、发声位置对照、F0/能量验证与自然音轨评分 | September 25, 2026 | user（实验请求）；agent（执行） |
