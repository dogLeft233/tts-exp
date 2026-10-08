---
title: 跨数据集 TFG 测评（5×50 multiset）
type: experiment
permalink: tts-exp/experiments/跨数据集-tfg-测评-5x50-multiset
tags:
- multiset
- cross-dataset
- ditto
- leaptalk
- syncnet
- tfg
---

# 跨数据集 TFG 测评（5×50 multiset）

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| Initial Document | August 14, 2026 | user |

在 5 个权威数据集 × 50 样本（MEAD / VFHQ / TalkVid / LRS3 / GRID，共 250 条视频+自然音频配对）上，用两个 TFG 模型（Ditto、LeapTalk）检验 TTS-vs-Natural 的唇形同步优势是否跨数据集成立。数据准备见 [[multidataset-samples]]。

## 数据与方法

- 样本：`data/dataset_samples/video_manifest_250.json`，MEAD(情绪表演)/VFHQ(YouTube)/TalkVid(真实视频)/LRS3(TED 演讲)/GRID(固定句法) 各 50 条；音频语言混杂（中/英/德/西/韩）
- TTS：faster_qwen3 0.6B ICL（seed 42, 24kHz），文本 = 云端 qwen3-asr-flash 转录（统一走 ASR，不用 LRS3 官方整句 txt）；250/250 成功
- TFG：远端 Ditto TRT（multiset_pipeline）、本地 LeapTalk（leaptalk_multiset）
- 评分：SyncNet V2，`min_track=25`（GRID 3s / 短视频低于 100 帧）

## 结果（2026-08-04 两套独立评分）

| TFG 模型 | n（配对） | Natural C | TTS C | ΔC | C 正向 | ΔD |
|---|---:|---:|---:|---:|---:|---:|
| Ditto | 250 | 4.032 | 5.012 | **+0.980** | 204/250 | +0.117 |
| LeapTalk | 248 | 5.221 | 6.477 | **+1.256** | — | −0.338 |

- LeapTalk 失败 2 条（ID 134 解析失败、248），Ditto 500/500 成功
- GRID/LRS3 低分辨率（360×288 / 224×224）压低绝对 Sync-C，配对 Δ 不受影响

## 结论

1. 两个 TFG 模型、5 个数据集、250 样本规模下 TTS 优势均稳健成立（ΔC +0.98~+1.26）——这是项目史上最大配对样本量的跨数据集证据，显著强于英文 LibriSpeech/HDTF 的负结果
2. 与「中文朗读域成立、自发域失效」形成对照：multiset 是朗读/独白/受控语料混合，说明 TTS 红利在规整语域跨数据集泛化
3. LeapTalk 绝对分高于 Ditto（5.22 vs 4.03 natural），ΔC 也更大

## 产物位置

- Ditto 评分：`runs/multiset_pipeline/04_eval/`（eval_meta.json + scores_250.json）
- LeapTalk 视频：`runs/leaptalk_multiset/{natural_raw,tts_raw}/`（500 mp4）
- LeapTalk 评分：`runs/leaptalk_eval/04_eval/`（scores_leaptalk.json）
- 准备脚本：`scripts/setup_multiset_run.py`；配置：`scripts/configs/multiset250*.yaml`

## Observations

- [status] concluded
- [hypothesis] TTS 唇同步优势在多种 TFG 数据集（非单一 ASR 语料）上依然成立
- [result] Ditto n=250 跨 5 数据集：ΔSync-C +0.980，204/250 正向；LeapTalk n=248：ΔSync-C +1.256，ΔSync-D −0.338
- [conclusion] TTS 优势跨数据集（MEAD/VFHQ/TalkVid/LRS3/GRID）与跨 TFG 模型（Ditto/LeapTalk）稳健，是目前最大的跨数据集证据
- [boundary] 音频语言混杂、GRID 固定句法、LRS3 TED 独白——不覆盖自发对话域；绝对分受分辨率影响，只看配对 Δ

## Relations

- relates_to [[multidataset-samples]]
- relates_to [[tts-exp]]
- relates_to [[多语言 Ditto 测评（日法葡俄）]]
