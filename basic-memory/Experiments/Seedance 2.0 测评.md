---
title: Seedance 2.0 测评
type: experiment
permalink: tts-exp/experiments/seedance-2.0-测评
tags:
- seedance
- doubao
- video-gen
- syncnet
- english
---

# Seedance 2.0 测评

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| Initial Document | August 14, 2026 | user |

2026-08-05 用豆包 Seedance 2.0（视频生成模型）检验 TTS-vs-Natural 唇形同步优势是否存在于该模型，n=6 配对。此前记忆与文档均未记录本次测评，现从会话导出与 `runs/seedance_validation/` 产物补录。

## 方法

- 模型：`doubao-seedance-2-0-260128`（API 生成，task id 见 results.json）
- 参考图：seedream-4-0 生成的照片级动漫脸（`runs/seedance_validation/real_anime.png`）
- 音频：LRS3 样本 152 源音轨前 5s（natural）；`data/tts_audio_250/152.wav`（faster_qwen3 自克隆，Obama 就职演讲文本，英文）
- 每条件生成 6 条视频（nat_1..6 / tts_1..6，其中 tts_1 生成失败无视频）
- 评分：本地 syncnet_python + syncnet_v2.model (CUDA)

## 结果

| 条件 | Sync-C ↑ | Sync-D ↓ |
|---|---:|---:|
| natural | 6.093 | 9.867 |
| tts | 6.098 | 9.713 |
| Δ (tts−nat) | +0.005 (p=0.992) | −0.155 (p=0.724) |

- 辅助指标：ASR fidelity（nat_1 0.262 / nat_2 0.368 / tts_1 0.507）——长句语义漂移，natural/tts 无系统差异

## 结论

**TTS 音频对 Seedance 2.0 生成无优势**（口型同步与语义保真均无差异）。与英文 LibriSpeech/HDTF 负结果方向一致：英文 + 视频生成模型场景下 TTS 红利不成立。n=6 小样本、长句（5s）、动漫风参考图，属探索性证据。

## 产物位置

- 视频：`runs/seedance_validation/videos/`（nat_1..6、tts_2..6、real_anime.mp4，共 12 条）
- 分数与 task id：`runs/seedance_validation/results.json`

## Observations

- [status] concluded
- [hypothesis] TTS 音频对 Seedance 2.0 的唇形同步也有优势
- [result] Sync-C natural 6.093 vs tts 6.098，Δ+0.005 (p=0.992)；Sync-D Δ−0.155 (p=0.724)——无差异
- [conclusion] Seedance 2.0 上 TTS 无优势；与英文跨模型负结果一致
- [boundary] n=6、英文长句、动漫参考图；探索性证据

## Relations

- relates_to [[tts-exp]]
- relates_to [[多语言 Ditto 测评（日法葡俄）]]
