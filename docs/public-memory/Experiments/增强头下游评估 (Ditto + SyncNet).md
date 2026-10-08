---
title: 增强头下游评估 (Ditto + SyncNet)
type: experiment
permalink: tts-exp/experiments/增强头下游评估-ditto-sync-net
---

# 增强头下游评估 (Ditto + SyncNet)

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| Initial Document | August 11, 2026 | user |
| Restructured to Experiments/ with schema observations | August 12, 2026 | user |

两阶段增强头(29 号报告)在 HuBERT L6 特征空间上验证了有界、可复现的小幅移动(enhanced 比 natural 接近 raw-TTS 约 1.4%, 50/50 全 closer), 但从未经过下游验证。本次将 seed-29 的 50 个 enhanced WAV + 配对的 natural/TTS 音频交给 Ditto 生成口型视频, 再用 SyncNet 评分, 检验特征位移是否转化为唇形同步改善。

## 结果 (SyncNet, 50 valid pairs, 固定参考脸, seed 42)

| 条件 | Sync-C (越高越好) | Sync-D (越低越好) |
|---|---|---|
| natural_raw | 5.059 | 8.078 |
| tts_raw | **6.140** | 7.533 |
| enhanced_raw | 5.121 | 8.033 |

- TTS vs natural: ΔSync-C = **+1.081** (复现核心结论, 中文 TTS 优势成立)
- enhanced vs natural: ΔSync-C = **+0.062** (几乎无提升, 负结果)

## 结论

1. 增强头 1.4% 的 HuBERT L6 特征位移 **未能穿透 Ditto 原生 1024-D frontend 转化为下游改善** — 与 29 号报告的边界声明完全一致
2. 佐证了 "768-D HF HuBERT 接近 ≠ Ditto 原生表示有效" 的鸿沟
3. 增强音频身份漂移极小(≈0.0009), 本质仍是 natural 音频 + 微量 TTS 方向偏移 — 量级不足以改变唇形

## 过程要点

- 服务器: autodl RTX 4080 SUPER 32GB, Ditto TRT online, 150 视频全部成功
- SyncNet 踩坑: 短音频(<4s)视频转 25fps 后 <100 帧, run_pipeline 的 min_track=100 导致 track 为空 → run_syncnet 静默无输出; 修复: `--min-track 50`
- 评测链路: run_pipeline.py (S3FD 人脸检测+裁剪) → run_syncnet.py (SyncNet V2)
- 服务器已 shutdown; 服务器 [redacted-local-path] 保留代码+音频副本

## 产物位置

- 视频: `runs/two_stage_hubert_aishell1_20260810/ditto_videos/{natural_raw,tts_raw,enhanced_raw}/`
- 评分: `runs/two_stage_hubert_aishell1_20260810/ditto_videos/syncnet_eval/`
- 特征审计脚本: `scripts/41_hubert_proximity_audit.py` (可复用)
- 样本映射: `ditto_videos/sample_mapping.json` (id → paired_key/speaker)

## Observations
- [status] concluded
- [hypothesis] 增强头 HuBERT L6 特征位移能穿透 Ditto 原生 frontend 转化为唇形同步改善
- [result] enhanced vs natural ΔSync-C = +0.062, 无下游改善 (负结果)
- [conclusion] 1.4% 特征位移不足以改变唇形; 需要更大尺度位移或直接以 Ditto 原生 frontend 为目标
- [report] 记忆笔记, 无对应 docs/experiments 编号报告

## Relations
- relates_to [[tts-exp]]
- relates_to [[TTS Feature Alignment and Duration Model Research]]
- [inputs] 50 valid pairs: results/rhythm_style_500/aishell1_test_400/{natural,tts}/; enhanced WAV runs/two_stage_hubert_aishell1_20260810/stage2_feature_targeted_20260810_v1_seed29/valid_enhanced_wav/ (16/24 kHz, seed-29 Stage-2)
- [outputs] Ditto videos runs/two_stage_hubert_aishell1_20260810/ditto_videos/{natural_raw,tts_raw,enhanced_raw}/ (150 mp4); SyncNet scores syncnet_eval/
- [model] Ditto TRT online (hubert_streaming_fix_kv frontend); SyncNet V2 data/syncnet_v2.model; reference face data/data/image/1.png
- [code_paths] scripts/03_ditto.py, scripts/ditto_seeded_inference.py, scripts/04_eval.py, scripts/41_hubert_proximity_audit.py

## MDC 英文 heldout 远端验证（2026-08-09 补录）

增强头在 MDC 英文语料（100 条记录 / 50 对：train 30, valid 6, heldout 14）上训练后，对 14 条 heldout 做远端 Ditto 三臂验证（natural_raw / enhanced_raw / tts_raw，min_track=25）：

| 条件 | Sync-C | Sync-D |
|---|---:|---:|
| natural_raw | 4.134 | 7.998 |
| enhanced_raw | 4.088 | 8.021 |
| tts_raw | **5.209** | 8.101 |

- TTS vs natural ΔSync-C ≈ +1.075（英文 MDC 上 TTS 优势成立，与 LibriSpeech/HDTF 不同）
- enhanced vs natural ΔSync-C ≈ −0.046——增强头输出仍未穿透 Ditto，与 AISHELL-1 下游评估的负结果一致（跨语料重复）
- 产物：`runs/hubert_waveform_mdc_20260809/remote_ditto_syncnet_heldout/`

- [result] MDC 英文 heldout n=14：natural 4.134 / enhanced 4.088 / tts 5.209——TTS +1.075 成立，enhanced 无提升。 #mdc #heldout #ditto
