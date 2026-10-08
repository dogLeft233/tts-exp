---
title: tedlium3-tts-augmentation
type: note
permalink: tts-exp/docs/experiments/tedlium3-tts-augmentation
tags:
- tedlium3
- qwen3
- tts
- 增强头
- 训练数据
---

# TED-LIUM 3 → Qwen TTS 增强头训练数据

## 目标

用 TED-LIUM 3 语料 + faster_qwen3 (0.6B ICL) 自克隆 TTS 生成增强头 (waveform HuBERT enhancer) 训练配对数据 (natural / tts / weak phone-local target)。

## TED-LIUM 3 数据 (data/tedlium3_30pct/)

- 32 个 HF parquet 分片, 81,760 条, 15.97GB, ≈277h, 均值 12.2s
- 字段: `audio(bytes: 16kHz mono PCM wav)`, `text` (ASR 转写, 含 `<unk>`), `speaker_id` (如 `CKWilliams_2001`), `gender` (int64), `file`, `id` (如 `CKWilliams_2001-11.95-18.05-<NA>`)
- 来源: SpeechColab TED-LIUM 3 (TED 演讲) 30% 子集

## 适配结论 (已验证)

- **采样率 ✓**: `build_waveform_hubert_targets.py:250` 显式接受 "16 kHz or explicit 24 kHz Faster-Qwen3 output"; Qwen 24kHz 输出会被 resample_poly 到 16k
- **训练脚本 ✓**: `train_waveform_hubert_enhancer.py` 对 tts_path 用 `allow_24k_resample=True`
- **ICL 自克隆 ✓**: `ref_text == text` (与 02_tts.py:140 同款), 需要清洗 `<unk>`
- **约束**: natural/tts 两臂 transcript 必须一致 (`validate_pair` 强校验); 两臂都需要 MFA spans (build_waveform_hubert_targets.py `_span_records`); speaker 不能跨 train/valid/heldout split (`_split_items` 泄漏检查); target 定义 `weak_phone_local_tts_target` schema v1

## 数据流

```
TED parquet → 抽 audio(bytes)/text → 清洗 text(<unk>)
→ Qwen ICL 自克隆生成 24kHz TTS
→ MFA 对齐 (natural + tts)
→ build_waveform_hubert_targets.py (phone_local_warp)
→ train_waveform_hubert_enhancer.py (V100, fp16 推理 / fp32 训练)
```

## 相关

- Qwen 部署: docs/deployment/qwen3-tts
- 增强头脚本: scripts/common/build_waveform_hubert_targets.py, scripts/common/train_waveform_hubert_enhancer.py
## 断点 (2026-08-12 00:18 暂停)

- natural 全部提取: 81,760/81,760 (16GB)
- TTS 已生成: 3,283/81,760 (4.0%, 1.3GB), 速率 ~690 条/h (V100 单进程 ~5s/条)
- checkpoint: `runs/tedlium3_tts/02_tts/checkpoint.json` (含 resume 命令与 id 哈希)
- 断点续跑: 重跑同一命令, 已存在 wav 自动跳过

```bash
~/.venvs/qwen3/bin/python scripts/experiments/tedlium/generate_tedlium3_tts.py \
  --run-id tedlium3_tts --all --config scripts/configs/tedlium3_local_v100.yaml
```

- 预计全量总耗时 ~5 天; tts_meta.json 只在跑完后写一次
