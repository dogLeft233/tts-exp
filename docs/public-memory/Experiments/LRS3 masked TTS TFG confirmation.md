---
title: LRS3 masked TTS TFG confirmation
type: report
permalink: tts-exp/experiments/lrs3-masked-tts-tfg-confirmation
status: confirmed-modality-only
date: '2026-09-02'
experiment: masked-tts-tfg-confirmation
tags:
- masked-tts
- tfg
- syncnet
- confirmation
- modality-gain
---

# LRS3 masked TTS TFG confirmation

2026-09-02 的确认性 direct-mel frozen-Wav2Lip 实验在首轮 probe 未使用的 16 条 parent evaluation records 上完成；8 个 source groups 每组 2 条，共 160 个 drivers/renders/scores。所有 rendered videos 都经过 untouched natural audio 的 strict replacement；PCM exact match 为 160/160，video stream copy 为 160/160，Wav2Lip 与 SyncNet frozen hashes 全部一致，sealed splits 未访问。

**Why:** 首轮 8-group probe 的 `PAIRED_TTS > NAT_ONLY` modality gain 需要在未参与首轮下游评分的记录上复现，才能决定是否值得推进 waveform decoder；同时要继续检验 `PAIRED_TTS > PHONE_CENTROID` 是否存在 token-specific signal。

**Result:** modality Sync-C 为 8/8 groups positive，中位 gain `1.50075`，95% CI `[0.79750, 1.65100]`；modality Sync-D 为 8/8 positive，中位 gain `0.99800`，95% CI `[0.46700, 1.32250]`。token Sync-C 为 5/8 positive，中位 gain `0.17950`，95% CI `[-0.27250, 0.38400]`；token Sync-D 为 5/8 positive，中位 gain `0.06950`，95% CI `[-0.40950, 0.22800]`。

**Decision:** `engineering=GO`, `science=CONFIRMED_MODALITY_ONLY`。结果确认 TTS-side conditioning 相对 `NAT_ONLY` 具有 direct-mel frozen-Wav2Lip 下游价值，但没有确认超越 phone centroid 的 token-specific TTS signal；waveform decoder 不应作为细粒度 TTS retention 实验推进。

**How to apply:** 后续可以把 modality-level downstream utility 作为已复现现象继续研究；若目标是证明细粒度 TTS retention，当前证据仍不足。该结论不代表 audible TTS-feature retention、waveform reachability、audio quality、natural prosody preservation、population generalization 或 deployable replacement system。结果文件位于 `runs/lrs3_masked_tts_tfg_confirmation_20260902/`，分析报告为 `05_analysis/report.md`，决策为 `decision.json`。

## Relations

- relates_to [[TTS Enhancer Feasibility Diagnosis]]
- follows [[LRS3 masked TTS retention scale-up preflight]]
