---
title: LRS3 ASR-SyncNet correlation result
type: report
permalink: tts-exp/docs/experiments/lrs3-asr-sync-net-correlation-result
engineering_decision: GO
run_dir: runs/lrs3_asr_sync_error_correlation_20260831
scientific_overall: SUPPORTED_NATURAL_ONLY
status: complete
tags:
- lrs3
- asr
- syncnet
- correlation
- experiment
---

实验已于 2026-08-31 在 LRS3 frozen fit-only fresh_confirmation 24-sample / 48-arm cohort 完成。工程门为 GO：preflight、strict mux、offline Wav2Vec2-CTC greedy ASR、per-arm CTC forced alignment、完整 SyncNet distance matrix/local_c、alignment、analysis、48 arm outputs、marker/hash audit 和 24 plots 均通过；未访问 sealed split。

科学结论：natural arm 为 SUPPORT，median Spearman 的 10,000-draw whole-sample bootstrap 95% CI 为 [0.0555, 0.1799]，median badness contrast CI 为 [0.1640, 1.4841]；TTS arm 为 NO_SUPPORT，两个 CI 分别为 [-0.0283, 0.1595] 与 [-0.1315, 1.7042]。Micro IoU / error recall / low-sync precision：natural 0.0943 / 0.2321 / 0.1371，TTS 0.0419 / 0.1491 / 0.0551。整体状态为 SUPPORTED_NATURAL_ONLY，不应概括为 TTS 普遍支持该相关性。

**Why:** 该结果回答了 ASR 错误时间段与低局部唇形同步质量时间段是否重合，并保留了 natural-vs-TTS 的对照边界。
**How to apply:** 后续讨论应以 run `runs/lrs3_asr_sync_error_correlation_20260831` 的 JSON/NPZ/plots 为依据，区分工程完整性与两种 arm 的科学门，不把 frame-level 相关性当独立样本显著性。