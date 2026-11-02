---
title: LRS3 masked TTS trajectory-specificity diagnosis
type: report
permalink: tts-exp/experiments/lrs3-masked-tts-trajectory-specificity-diagnosis
tags:
- masked-tts
- trajectory-specificity
- tfg
- syncnet
- hard-negative
- exploratory
---

# LRS3 masked TTS trajectory-specificity diagnosis

2026-09-02 完成 masked-TTS trajectory-specificity exploratory diagnosis。该 change 复用了此前已经 inspect 过的 16 条 confirmation records，因此不是新的独立 confirmation；所有 downstream 视频均使用 frozen direct-mel Wav2Lip，随后替换为对应 untouched natural audio，再由 official SyncNet V2 评分。sealed splits 未访问。

**Why:** 既有 `PAIRED_TTS > NAT_ONLY` 只确认 modality-level downstream utility，不能说明 paired instance trajectory 优于 phone centroid；本实验审计 unchanged natural-mel reference，加入 same-phone wrong-instance 和 within-phone reversed controls，并在冻结模型缺乏 reconstruction trajectory signal 时尝试一个固定 hard-negative objective。

**Part A:** `PAIRED_NOT_SHOWN_TO_BEAT_NATURAL_REFERENCE`。`PAIRED_TTS` 相对 unchanged `NATURAL_MEL` 没有通过两个指标的 reference rule；这不等价于证明 natural mel 统计显著优于 paired。`PHONE_CENTROID` 相对 `NAT_ONLY` 通过 practical rule：Sync-C median `1.28700`, 95% CI `[0.74150, 1.61350]`, 8/8 positive；Sync-D median `1.01175`, 95% CI `[0.56400, 1.34750]`, 8/8 positive。

**Part B:** 在 229 个 frozen evaluation masks 上构造 458 个 deterministic control features，完成 1,374 control reconstruction cells 和 96 downstream cells。`SAME_PHONE_WRONG_INSTANCE` 的 reconstruction gain median `0.05766`, 95% CI `[-0.05490, 0.15016]`, 5/8 positive；`WITHIN_PHONE_REVERSED` median `0.01519`, 95% CI `[-0.12149, 0.04087]`, 5/8 positive，因此预注册状态为 `NO_TRAJECTORY_SIGNAL`。有趣的是两类 control 在 frozen downstream 上 paired contrast 为正，但因为 reconstruction gate 未通过，不能标记为 trajectory-sensitive TFG result。

**Part C:** 按唯一允许的固定 objective 训练三个 600-step hard-negative models：margin `0.01`, ranking weight `0.10`, architecture/split/masks/optimizer/seed unchanged。reconstruction gate 通过：wrong-instance gain median `0.09799`, 95% CI `[0.00708, 0.19597]`, 7/8 positive；reversed gain median `0.03936`, 95% CI `[0.00218, 0.10024]`, 7/8 positive；new paired loss median `0.61478`，original `0.61600`，低于 5% guard 上限 `0.64680`。完成 192 个四条件 downstream cells 后，hard-negative model 的 paired-over-wrong/reversed 均通过 Sync-C/Sync-D rule，且 paired-over-NAT_ONLY modality rule 通过：wrong-instance Sync-C median `0.54775`, CI `[0.29800, 0.69150]`, 8/8；Sync-D median `0.39850`, CI `[0.20100, 0.67000]`, 8/8；reversed Sync-C median `0.36275`, CI `[0.15800, 0.60300]`, 7/8；Sync-D median `0.29650`, CI `[0.17050, 0.44800]`, 7/8；modality Sync-C median `1.33050`, CI `[0.97150, 2.08100]`, 8/8；Sync-D median `0.79250`, CI `[0.30150, 1.39150]`, 8/8。Part C 状态为 `TRAINING_TFG_GO`。

**Decision:** `CONFIRM_TRAJECTORY_MODEL_ON_NEW_RECORDS`。这只允许在新的独立 records 上确认修改后的 mel model；本 change 不打开 waveform-decoder gate。

**How to apply:** 下一步只做新独立记录 confirmation，保持 hard-negative model 和 frozen evaluation protocol；不要把当前结果表述为 audible TTS retention、waveform reachability、audio quality、natural-prosody preservation、population generalization 或 deployable replacement system。结果文件位于 `runs/lrs3_masked_tts_trajectory_specificity_20260902/`，最终决策为 `decision.json`，Part B 分析为 `03_frozen_diagnosis/analysis.json`，Part C 分析为 `04_training/analysis.json`。

## Relations

- extends [[LRS3 masked TTS TFG confirmation]]
- relates_to [[tts-exp]]