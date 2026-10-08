---
title: LRS3 masked TTS retention scale-up preflight
type: experiment
permalink: tts-exp/experiments/lrs3-masked-tts-retention-scale-up-preflight
code_paths:
- scripts/experiments/masked_tts_reconstruction/scaleup_run.py
- scripts/experiments/masked_tts_reconstruction/train.py
- scripts/experiments/masked_tts_reconstruction/protocol.py
- tests/experiments/masked_tts_reconstruction/test_scaleup.py
hypothesis: Paired token-level TTS-L6 may improve masked natural-mel reconstruction
  beyond natural context and a train-only phone centroid.
inputs: Fit-only LRS3 source pool, MFA token alignment cache, local pinned kNN-VC
  WavLM-L6; exploratory run reuses the formal frozen masks.
model: bshall/knn-vc revision c616845c4e309e24d5927f15adbdf277a3d65358
outputs: runs/lrs3_masked_tts_retention_scaleup_20260901/; runs/lrs3_masked_tts_retention_exploratory_20260901/
status: concluded
tags:
- lrs3
- masked-reconstruction
- tts-retention
- fit-only
---

# LRS3 masked TTS retention scale-up preflight

## Context

按 `scale-lrs3-masked-tts-retention` OpenSpec 执行扩大规模的 fit-only masked natural-mel reconstruction 实验，目标是复现 paired TTS gain，并用 phone-centroid control 区分 token trajectory 与 phone identity。正式 run 严格执行了 readiness gate；随后按用户要求，在不修改正式 run 的前提下，用同一批不足门槛的数据完成了独立 exploratory descriptive run。

## Frozen execution

- Cohort：12 个 train source groups、8 个相对 prototype 十组全新的 evaluation groups；47 train records、24 evaluation records。
- 数据快照：71 个样本均生成了 MFA token alignment 和 TTS WavLM-L6 snapshot；WavLM/HiFi-GAN 使用本地 pinned kNN-VC revision，不修改旧 cache。
- Support filter：只保留至少 20 个 train mask instances 且覆盖至少 3 个 train groups 的 lexical phones；最终保留 20 个 phones。
- 共享 manifest：805 train masks、392 evaluation masks；每个 evaluation group 均至少 30 masks，但低于正式要求的 900/400 总分母。

## Formal gate and exploratory result

正式 scale-up run `runs/lrs3_masked_tts_retention_scaleup_20260901/` 按 spec 在训练前停止：`engineering=GO`、`science=INSUFFICIENT`，没有生成 schedule 或 checkpoint。

按用户要求创建的独立 exploratory run `runs/lrs3_masked_tts_retention_exploratory_20260901/` 使用完全相同的不足门槛 masks，完成了 3 seeds × 3 arms、每个 arm 固定 1,200 steps，以及 3,528 个 evaluation cells。每个 seed 的三个 arm 共享 byte-identical initial state 和 sampler schedule，`sealed_splits_accessed=false`。

- `modality_gain = L_total(NAT_ONLY) - L_total(PAIRED_TTS)`：median `0.0931845`，whole-group bootstrap 95% CI `[0.0621131, 0.1365000]`；8/8 evaluation groups 为正，3/3 seeds 为正；相对 NAT_ONLY 的 median reduction 为 `15.876%`。
- `token_gain = L_total(PHONE_CENTROID) - L_total(PAIRED_TTS)`：median `0.0081968`，95% CI `[-0.0360594, 0.0262510]`；只有 6/8 groups 为正，3 seeds 中仅 2 个 overall median 为正；相对 phone centroid 的 reduction 为 `-3.065%`。
- exploratory decision：`EXPLORATORY_DESCRIPTIVE_ONLY`。这说明 paired TTS branch 在当前 masked reconstruction task 中仍明显优于 separately trained NAT_ONLY，但没有显示 paired token trajectory 稳定超过 train-only phone centroid。

## Conclusion boundary

这次 exploratory 结果仍不能作为正式 scale-up 的 `TOKEN_LEVEL_TTS_SIGNAL_SUPPORTED` 或 `PHONE_LEVEL_ONLY_SUPPORTED` 结论，因为正式 denominator gate 未通过。它也不证明或否定 waveform 中的 TTS clarity、natural prosody preservation、waveform reachability、TFG/SyncNet gain 或 replacement effect。不得据此直接进入 waveform bridge、vocoder、TFG、SyncNet 或 replacement experiment。

## Artifacts

- Formal run：`runs/lrs3_masked_tts_retention_scaleup_20260901/`
- Formal decision：`runs/lrs3_masked_tts_retention_scaleup_20260901/decision.json`
- Exploratory run：`runs/lrs3_masked_tts_retention_exploratory_20260901/`
- Exploratory decision：`runs/lrs3_masked_tts_retention_exploratory_20260901/05_analysis/decision.json`
- Exploratory analysis：`runs/lrs3_masked_tts_retention_exploratory_20260901/05_analysis/analysis.json`
- OpenSpec：`openspec/changes/scale-lrs3-masked-tts-retention/`
- Code：`scripts/experiments/masked_tts_reconstruction/scaleup_run.py`、`train.py`、`protocol.py`
- Tests：`tests/experiments/masked_tts_reconstruction/test_scaleup.py`

## Observations
- [status] concluded

- [decision] 正式 run 遵守 denominator gate 并停止；exploratory run 单独使用不足分母完成 descriptive contrasts #masked-tts-retention
- [result] paired TTS 相对 NAT_ONLY 的 exploratory modality gain 为 `0.0931845`，95% CI `[0.0621131, 0.1365000]`，但不具备正式 scale-up 推断资格 #modality
- [result] paired TTS 相对 PHONE_CENTROID 的 token gain 为 `0.0081968`，95% CI 跨零，且相对 centroid reduction 为 `-3.065%` #token-level
- [insight] 当前数据支持 TTS branch 作为 masked natural-mel reconstruction 的辅助输入，但没有支持 paired token trajectory 超过 phone identity centroid #fit-only
- [constraint] exploratory descriptive result 不授权 waveform bridge、vocoder、TFG、SyncNet 或 replacement experiment #claim-boundary

## Relations

- follows [[Masked TTS reconstruction feasibility prototype]]
- constrained_by [[Raw TTS strict replacement NO-GO]]
