---
title: LRS3 natural-to-TTS bridge confirmation result
type: report
permalink: tts-exp/experiments/lrs3-natural-to-tts-bridge-confirmation-result
status: complete
date: '2026-09-04'
run_root: runs/lrs3_natural_to_tts_bridge_confirmation_20260904
scientific_decision: CONTROL_FAILED
engineering_decision: GO
tags:
- lrs3
- replacement
- natural-to-tts
- syncnet
- experiment
---

# LRS3 natural-to-TTS bridge confirmation result

## Context

按 OpenSpec `confirm-lrs3-natural-to-tts-bridge` 执行了一个不训练、不生成新 TTS、不重跑 MFA/DTW 的 fit-only confirmation run。实验使用 parent 133-record cohort 中每个 source group 的第二条记录，形成 22 records / 22 source groups，并与 discovery 的 23-record cohort disjoint。

**Why:** 第一轮 `MAG_075`/`MAG_100` 有 promising movement 与 replacement 子门槛，但 `SHIFT_200` 没有验证 endpoint 的 timing sensitivity；本实验用独立 cohort、`N_REPEAT` 和 `LOCAL_SWAP` 重新校准 endpoint。

**How to apply:** 后续讨论只能把本次结果解释为固定 22-record fit-only cohort、固定 `BRIDGE_075` 构造和固定 Wav2Lip/SyncNet endpoint 下的结果；不得外推到 identity transfer、整个 LRS3、训练或部署。

## Execution

- OpenSpec: `openspec/changes/confirm-lrs3-natural-to-tts-bridge/`
- Run root: `runs/lrs3_natural_to_tts_bridge_confirmation_20260904/`
- Arms: `N`, `N_REPEAT`, `LOCAL_SWAP`, `BRIDGE_075`
- Media matrix: 88 videos, 132 unique SyncNet cells
- Independent validator: Stage 00–04 all valid
- Focused tests: 20 passed；Ruff、Python compilation、OpenSpec strict validation passed
- Forbidden-operation ledger: no sealed media、training、fine-tuning、new TTS、MFA/DTW、score-based retry 或 outcome-based filtering

## Registered results

### Controls

- `N_REPEAT` repeatability passed。`gap_C` mean 0.0195，95% CI `[-0.0140, 0.0554]`；`gap_D` mean 0.0075，95% CI `[-0.0233, 0.0397]`；offset agreement 22/22。
- `LOCAL_SWAP` own-audio validity failed。own-audio `gap_C` 95% CI `[-3.2993, -2.5611]`，`gap_D` 95% CI `[-3.5761, -2.8754]`；local sensitivity `damage_C` 95% CI `[-3.1383, -2.4857]`，`damage_D` 95% CI `[-3.3337, -2.7171]`；两项 damage 同时为正的记录为 0/22。

### Bridge descriptive outcomes

- `BRIDGE_075` movement passed：progress 22/22 达到 0.15，mean 0.5118，95% CI `[0.4539, 0.5685]`。
- Replacement compatibility 子门槛通过：`gap_C` mean 0.0315，95% CI `[-0.0342, 0.1047]`；`gap_D` mean 0.0172，95% CI `[-0.0332, 0.0683]`；offset agreement 22/22。
- Primary SyncC useful-effect gain 未通过：mean 0.0315，但 95% CI 下界为 -0.0342，不满足严格大于 0 的要求。

### 两轮改善样本比例与视觉输入（September 13, 2026 复核）

统一口径为：候选驱动生成的视频与原 natural 音轨评分，再减去 natural 驱动视频与同一 natural 音轨的分数。C 越高越好；D 改善量定义为 baseline D − candidate D。

| 队列 | C 提升 | D 改善 | C/D 同时改善 | mean ΔSync-C | C 95% CI |
|---|---|---|---|---|---|
| discovery MAG_075，23 条 | 18/23，78.3% | 15/23，65.2% | 15/23，65.2% | +0.078 | [+0.016,+0.136] |
| confirmation BRIDGE_075，22 条 | 9/22，40.9% | 11/22，50.0% | 9/22，40.9% | +0.031 | [−0.034,+0.105] |

从两轮 `05_final/analysis.json`（discovery）与 `04_final/analysis.json`（confirmation）的 per_record/compatibility 重算。discovery 根为 `runs/lrs3_phase_preserving_replacement_envelope_20260904/`。探索轮考察多个强度，区间未经多重选择校正；确认轮记录不同但来源组重合，不是全新来源验证。约 +0.07 的历史记忆属实，但改善比例和均值未在确认轮稳定复现；mel movement 22/22 不是同步改善 22/22。

这两轮 bridge 的 Wav2Lip --face 是 LRS3 原始动态 MP4。不能外推为全项目一直用视频：早期 AISHELL/Ditto 主实验用静态肖像 PNG，部分早期 Wav2Lip 下游用 natural-driven Ditto 视频；后续全局 shift 与参考交互诊断也有固定第 0 帧的 STATIC 条件。

原视频动作残留/音频驱动传递不足是合理假设；固定生成 crop 的三样本诊断支持继续检查，但并未证明 mouth leakage，也不能否定全部静态输入实验。新设计 [[静态图片 Natural-to-TTS bridge 实验]] 单独测冻结 B 波形；本次仅 spec，尚未运行。

## Decision

终态 artifact 的 `scientific_decision` 为 `CONTROL_FAILED`，`engineering_decision` 为 `GO`，`reference_conditioned_audio_head_spec_eligible=false`。

由于 `LOCAL_SWAP` control 失败，`BRIDGE_075` 的 movement、replacement compatibility 和 primary SyncC 结果只能作为 descriptive observations，不能解释为科学确认，也不能授权后续 reference-conditioned audio-head OpenSpec、训练、identity transfer 或 deployment。

## Observations

- [decision] 本次确认实验按注册 cohort、arms、matrix 和终态规则完成，未产生科学确认 #experiment
- [result] 从 natural 保留时间/phase scaffold 并向 MFA-linear TTS magnitude 移动仍能稳定产生 Wav2Lip mel movement #audio-visual
- [problem] 当前 `LOCAL_SWAP` 负 control 没有满足 own-audio validity，且没有显示预期的 replacement damage；该 endpoint calibration 仍未通过 #syncnet
- [insight] replacement compatibility 的非劣性与 primary useful-effect gain 是不同门槛；本次前者通过、后者失败 #replacement

- [result] discovery MAG_075 的 Sync-C 改善 18/23（78.3%），confirmation BRIDGE_075 为 9/22（40.9%）；两轮 mean ΔC 分别 +0.078/+0.031。
- [boundary] bridge 历史动态输入、静态肖像主实验和后续 STATIC 诊断必须分开；不能把动态嘴型残留假设推广到全部工作。
- [next] 按用户要求交付静态图片 bridge OpenSpec；只 planned，不新增实验结果。

## Relations

- relates_to [[LRS3 MFA-linear TFG native/replacement 2026-09-13|LRS3 MFA-linear replacement NO-GO]]
- relates_to [[Natural slot oracle result]]
- implements [[confirm-lrs3-natural-to-tts-bridge]]

- relates_to [[LOCAL_SWAP 最小重放诊断 2026-09-13]]
- motivates [[静态图片 Natural-to-TTS bridge 实验]]

## Changelog

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 从历史逐样本结果核对两轮提升比例，澄清视觉输入类型与解释边界，链接静态 bridge spec；原 CONTROL_FAILED 保留 | September 13, 2026 | user |
