---
title: MFA-linear existing-data diagnostic outcome
type: report
permalink: tts-exp/experiments/mfa-linear-existing-data-diagnostic-outcome
tags:
- mfa-linear
- tts-only
- lrs3
- syncnet
- diagnostic
- wav2lip
---

# MFA-linear existing-data diagnostic outcome

## Context

为验证 TTS-only MFA-linear 音频模型是否能在真实视频上改善同步，在现有合规 LRS3 资产上执行了放宽后的记录留出实验。模型训练和推理始终只接收 exact-length 的 MFA-linear TTS 波形；natural audio 没有进入模型输入、训练损失或 cross-attention，只用于既定的对齐构造、脱离计算图的坐标校准，以及官方比较中的 natural-audio 参考臂。

## Protocol

- 从 102 条完全合格记录中固定选择 94 条训练记录和 8 条评估记录。
- 训练记录与评估记录的 sample ID 不重叠；由于现有数据不足以保证 source-group 不重叠，训练/评估允许 source-group overlap，因此结果只能称为 record-heldout，不能称为 unseen-source-group、unseen-speaker 或总体泛化。
- 重新训练 100 步，保持模型、冻结 SyncNet、31 个内部偏移、官方偏移符号、目标间隔损失、曲线指标、PCM16 和 waveform QC 规则不变。
- 原始候选音频 normalized log-mel QC 上限为 0.10；其中一条候选为 0.1005600542，仅超出约 0.00056。随后单独创建 post-hoc 诊断，将该项诊断上限放宽至 0.11，不重新训练、不重新选样本，也不放宽残差峰值、PCM 饱和、形状、有限值、目标偏移或 SyncNet 规则。

## Execution

- 94-record fresh training completed for the full 100 steps, including step-zero and step-100 checkpoints.
- The original relaxed run was correctly sealed as `RECORD_HELDOUT_REAL_VIDEO_NOT_EVALUATED` because the 0.10 QC gate rejected one candidate.
- The clean post-hoc diagnostic completed all 8 official real-video SyncNet curves and wrote a valid artifact graph.
- Run root: `runs/lrs3_mfa_linear_existing_data_audio_tolerance_20260903_final/`.
- Validation: `validation.json` reports `status: valid` and `artifact_graph_valid: true`.
- Terminal status: `DIAGNOSTIC_REAL_VIDEO_COMPLETE`; process exit code 2 reflects `pass: false`, not a runner crash.

## Result

- Official real-video records: 8.
- Engineering-valid records: 8/8.
- Records satisfying both scientific gains: 0/8.
- Median SyncNet distance gain, baseline distance minus candidate distance: -0.0205335617. Negative means the candidate was worse on the median minimum distance.
- Median SyncNet curve-separation gain, candidate separation minus baseline separation: -0.0126166344. Negative means the candidate’s best-offset preference was less distinct on the median record.
- The unchanged real-video gate did not pass.
- All 8 records matched the target offset, but target-offset agreement alone does not establish improvement. Several records improved one curve metric while degrading the other; none met both minimum gains of 0.003 together with the remaining gates.
- The candidate that triggered the original QC boundary was usable under the diagnostic 0.11 limit, but relaxing that boundary did not produce evidence of reliable transfer.

## Downstream replacement

Wav2Lip replacement was not run. The protocol permits replacement only after the complete real-video transfer gate passes; because it failed, there is no valid evidence for candidate audio improving a frozen Wav2Lip downstream model. Diagonal candidate/candidate behavior is therefore unavailable and cannot be used as a substitute claim.

## Interpretation

The fresh 94-record training run and the relaxed audio-quality diagnostic establish that the pipeline can train and score a record-heldout cohort with exact official curves. They do not support the claim that this TTS-only adapter reliably improves synchronization on real video. Because the tolerance change was chosen after observing the failing candidate and because source groups overlap, the result is post-hoc descriptive evidence only, not a pre-registered scientific transfer or generalization result.

## Observations

- [decision] Keep the 0.10 result as the original strict QC outcome and report the 0.11 run only as a separately labeled post-hoc diagnostic.
- [insight] The main failure is not missing candidate generation or a broken official scorer: all eight diagnostic candidates passed engineering checks and all eight official curves were completed.
- [insight] Target-offset signs and agreement were correct across all eight records, while paired improvement was absent.
- [decision] Do not run or claim Wav2Lip replacement after a failed real-video gate.
- [problem] Current fit-only inventory supports 102 eligible records across 10 source groups, far below the strict 200-record/40-group expansion requirement.

## Relations

- relates_to [[MFA-linear 200-record generalization expansion]]
- relates_to [[LRS3 MFA-linear TFG native/replacement 2026-09-13|LRS3 MFA-linear replacement NO-GO]]
- relates_to [[Natural-slot oracle result]]
