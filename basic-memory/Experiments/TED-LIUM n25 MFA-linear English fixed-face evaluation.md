---
title: TED-LIUM n25 MFA-linear English fixed-face evaluation
type: experiment
permalink: tts-exp/experiments/ted-lium-n25-mfa-linear-english-fixed-face-evaluation
status: concluded
completed: '2026-08-16'
dataset: TED-LIUM 3
sample_count: 25
speaker_count: 25
tags:
- tedlium
- mfa-linear
- english
- syncnet
- tfg
---

# TED-LIUM n25 MFA-linear English fixed-face evaluation

## Context

在 TED-LIUM 3 30% 子集已有的本地 faster-Qwen TTS 配对缓存中，冻结 25 条不同说话人的 natural/TTS 配对，使用英文 MFA 3.4.1 生成两侧独立对齐，再生成 MFA-linear 音频，并用同一固定人脸和固定 Wav2Lip/SyncNet 协议比较 natural、raw TTS、MFA-linear 三臂。

## Protocol

- Cohort: 25 utterances, 25 speakers, 每个 speaker 1 条；按预先固定的 speaker/时长规则选择，不依据评分换样本。
- TTS: 本地 faster_qwen3 已有缓存；本次未调用云端 TTS。
- MFA: English MFA 3.4.1，natural/TTS 各 25 条独立 TextGrid，tokens 25/25。
- MFA-linear: frozen WavLM-Large L6 + prematched HiFi-GAN；25/25 生成，输出与 natural 精确等长。
- TFG diagnostic: Wav2Lip + SyncNet，固定 face `data/data/image/1.png`，`min_track=50`，每个 sample/arm 独立工作目录；75/75 score 完整。

## Results

| 条件 | n | mean Sync-C | SD Sync-C | mean Sync-D | SD Sync-D |
|---|---:|---:|---:|---:|---:|
| natural raw | 25 | 6.87576 | 1.07057 | 6.96096 | 0.60449 |
| raw TTS | 25 | 6.63164 | 1.79310 | 6.72564 | 0.76528 |
| MFA-linear | 25 | 6.74792 | 1.23050 | 6.78104 | 0.46609 |

Sync-C 越高越好，Sync-D 越低越好：

- MFA-linear − natural: ΔC = −0.12784，bootstrap 95% CI [−0.57656, 0.23344]，C better 13/25；ΔD = −0.17992，CI [−0.37464, 0.00684]，D better 18/25；joint better 10/25；paired t-test p(C)=0.549，p(D)=0.0824。
- MFA-linear − raw TTS: ΔC = +0.11628，CI [−0.26536, 0.50964]，C better 12/25；ΔD = +0.05540，CI [−0.18660, 0.31520]，D better 14/25；joint better 5/25；paired t-test p(C)=0.575，p(D)=0.676。
- raw TTS − natural: ΔC = −0.24412，ΔD = −0.23532；两项均未达到显著性。

## Conclusion

本次 25 条结果不能作为干净的英文 TFG 总体结论：3 条 raw TTS 出现严重失控时长（相对 natural 分别约 2.36×、26.36×、25.72×），且当前 MFA-linear QC 没有设置 TTS 时长/尾部静音门槛。全量均值因此被质量异常样本污染。对这 3 条做的事后敏感性检查（仅诊断，不替代预注册主分析）显示剩余 22 条中 raw TTS 相对 natural 为 ΔC=+0.281、ΔD=−0.350；MFA-linear 为 ΔC=+0.208、ΔD=−0.256。下一次正式比较必须在生成后、评分前冻结 TTS 质量门槛并重新选定独立 cohort。

## Artifacts

- Run: `runs/tedlium3_mfa_linear_n25_20260816/`
- Scores: `05_wav2lip_syncnet/summary.json`
- Analysis: `05_wav2lip_syncnet/analysis.json`
- Evaluator: `scripts/eval_tedlium_n25_wav2lip_syncnet.py`
- First failed attempt was preserved in `05_wav2lip_syncnet_failed_missing_video_parent/`; Wav2Lip itself had succeeded, but the evaluator had not created the output parent directory. The evaluator was fixed and the complete matrix was rerun without changing the cohort or thresholds.

## Limitations

这是单一固定人脸、25 个 speaker 各 1 条 utterance 的 paired audio diagnostic，不是 speaker-matched 视频泛化实验；SyncNet/Wav2Lip 分数不能单独证明真实嘴型因果。MFA-linear 音频覆盖率和 fallback 仍需结合音频质量解读。

## Post-hoc data-quality audit

- 25 natural/TTS source pairs have unique paired keys and speakers; natural/TTS source hashes are all different, so no literal waveform duplication was found.
- Natural duration mean is 5.99 s; TTS duration mean is 18.02 s, median 5.28 s; duration-ratio median is 0.869 because three extreme outliers dominate the mean.
- `ted_013` TTS is 14.48 s for 6.14 s natural (2.36×); `ted_015` is 156.56 s for 5.94 s (26.36×); `ted_025` is 156.40 s for 6.08 s (25.72×). The latter two contain long trailing silence intervals in their MFA labels.
- The existing TTS generator uses the exact natural utterance as `ref_audio` and the same cleaned transcript as `ref_text`; this is a deliberate self-clone protocol, but it is not an independent text/audio evaluation and should be documented as reference leakage by design.
- The current run label `tedlium_mfa_linear_n25` is synthetic; no official held-out split was used. The cached TTS metadata is also stale (`limit=10`) while the directory contains 3,283 files, so provenance for the full cache is incomplete.
- TED-LIUM/TED-talk source overlap with common LRS/SyncNet training corpora is a separate evaluator-contamination risk; it does not explain the three duration failures and is not resolved by the fixed-face protocol.

## Observations

- [result] English MFA 50/50 alignments and MFA-linear 25/25 outputs completed.
- [result] Fixed-face SyncNet matrix completed 75/75 with no failures.
- [result] MFA-linear improved Sync-D versus natural on average but did not improve Sync-C.
- [decision] Do not present this TED-LIUM run as an overall TFG improvement; retain it as a negative/qualified English-language result.
- [protocol] Keep the fixed-face and one-utterance-per-speaker limitations explicit.

- [status] concluded

## Relations
- relates_to [[MFA-linear 冻结 VC 重合成与下游验证]]
- contrasts_with [[跨数据集 TFG 测评（5×50 multiset）]]
