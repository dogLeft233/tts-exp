---
title: 010-audio-content-check-before-render-completion
type: decision
permalink: tts-exp/docs/adr/010-audio-content-check-before-render-completion
git_sha: 1679f680278a291d5e2a9091d2a2ceac2a41a70c
repository: dogLeft233/tts-exp
status: open
project: tts-exp
decided: '2026-10-04T05:35:21.9994202+00:00'
branch: codex/cross-tfg-audio-head-20261003
---

# 010-audio-content-check-before-render-completion

## Decision
冻结候选后的内容检查独立于视频生成完成时间。先使用生产路径在CPU上生成开发集20条完整候选音频，并封存每条原自然F32、候选F32、sample count、配置、权重和转换源码；随后为每条新候选独立调用Qwen ASR。最终用于方法证据时，必须与实际生成ledger的conditioning PCM逐样本完全一致。

## Context and rationale
跨三frontend、三paired seeds的候选开发验证有180次生成，同一候选在同一clip上的波形却是确定的。等待全部视频完成才检查内容会延后发现损伤。已冻结的v4权重及完整音频CLI等价结果允许提前独立检查音频，而视频/SyncNet效果仍等待完整原始矩阵与共同支持审计。用户已授权Qwen ASR；本决策改变执行顺序，不改变开发/确认cohort、内容normalizer、主要评分或确认标准。

## Alternatives
等待完整单frontend合并ledger后调用ASR可以直接接原guard，但增加串行等待。把candidate-only行伪装成含baseline的合并ledger会失去真实生成来源，因此拒绝。用旧v3的零CER代替v4或共享profile会混淆不同PCM，因此拒绝。新候选彼此不复用hypothesis；自然identity结果仅在F32和direct PCM16都严格相同时复用。

## Consequences and evidence boundary
音频阶段明确audio-only，不写video/SyncNet PASS。每次API回执append、flush、fsync后再调用下一条，所有20条结束后才原子封存最终结果；中断保留partial与完整计划分母。API失败行保留，不自动重试或换hypothesis。实际视频ledger与staging PCM不一致时，该ASR回执不能作该生成条件的内容证据，须重新建立真实绑定。确认数据保持封存。API key只走进程环境及SSH channel stdin，不能进入文件、argv、请求JSON或笔记。

## Affected work
- `scripts/audio_enhancement_eval/stage_v4_dev_audio_only_attempt02.py`：生产转换与诊断stack的逐样本F32等价，以及RMS-stage/末级peak limiter分别审计。
- `scripts/audio_enhancement_eval/run_v4_audio_only_asr_guard_v2.py`：20个新调用、官方文本评分、自然hypothesis严格reuse和逐回执journal。
- `openspec/changes/develop-cross-tfg-audio-head-20261003/spec.md`：内容guard仍是独立端点；最终方法须结合完整跨模型生成效果与口部运动guard。

## Observations
- [decision] 新候选内容检查可在完整视频开发验证结束之前实际运行，最终需与生成conditioning PCM精确对接。
- [rationale] 同clip的确定波形不必等待三frontend与三seed的视频全部结束，提前内容检查缩短串行等待。
- [consequence] audio-only产物、实际API结果、生成ledger分别保存，并通过PCM和源码/权重封印关联；中断和失败不伪装为完整PASS。

## Relations
- affects [[开发跨 TFG 的自然音频增强头]]
- affects [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[006-portable-waveform-enhancement-head]]
- relates_to [[007-original-pcm-scoring-for-waveform-head]]