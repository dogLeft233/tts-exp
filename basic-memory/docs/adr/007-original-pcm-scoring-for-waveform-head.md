---
title: 007-original-pcm-scoring-for-waveform-head
type: decision
permalink: tts-exp/docs/adr/007-original-pcm-scoring-for-waveform-head
scope: new cross-TFG waveform enhancement objective; existing frozen runs unchanged
repository: dogLeft233/tts-exp
project: tts-exp
status: open
decided: '2026-10-03T05:09:51.6181874+00:00'
---

# 007-original-pcm-scoring-for-waveform-head

## Decision

新跨TFG增强头开发/确认的主端点采用 `native_roi_original_pcm16`：真实成片保留原canonical F32 PCM；从原音直接派生PCM16并按官方selected native track时钟裁剪，与同一cropped video streamcopy组成新的评分输入，再fresh SyncNet。默认官方pipeline的 `native_official_mp3` 保留为单独基准诊断，固定commonROI再单独列出。该选择在任何fresh dev科学效应读取前作出；旧机制与QC产物均不改变。

## Context and rationale

用户要增强音驱动视频、成片换回自然音并提高与自然音的同步分数。QC04三次F32 mux均精确，但官方selectedcrop PCM16与直接原音slice不同。ROOT实际只读 `ffprobe` 查明，pinned `run_pipeline.py` 的第一步ffmpeg转AVI没有指定音频codec，实际intermediate `video.avi` audio codec=mp3/16kmono/duration5.868sec，video5.720sec；后续从该AVI提取PCM16再裁剪/合并。该实际lossy路径与直接原音派生不同，不能把它的PCM16宣称为原F32直接量化slice。

官方cropN91520/raw91321866746580de045feb8fdcc83f7993aeff9e2ed6282385a64a7aa6be657f，directsource同N rawd95aa787d7757b01e2bebc26889a1c8f517429671cc4fc1db645720e0f04a1ba。paired identity/repeat官方crop音频相同，原工程Cbest结果仍是明确的native_official_mp3估计；此处不把所有样本差异归因到某个单独async/延迟操作。主评分从原音明确一次PCM16派生，符合增强头的目标与跨模型统一音频条件。

## Alternatives

继续以默认官方MP3路径为主可直接对齐老benchmark，但会让所称原音评分包含未声明的有损转换。修改旧SyncNet源码/覆盖旧产物会破坏历史可复现性，不采用。只用fixed commonROI可减少检测波动，但改变native跟踪估计，保留为独立端点而非混用。仅核验最终F32 mux仍不足以验证评分实际使用的音频。

## Consequences and affected work

生产者记录selected track/index/crop视频stream、source-frame interval、directPCM16 full/slice/file hash、freshscore输入与raw31lag矩阵；消费者独立从原F32派生并核对primary PCM16，不再误要求官方MP3 path等于direct slice。成片F32精确检查与评分PCM16量化检查分别成立。训练engineering smoke仍可使用已绑定MP3 PCM16 context，仅工程梯度证据；科学fit采用direct原音MFCC并标注训练proxy。开发/确认计划及统计族在对应效应读取前冻结。

影响 `openspec/changes/develop-cross-tfg-audio-head-20261003/spec.md`、`scripts/audio_enhancement_eval/`、新实验producer与fit context。原SyncNet/Wav2Lip源码、QC04和报告25不改。

## References

- `.codex/scratch/cross_tfg_execution_attempt02/root_probe_intermediate_audio_codec_attempt01.json.response.json`：实际ok/code0 ffprobe MP3 codec证据。
- `.codex/scratch/cross_tfg_execution_attempt01/full_source_contract_actual.json`：pinned run_pipeline完整source。
- `.codex/scratch/cross_tfg_execution_attempt01/26_read_compact_noise_track_model_status.json.response.json`：实际track/source/PCM16差异。
- [Pinned SyncNet source](https://github.com/joonson/syncnet_python)

## Observations

- [decision] 新增强头主评分明确directoriginalPCM16/nativeROI，默认官方MP3 benchmark与commonROI分别报告；旧冻结run保持。
- [rationale] 实际官方intermediate AVI选择MP3，使评分PCM16不同于原F32直接量化；用户目标要求固定自然音条件。
- [consequence] 独立消费者验证评分真实audio lineage，fit自然音目标与主评价保持相同direct派生政策。

## Relations

- affects [[开发跨 TFG 的自然音频增强头]]
- affects [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- refines [[006-portable-waveform-enhancement-head]]
- relates_to [[005-separate-score-motion-and-native-estimands]]