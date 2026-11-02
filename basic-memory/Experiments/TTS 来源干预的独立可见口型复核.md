---
title: TTS 来源干预的独立可见口型复核
type: experiment
permalink: tts-exp/experiments/tts-来源干预的独立可见口型复核
status: concluded
started: 2026-09-30
concluded: 2026-10-01
repository: dogLeft233/tts-exp
---

# TTS 来源干预的独立可见口型复核

## 问题与冻结定义

用CPU MediaPipe形状/运动端点，检查新男性80 N clips的Wav2Lip来源方向八臂视频是否增强phone-linked口型。此实验测统一d的latent干预，不是原生TTS视频对自然视频。原N MFA phone/timestamps与全部八轨迹共同有效帧冻结支持；双唇类b/m/p、open-vowel类a/æ/ɐ/ɑ/ɒ，冻结latency=-0.085s、计数与clip→双clip→speaker等权。20,000 PCG64/seed20260930、99%bootstrap。v2新adapter剥五个tone bars后调用旧normalizer，旧规则剥ː/ʰ/ʲ/ʷ/̪；最终视觉宽类映射不保留所有非tone标记。物理phone身份仍原NFC/strip exact，保留IPA差异。

## 支持集与结果

640/640视频hash和轨迹完成；80clip×8轨迹共有8707帧且100%共同valid。原MFA资格79/80，单条S0064含spn，不用physical-only新overlay补原visual primary。按原两phone类与全八臂规则，52clips有endpoint，只有16完整两clip speakers/32clips；24speakers不完整。因此全80/40 primary=INCOMPLETE_PAIRED_SPEAKER_SUPPORT，不能靠轨迹完整代替phone类别支持。

可测16paired-speaker secondary，orth+−baseline frozen-latency separation=-0.0001168，99%[-0.003126,+0.003903]；zero-latency=+0.0002753[-0.002343,+0.002900]；AUC=-0.00104[-0.06632,+0.05972]。统一方向干预在这三个限定端点没有明确正向响应；不是全人群严格等效或人类视觉同步无变化的结论。

orth+−mean3random separation=-0.0000396[-0.001340,+0.001109]、zero-latency=+0.0007150[-0.001588,+0.002932]、AUC=+0.01267[-0.02407,+0.05480]均保留；但旧random实际F32剂量51帧/37clips略超公差，这些是out-of-spec sensitivity，不能作严格equal-dose specificity确认。新random aux独立执行，不修改旧视频。

全80描述性baseline/orth+平均aperture=.055058/.054748、绝对velocity=.332414/.326952 per second，width=.705165/.705050，yaw=.002295/.002281，pitch=.423188/.423189。总运动量、phone关联与事件时序分开解释。CAL event-rematching±5控制失败，event_metric_calibrated=false；固定索引timestamp工程变换不替代事件匹配控制，不写事件同步确认。

## 可复现来源

本地目录 runs/recovered_20260930/independent_validation/eval_visual_20261001/。
- eval_visual_tone_analysis_20261001_v2.json SHA7b6f0e7b7f32083605630e92a4141518eaacbe23057f0d8454f86505b72c7d59。
- 80 feature manifest SHA8a3815835489241bb1231fe89b8c2adc7a052a0ca28cc752be9b57053c8df3e6。
- 预效果phone availability/support v2均另封存，依据原phone labels和全轨迹valid mask，不依据aperture或评分选择。
- adapter执行SHA b725b5feb5c3f30bea754f73bf897d606e04f59940fe8f4fdb13f269306fce98；旧快照与新规则同时保存。
- Py3.13.5/MediaPipe1.0.1/np2.5.3/OpenCV5.0.0.93，CPU XNNPACK；不读SyncNet构造视觉指标。

Root在完整资格检查后读取本结果，与sham及主08结果共同指导新物理/动态假说；不声称此global-axis视频代表全部TTS现象。

## Observations

- [status] concluded
- [result] 全640轨迹有效，但phone endpoint只16完整paired speakers，full80/40视觉primary支持不合格。
- [result] 合格16subset中来源方向phone-separation与AUC的99%区间跨零；总aperture/velocity变化与时序精确度不可混称。
- [decision] 原random视觉估计标out-of-spec sensitivity；物理MFA overlay不改原视觉资格。
- [limitation] CAL event控制失败、无感知盲评；不能把结果改写为人类同步阴性或原生TTS对照阴性。

## Relations

- tests [[TTS 来源方向与声学机制确认]]
- advances [[解释 TTS 的 TFG 与 SyncNet 优势]]