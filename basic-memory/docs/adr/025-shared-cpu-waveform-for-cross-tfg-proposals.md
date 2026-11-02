---
title: 025-shared-cpu-waveform-for-cross-tfg-proposals
type: decision
permalink: tts-exp/docs/adr/025-shared-cpu-waveform-for-cross-tfg-proposals
status: open
project: tts-exp
decided: '2026-10-09T10:09:08Z'
tags:
- audio-enhancement
- cross-tfg
---

# 025-shared-cpu-waveform-for-cross-tfg-proposals

## Decision

ADR022后续SPSA候选由一个固定CPU运行环境应用同一九参数v3头，每个sample/point只生成一次完整float32 WAV，三个TFG都消费这一个实际PCM。生产DSP采用执行机的共享Python3.10/Torch2.5.1+cu121环境、明确CPU device、CUDA_VISIBLE_DEVICES为空、Torch intra/inter线程各1；环境和实现完整身份写入application receipt。后续实际CPU资格和源闭合读回完成后才执行候选，不能把本地marker测试代替远端资格。

已冻结四legacy快照160个GPU生成WAV和720初筛计划保持原字节，不重生成或以CPU产物覆盖。初筛缓存仍表示其原实际application/runtime；新候选表示同一数学头及θ在新明确DSP runtime中的实际波形，两者记录不同。正式选择门限、端点/分母、完整原自然音评分与时钟、40Fit/20人和sealed新Dev/Confirm界限沿用ADR022/023。

## Context and rationale

原160次application实际receipt记录marker Python3.11.15/Torch2.7.1+cu118/NumPy2.4.4、RTX4060 cuda:0、intra16/inter32。ROOT本轮直接读回该runtime；exacthead32233B/SHA77acdff03cca22261852f48364cbc75263db58a14cb565c08aa0d268afca6e87、constructor eaf51303012125c2de40dc102fc04a983c130b6d2aac31cd96b6e5648c3e27b8、n_fft512/hop128/global3dB/targetRMS0.0376838172675746均已固定。

Signal实际本地CPU1thread seed120重放保留N62639，报告与原GPU波形maxabs PCM差1.1920928955078125e−7；它没有运行完整SPSA application API，不能被称作API通过或远端DSP资格。ROOT全文审查共同helper04发现undefined source_head_pin、plain-int flux诊断误调detach、derived单method计划被误要求包含legacyseed等工程缺陷；freshhelper05正在修复和整条实际CPU测试。保留这些失败，不能将它们写成方法科学阴性。

同一candidate的声学输入必须跨三模型相同。每个模型自行生成增强WAV会引入不同Torch/NumPy版本、设备和线程的数值差异，增加本来只应比较nativefrontend和视频生成差异的变项。固定CPU生成、全文件和PCM hash验证后共享波形，可以明确定位输入与后端差异；CPU应用也能在V100运行视频时并行，无需抢占GPU。

## Alternatives and consequences

继续本地GPU为每个候选生成可接近原cache runtime，但需要跨机搬运并绑定实际路径，且该机器并非三个模型的执行机。每个TFG单独执行DSP会使同θ产生不同输入。为了逐bit重复而覆盖原cache会破坏初筛的冻结输入身份。选择单个远端CPU DSP，使重复资格、生成WAV和三个消费器的byte/PCM证明有一个明确来源。

新执行前，在SHM create-only补齐exactpinned head/extractor及必要source closure，旧W2L项目计划和既有manifest不覆盖；实际逐文件读回并运行同一seed/sourceθ两次CPU应用，核验N/clock/RMS/global3dB、receipt raw/self、全WAV/PCM相同。完整apply→receipt→共同lineage验证和错误θ/config/receipt/PCM/N拒绝测试须通过；框架测试和原seed直接transform重放是不同证据。

公共helper保管snapshot/θ/point/PCM证据，modelhook保管nativefrontend/runtime/capacity/decodedframes/scorer。Code/snapshot closure与stage音频root各自固定，实际source自然路径必须等stage projectroot加sample-specific sciencepin，不凭字节相同开放任意文件。模型descriptor.generator和protocol.generator分别匹配各自冻结源对象，不能强行等值。新候选应用receipt记录真实DSP环境并由外层执行闭合验证。

远端生产DSP实际重复资格尚待执行，尚无新科学初始化、SPSA模型调用或收益。本记录补充执行运行环境，不替换ADR022的联合优化设计、ADR021未来Dev准入或已保存分数。

## Evidence and affected work

- `.codex/scratch/fit40_snapshot_audio_application_attempt04/application_output/application_receipt.json`，raw74369d24f09a38e47f830d724bb2bb896f33d47791a7d435bbf4a0d482c60743；实际GPU应用环境与160cache的权威出处。
- `.codex/scratch/fit40_snapshot_seed_runtime_replay_attempt01.py`，source-bound seed120 CPU/GPU诊断重放工具；生产application不同于这个直接transform工具。
- `.codex/scratch/root_actual_packet04_composition_replay_attempt04/actual_readback_qc.json`，ROOT实际17sourcepins及720metadata逐byte复现，rawcc934871ef8c6352e7cca9d30daf115479640bccce32fde7019593c2822e0373。
- `scripts/audio_enhancement/fit40_snapshot_spsa_application_attempt05.py`及后续实际闭合版本、三个SPSA modelhooks、共享窄dispatcher受本决策约束；05尚未封存，不预造SHA。
- [Repository](https://github.com/dogLeft233/tts-exp)，branch codex/cross-tfg-audio-head-20261003，local/unpushed HEAD5688784dd5c6c1272074f0c4b6c5ff79f117c907。

## Observations

- [decision] 后续同sample/point只在固定远端共享CPU1thread环境生成一次candidate完整F32 WAV，三模型共用同一actualPCM；原GPUcache身份保持。
- [rationale] 去除model各自DSP运行时差异，明确输入与nativefrontend差异边界，并允许CPU应用与GPU视频串行主线并行。
- [consequence] 生产运行前新增exactsourceclosure读回和真实CPU重复应用资格，应用receipt保存实际环境；本地单seedtransform数值重放不代替完整API或远端资格。

## Relations

- relates_to [[022-fit-only-cross-tfg-native-score-refinement]]
- relates_to [[023-decoded-video-window-support-for-native-syncnet]]
- affects [[开发跨 TFG 的自然音频增强头]]
- affects [[跨 TFG 自然音频增强头的训练与原音换轨验证]]