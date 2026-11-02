---
title: 018-native-repeat-basis-and-candidate-diagnostic
type: decision
permalink: tts-exp/docs/adr/018-native-repeat-basis-and-candidate-diagnostic
git_sha: c93e2a20655775230861f22f69abd34738b6d060
decided: '2026-10-08T10:16:06.6329654+00:00'
repository: dogLeft233/tts-exp
status: open
project: tts-exp
tags:
- audio-head
- determinism
- admission
---

# 018-native-repeat-basis-and-candidate-diagnostic

## 上下文

用户要求持续完成多TFG自然音频增强头。冻结controls360 plan09和candidate180 plan12都明确不声明generation_repeat_qc；Stage-A core06在这类子批次写空generation_repeat_qc.jsonl，并原样记录NOT_RUN_SUBRUN。要求该字段为PASS会使真实成功的冻结子批次无法准入，属于消费接口错误。当前尚无候选180终态，不能把计划当成实际重复结果。

同GPU严格重复资格已由独立实际运行建立：MuseTalk和Ditto使用strict09各两次PASS，Wav2Lip使用strict12两次PASS_EXACT_MATCH。binding09包含每模型两条实际repeat_rows、全输入/权重/源码/前端/政策/GPU来源及repeat_gate。controls18的validate_backend_determinism_qualifications逐文件全SHA并检查该组合，candidate13严格逐行attestation记录相同政策及原生入口。该证据证明后端政策下的重复资格；不把它扩大为对任意增强波形的实测重复保证。

## 决策

ASR内容检查准入使用真实candidate180和controls360的完整有序PASS分母、COMPLETE、子进程rc0、各自严格逐行attestation，以及binding09中三个后端的实际精确重复资格。两份子批次NOT_RUN_SUBRUN保持原样，并核对对应空文件SHA。不创建虚构repeat PASS或归一化终态schema。

在完整方法验收前，另做增强候选同种子重复诊断：预先选各模型冻结180计划的首个有序候选row，固定该row的样本、种子、来源、音频前端、几何和冻结参数，使用两个新输出目录重复真实生成与官方评分。直接核对实际增强generator F32 PCM、elementary video/pixels、帧时钟、query map及全距离矩阵。诊断行在效应540行之外，不能用于修改参数、替换样本或选择结果。内容API可在组合工程准入后执行，不必等待这个额外诊断；最终方法准入需诊断结果。

## 理由与替代方案

直接要求子批次repeat PASS与现有生产契约矛盾；改写manifest会破坏原始证据。重复完整540批次成本高且不是最小诊断。只沿用后端pilot无法直接测量增强音频条件下的输出重复；因此保留已核验政策资格，并补三模型明确预声明的候选重复实验。不修改冻结生产脚本、原始计划或效应端点。

## 后果与来源

私有ASR客户端需要读取真实schema并单独说明两类重复证据的作用。新重复诊断由GPT-6 Luna/max子代理实现，ROOT审阅后在真实候选终态运行；失败保留原分母和诊断证据，进行工程修复，不伪造成功。Confirm80/40人与LatentSync封存。

- scripts/experiments/cross_tfg_audio_head_20261003/run_stagea_dev_attempt06.py，raw291f8b5cd48d8ebc63aafd814dcd37450eecba949fab98283cf20a27c6335483，重复生成与manifest逻辑见1278–1346。
- scripts/experiments/cross_tfg_audio_head_20261003/run_v3_step120_candidate_samehost_attempt13.py，raw8c3b45f73d420b1dac6db0d5676eaf3343324cbb582d17cad7a3868d27e42708，strict_receipts及execution准入逻辑。
- scripts/experiments/cross_tfg_audio_head_20261003/run_v3_samehost_dev360_controls_attempt18.py，raw30381ee8deb6d0e7880379fc91a042e1e2bb3cca375dd05423319259fd9e8756，262–420的后端重复资格校验。
- 当前binding09 rawdd07fa3d2b3b5940df25fd3521313f7b2c3f1c6b99494b78a688e1ce39e527c5，controlsplan09 raw8a592cc54dc981bec5e2398108e6873190a54c7c1ed2bb7901cd31aad6cc86eb，candidateplan12 raw5a01e0ab1d6d4545ba4f6de332aced0ee1920f2001e51f5f552684d8b9926038。

## Observations

- [decision] 组合真实终态和三原生后端精确重复资格准入ASR；保留NOT_RUN_SUBRUN，额外候选重复诊断在最终方法验收前完成。
- [rationale] 后端政策资格与特定增强波形的实测重复是不同证据，消费接口不得制造生产端未产生的状态。
- [alternative] 不改写原manifest，不要求不存在的子批次repeat PASS，不重复完整540效应批次来替代三模型诊断。
- [consequence] 新诊断行独立于效应540分母，冻结方法、样本、种子、端点和Confirm封存不变；仍需实际执行后才能报告结果。

## Relations

- relates_to [[015-same-gpu-deterministic-controls]]
- affects [[开发跨 TFG 的自然音频增强头]]
- affects [[跨 TFG 自然音频增强头的训练与原音换轨验证]]