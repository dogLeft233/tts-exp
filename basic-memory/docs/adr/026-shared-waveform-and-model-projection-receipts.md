---
title: 026-shared-waveform-and-model-projection-receipts
type: decision
permalink: tts-exp/docs/adr/026-shared-waveform-and-model-projection-receipts
status: open
project: tts-exp
decided: '2026-10-09T11:49:05Z'
---

# 026-shared-waveform-and-model-projection-receipts

## Decision

同一音频和同一优化点在固定共享CPU环境只计算一次增强WAV，三种TFG使用同一物理文件。新增公共应用回执及三个显式模型投影回执，分别记录实际变换和每个模型的使用身份；保留helper05实际产生的原始应用回执及所有历史评分。

## Context

ADR025已冻结固定remote Python3.10.20/Torch2.5.1cu121/NumPy1.26.4 CPU1/1生成同WAV；两次seed120实际输出逐bit相同并经ROOT独立wholePCM/时钟/θ/receipt复核。新Muse07/Ditto08已通过真实composer24提案/120中心计划检查。

连接实际共同WAV时发现helper05.apply_snapshot_theta的回执绑定parent_row_id和model_id，而且只写新WAV；validate_application_lineage要求对应的精确模型父行。现有回执不能直接代表另两个模型使用同一文件。逐模型重复调用可产生逐bit一致波形，但尚未实现ADR025要求的一次计算、共同物理文件。这是工程接口缺口，并不改变已完成评分的科学解释。

## Rationale

公共回执保存变换的真实来源、优化点及波形身份；模型投影回执保存TFG身份和评分行身份。两层显式绑定使三个前端使用同一数值输入，同时维持各自精确parent/point/model来源。变换仍使用已核验的head/helper05，不引入新谱形算法或改变θ、RMS、3dB、原音时钟约束。

计算前校验三个模型的实际父行和derivedpoint上下文，确认相同优化method/θ/snapshot、自然PCM/N/clock、scientific input pins及共享运行环境。仅执行一次实际变换，原始helper05回执保持真实来源。新的公共回执用自己的schema/raw/self hash绑定原回执和实际WAV；每个模型投影回执用自己的schema/self hash绑定公共回执raw/self pin、该模型精确family identity行和derivedpoint行。新validator再次核对原音、WAV完整字节/PCM、θ、运行环境、RMS/谱mask/时钟与模型身份，随后才调用模型生成hook。

## Alternatives

- 每模型单独计算同一变换：适用于现有CPU工程诊断；重复计算且三文件，未作为新的正式共同波形执行方案。
- 将helper05回执改写model_id/parent_row_id并重新封hash：丢失真实应用来源，不能作为三个模型的真实helper05执行证明。
- 删除模型父行身份或放宽旧lineage验证：使模型使用及评分行身份脱离冻结计划，无法保证正确接线。

## Consequences

新增helper06及fresh Muse08/Ditto09接口，旧helper05和旧adapter、回执、输出保持可追溯。ROOT复核窄diff、实际公共WAV以及三个模型的pre-render验证后，才将新契约接入正式优化。实现由Signal负责公共helper/Muse，Dev负责Ditto/common dispatcher/独立推理接口；ROOT负责协议、BM及独立核。

三模型Fit六方法720评分仍在自动队列执行；完整720前不选择初值、不生成实际SPSA提案、不进行更新。独立Dev/Confirm继续封闭。本决策当前实现中，未声明新的三模型收益或已完成共享物理执行。

## Affected Work

`scripts/audio_enhancement/fit40_snapshot_spsa_application_attempt05.py`为已验证来源；新增helper及Muse/Ditto adapters、联合优化dispatcher使用上述两层契约。真实remote family03在 `/dev/shm/cross_tfg_audio_head_20261009_v100_native_deployment_attempt01/fit40_spsa_remote_family_composition_attempt03/`，family raw9f7914033f9e7e89a60ae2618f7d429860d11de902a8400855fe57e37eb216a2。ROOT实际remote family/source/import QC在 `.codex/scratch/root_actual_remote_family03_code_readback_attempt01/receipt.json`。这些远端SHM和本地scratch均为machine-local/fragile，需按Task指针恢复。

## Observations

- [decision] 共同WAV实际变换来源与各模型使用身份通过公共应用回执和显式模型投影回执分别绑定。
- [rationale] 现有helper05回执严格绑定单个模型父行，不能改写后冒称另两个模型的实际应用。
- [alternative] 逐模型重复生成只用于明确标记的CPU工程诊断，不作为新正式共同物理WAV执行。
- [consequence] 新helper/adapter契约需实际公共WAV及三模型pre-render验证，旧应用回执和分数保持真实来源。
- [status] open

## Relations

- relates_to [[025-shared-cpu-waveform-for-cross-tfg-proposals]]
- affects [[开发跨 TFG 的自然音频增强头]]
- affects [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[022-fit-only-cross-tfg-native-score-refinement]]