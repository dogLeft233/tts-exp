---
title: 031-selected-spsa-center-dev20-provenance
type: decision
permalink: tts-exp/docs/adr/031-selected-spsa-center-dev20-provenance
status: open
project: tts-exp
decided: '2026-10-10T01:52:16Z'
tags:
- audio-enhancement
- dev20
- provenance
---

# 031-selected-spsa-center-dev20-provenance

## 上下文

用户要求自主推进可用于多个TFG的自然音频增强头。ADR022/027已将Fit路线确定为真实三模型联合SPSA与四个完整中心选择。现有Dev20 attempt06仍是候选未绑定的SOURCE_AND_COHORT_VERIFIED_DRAFT，旧builder/analyzer要求DirectD0 training step120及checkpoint SHA，不能准确表示SPSA选中的k5/10/15/20中心。此时新的20步尚无科学score/tell，未打开新Dev candidate effects；决定不由Dev结果导出。

权威旧草案 .codex/scratch/dev20_future_cohort_protocol_attempt06/future_dev20_three_method_protocol_attempt06.json 为35244B/raw73ac37eee2c8c728a7123fa85834dc53104a4b809cbc017593bd86939816eb17/selfe513f565618fed124677cb75a97c2ff8b8d7a517521f60a96a0438f05c840650。它保留为历史证据，不原地改写或冒充已冻结候选。

## 决策

新增独立的selected_spsa_center Dev20协议schema、candidate plan builder和analyzer。候选只能来自冻结Fit-only final720 selector的ADMITTED_FIT_CANDIDATE_NOT_FINAL_DEV，真实point_kind=spsa_center/source_step=k5|k10|k15|k20，checkpoint_sha256=null；以实际method_id、config/artifact/binding/input-pins/canonical-theta SHA及selected portable artifact表示身份。source snapshot step与选中中心step独立记录。

保留原cohort_and_source完整对象及其20段/10说话人、每人2段；三TFG、种子20261003/04/05、identity/RMS/唯一选中中心三条件，共27jobs/540行。只将第三条件及其真实生成路线/provenance改为联合SPSA中心。原FULL/GUARD20/31lags/zero-anchor15、native ROI、原始PCM16评分/完整原自然PCM最终音轨、完整失败分母、speakerbootstrap与同时检验family、ADR021 numeric/content/visual gates及Cbest每模型>0/等模型均值≥+0.050均不变。Dev仍探索性，Confirm保持封闭。

真实中心选定后、任何新Dev candidate effects读取或生成前，create-only冻结：selector receipt、final-center index、选中terminal、center plan/full_center_spec的raw+payload-self pins；selected portable artifact、application receipt与每段output F32PCM/N/clock/RMS；Dev common plan/method config、原20段cohort、ROOT admission policy与implementation source pins，以及freeze commit/time。runtime/GPU/native资产必须按执行机器重新资格，不能借旧Dev360或旧AutoDL结果代替。选中artifact不是checkpoint替身；不得按旧step120分支伪造训练身份。

## 理由与替代方案

真实科学对象是Fit选择后的9参数增强函数，训练快照只是初始谱系。新增明确schema可复用冻结评价设计，同时阻止把不同训练对象写成同一个checkpoint。直接改旧草案或将center塞进DirectD0_step120会破坏可追溯身份；另选Dev cohort或改门槛会改变评价问题；继续只评旧Wav2Lip训练快照也无法落实三模型联合路线。这些方案均不采用。

## 后果与受影响工作

ASR子代理可立即实现additive schema/builder/analyzer和严格portable导出，但实际冻结/Dev dispatch必须等待真实admitted Fit中心与完整source/PCM绑定。不提前绑定虚构hash、不打开候选效果、不以Dev调参。所有旧草案、旧step120代码、失败尝试保留；同一科学评价门槛不因接口更新放宽。此typed decision记为open以遵守BM状态约定；用户已授权方法开发，实施无需新的人工许可。

## Observations

- [decision] 新增selected_spsa_center Dev20候选身份与独立执行/分析接口，checkpoint_sha256=null，真实参数与完整选择谱系late-bound。
- [rationale] 当前候选未设定且未打开新Dev效果，能够在保留科学设计的同时准确对接联合SPSA路线。
- [alternative] 不采用改写旧step120草案、伪造checkpoint、改Dev cohort或放宽评价门槛。
- [consequence] 必须在任何Dev candidate effects之前冻结真实中心/portable/application/原PCM时钟与ROOT policy source closure；Dev仍探索性，Confirm保持封闭。

## Relations

- refines [[021-future-dev20-natural-audio-admission-policy]]
- relates_to [[022-fit-only-cross-tfg-native-score-refinement]]
- relates_to [[027-durable-spsa-journal-and-final-center-selection]]
- affects [[开发跨 TFG 的自然音频增强头]]
- affects [[跨 TFG 自然音频增强头的训练与原音换轨验证]]