---
title: 核查 SyncTalk FP16 适配与官方环境
type: note
permalink: tts-exp/tasks/核查-sync-talk-fp16-适配与官方环境
status: in-progress
priority: high
current_step: 4
started: '2026-10-18'
tags:
- synctalk
- fp16
- audit
---

# 核查 SyncTalk FP16 适配与官方环境

用户明确授权：“那先核查我们加的适配代码，并在官方原始环境下做对照，查清失败原因。先不要改成 FP32，保留已有失败记录。”此指令取代父任务等待一般诊断范围答复；正式24模型/27格仍未完成。

## Requirement
在不改原FP16训练政策、不覆盖原失败的前提下，审计新增数据/上下文/RNG/epoch恢复/记账/验证适配，以固定官方commit 43079fb3f349675f6cb33031bb07262f7566cdd5和原版环境做同输入同state对照，区分触发点、数值机制与尚未证实的原始根因。

## Scenario
Brown/seed42/coarse ep5/+6原job在非有限loss断言退出，4636真实prefix后失败行未写，丢失的失败时完整state不可复原。原failure.tar.gz SHA55855cf8b3f28e088df0d8a056336a34ce6bb4acfaef4420feb9de97eab21e72及20成员/原路径保留。任何ep4重放都是新的独立诊断，不合并到旧训练，不宣称无损恢复。

## Requirements
### Requirement: 保持FP16和原失败完成适配及官方环境对照
#### Scenario: 核查已有失败
- **WHEN** 原Brown/+6/ep5非有限loss停止且失败state不完整。
- **THEN** SHALL核验新增适配、在隔离官方依赖中执行独立对照，保留全部失败与实际预算，分别记录直接机制证据和无法精确追溯的边界。

## Current evidence

最新用户优先要求先试其他三人，见 [[试跑 Patrick Chris Cathy 的 FP16 粗训练]] / [[045-synctalk-other-identities-fp16-diagnostic-training]]；这是独立有界诊断训练授权，不再由下文旧完整24模型稳定性门阻止此试跑。Brown上游原因仍待查。
用户进一步明确授权保持FP16、另立更新前保护版本、继续上游诊断，验证后才重训。独立修复及本轮诊断完成并封存，见 [[SyncTalk FP16 更新前保护资格与密度溢出因果核查]] / [[044-synctalk-fp16-preupdate-guard-revision]]。双环境各10/10保护测试通过，真实state拒绝0更新且SHA不变，原controller trace/记账fixture精确；同一forward图密度logit梯度干预使非有限参数28→0。首次密度点eye/sigma权重干预定位局部放大路径，仍待解释完整增长轨迹。原失败archive SHA保持。新32/32模型零更新案例、24/32微fixture更新；不增加正式训练更新。正式训练未重启。

## Observations
- [status] in-progress
- [result] 保护修订已资格，密度指数与梯度传播有双环境同图因果证据；不等于训练稳定。
- [decision] 保持FP16与原失败，后续只在独立版本中处理；用户已授权范围，不再重复等待原问题回答。
- [blocker] 未完全隔离eye_att/sigma训练轨迹增长来源，且没有统一稳定性合格证据，因此不启动24模型重训。
- [next_step] 只读追查原眼部输入适配及已有state权重演化，冻结必要的下一组因果对照；不把32案例完成写成全目标完成。

## Relations
- depends_on [[准备并核验四个新身份的 SyncTalk 完整训练数据]]
- implements [[044-synctalk-fp16-preupdate-guard-revision]]
- relates_to [[SyncTalk FP16 更新前保护资格与密度溢出因果核查]]
- relates_to [[研究交接总览]]