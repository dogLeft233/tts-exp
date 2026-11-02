---
title: 045-synctalk-other-identities-fp16-diagnostic-training
type: decision
permalink: tts-exp/docs/adr/045-synctalk-other-identities-fp16-diagnostic-training
status: accepted
project: tts-exp
decided: '2026-10-18T02:55:04.970542+00:00'
---


# 045-synctalk-other-identities-fp16-diagnostic-training

## Decision
用户明确要求“先把Patrick、Chris、Cathy的训练跑了试试”。接受先行三身份独立FP16诊断训练；此授权解除这三人的试跑等待，不声称Brown上游已修好，不启动或覆盖原24模型实验版本。当前操作化范围固定每人seed42、0/-6/+6dB、原粗10epochs，每档60000attempts，上限540000尝试；fine和seed2026不在本轮。完成后按实际证据决定后续，不以初期无溢出宣称完整稳定性。

## Context and rationale
持续溢出仅在Brown被观察和双环境复现，其他三人只完成准入；没有证据可称1/4失败。用户优先希望知道其他身份的实际训练表现。既有FP16更新前保护已资格，本轮将其接入独立epoch入口，附加first AMP skip与terminal完整state/gradient保存，正常更新与原trace精确验证通过。保持原65536rays/maxsteps16、loss/AdamW/EMA/defaultGradScaler、共同完整随机初始化；不以Brown或May state warmstart。

## Alternatives and consequences
继续只诊断Brown会延后身份范围证据，按用户最新要求先试其他三人。直接重启原supervisor会混合原失败与新保护，拒绝。FP32、clip、sigma/scaler调整仍未实施。各identity串行，首terminal失败保留job/epoch/状态并终止该identity剩余条件，随后下一人；这意味着尚未运行的条件必须记为未测。失联job须人工核对不重放。正常phase结束仍按原实际成功更新配对门记CONTROL_FAILED，不补步。

空间只处理已经完整F归档逐成员复验的本任务监督缓存，独占Linux lease后退休非控制GT/torso文件；保留72control及原素材/模型/失败，未来加载显式full恢复。本轮Brown已退休16428文件逻辑1820139867B，F archive SHA2a6f3bcfdcd9c6c0def6d077bfcd83b0e6314f757a73604c211f89870a223aef；这不是旧成功identity handoff，不撤销原失败。中间完整checkpoint只在同臂新完整state独立F验证且已消费后轮转，coarse final与异常state保留。

## Observations
- [decision] 先行三身份seed42三档完整粗阶段诊断试跑，独立版本fp16_other_identities_trial_v1，原科学实验未完成。
- [verification] 入口保护/capture资格4个微型optimizer更新，正常trace/state与原版精确；另有3项测试验证terminal后继续下一人、lost job禁止重放和Windows主机生成POSIX远端checkpoint路径。
- [constraint] 原失败不改pass，未运行身份或增益不称稳定，无自动数值政策调整。

## Relations
- affects [[试跑 Patrick Chris Cathy 的 FP16 粗训练]]
- affects [[核查 SyncTalk FP16 适配与官方环境]]
- relates_to [[044-synctalk-fp16-preupdate-guard-revision]]
- relates_to [[SyncTalk 四新身份官方完整训练电平实验 Spec]]