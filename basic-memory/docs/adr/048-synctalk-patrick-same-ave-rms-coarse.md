---
title: 048-synctalk-patrick-same-ave-rms-coarse
type: decision
permalink: tts-exp/docs/adr/048-synctalk-patrick-same-ave-rms-coarse
status: accepted
decided: '2026-10-19T02:03:58.897454+00:00'
project: tts-exp
---

# 048-synctalk-patrick-same-ave-rms-coarse

## Decision
按用户2026-10-19明确选择，在原同一AVE编码器前加入逐完整录音RMS归一化，独立运行Patrick seed42的−6/0/+6dB coarse10并完成新27格。此新授权不恢复原Patrick fine14或Chris/Cathy。

## Context and rationale
旧AVE保留增益响应，用户要求以加入音量归一化的编码器重跑同样三档。保留AVE架构/权重和共同完整随机初始化可避免同时更换编码器家族；gain在归一化前施加，直接观察该前处理是否消除电平差及剩余数值/训练/评分响应。用户从“同AVE加RMS / 同AVE加LUFS / 改HuBERT”中明确选RMS。

固定RMS0.1（−20dBFS），float64统计/标量缩放后float32进入原mel；无去均值、epsilon、clip或limiter，不改原时钟，静音/非有限拒绝。同一算法用于train/val/test完整录音，目标不由test分数选择。每档真实独立AVE重复前向及训练，即使特征相同也不复用0档模型。FP16/defaultGradScaler/guard、原loss/AdamW/EMA、65536rays/maxsteps16、6000attempts×10/arm保持；共同原seed42全随机state，不使用已训练coarse/fine/May/Brown权重。

## Alternatives and consequences
更换HuBERT会同时改变表示维度/架构/权重和增益入口；LUFS采用不同统计/门控定义，因此本轮均不采用。固定RMS会消除纯标量gain，但不抹去段内能量变化；不将编码不变当人类同步/感知结论。评分端保持原PCM单位的独立gain轴，因此不能把driver归一化与scorer归一化混为一谈。

旧CONTROL_FAILED/Brown失败和旧fine14暂停state全部保留。新phase是否实际更新相等按真实trace验收，无补步/precisionfallback/scalertune；首terminal/lost封存后停止全部新job，不自动重放。原24full目标未完成。

## Verified implementation
OUT=F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017/trials/patrick_ave_rms_coarse_v1；contractSHA5cc4d22b57a32e5b6c3672d1e782dd40914b33850da5ae9b1e819b34ebb17859，14冻结源。全部41447原输入pins恢复鲜验，stage1全部41251与derived16562成员/资格archive原字节复核，8250masks原公式+原SHA恢复。实际main/test六条件各两次完整AVE，三gain归一化样本和特征均逐位相同；独立CPU公式误差0。

真实三gain各2次FP16尝试，6实际更新/0skip、full model/optimizer/scheduler/scaler/EMA/counters roundtrip exact和实际采样相同；独立鲜核13archive成员+3standalone cp和rawresult/trace通过。入口资格不等于整阶段稳定性。首次入口job12546仅import缺geometry，构造trainer前0attempt/0optimizer；原job及bootstrap_failure保留。deployment_repair02补齐同原依赖后用新attempt02 job13408，未改冻结14源或重放optimizer。

controller217680当前真实0gain coarse1/job15249运行；完整live前缀1130尝试/1130成功/0skip/loss有限，非整轮完成。后处理host190252/219064等待真实coarse10和new27；0评分GPU/视频/分数，不能称候选完成真实渲染资格。

## Observations
- [decision] 同AVE逐录音RMS0.1、gain先施加；独立新训练和27格，保留旧fine14/Chris/Cathy暂停。
- [result] 新前端实际六条件重复与独立公式通过，真实六FP16入口/roundtrip及独立receipt QA通过，formal已启动。
- [constraint] 无补步/数值政策修改，原控制失败保持；不推单身份总体因果或感知改善。

## Relations
- affects [[运行Patrick的RMS归一化AVE三档粗训练并评测]]
- implements [[Patrick同AVE逐录音RMS归一化三档粗训练与27格评测]]
- relates_to [[047-synctalk-patrick-fp16-fine-and-safe-pause]]
