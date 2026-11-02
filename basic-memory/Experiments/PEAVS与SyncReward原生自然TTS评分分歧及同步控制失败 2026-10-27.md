---
title: PEAVS与SyncReward原生自然TTS评分分歧及同步控制失败 2026-10-27
type: note
permalink: tts-exp/experiments/peavs-与-sync-reward-原生自然-tts-评分分歧及同步控制失败-2026-10-27
status: CONTROL_FAILED
scientific_status: NATIVE_SCORING_COMPLETE_SYNC_CONTROL_FAILED_CROSS_PENDING
training: false
---

# PEAVS与SyncReward原生自然TTS评分分歧及同步控制失败 2026-10-27

## 问题与设计
用户要求复做CoGenAV的自然/TTS偏好比较，判断Sync-C偏好还是TTS口型优势。沿用20来源138可测原生视频、118自然–TTS配对，重复draw不增加独立来源。另472交叉换轨格和40个±400ms固定样本数零填充控制。原生视频stream copy，音轨源PCM无损ALAC并逐样本校验。SyncReward为官方Stage2/V100 FP32/中心4.75秒，PEAVS官方整段评分；与旧SyncNet时间支持不同，非同窗口复现。

## 完成阶段与实际数值
两模型各138原生格与40时移格完整完成。2026-10-27检查总进度SyncReward442/650、PEAVS178/650，交叉仍未完成；run活跃，等待ComfyUI占用约5.3GiB释放。不结束该外来进程。

118原生自然–TTS配对，以原始差值正负统计，无平局：SyncReward与Sync-C冲突58/118=49.2%，其中Sync选TTS/新模型选自然41，对向17；两者都TTS32、都自然28。PEAVS冲突66/118=55.9%，其中Sync选TTS/新模型选自然44，对向22；两者都TTS29、都自然23。20来源等权冲突率49.5%/55.7%。Sync-C给TTS更高73/118，新指标分别49/118、51/118。L06冲突27/58和29/58；L17冲突31/60和37/60。旧CoGenAV Eq1同支持冲突74/118。

来源等权原生TTS−N均值：SyncReward −0.020355，描述性95%CI[−0.043693,+0.002191]；PEAVS −0.170340，CI[−0.373764,+0.026337]。不作确认性优势声明。

## 控制与结论边界
时移后评分降低仅SyncReward16/40=40%、PEAVS26/40=65%，都未达到评分前analysis_protocol.json锁定的32/40=80%诊断门槛。状态CONTROL_FAILED，不能把新模型当感知真值，也不能从冲突认定Sync-C偏好或TTS口型差。可能涉及模型领域、时间敏感性、不同评价窗口；未识别原因。未经人工盲评。

## 工程复核与产物
权威run runs/av_metric_preference_20261026_attempt02/（Ubuntu /home/wjj/tts-audio/tts-exp，同项目Windows .）。native_conflict_audit_20261027.json独立从12个实际原生批次重算，核验每模型138唯一ID齐全，记录12批SHA。native_conflict_rows_20261027.json保留236行两模型逐配对差值，native_conflict_report_20261027.md报告。三文件已复制本地。该原生阶段审计独立于待完成总汇总，不声称新模型数值独立实现重放或全交叉完成。

2026-10-26资源门槛曾误将SIGSTOP暂停controller判活跃，已修复排除T/t/Z状态，四个直接测试通过；未恢复/中断用户暂停的旧生产。修复收据paused_controller_gate_repair_20261026.json。当前runner PID1638959。

## Observations
- [result] 原生配对的SyncReward/PEAVS与Sync-C方向冲突49.2%/55.9%，冲突多数为Sync-C偏TTS而新模型偏自然。
- [result] 两新模型均未通过本次±400ms同步敏感性控制，不能作为裁判认定SyncNet偏好。
- [next_step] 保留原生审计结果，待交叉评分资源释放继续，并先调查控制失败与窗口/领域适用性；不据此启动新训练。

## Relations
- relates_to [[peavs]]
- relates_to [[syncreward]]
- relates_to [[Codex checkpoint - 2026-10-26T08-29-54Z - PEAVS SyncReward preference evaluation queued]]
