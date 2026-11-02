---
title: 云端Qwen Audio3.1Plus的20来源原生四指标分歧 2026-10-27
type: note
permalink: tts-exp/experiments/云端-qwen-audio3.1-plus-的-20-来源原生四指标分歧-2026-10-27
status: CONTROL_FAILED
scientific_status: NATIVE_SCORING_COMPLETE_PEAVS_SYNCREWARD_SYNC_CONTROL_FAILED
run_id: cloud_av_preference20_20261027_attempt05
training: false
tags:
- tts
- cloud-qwen
- syncnet
- cogenav
- peavs
- syncreward
---

# 云端Qwen Audio3.1Plus的20来源原生四指标分歧 2026-10-27

## 问题和设计
用户要求将此前CoGenAV/PEAVS/SyncReward同20来源的本地TTS换成云端qwen-audio-3.1-tts-plus，判断原生Sync-C优势是否获得独立指标支持。已完成20来源x3调用=60配对，provider仅21种唯一字节，全部重复保留；独立单位仍为20来源。原生为N驱动视频配N音频 versus cloudT驱动视频配同条cloudT，无换轨、DTW或时间编辑。固定RMS0.0376838172675746/16k monoPCM16；云端自带preprocess与volume_normalization关闭，下载后统一RMS。相同Wav2Lip GAN/图3/25fps/nosmooth/batch4；沿用各已部署官方评分入口。CoGenAV Eq1为本项目论文公式实现，bestlag为自定义±600ms派生诊断。

12个来源单句原参考>=5秒，严格沿用相同克隆参考，为预设主层；8个短目标拼接同说话人另一句作克隆参考，为扩展参考层，不能当严格参考匹配。三个层全部报告，不按分数选样。均值及分差来源等权；描述性95%来源bootstrap10000次seed20261026。

## 2026-10-27 推理审计补充

下方PEAVS均值为历史batch1，SyncReward为历史FP32，不能称作完全官方默认配置。后续审计发现PEAVS上游长度掩码导致批次依赖，默认64重算N2.252396/T1.844311/Δ-0.408085，TTS较高6/60、反向42/60不变；严格12来源Δ改为-0.474832，3/36和30/36不变。SyncReward在V100上可用Torch2.6执行BF16，完整40时移控制14下降/7持平/19上升；历史FP32为16下降，仍均失败。本次没有重算整批cloudT为BF16，历史原生胜率仅属于FP32版本。详见 [[PEAVS与SyncReward官方推理审计及批量掩码问题 2026-10-27]]；原历史数值和冻结收据保留。

## 全20来源原生结果
|指标|自然均分|云TTS均分|TTS-N|TTS更高|自然更高|与Sync-C反向|
|---|---:|---:|---:|---:|---:|---:|
|Sync-C|7.006|7.751|+0.745|48/60 80.0%|12/60 20.0%|—|
|CoGenAV Eq1|0.317945|0.317109|-0.000836|31/60 51.7%|29/60 48.3%|23/60 38.3%|
|CoGenAV bestlag 自定义|0.466360|0.476832|+0.010472|28/60 46.7%|32/60 53.3%|26/60 43.3%|
|SyncReward|1.494481|1.474959|-0.019522|19/60 31.7%|41/60 68.3%|35/60 58.3%|
|PEAVS|2.217287|1.813750|-0.403537|6/60 10.0%|54/60 90.0%|42/60 70.0%|

两种冲突方向按“Sync-C选TTS/其他选自然；Sync-C选自然/其他选TTS”：Eq1=20/3；bestlag=23/3；SyncReward=32/3；PEAVS=42/0。无分数平局。
全20分差95%描述区间：Sync-C[+0.378809,+1.095653]；Eq1[-0.016507,+0.014700]；bestlag[-0.011068,+0.031881]；SyncReward[-0.042717,+0.006511]；PEAVS[-0.642758,-0.177755]。

## 严格同参考12来源/36配对
|指标|自然均分|TTS均分|TTS-N|TTS更高|与Sync-C反向|
|---|---:|---:|---:|---:|---:|
|Sync-C|6.901|7.641|+0.740|33/36 91.7%|—|
|CoGenAV Eq1|0.320104|0.321792|+0.001689|19/36 52.8%|20/36 55.6%|
|CoGenAV bestlag 自定义|0.465037|0.474509|+0.009473|16/36 44.4%|23/36 63.9%|
|SyncReward|1.492633|1.461146|-0.031487|7/36 19.4%|32/36 88.9%|
|PEAVS|2.414602|1.888835|-0.525767|3/36 8.3%|30/36 83.3%|

严格层Δ95%：Sync-C[+0.399160,+1.068167]；Eq1[-0.015830,+0.019705]；bestlag[-0.010042,+0.031129]；SyncReward[-0.046649,-0.016069]；PEAVS[-0.866529,-0.185904]。扩展8来源详细表在报告及audit JSON中保留。

## 验证与边界
父侧独立读取60个cloud评分、60组CoGenAV嵌入及自然基线和PEAVS/SyncReward原batch，重建配对/原始均值/来源均值/两个方向/冲突，逐字段与summary一致。独立float64 Eq1 max误差6.484090842e-8，31lag曲线<1e-6；新CoGenAV真实波形时移120/120通过。合成60/60的provider/PCM/参考SHA、RMS、样本数独立审计通过；既有完整合成C归档462成员已读回核验。Sync-C官方float32重放先前20自然max差4.768e-6。

工程上四项60/60完成，科学上PEAVS/SyncReward仍保留CONTROL_FAILED：先前自然视频±400ms错位使评分下降仅26/40、16/40，低于预设32/40。不得将冲突解释为新模型更可信、SyncNet已证实存在纯声学偏好、TTS真实口型更差。SyncReward中心4.75秒、PEAVS全片、SyncNet/CoGenAV时间支持也不同。当前证据是不同评价器对同一原生比较显著分歧；CoGenAV Eq1平均接近0差。

CoGenAV首次因新run少链接20words_mean_face.npy在任何特征提取前失败，已链接原成功部署的相同SHA dbf68b2044171e1160716df7c53e8bbfaa0ee8c61fb41171d04cb6092bb81422 后完成；未重跑已完成三项。失败日志和修复收据保留。

## 产物与当前状态
Ubuntu根/home/wjj/tts-audio/tts-exp，run runs/cloud_av_preference20_20261027_attempt05。
权威报告 native_cloud_report_20261027.md；原始配对 paired_scores.json；统计 summary.json；独立核验 native_cloud_independent_audit_20261027.json；CoGenAV原始 scores在cogenav/scores.json。Windows . 同run有报告/统计/审计及cogenav_scores.json元数据副本，完整GPU媒体仍在Ubuntu，不能称已完整本地归档。合成C备份地址及SHA见既有checkpoint/合成报告，不冒称包含后生成视频。

最新用户明确说不用交叉音轨；已终止旧av_metric_preference_20261026_attempt02的交叉runner PID1638959，保留已完成数据，以user_cancel_cross_20261027.json及final收据记录。其原生分数已完整，不再恢复交叉评分。ComfyUI此前按用户授权关闭。当前GPU仅桌面进程313MiB；用户暂停的生产controller320414保持不变，不恢复。新cloud四指标run终态COMPLETE，不需再调用云TTS或启动GPU任务。

## Observations
- [result] Cloud原生Sync-C平均+0.745、80%配对TTS更高；PEAVS平均-0.403537、仅10%TTS更高，70%方向与Sync-C反向。
- [result] 严格同参考12来源并未消除分歧：Sync-C91.7%选TTS，SyncReward19.4%、PEAVS8.3%，反向率88.9%/83.3%。
- [decision] 因已知时移控制失败，不根据评价器投票认定感知同步真值。
- [decision] 用户取消交叉音轨，保留原生结果和已完成交叉数据，不继续交叉GPU评测。

## Relations
- relates_to [[PEAVS与SyncReward原生自然TTS评分分歧及同步控制失败 2026-10-27]]
- relates_to [[Codex checkpoint - 2026-10-27T03-14-44Z - Cloud Qwen20 corpus complete and AV scoring queued]]
