---
title: TalkLikeYou中文20来源九评分器实验2026-10-30
type: task
permalink: tts-exp/codex/tts-exp/talk-like-you-中文-20-来源九评分器实验-2026-10-30
status: done
current_step: 5
started: '2026-10-30'
assigned_to: codex
steps:
- 确认模型与服务器规范
- 冻结输入与单对验收
- 完整生成和九评分器
- 独立复核和跨模型比较
- 本机证据交付与正式报告
completed: '2026-10-31'
---

# TalkLikeYou中文20来源九评分器实验2026-10-30

## Observations
- [description] 用户确认测试192已部署的TalkLikeYou，沿用中文20来源九评分器协议；完整执行、复核和本机交付已完成。
- [status] done
- [assigned_to] codex
- [current_step] 5
- [conclusion] 主要同句12来源九项Holm9均未确认；全20仅VocaLiST K5正差过校正。主要12相对Echo的Sync-C效应差+0.735、Holm18 p=.01818，不支持跨模型效应笼统一致，也非感知同步结论。

## Steps
1. [x] 读取当次飞书1048手册；确认LiveTalk未找到，用户明确选择TalkLikeYou；旧InfiniteTalk/MultiTalk停止记录保留。
2. [x] 80 RMS−24dBFS输入/3.png、17权重与129源码freshSHA通过；首对N/T真实生成/PCM/帧数/运动/预览验收PASS。
3. [x] 80逻辑媒体/41唯一推理、九评分器全部完成。七项20来源完整，两个5秒分类器28媒体/6完整来源；616原生分数。六项时移控制通过，其余旧控制不足边界保留。
4. [x] 七类原始归约、80媒体PCM/SHA、25分层统计和50模型间效应对比独立复算PASS，最大统计差0。170科学文档在精度复核修复前后SHA不变。
5. [x] 260554096B证据包/2784内容文件全SHA读回、80标准播放视频导出、正式报告79、研究地图/结果表/索引、final_delivery_receipt均已完成。

## Context
正式报告basic-memory/docs/experiments/79-talklikeyou-chinese20-nine-scorers.md。最终验收入口results/talklikeyou/chinese20_rms24_20261030_attempt01/final_delivery_receipt.json；同目录report.md、summary.json、paired_scores.csv、per_video_scores.csv、cross_model_comparison.md、video_index.json、paired_differences.png及evidence/。标准播放媒体在results/talklikeyou/{natural_raw,tts_raw}/<source>/。证据包SHA87e25057f9443c3428f751bc777f0bf32215e1bd6821a0e4c7a885da7fbad58f，未声明网盘备份。

远端run /home/data2/wangjiajun/av-evaluation/jobs/talklikeyou_chinese20_rms24_20261030_attempt01；源码av-evaluation/experiments/talklikeyou_chinese20_20261030，本机scripts/experiments/同名目录，tests/test_talklikeyou_chinese20_20261030.py。官方d1088c49071148853c76d8a8f0c1fb97d895758b/权重652b744bbf86510c370ff6f302d046bdd63632e1；habit192、stage0+neutral、单步lip/pose、audio pose、seed42。原生57–193帧、少194–237ms画面尾部，完整PCM不动、不补帧/移音。私有目录、nohup、单GPU UUID GPU-3464c5ee-e2ed-81e4-e275-b612a929bfe1、CPU4、torch60%；生成累计冷进程/封装验证829.509秒，NVIDIA本人采样峰5856MiB。终局无本人GPU计算进程，不按历史pid或RUNNING字段重复launch。

Sync-C主要12自然4.937/TTS5.446、Δ+.509、CI[+.170,+.796]、Holm9 .15345；全20Δ+.495、15/20提高、Holm9 .37247。全20VocaLiST K5Δ+6.416个百分点、17/20提高、Holm9 .04635，仅补充分层。主要12相对Echo的Sync-C效应差+.735、CI[+.408,+1.110]、Holm18 .01818；对FlashHead未确认，不能等效。旧Wav2Lip电平/运行时不同，仅历史背景。

原audit.py对FP64/CPU-FP32 softmax要求1e−7而失败，原源码/log/stage/finish_failure.json保留；audit_precision_replay.py要求102组完整CPU FP32概率精确重放并独立FP64核对，PASS（最大1.598e−7）。没有新TFG/评分模型前向；170科学SHA不变。finish_repair02.py全部后处理完成，terminal_delivery_receipt.json确认41生成exit0、九阶段完成和GPU释放。档内旧RUNNING/失败为历史快照，最终以terminal/final receipt为准。服务器4测试PASS，本机3PASS/1因无Torch跳过；源码编译PASS。新增未提交，继承dirty未清理，无PR。

## Relations
- completed_by [[79-talklikeyou-chinese20-nine-scorers]]
- uses [[server192-nine-scorers]]
- compares_with [[76-flashhead-chinese20-nine-scorers]]
- compares_with [[77-echomimic-v3-chinese20-nine-scorers]]
