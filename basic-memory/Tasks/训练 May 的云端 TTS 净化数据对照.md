---
title: 训练 May 的云端 TTS 净化数据对照
type: task
permalink: tts-exp/tasks/训练-may-的云端-tts-净化数据对照
status: done
current_step: 3
description: Train one May cloud TTS purified no-normalization arm with baseline50
  60000 successful FP32 updates
steps:
- AVE and real loader qualification
- 60000 training updates with F full checkpoints
- Independent completion audit and baseline comparison
phase: CONCLUDED
evaluation_outputs: F:/Datasets/tts-exp/may_tts_purified_eval_20261020_v1
report: docs/experiments/64-may-purified-tts-training.md
completed: October 20, 2026
---




# 训练 May 的云端 TTS 净化数据对照

## 当前交付

评测已完成并独立验收，科学状态CONCLUDED。一个TTS/NAS训练0dB模型，原AVE/同initial/60000 FP32/0skip，固定最终EMA60；原自然test三档驱动、原PCM统一评分，与报告50自然训练0dB模型匹配比较。-6dB：新C5.296、自然baseline 8.669、Δ-3.373 / +0dB：新C5.703、自然baseline 8.909、Δ-3.206 / +6dB：新C5.911、自然baseline 9.036、Δ-3.125。权威F:/Datasets/tts-exp/may_tts_purified_eval_20261020_v1；新三档3300 test帧、1650 val帧、12矩阵91140距离、96个最终val浮点帧/曲线、原PCM/全像素/GT ROI和250帧重复通过。旧baseline9帧float/RGB完全复现，四原音频SyncNet特征重算逐位相同，F完整成员SHA及第二次独立矩阵重建通过，真实图/代表帧检查完成。未声明云备份，所有新媒体/训练断点保留，无训练或评分重放；原source WAV路径准备失败在remote部署/GPU前修正，保留。新数据3849监督行与自然4150行不同，NAS同SyncNet数据QC不等效果，单May/seed42不能推广总体/感知结论，旧Patrick等继续暂停。正式报告[[64-may-purified-tts-training]]。

## 当前训练验收

训练阶段已完成并独立验收，phase=TRAINING_COMPLETE_PENDING_EVAL。2026-10-20T07:13:04Z（北京时间15:13）job rc0自然退出，60000attempts/60000success/0skip，耗时约47.3分钟，无本任务活动训练进程。F根may_tts_purified_training_20261020_v1的completion_review_v1/review.json为独立验收入口：128成员fresh SHA、60000完整trace有限loss/逐行source图像与audio绑定/冻结order/LR精确、60000 ray hash记录通过；全部30个fullcp CPU真实load，model/optimizer/EMA有限，scheduler/metadata/RNG/counters齐全，最终EMA60。最终step060000.pt SHA32e4be04483a290da463a20d98a1c28c42c104959ab0e2aeebcb0f1004ce3ba2。93条验证摘要/2976帧检查有限；三自然val gain后段MSE变化−0.877%/−1.193%/−0.986%，均通过预定±2%平台筛查，此筛查不等统计收敛或收益。正式冻结源、initial、cache/WAV/JSON fresh remote SHA均精确。只读备份host已结束，全部新checkpoint保持F，无cloud备份声明。

新训练仍为报告50原AVE、同initial/seed42、FP32/4096rays/maxsteps16和原优化合同，无输入归一化；新训练数据为云端TTS/NAS净化3849行，自然val/test保持。0新视频/0新SyncNet，不能把净化数据C最低6.121或训练loss当作新生成器Sync-C，尚未判断是否优于自然baseline。此处只发布训练验收，不自动新增GPU评测或重训，旧Patrick/fine14/Chris/Cathy/24full持续暂停。

## Context
用户已授权通过AVE缓存和真实loader检查后训练；新run F:/Datasets/tts-exp/may_tts_purified_training_20261020_v1已启动一个0dB无归一化模型。沿用报告50原AVE、seed42原完整初始化tensorSHA15e74e38abec740a7e9790917658da13a0d3a0c0bf426faeb0c119b252eabbc1、60000成功FP32更新、4096rays/maxsteps16、原L1及正则/AdamW/LR/EMA。自然val/test全部不改，正式执行调用原baseline trainer，未加载资格权重或旧训练权重。

新TTS AVE全量两次逐位重复，train padded[3915,512]→aligned[3849,1,512]；自然val重算与历史cache逐位一致。真实train3849/val550/test1100加载、36像素/pose/audio索引检查、独立2success/0skip FP32入口和model/optimizer/scheduler/EMA/RNG保存读回通过。F已逐成员fresh SHA且独立NumPy复核；5项CPU真实产物/参数/路径拒绝检查直接调用通过（未安装pytest，不称pytest通过）。共同initial文件与官方network/utils/renderer/provider SHA同baseline。3849新样本域按同seed42 repeated permutation生成60000索引，SHA7c9e27b78945c88a1ed47f40149b07c42ca5f46d462cec3cf772dbbeb37febd3；不直接用旧4150域，训练数据时长/采样序列差异明确。

权威formal_admission.json、formal_launcher.json、startup_review_v1.json及controller/receipt.json。远程/dev/shm/may-tts-purified-training-20261020-v1是易失执行副本，SSH需-b192.168.50.1；只读host observer把每2000步全部fullcp/验证/trace持久F并fresh SHA，保留所有新checkpoint、不自动重启、无评分/视频启动。科学失败必须封存再停，0新cloud备份声明；旧Patrick/fine14/Chris/Cathy及24full继续暂停。

准备失败均保留：CUDA缺cusparse头由隔离symlink恢复；修复helper命名空间碰撞发生在编译前；等待300秒超时未运行qualify；CRLF启动失败未进Python；首loader reviewer错误地把val全图HWC与flatten比较，在模型构造前拒绝，修订qualify_v2按原provider布局通过，provider不改。全部发生于formal之前、0正式optimizer重放。当前已记录正式启动前缀由startup_review_v1.json固定核验，不能称60000训练完成。


## Requirements

### Requirement: SHALL在正式训练前完成新TTS AVE与真实loader资格
#### Scenario: 输入准入
- **WHEN** 使用已封存云端TTS净化数据
- **THEN** 全量AVE两次重复一致、自然val缓存复现、36像素/pose/audio检查和2success/0skip入口及F独立读回通过。

### Requirement: SHALL按首次May baseline完成60000成功FP32更新并保存全部断点
#### Scenario: 正式训练
- **WHEN** 原initial加载并进入新0dB训练臂
- **THEN** 原AVE/网络/loss/AdamW/LR/EMA/4096rays/maxsteps16保持、无归一化，60000success/0skip及30个每2000步完整state与验证产物F逐成员核验；未全部满足前不能done。

### Requirement: SHALL在故障时保全证据并停止而不重放
#### Scenario: 数值或job故障
- **WHEN** 首次实际训练失败
- **THEN** 保存state/trace/job到F并独立核验，标blocked并等待人类新方向；不自动换精度/归一化/重启，不恢复旧实验。

## Observations
- [status] done
- [result] 三档新模型Sync-C评测与独立QA及F交付完成；-6dB：新C5.296、自然baseline 8.669、Δ-3.373 / +0dB：新C5.703、自然baseline 8.909、Δ-3.206 / +6dB：新C5.911、自然baseline 9.036、Δ-3.125。
- [verification] 原AVE/initial/优化预算/heldout同报告50；9帧baseline replay、250帧重复、12矩阵/像素/PCM/ROI和F全成员passed。
- [limit] 单May/seed42，TTS时长与图像映射改变；不推感知或总体因果，不恢复旧实验。
- [close_date] October 20, 2026

## Relations
- project [[May 云端 TTS 净化训练数据对照]]
- compares_to [[50-synctalk-may-gain-training-pilot]]

- reported_in [[64-may-purified-tts-training]]
