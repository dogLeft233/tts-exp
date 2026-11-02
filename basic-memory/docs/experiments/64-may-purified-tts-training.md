---
title: 64-may-purified-tts-training
type: experiment
permalink: tts-exp/docs/experiments/64-may-purified-tts-training
status: concluded
result: -6dB：新C5.296、自然baseline 8.669、Δ-3.373 / +0dB：新C5.703、自然baseline 8.909、Δ-3.206
  / +6dB：新C5.911、自然baseline 9.036、Δ-3.125
inputs: cloud Qwen-Audio3.1 train-only NAS; original natural heldout
outputs: F:/Datasets/tts-exp/may_tts_purified_eval_20261020_v1
model: Original SyncTalk AVE; FP32 60000 final EMA60
---

# 64-may-purified-tts-training

## 问题与对照
用户要求仅更换净化后的云端TTS训练数据，与首次成功May baseline尽量对齐。一个新0dB模型从报告50同seed42完整initial训练60000成功FP32更新/0skip，原AVE、4096rays/maxsteps16、loss/AdamW/LR/EMA保持，无波形归一化。自然训练baseline为报告50的0dB模型，未挑选历史最佳训练增益。新3849监督样本与自然4150样本数量/时钟不同，这是数据处理差异；自然val/test字节保持。

## Sync-C
固定60000步EMA60，四个原test clip各250帧，source start4929/5179/5429/5679；−6/0/+6只改变自然驱动输入，评分统一原自然PCM16。guard20主端点四clip等权，无新p值或身份级推广。

| 驱动dB | 自然训练baseline C | TTS净化训练 C | 新减旧 |
|---:|---:|---:|---:|
| -6 | 8.669 | 5.296 | -3.373 |
| +0 | 8.909 | 5.703 | -3.206 |
| +6 | 9.036 | 5.911 | -3.125 |

## 运行与独立核验
旧baseline9固定帧float/RGB逐位复现，新模型9帧repeat及FFV1/PCM像素资格通过。新模型3300 test帧/1650 val帧，train0/test0首250帧独立repeat float/RGB/SyncNet矩阵逐位同。评分重用历史原GT S3FD/median13轨迹，四原音频MFCC/SyncNet特征实际重算并与baseline逐位一致；原权重、FP32/TF32off、25fps/原时钟、guard20及full敏感性保持。12个245×31矩阵共91140距离独立float64重建及C/Dmin/D0/B/offset通过；96个保存val浮点帧与原训练final曲线核验通过，完整视频像素/PCM及裁剪重建通过。新F全成员fresh读回另有manifest，不声称云备份或清理媒体。

## 辅助诊断（不混入Sync-C主表）
新模型12个clip的最佳offset均为0，匹配自然baseline均为−3帧。固定0dB驱动，新D0=8.549，自然baseline D0=14.058；全test口部框MSE新0.004943、旧0.002821（变化+75.19%）。这些评价端点并不一致：C更低不能直接解释为时序更差或感知更差；没有物理对齐校正或盲评结论。

## 证据边界
SyncNet也用于训练数据NAS净化和数据QC；数据最低C6.121不是本次生成器测试分数。当前结果只支持单May/seed42、当前预算下的匹配评价差异，不证明感知嘴型、总体TTS训练收益或排除时长/帧重排等数据因素。保留原自然PCM和实际offset记录，不以更高C修复或重对齐主结果。所有旧Patrick/fine/Chris/Cathy保持暂停，无新训练重放。

CPU独立审核初次因评测目录缺少训练frame_order只读引用而在视频/矩阵审核前停止，原失败与日志保留；核对原order SHA后仅增加指向训练原件的symlink，使用同审核源完成剩余QA，没有重放训练、渲染或评分。

权威：evaluation_contract.json、remote/qualification/receipt.json、remote/independent_results_qa.json、readback_manifest.json及analysis/analysis.json。API key不在产物中，0本地TTS。

## Observations
- [status] concluded
- [result] -6dB：新C5.296、自然baseline 8.669、Δ-3.373 / +0dB：新C5.703、自然baseline 8.909、Δ-3.206 / +6dB：新C5.911、自然baseline 9.036、Δ-3.125。
- [verification] 60000 FP32更新/0skip和30完整cp；新三档3300test帧/12矩阵91140距离、PCM/像素/ROI/重复/F全成员QA通过。
- [limit] 单May/seed42，TTS/NAS同时改变波形、时长与监督图像映射；Sync-C不是感知或普遍训练收益。

## Relations
- compares_to [[50-synctalk-may-gain-training-pilot]]
- concludes [[May 云端 TTS 净化训练数据对照]]
- completes [[训练 May 的云端 TTS 净化数据对照]]
