---
title: 准备 May 的云端 TTS 净化训练数据
type: task
permalink: tts-exp/tasks/准备-may-的云端-tts-净化训练数据
status: done
current_step: 3
completed: October 20, 2026
description: Prepare train-only cloud TTS/NAS dataset with each Sync-C >5; no training
---

# 准备 May 的云端 TTS 净化训练数据

## Context

用户要求先找历史净化方法，再准备May的云端TTS净化训练数据，保持原AVE与无归一化；仅准备，不训练。权威F数据入口和限制见父实验及DATASET_README.md。

## Requirements

### Requirement: SHALL绑定首次成功May baseline并隔离heldout
#### Scenario: 数据绑定完成
- **WHEN** 原baseline与源资产核验
- **THEN** 原seed42 initial/frame order精确保留，29400训练侧源SHA通过，val/test WAV及JSON逐字节不变。

### Requirement: SHALL仅用云端Qwen-Audio3.1准备内容完整的TTS
#### Scenario: 音频准备完成
- **WHEN** 12段训练源ASR及TTS生成并复核
- **THEN** 使用qwen-audio-3.1-asr-flash/tts-flash，关闭音量归一化与预处理，无本地TTS，全部回转文字WER0，漏句原件和修复证据保留。

### Requirement: SHALL净化并逐段验证Sync-C严格高于5且不训练
#### Scenario: 数据门通过
- **WHEN** 固定NAS短窗净化完成
- **THEN** 12/12段guard20与均值均>5，独立119474距离格重建passed，完整F逐成员核验，无生成器训练。

## Observations
- [status] done
- [result] 12/12门通过，最低C6.121、等权均值7.257；原生TTS156.56秒、3849监督行，数据差异明列。
- [verification] 416成员SHA、baseline/heldout字节、距离重建及5项实际数据检查通过；pytest模块缺失，使用同5函数直接调用。
- [decision] 仅数据任务done，训练研究仍planned；AVE新特征与loader资格不冒称已完成，不恢复旧实验。
- [close_date] October 20, 2026

## Relations
- project [[May 云端 TTS 净化训练数据对照]]
