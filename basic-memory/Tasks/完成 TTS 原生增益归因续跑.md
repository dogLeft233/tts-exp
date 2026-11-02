---
title: 完成 TTS 原生增益归因续跑
type: task
permalink: tts-exp/tasks/完成-tts-原生增益归因续跑
status: blocked
priority: high
change_id: complete-tts-native-gain-attribution
run: runs/tts_native_gain_attribution_completion_20260916_v1
engineering_status: PARTIAL/DEPENDENCY_BLOCKED
tags:
- tts
- native-gain
- openspec
- blocked
---

# 完成 TTS 原生增益归因续跑

## Requirements

### Requirement: 完成补全 spec 的可复现实现
实现 `openspec/changes/complete-tts-native-gain-attribution/` 中的父证据传递、A 复验、LeapTalk B 生成与交叉评分、统计、盲评包、资源控制和独立验收。

#### Scenario: 外部 LeapTalk 依赖可用
提供 provenance-bound `LEAPTALK_CONFIG` 后，先完成 ID151 smoke，再运行 144+4 个 B 视频、336+8 个评分和控制，生成六项注册对比并通过成功态 validator，才可将任务标为 done。

#### Scenario: 外部 LeapTalk 依赖不可用
保留 A 结果和只读父 run，B 写出完整固定分母与逐格 `DEPENDENCY_BLOCKED` 台账，盲评媒体标为 `PARTIAL_MEDIA`，validator 必须通过且 `automatic_complete` 保持 false。

## Observations

- [status] blocked
- [progress] 正式续跑已完成 A 复验和阻塞态全链路；B 没有伪造数据，当前工程状态为 `PARTIAL/DEPENDENCY_BLOCKED`。
- [result] A 保留 180+18 个 cell、72 个操纵检查、24 个 v15 重放和 396 个错内容诊断；B 规划 144+4 个视频与 336+8 个评分，实际完成数均为 0。CPU success fixture 已覆盖成功接口，但不替代正式 B 数据。
- [blocker] 缺少可核验 `LEAPTALK_CONFIG`，包括官方 repo commit、base/LoRA/audio projection/audio encoder/decoder/VAE/TAE revision 与文件 hash、运行时设置，以及每格 frontend、RNG、PCM 消费和 chunk-to-frame 证明。
- [verification] `python -m pytest -q tests/experiments/tts_native_gain_attribution`：19 passed；ruff、py_compile、`git diff --check`、strict OpenSpec 和独立 validator 均通过。GPU 共享锁、foreign PID 资源等待、独立 GPU/磁盘预算和中断隔离已实现并有测试。

## Relations

- implements [[complete-tts-native-gain-attribution]]
- updates [[TTS 原生增益来源的生成端与评估端交叉诊断]]
- tracks [[Task Board]]