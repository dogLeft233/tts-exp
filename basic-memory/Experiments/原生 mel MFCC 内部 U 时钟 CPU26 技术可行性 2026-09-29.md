---
title: 原生 mel MFCC 内部 U 时钟 CPU26 技术可行性 2026-09-29
type: experiment
permalink: tts-exp/experiments/原生-mel-mfcc-内部-u-时钟-cpu26-技术可行性-2026-09-29
status: technical_feasibility_pass
date: '2026-09-29'
engineering_status: CPU26_INDEPENDENT_PASS
scientific_status: NO_NEW_EFFECT_GPU_LOCKED
tags:
- tts
- tfg
- syncnet
- native-frontend
- u-clock
- technical-feasibility
---

# 原生 mel/MFCC 内部 U 时钟 CPU26 技术可行性 2026-09-29

## 问题与设计

WORLD 域内部 U 时距曾使自配 Sync-C 上升，但 WORLD 恒等重建未桥接原波形。本实验的下一条路线直接使用旧 uniform N_WARP/T_ID 原生 FLOAT WAV 在 Wav2Lip 与 SyncNet V2 正式前端生成的完整 mel/MFCC，按已封 E/U 区间在特征时间网格上双向改变 U 时距，同时保持总长及 E；它是**模型输入特征层**操作，不是可播放波形或原 Ditto 复现。根科学协议 SHA `1d82c867015cde331dc4142c61136dd80e26d90cbc96df357d6d519f2e4f43d4`，父 FLOAT WAV 音频路径说明 SHA `a61e33dc411c38d5d29734973306a75d582c415fc1a3a716b61d15d67d0db8f0`。

## 结果与边界

CPU 26cal 技术实施及不同 agent 独审 **PASS**。26 对保留7个空 U、19个非空 U；非空双向共38个换时臂，U/E/S 锚、支持与合成 pulse 时点门通过，最大 mel 6.900 ms <12.5 ms、MFCC 5.553 ms <10 ms。52 个母 FLOAT WAV 解码 PCM 与完整 mel/MFCC、Wav2Lip chunks、SyncNet windows 的 identity 全部一致；a1_001 的 N/WARP 与 T/ID 两母角源 SHA 精确匹配。74eval 旧父 V/A 的 2臂×4政策共592条**既有**距离曲线/端点由旧 scorer 逐值重算 exact；没有计算任何新 U 效果。

实施代码 `runs/tts_native_frontend_u_clock_20260929/prepare_cpu.py` SHA `e98e6fb61fca0929ee46c5ccdfd5166f63051253fdeb8ea3747f909c7c9693fe`；技术协议 SHA `55ddeae920d9f70d63b5997515ef8eb2704f749b50c2a2f2a9161e5576d74efe`；最终 CPU 收据 SHA `f1d9e2313ec3dc0e645eb69841c389d8c7f7b0f1509d6a084fb3621e14a2b68c`。不同 agent 独审回执 SHA `7329248c24f8aca57ce45b7397d0b4fc3f18216f5e8bba99be18f64d1236ef95`，独审报告 SHA `2ad9d18f77dc9ea2d1044dc84c922abd18a4d16680a67ad463eaa814cae767d4`。早期128KiB资源低估及首次Gaussian argmax pulse方法失败留存，所有修订发生在新效果前。持久技术资产462,848B<1MiB，GPU0。

**尚未证明**：旧74eval临时 FLOAT WAV 已释放，必须原代码重建并逐字节 SHA 桥；新视频像素/PTS/V/A、母角端点 identity 与不同 agent 的现场审未做。故不能称原生 U 时钟已使 Sync-C 增加，更不能称物理声学波形因果归因。后续 GPU/新效果仍锁定。

## Observations

- [status] CPU 26cal 技术可行性独审 PASS；正式 GPU/效果未开始。
- [bridge] 52个母角 PCM/mel/MFCC/chunks/windows identity、a1_001双母角 SHA、592条旧父曲线/端点 exact。
- [timing] 非空U的38个双向时钟映射 pulse 在 mel/MFCC 一个步长内。
- [scope] 仅原生模型输入特征层；历史共享队列；不证明可播放波形、真实嘴型或原 Ditto 机制。
- [remaining] eval74 WAV 精确再生、GPU现场媒体/V/A母桥与独立审计仍需完成。
- [artifacts] `runs/tts_native_frontend_u_clock_20260929/` 及 `runs/tts_native_frontend_u_clock_20260929_reviewer/`。

## Relations

- follows [[原波形内部 U 分段变时 26cal 技术可行性 2026-09-29]]
- relates_to [[WORLD 内部 U 错位背景距离几何定位 2026-09-29]]
- relates_to [[TTS 增强 TFG 的原因：证据、排除项与因果分叉]]

## Changelog

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| CPU技术实施和不同agent独审PASS；界定原生前端层与未完成GPU门 | September 29, 2026 | root理论协议；GPT-6 Sol high实施与独审 |
