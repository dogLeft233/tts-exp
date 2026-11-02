---
title: Codex checkpoint - 2026-10-18T12-24-36Z - Gain direction interpretation and
  transfer boundary
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-18-t12-24-36-z-gain-direction-interpretation-and-transfer-boundary
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: 2026-10-18 12:24:36.564938+00:00
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
trigger: compact
model: gpt-6.1-sol
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-18T12-24-36Z - Gain direction interpretation and transfer boundary

## Summary
报告52的纯音量响应具有明确编码器边界；跨人共同方向与跨编码器迁移必须分开解释。

## Story
用户此前要求ROOT直接完成主流编码器扩展，现追问是否所有编码器都因音量改变方向、是否存在例外，以及变化是否跨编码器生效。本轮只回读BM地图、实验指令和报告52实际数表，不启动新实验。研究主线由表示共性走向可干预机制；这一问题需要明确测量层级，避免把TTS与自然的方向共性当作纯增益结果。

同一旧CAL20十位男性二十句、自然FIXED基线加3dB，末层单位向量方向变化RMS在Wav2Lip/WavLM base-plus/Whisper base/small为0.0721/0.0350/0.0275/0.0311，相应跨人cos0.884/0.479/0.613/0.577。HuBERT英base/中文base、wav2vec2 base与W2v-BERT2只有约7e-7至7e-6，低于预设方向分辨率；不能称数学严格零。data2vec、HuBERT large、XLS-R53、WavLM large约1e-5至5e-5，虽部分过门仍是微弱变化。原生wave归一化或CMVN与低响应相符；模型权重也不同，未做相同权重切换前处理的单因实验。

多个家族均存在音量导致方向变化这一现象，但所有方向cos都在各编码器自身空间内按人计算；不同维度/坐标从未直接比较。本次0新TFG/SyncNet，也未将某编码器轴映射注入另一编码器。因此跨编码器可迁移共同轴或共同生成收益仍是待执行问题。上一轮0.165–0.506是统一RMS的TTS减自然方向共性，不是纯音量响应。

## Working State
编码器扩展Task已经done。实际产物F:/Datasets/tts-exp/encoder_survey_20261018，轻量镜像runs/encoder_survey_20261018；科学原数组和source/receipts保留，公共权重body已回收。报告52、证据地图与存储参考是入口。上次1331正式前向、W2v-BERT两次失败路径及修复、存储处理和恢复位置详见上一checkpoint，不重复制闭包。本轮无代码改动、无新科学计算或测试。通用head仍按用户要求暂停；不触碰其他聊天SyncTalk训练。

## Changed Files
仅创建本不可变checkpoint；七个encoder_survey实验脚本及runs为上轮未提交本地产物。当前九项tracked dirty保持，包括CONTEXT和继承的audio_enhancement/portable_source_axis等文件；未提交或push。

## Verification
本轮回读报告52和地图实际数值，未重跑验证。上一轮独立消费者验证240 NPZ、448统计组，公式最大误差4.44e-16。Git实际分支/SHA/root已核验。只读GitHub确认仓库canonical URL和当前head无PR；GitHub不解析本地HEAD，故SHA仅作本地锚点。BM全局用户配置不存在，仓库配置primaryProject=tts-exp/coding生效。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- basic-memory/docs/experiments/52-encoder-direction-commonality.md
- runs/encoder_survey_20261018/{results.json,qa.json,artifact_manifest.json}
- 本地未获远端解析的HEAD：`2b71639c3b555293375d248350999377168d515d`

## Observations
- [result] 纯音量方向响应在编码器之间从接近数值分辨率到0.07量级相差多个数量级，不能统称所有编码器都有明显方向变化。
- [decision] 跨编码器现象覆盖、编码器内部跨人共性、跨编码器干预迁移与生成收益分开报告。
- [next_step] 用报告52回答当前解释问题；若用户进一步要求迁移验证，再围绕两条强响应原生路径制定小型生成实验，不自动扩规模或恢复暂停head。

## Relations
- continues [[Codex checkpoint - 2026-10-18T12-17-20Z - Mainstream encoder commonality completed]]
- relates_to [[52-encoder-direction-commonality]]
- relates_to [[研究主线与证据地图]]