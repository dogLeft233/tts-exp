---
title: Codex checkpoint - 2026-10-23T13-10-41Z - Generator hypotheses for normalized
  Sync-C gain
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-23-t13-10-41-z-generator-hypotheses-for-normalized-sync-c-gain
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-23T13:10:41Z'
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a16144-7517-77a1-b372-027fba42da62
model: gpt-6-astra
trigger: compact
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-23T13-10-41Z - Generator hypotheses for normalized Sync-C gain

## Summary
共享codec将1.7B对0.6B归一化Sync-C差异的候选来源收敛到生成输出与下游响应，但尚未定位因果机制。

## Story
研究希望解释TTS为什么在口型生成后得到较高Sync-C。用户最新要求基于1.7B主要扩大生成器这一事实提出可解释性猜测。本轮核对研究地图、1.7B同10人正式报告、固定粗码技术报告及此前分数分解，在线核对Qwen官方技术报告架构与训练描述；未启动合成、评分或训练。

同10人norm L17−L06为+0.656，9/10提高；排除两停止异常仍+0.418。此前经CSV核验的分解为最佳距离改善+0.118、偏移距离中位数增加+0.538。这是分数恒等式，不是因果占比。L17相对自然+0.409的严格9人区间跨零，应把解释对象首先限定为当前两模型差异。

优先研究假设是生成器使局部音素转换及相邻帧声学线索更协调、更易被Wav2Lip/SyncNet辨别。必须并列两个竞争解释：语速/节奏使固定毫秒错位跨过更多发音变化；声谱/音色改变SyncNet表征尺度或分布，提高距离反差但不改善独立视觉同步。更大上下文表示可能改变粗码预测及细码条件，不证明纯参数量因果，也不意味着只改粗码；官方训练阶段描述不能充当两checkpoint训练完全一致的证明。旧响度/停顿/能量损伤实验不足以辨别这些解释，不能再把“破坏后掉分”当机制闭合。

## Working State
科学产物仍封存于runs/qwen17_level_comparison10_20261023_attempt02/与runs/qwen_fixed_coarse_20261023_attempt01/。后者自重放通过，但交叉补全内容/局部时序门HOLD，不进入完整评分；其自动边界异常不是人工确认的时间位移。共享codec无需因1.7B名字再做同一个自然重建，现有R与当前10人官方链的逐项绑定尚未审计。

本轮没有修改代码或实验数据。Git直接核验为codex/cross-tfg-audio-head-20261003、HEAD 2b71639c3b555293375d248350999377168d515d；21个修改、1152个untracked条目均为继承状态，不清理/提交。GitHub只读确认repo及分支PR查询空；HEAD远端可解析性本轮未确认，不构造commit链接。

## Verification
本轮读取Router→研究地图→实验指令→正式两实验与分解检查点。官方arXiv2601.15621v1确认12Hz主干预测零号码本、MTP补残余码本，非直接预测嘴型。现代与legacy会话检索完成、前检查点agent/session精确读回。没有重算分数、运行测试或对现有视频做新测量；数值来自已核验历史记录。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- [Qwen3-TTS technical report](https://arxiv.org/html/2601.15621v1)
- runs/qwen17_level_comparison10_20261023_attempt02/report.md
- runs/qwen17_level_comparison10_20261023_attempt02/person_means.csv
- runs/qwen_fixed_coarse_20261023_attempt01/report.md

## Observations
- [decision] 将局部发音转换可辨识度列为优先假设，同时保留语速时钟和评价表征尺度两种竞争解释，不声称已证明发音或嘴型更清楚。
- [decision] 下一步优先复用原音频、视频与距离曲线做定位，独立核验可见闭唇/释放等事件；不自动恢复粗码补全评分，不训练。
- [blocker] 尚缺同一批样本局部事件与独立视觉证据，原生自配评分无法分离生成端与评价端；两模型比较不能隔离纯参数量。
- [next_step] 若用户要求继续执行，先对已有同10人三draw原生音视频及距离曲线作有界局部定位方案，区分转换区、语速及距离尺度响应，再决定是否值得做保持内容时序的干预。

## Relations
- continues [[Codex checkpoint - 2026-10-23T13-02-00Z - Shared codec evidence and reconstruction reuse]]
- interprets [[同一10人上的Qwen1.7B与音量归一化对照2026-10-23]]
- qualified_by [[固定粗编码的双模型条件补全技术验证 2026-10-23]]
