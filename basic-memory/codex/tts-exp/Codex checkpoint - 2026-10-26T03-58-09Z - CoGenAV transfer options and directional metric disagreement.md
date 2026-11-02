---
title: Codex checkpoint - 2026-10-26T03-58-09Z - CoGenAV transfer options and directional
  metric disagreement
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-26-t03-58-09-z-co-gen-av-transfer-options-and-directional-metric-disagreement
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-26T03:58:09Z'
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a16cf2-be54-7ba0-bcfb-be8433abd951
model: gpt-6.1-sol
trigger: compact
tags:
- codex
- checkpoint
- cogenav
---

# Codex checkpoint - 2026-10-26T03-58-09Z - CoGenAV transfer options and directional metric disagreement

## Summary
CoGenAV解码器先导已完成但声学保真度低；评分冲突以Sync-C偏TTS、CoGenAV偏自然为主，当前讨论怎样将假定更好的TTS口型迁移到自然音轨。

## Story
本聊天从核验CoGenAV官方编码器与SyncNet评分差异，推进到已知变形对齐、TTS视频重定时以及五小时解码器训练。用户最新问题是假定TTS确实带来更好的口型，利用论文是否有合适的迁移路径。当前是方法讨论，尚未授权新一轮训练或生成。

论文对齐是同步音视频表示的对比学习，生成目标是冻结Whisper上的文本预测。官方没有将不同语速的TTS视频重定时到自然音频的完整算法。我们的DTW、路径约束、选帧与最佳时移搜索均为项目适配。特征相似不代表可互换或可无损还原。

六来源视频后处理实验未转移优势：自然原生Sync-C 6.649、同流程自然6.660；TTS配自身音轨7.448仅作背景；CoGenAV跨模态DTW换回自然音轨5.814、音频间DTW5.360、MFCC5.146、线性2.183，全部0/6超过自然。所选来源按历史T>N挑选；重复跳帧和路径误差未分解，不能视为理想对齐的上限。

最新建议方向是先将TTS音频局部时长/停顿匹配自然时间轴再驱动连续视频，最终配回完整自然音频；可用音素边界作骨架、CoGenAV局部特征作辅助。此为研究建议，尚未验证，也不是论文官方功能。更长线可考虑自然音频条件下蒸馏TTS运动，需要先解决可靠对应与教师质量。

## Working State
已完成的权威产物：
- runs/cogenav_sync_full20_20261024_attempt01/pairs.json与scores.json：20来源、138视频、176配对，其中118个TTS-N比较，58个TTS-TTS不能混入自然偏好统计。
- runs/cogenav_video_retime_20261025_attempt01/parent_report.md：六来源迁移结果与独立审计。
- runs/cogenav_decoder_pilot_20261025_attempt01/report.md：五小时AISHELL-3冻结CoGenAV+适配器+预训练普通kNN-VC HiFiGAN，2000步适配器预热、4000步联合重建微调，无对抗训练。训练4202句40说话人5.03h，验证429句4独立说话人。固定40句质量集CER3.07%、STOI.665、PESQ1.116；自然CER1.53%，WavLM直接声码器CER2.73%、STOI.928、PESQ2.255。内容可辨而保真度低；未做人评、TTS重定时或下游TFG验证。
- C:/Users/dogle/.codex/tts-exp-artifacts/cogenav-decoder-pilot-20261025/science_delivery.tar 为完整本机备份，SHA256 824e606263ac8c6d52451e08f6d21c3098e968d8756a60d0f47826f0103f2761；decoder_best.pt SHA256 1127aa8ed9f33ce25562df2cfc8d599cd49eea726d57b9e6ed6eae8094f9ac65。教师只在训练使用，推理解码无需WavLM。
Ubuntu旧名路径/home/wjj/tts-audio/tts-exp有效；Windows唯一入口E:/Documents/tts-exp。E盘空间紧张，完整产物在C盘。另一个聊天的audio200/TFG任务不得干预。

## Changed Files
最近评分方向统计及本次解释没有改代码、没有提交推送、没有启动GPU工作。本次新增仅此不可变检查点。五小时训练先前增加scripts/cogenav_decoder_pilot相关脚本和对应测试，细节以实验报告为准。
当前工作树有大量继承修改/未跟踪文件；本次读取git status已确认，不能将它们归因于本轮或清理。

## Verification
前一轮实际重新读取所有176个pairs及138个scores，核验三个端点score_float32/eq1/max_score均为a-b，最大误差0。118个TTS-N比较无平局：
|端点|Sync-C偏TTS/CoGenAV偏自然|反方向|都偏TTS|都偏自然|
|---|---:|---:|---:|---:|
|Eq1|51|23|22|22|
|最佳时移max_score|33|24|40|21|
Eq1细分L06-N 27/9、L17-N 24/14；max细分14/10、19/14。冲突中主方向分别68.9%、57.9%。118比较共享20来源与自然基线，不是118独立来源，不能据此判定指标真伪。

本次重新读取Startup Router、技能、配置、前序同聊天检查点元数据及六来源实验全文，并在线核对论文v2与官方仓库。重新确认本地根/分支/SHA。GitHub连接器确认仓库dogLeft233/tts-exp及当前分支open PR为空；本地SHA未查远程，不构造提交链接。gh不可用，已用连接器替代。Router指向的note-instructions在项目及main均未找到，搜索也无结果；本次按已加载bm-writing与用户项目规则记录。历史训练审计和质量数字来自已完成工作上下文，非本轮重复运行。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- [CoGenAV v2](https://arxiv.org/html/2505.03186v2)
- [Official implementation](https://github.com/HumanMLLM/CoGenAV)

## Observations
- [result] Eq1的74个TTS-N冲突中51个是Sync-C偏TTS、CoGenAV偏自然；最佳时移57个冲突中33个同方向。
- [decision] 分清实际口型质量、时间匹配和评分器声学偏好；当前分歧不足以认定任一指标是真值。
- [decision] 当前低保真CoGenAV波形解码器不适合作为下一轮迁移试验唯一音频处理通道；先对齐TTS波形再驱动可避免引入该混淆。
- [blocker] 尚无真实TTS-N精确时间路径或人评证据，理想口型优势能否迁移未确定。
- [next_step] 回答用户迁移路径问题，说明先对齐TTS驱动与运动层迁移的区别、所需验证；未获实施请求前不启动新实验。

## Relations
- continues [[Codex checkpoint - 2026-10-25T10-56-11Z - CoGenAV decoder initialization and available audio inventory]]
- relates_to [[CoGenAV特征重定时TTS视频并换回自然音频的六来源先导 2026-10-25]]
- relates_to [[CoGenAV 五小时解码器微调的重建质量 2026-10-25]]
