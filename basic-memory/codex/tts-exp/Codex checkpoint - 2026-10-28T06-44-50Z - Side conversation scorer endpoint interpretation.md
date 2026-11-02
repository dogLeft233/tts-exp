---
title: Codex checkpoint - 2026-10-28T06-44-50Z - Side conversation scorer endpoint
  interpretation
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-28-t06-44-50-z-side-conversation-scorer-endpoint-interpretation
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-28T06:44:50Z'
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a1813d-f78a-7f40-b798-b3bd1f3672a1
trigger: compact
model: gpt-6.1-sol
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-28T06-44-50Z - Side conversation scorer endpoint interpretation

## Summary
侧聊仅解释现有同步指标的评价重点，不接续主线程部署和评分。

## Story
用户在侧聊边界后询问各指标检测重点。读取项目记忆及原论文，将特征对应、零时移定位、最佳匹配突出程度和人类评分预测区分。Sync-C的相对峰形不能当作零时移准确率；CoGenAV Eq1与AVS_u更接近跨模态特征对应；MTD官方K指标关注预测时差落在零附近比例；Synchformer当前端点为零时移类别概率；SyncReward预测整体人类同步评分。模型训练任务与项目汇总端点必须分别描述。

## Working State
主线程未完成的原版VocaLiST下载/验证仅是继承上下文，本侧聊未查询或操作其进程，不能确认最新部署状态。没有修改源码、输入媒体、Git状态或配置。工作区已有大量dirty/untracked文件，不能归因于侧聊。仅依开发者明确要求新增本检查点。

## Verification
读取Startup Router、实验指令、实验70、云Qwen20来源四指标笔记及既有机制检查点；打开CoGenAV、AVS、SyncReward论文和Synchformer/MTD官方仓库。未运行评分或测试。只读Git确认root、branch、SHA；GitHub REST确认仓库URL和当前分支无open PR。gh命令不可用，使用REST替代；SHA未验证为已推送，不构造commit链接。同会话身份与legacy两次检索无已有检查点。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- [CoGenAV](https://arxiv.org/html/2505.03186v1)
- [AVS](https://arxiv.org/html/2405.04327v1)
- [SyncReward](https://arxiv.org/html/2610.16338v1)
- [Synchformer](https://github.com/v-iashin/Synchformer)
- [MTDVocaLiST](https://github.com/xjchenGit/MTDVocaLiST)

## Observations
- [decision] 仅回答侧聊问题，不恢复任何继承的部署、后台进程或评分任务。
- [result] 官方MTD K5与旧平均logit端点不同；K7至K15增加上下文，不放宽±1帧容差。现有绝对K分数存在低分问题，不能由冲突判定哪方是真值。
- [next_step] 向用户说明各指标的重点及分数分歧为何不能直接判定SyncNet偏置。

## Relations
- relates_to [[70-mtd-official-native-cloud-preference]]
