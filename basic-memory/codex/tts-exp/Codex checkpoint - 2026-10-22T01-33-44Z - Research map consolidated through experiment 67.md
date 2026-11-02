---
title: Codex checkpoint - 2026-10-22T01-33-44Z - Research map consolidated through
  experiment 67
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-22-t01-33-44-z-research-map-consolidated-through-experiment-67
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:\Documents\tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-22T01:33:44.146760+00:00'
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a139e9-67ce-7b41-90f2-5b7019492a69
trigger: compact
model: gpt-6.1-sol
tags:
- codex
- checkpoint
- research-map
---

# Codex checkpoint - 2026-10-22T01-33-44Z - Research map consolidated through experiment 67

## Summary

项目记忆已按原生现象、生成／评价机制、训练效应重组，补上噪声训练已完成及低尺度训练反向基线的证据边界。

## Story

本 chat 起初协助编辑中英文桌面 proposal，最近一次编辑已按用户要求只保留噪声后续实验；用户曾明确不要将桌面文档编入 BM，该要求仍保留，本次没有修改或复制桌面正文。最新请求是整理项目记忆和研究路线图，因此此次整理直接针对项目导航及既有实验记录。

读取 Startup Router／实验指令，全文搜索 Experiments 与 docs/experiments，追读正式报告56、62、66、67及相关专题，发现旧地图同时保留过时主线、运行中噪声状态与已完成结果。现地图按三个问题导航，压缩约550行到186行；历史路线及停止边界另存，实验原件不删改。特别保留56低尺度C正向与67无添加组反向的差异，不据此擅自判定严格重复失败，也不预设低尺度训练有稳定收益。

## Working State

权威入口为 Research/研究主线与证据地图.md。低尺度／噪声综合判断见 Research/低尺度训练与噪声解释的证据边界.md；旧迁移、增强头、MFA与工程失败路线见 Research/历史研究路线与停止边界.md。编号索引补齐32–38／62并列出两个25文件；Task Board明确暂停与历史未结任务，旧写作综合标为reference，保留原定量表。CONTEXT.md只替换最前面的旧近期摘要，仍99行。

本次0训练／0生成／0评分，未修改自动化或恢复任何暂停任务。Patrick RMS coarse7、旧精调、Chris／Cathy及增强头的既有暂停保持。18模型噪声训练由正式报告67确认完成；这不是本轮新执行，也不据此声称现场GPU状态。

本地工作树继承大量未提交、未跟踪文件；AGENTS.md、音频增强源码及相关测试的既有dirty不是本次改动，未reset／clean／commit／push。BM变更已写入并通过read_note逐篇读回；草稿、旧地图备份、发布回执和QA在.codex/scratch/research_memory_audit_20261022，属机器本地辅助文件，不是云备份。科学证据的F／E／Ubuntu恢复位置继续以原报告receipt为准。

## Changed Files

- basic-memory/Research/研究主线与证据地图.md
- basic-memory/Research/低尺度训练与噪声解释的证据边界.md（新增）
- basic-memory/Research/历史研究路线与停止边界.md（新增）
- basic-memory/docs/experiments/README.md
- basic-memory/Tasks/Task Board.md
- basic-memory/Research/论文实验数据整理与写作主线.md（历史标记与导航）
- CONTEXT.md（仅首条摘要的语义变更）

## Verification

六篇BM笔记写入并读回正文一致，链接目标全部存在且无歧义；新综合笔记通过BM text搜索可检索。编号1–67除预留07外全部有入口，两份报告25分开列明。综合表八组噪声数值与报告67逐项一致；正式报告56／67文件SHA未变。CONTEXT.md为99行。没有运行生产代码或模型测试，本次仅做文档、链接、数值转录及索引核验。

## References

仓库身份由只读GitHub API核对：[dogLeft233/tts-exp](https://github.com/dogLeft233/tts-exp)。同一查询确认当前分支没有开放PR；本次未创建PR。本地HEAD为 `2b71639c3b555293375d248350999377168d515d`，未核验其远端commit可解析性，故不构造commit链接。

## Observations

- [result] 噪声训练67已经完成；八个加噪条件中仅背景讲话30dB直接表现为低尺度C较高，其余七个为负；正交互不等于低尺度模型直接受益。
- [limitation] 56与67无添加组的方向相反；编码特征已有逐值桥接，但跨批训练初始化、执行版本与数值轨迹可比性尚未完整核对，不能给稳定低尺度训练收益或噪声因果结论。
- [decision] 当前导航按现象、机制、训练组织；保留历史路线、反向条件和原失败，不恢复旧任务，不把Sync-C当感知收益。
- [next_step] 只读核对56／67训练身份与可比性，再决定是否值得提出背景讲话30dB／无添加两尺度的有限配对重复；不要据本交接启动实验。

## Relations

- continues [[Codex checkpoint - 2026-10-20T11-48-53Z - Desktop proposal template editing delivered]]
- relates_to [[研究主线与证据地图]]
- relates_to [[低尺度训练与噪声解释的证据边界]]
- relates_to [[历史研究路线与停止边界]]