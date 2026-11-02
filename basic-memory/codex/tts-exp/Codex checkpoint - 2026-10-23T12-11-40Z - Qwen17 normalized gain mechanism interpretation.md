---
title: Codex checkpoint - 2026-10-23T12-11-40Z - Qwen17 normalized gain mechanism
  interpretation
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-23-t12-11-40-z-qwen17-normalized-gain-mechanism-interpretation
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-23T12:11:40Z'
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a16144-7517-77a1-b372-027fba42da62
model: gpt-6.1-sol
trigger: compact
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-23T12-11-40Z - Qwen17 normalized gain mechanism interpretation

## Summary
已完成1.7B同10人raw/norm实验；本轮核对架构与SyncNet公式，发现归一化1.7B对0.6B的+0.656分主要体现为时间偏移距离对照与最佳匹配的间隔扩大，尚未识别声学因果机制。

## Story
原目标是解释TTS原生Sync-C优势。上一阶段按用户要求复用云端10人自然/0.6B/云端，完成30条1.7B合成、130新视频/官方评分、总200单元，未训练。用户现问为什么换更大TTS后归一化Sync-C增益明显，是否可从音频和架构解释。此次只读BM canonical报告、部署记录、旧15号表征关联及历史停止边界，再核对官方Qwen3-TTS论文和两版config，以及SyncNet源码。未启动任何合成、视频、评分、训练或云调用。

norm L17-L06 +0.655867，调整98.333%CI[+0.198598,+1.212833]、9/10提高；排除两停止异常仍+0.418，说明整体RMS和两异常不足以解释。相对自然+0.409的证据脆弱，严格9人CI跨零；不可把问题前提写成普遍大模型TTS优势。

官方config：两版Talker均28层，hidden/intermediate由1024/3072增为2048/6144；speaker encoder enc_dim1024到2048。残余code predictor均5层、hidden1024、intermediate3072，仅结构相同不等于权重相同。论文12Hz体系由主干预测第0码本、MTP补其余15码本，codec重建波形。本地部署codec SHA相同、30参考codes独立核对一致。因此优先假设是生成token序列及speaker conditioning的联合变化，而不是换了codec；训练配方/数据/权重差异未单因素隔离，不能说纯参数量因果。

SyncNet定义 C=median(mdist)-min(mdist)，D=min(mdist)。从封存person_means.csv按10人等权，norm0.6B C6.860900,D7.3637667,median14.2246667；norm1.7B C7.5167667,D7.2455000,median14.7622667。故ΔC=+0.6558667=最佳距离降低+0.1182667+偏移距离中位数升高+0.5376000。逐人代数残差最大8.88e-16；独立从all_draw_scores.csv聚合相同（每人3draw均等）。这是打印三位分数的描述性恒等式，不是机制因果比例、82%评分器偏差或新显著性检验，也未在合格draw子集拆分。它建议先看偏移曲线、活跃语音/停顿及局部声学结构，避免将C提升直接解释为时间误差减少。

## Working State
原正式run runs/qwen17_level_comparison10_20261023_attempt02/及完整C盘tar保持封存。此前44897文件读回/200score审计/9tests是前阶段实际结果，本轮没有重跑测试。E盘只小镜像，完整tar在C:/Users/dogle/.codex/tts-exp-artifacts/qwen17-level-comparison10-20261023/qwen17_level_comparison10_20261023_attempt02.tar，SHA4ebe82b42e2a7576e6674417d9b1a876fb70a3013039c8d271612637cb341b7a。旧8点PCM例外及父包装退出原因/码未知仍保留，详见上一检查点和正式报告；原科学执行已完结。

工作区大量继承dirty/untracked，git status本轮只读确认，无提交或清理。分支codex/cross-tfg-audio-head-20261003，本地HEAD 2b71639c3b555293375d248350999377168d515d；此前GitHub不能解析该SHA，仍不构造远程commit链接。本轮GitHub canonical repo只读确认，分支PR查询空。无代码改动；本检查点是本轮唯一新增持久笔记。

## Verification
Startup Router→研究地图→实验指令已读。正式1.7B实验笔记、部署codec哈希、历史15号与停止边界已读。官方论文arXiv2601.15621v1和Qwen两版config在线核对。CodeGraph先于代码检索调用；未唯一命中新实验文件，返回现有SyncNet evaluate源码conf=median-min。Shell只读计算来源CSV SHA256 5bf50f5ad7025025406803cf2549d2e8f75f21967680e54f7ec0263b3dfd2a57，逐人公式核验与独立draw聚合实际完成。尚未比较这批1.7/0.6的MFA音素时长、局部有声RMS、谱过渡、F0或人工口型；不能写成已观察到“发音更清楚/时间更规则”。旧表征可分性和时长并未稳定预测下游分差，不能重新当既定机制。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- [Qwen3-TTS technical report](https://arxiv.org/html/2601.15621v1)
- [0.6B official config](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base/raw/main/config.json)
- [1.7B official config](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base/raw/main/config.json)
- [SyncNet confidence definition](https://github.com/joonson/syncnet_python/blob/master/SyncNetInstance.py)
- runs/qwen17_level_comparison10_20261023_attempt02/report.md
- runs/qwen17_level_comparison10_20261023_attempt02/person_means.csv

## Observations
- [result] 归一化L17-L06的C差+0.656在分数恒等式上分为最佳距离降低+0.118和偏移距离中位数升高+0.538；非因果占比。
- [decision] 以局部时序/声学结构及生成器-评价器共同可辨识度作为候选解释，不直接归因模型规模、codec或人类准确度。
- [decision] 整段RMS相同未控制有声段响度、停顿占比、频带分配、音素时长与局部动态；这些是待测候选。
- [blocker] 缺本队列局部声学测量与独立口型证据，尚不能区分真实生成改善和评价端响应。
- [next_step] 如继续机制研究，先复用已有波形/距离曲线做有声段与停顿、音素局部动态及匹配距离诊断，再选择单因素且保时序干预；当前只建议，不自动启动GPU或训练。

## Relations
- continues [[Codex checkpoint - 2026-10-23T12-02-57Z - Qwen17 same10 raw norm completed]]
- interprets [[同一10人上的Qwen1.7B与音量归一化对照2026-10-23]]
- relates_to [[研究主线与证据地图]]
