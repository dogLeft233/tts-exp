---
title: Codex checkpoint - 2026-10-27T06-27-12Z - AVS public backbone weights clarification
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-27-t06-27-12-z-avs-public-backbone-weights-clarification
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-27T06:27:12Z'
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a17bfa-f7da-7ca3-9538-8fd44e01fb91
trigger: compact
model: gpt-6.1-sol
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-27T06-27-12Z - AVS public backbone weights clarification

## Summary
旁支资料核对确认 AVS 可以依赖公开的 AV-HuBERT 权重实现，但尚未确认作者独立评测包及其确切 checkpoint。

## Story
本旁支只回答 SyncReward、PEAVS 和 TFG 专用评分的论文问题，不承接主线程实验或运行任务。最新问题是“AVS没有开源权重吗”。核查原论文与 AV-HuBERT 官方仓库，区分公开骨干模型、评分公式和精确复现所需配置。AVS_u/m/v 使用 AV-HuBERT 表征计算余弦相似度，无需另训独立评分头；论文使用经唇读微调的模型。官方 AV-HuBERT 提供预训练与微调权重，但多个版本不能随意认作 AVS 原文的唯一版本。

## Working State
没有下载、部署、评分或修改代码。工作区有大量既有修改和未跟踪文件，属于其他工作，未接触。此记录只保存文献查证；主实验状态没有重新核实。配置 primaryProject=tts-exp，coding profile。Startup Router 返回全局路由文本；所引用 note-instructions 在项目与 main 均未找到，未据此推断额外规则。

## Verification
已打开 AV-HuBERT 官方仓库与 AVS 原论文 HTML；此前搜索没有定位到可确认的独立官方 AVS 评测实现，搜索未命中不能证明其不存在。已只读核对 Git 根目录、分支、SHA 和 origin。gh 未安装，无法确认当前 PR 或远端 commit，故不构造 PR/commit 链接。没有运行测试，当前工作是资料核对。

## References
- [AV-HuBERT 官方代码与权重入口](https://github.com/facebookresearch/av_hubert)
- [AV-HuBERT 模型页面](https://facebookresearch.github.io/av_hubert/)
- [AVS 原论文及指标公式](https://arxiv.org/html/2405.04327v1#S4.SS1)

## Observations
- [result] AV-HuBERT 有官方公开权重，AVS 公式没有独立学习的评分头参数。
- [decision] 若自行实现，必须称作基于明确选择的官方 AV-HuBERT checkpoint 的 AVS 复现，不能宣称直接调用官方 AVS 评分包或精确复现原文数值。
- [blocker] 尚未确认 AVS 作者所用的确切 checkpoint 文件与完整官方评测预处理实现。
- [next_step] 回答用户权重可用性并解释骨干权重与官方评分实现的区别；不启动主线程实验。
