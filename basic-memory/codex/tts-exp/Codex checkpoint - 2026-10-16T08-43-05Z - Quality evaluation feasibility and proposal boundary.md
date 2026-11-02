---
title: Codex checkpoint - 2026-10-16T08-43-05Z - Quality evaluation feasibility and
  proposal boundary
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-16-t08-43-05-z-quality-evaluation-feasibility-and-proposal-boundary
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: c6c1e7bd005abeaa4a2cc52991391ef2aaec5223
started: 2026-10-16 08:43:05+00:00
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a139e9-67ce-7b41-90f2-5b7019492a69
trigger: compact
model: gpt-6-astra
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-16T08-43-05Z - Quality evaluation feasibility and proposal boundary

## Summary

用户要求有充分可行性依据的质量评估；文献与历史实验尚不能保证检出幅度干预的细微感知收益。

## Story

原任务是将项目整理成有明确贡献、适合投屏的中文 proposal。上一交接后已重写 proposal 与研究地图投稿段，保留 Abstract/Method/Experiments、短句、Mermaid 和关键表格：原生现象、幅度通路、MFA-linear 停顿诊断与修复。按用户明确授权，已给“推进实验”会话发送 P0 资产核查、P1 最多80视频幅度独立复现、P2 最多18视频既有停顿病例复核的目标指令。

随后用户认为 MFA-linear 的问题过于局部，又认为幅度机制的实际价值不充分。讨论已承认 Q3 不必作为主线，但尚未再次编辑 proposal。用户最新要求是调研能可靠评定微小质量收益的方法；若没有足够把握，应明确说难以实现。本轮核对独立读唇、真实口部轨迹以及新评估论文，不能把跨模型人类排名相关性转成同模型小干预的有效性保证。

## Working State

主要文件：basic-memory/Research/tts-tfg-resynthesis-proposal-zh.md；basic-memory/Research/研究主线与证据地图.md。这两份仍含 MFA Q3 和 P0/P1/P2，不代表用户已接受其最终定位。当前调研没有修改这两份文件，没有启动 GPU、读唇或人工评价。

历史证据入口：
- basic-memory/Experiments/电平生成效应的独立视觉内容校准 2026-09-27.md：66eval/15音频说话人，视频独立 CMLR 内容 ΔQ=0.031992，95%区间[-0.025682,0.097053]；ΔM同样跨零。N_RAW校准刚过、N_LEVEL辅助门未过，绝对CER约105–118%。不能当可靠准确读唇或零效应证明。
- basic-memory/Experiments/GRID 固定电平 aperture 相关性独立标签评估 2026-09-27.md：16标签，主Δr=0.005123，95%区间[-0.016918,0.027786]；同期ΔSync-C=0.143、99%区间正。支持评分与独立端点分开报告，不支持质量等效。
- basic-memory/Experiments/完整同机 RMS 对照的分数分解与边缘支持.md：3模型均值C增加主要来自距离曲线背景项，不能直接叫时序更准。
- basic-memory/docs/experiments/25-tts-tfg-mechanism.md：40人有效频谱坐标与编码方向的固定自然音轨响应，运动改变不等于准确性。

推进实验 threadId=01a0f26b-a3e7-7f91-b1e8-46813543042f，host=local。已发送目标并确认启动；最近讨论没有向该会话发送修改或取消，不能声称已停止P2。当前会话没有创建目标。

## Verification

此前改稿结构检查：334行、3个Mermaid、6个代码围栏、5个details配对、所有本地证据链接存在。当前仅阅读实验报告与公开论文/官方仓库，没有重算原实验或执行测试。Git核实本地SHA与分支，9修改项和1004未跟踪项；未清理、提交。gh不可用，远端PR/commit状态未确认，SHA仅本地证据。

Basic Memory CLI可用；按短标题读取Startup Router曾返回main同名笔记，改用精确tts-exp/instructions/startup-router已验证正确。项目配置路径为本仓库basic-memory。两类session元数据检索均空，说明不能把搜索空结果当无历史；此前checkpoint本地文件已读且身份吻合。MCP未提供，当前通过CLI非覆盖创建并需核对落盘。

## References

- [THEval论文](https://arxiv.org/html/2511.04520v4)：17模型整体真实感排名rho0.870；Lip-Sync单项rho0.404，95%区间[-0.143,0.775]；该单项是口部开合与音量包络差，不是逐音素正确性。
- [THEval官方代码](https://github.com/Newbyl/THEval)：已公开；整体分数需本数据集真实视频参考统计，不能直接借用其他数据集GT行。
- [AV-HuBERT评估论文](https://arxiv.org/html/2405.04327v1)：AVS_u音视频独立提特征余弦；AVS_v需要真实参考视频；鲁棒于平移不等于已验证细微幅度收益。
- [PVSync预印本](https://arxiv.org/html/2610.09223v1)：13模型排名rho0.83；代码/benchmark声明录用后发布，本轮未找到可确认官方发布；不能作为即刻可执行的关键依赖。

## Observations

- [decision] 当前不选某个新指标作为已具充分把握的质量收益主终点；可运行、相关于跨模型人评、足够灵敏于微小处理效应是三个不同条件。
- [decision] 不以人眼难区分或独立指标不显著宣称质量不变，也不继续挑指标直到显著。
- [blocker] 当前尚无经本项目验证、可可靠衡量幅度处理微小质量收益的独立方法；原MFA run本机资产缺口仍未由本会话确认恢复。
- [next_step] 向用户给出有来源的评估可行性结论：撤回保证能用盲评补齐链条的暗示，区分已证实评分响应与尚未证实质量收益；等用户决定研究定位，不自行追加实验。

## Relations

- continues [[Codex checkpoint - 2026-10-16T07-52-12Z - Negative results and transfer paper framing]]
- relates_to [[研究主线与证据地图]]
- relates_to [[电平生成效应的独立视觉内容校准 2026-09-27]]