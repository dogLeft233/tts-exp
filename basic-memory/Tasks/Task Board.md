---
title: Task Board
type: note
permalink: tts-exp/tasks/task-board
tags:
- task-board
- index
---

# Task Board

本页索引未结任务，不表示任务正在运行。2026-10-22 根据实验报告及后续暂停记录整理；不启动任务、不改变自动化、不把未完成目标标为完成。

近期结果先读 [[研究交接总览]]。May 54–67 的已完成实验从正式报告查结论，不重新列成待办；噪声训练 67 已交付，当前没有由本次整理新增的执行队列。

## 已明确暂停或执行受阻

| 任务 | 当前有效状态／边界 | 恢复或查证入口 |
|---|---|---|
| [[运行Patrick的RMS归一化AVE三档粗训练并评测]] | blocked：用户暂停，三档 coarse7 保存；coarse10／新评分未完成 | [[Patrick RMS归一化同初始化独立复跑]] |
| [[暂停 Chris 并精调 Patrick 三档及27格]] | paused：Patrick fine14 断点；fine17／新评分未完成，Chris 暂停，Cathy 未启动 | 任务内 `paused.json`／完整断点入口 |
| [[试跑 Patrick Chris Cathy 的 FP16 粗训练]] | 原任务元数据 in-progress；其继续调度被后续暂停记录覆盖，勿自动继续 | [[暂停 Chris 并精调 Patrick 三档及27格]] |
| [[tts-exp/tasks/native-axis-factor-pilot-20261013]] | paused：用户已暂停增强头及原生坐标后续 | [[接收端局部构音响应坐标]] |
| [[tts-exp/codex/tts-exp/proposal-validation-p0-p1-p2-20261016]] | 幅度实验已完成；P2 停放，原总目标未结 | [[46-amplitude-gain-independent-validation]] |
| [[准备并核验四个新身份的 SyncTalk 完整训练数据]] | 数据准入通过；原 FP16 数值失败，24 模型／完整评分未完成 | [[SyncTalk 四新身份官方完整训练电平实验 Spec]] |
| [[核查 SyncTalk FP16 适配与官方环境]] | 保护修订与局部定位已交付；完整上游轨迹解释未结，非重训许可 | [[SyncTalk FP16 非有限训练失败的适配与官方环境核查]] |

## 历史未结任务：保留入口，执行前查原证据

这些任务的旧 open／in-progress／blocked 标签不自动等于当前优先级或运行状态。只整理导航，不据此推断全部已停止或已完成。

| 任务 | 原记录状态 | 说明 |
|---|---|---|
| [[A1 hard-coordinate endpoint objective v3]] | in-progress | 历史目标修订，先核父任务与原 gate |
| [[MFA-linear real-video SyncNet prototype plan]] | in-progress | 历史对齐原型 |
| [[对齐 Wav2Lip–SyncNet proxy 与官方评估 policy]] | open | 代理与正式端点资格，不能当新科学结果 |
| [[建立可学习音频对齐模型]] | in-progress | 正文已暂停 residual 训练；旧目标未结 |
| [[对齐 SyncNet loss 与官方评分]] | in-progress | 先核已有校准结果 |
| [[执行 Exp 31 Week 1 特征诊断 (CKA + 线性探针)]] | in-progress | 历史特征诊断 |
| [[执行 replacement audio-head 泛化与 TTS 贡献验证]] | open | 泛化证据读正式报告30及后续修正 |
| [[执行 TTS 视觉教师到 replacement-safe 音频头链路]] | blocked | 资格不足不能当阴性结论 |
| [[执行 LRS3 TTS Visual Advantage Audit]] | blocked | 支持集／媒体资格分别核对 |
| [[Wav2Lip replacement-aware cross-attention v2 修复]] | blocked | 原固定样本 gate 失败保留 |
| [[固定真实视频 CEM 对齐验证]] | blocked | 不能自动恢复旧搜索 |
| [[完成 TTS 原生增益归因续跑]] | blocked | A 已复验；B 缺可核验 LeapTalk 配置，未执行 |

## Relations

- tracks [[研究交接总览]]
- relates_to [[历史研究路线与停止边界]]
- relates_to [[Patrick RMS归一化同初始化独立复跑]]
- relates_to [[暂停 Chris 并精调 Patrick 三档及27格]]