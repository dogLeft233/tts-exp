---
title: Codex checkpoint - 2026-10-26T04-14-10Z - Full fit reconstruction reduced to
  one failure with transcript risk
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-26-t04-14-10-z-full-fit-reconstruction-reduced-to-one-failure-with-transcript-risk
status: open
project: tts-exp
cwd: E:/Documents/tts-exp
repo_root: E:/Documents/tts-exp
repository: dogLeft233/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-26T04:14:10Z'
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a175f1-7164-7510-ae4b-f5af8e9d0fde
trigger: compact
model: gpt-6.1-sol
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-26T04-14-10Z - Full fit reconstruction reduced to one failure with transcript risk

## Summary
完整300句统一峰值候选复验将重建失败3→1，仍未准入；同时发现S0168冻结文字可能未覆盖完整发声，撤回此前非语音区解释。

## Story
原研究要估计逐音素静态、动态、联合TTS差方向并运行11臂区分生成与评分音轨响应。用户当前限定先fit技术校准，不降门、不剔除样本，开发与确认不动。本轮在固定20条控制加3条失败上分别测整句峰值0.95、可逆局部增益、分块编码；再把统一峰值规则实际应用到全部100fit/300自然源复验。没有逐样本挑参数、混合原音或裁剪评分。共享适配器仍未改，普通声码器和前处理仅候选。

权威最新报告 runs/phoneme_direction_20261026_repair02/report.md；full_peak95_quality.json记录全部原门下的300条读回。中位STOI0.9000667047410837，S0356W0277=0.7583232237430504、S0039W0124=0.7559552087556914过0.750；S0168W0257=0.7470257940624025仍失败，无新增失败。长度/finite/RMS误差/削波/全局包络lag均通过，中位门0.850及单句0.750不变。

localize_stoi_loss.py按实际pystoi原步骤分解，整句复算误差<1e-12。3句约42%弱能量窗口贡献79%–82%的1−相关性损失，其余窗口相关性约0.91；此处弱窗口标签只是诊断，不改原40dB移静音算法。宽带包络恢复更差，小固定平移±20/40/80样本更差；局部增益仍1失败，分块编码仍3失败。电平参与两句损伤，剩余编码/解码误差尚未分离，不能说原因完全解决。

重要修正：S0168参考只有“十一月二十日消息”，整句Paraformer却识别后续网站报道内容；独立0–3秒切段为参考短句，6–9.024秒再次识别网站报道内容。具体内容未有人类或独立识别器真值，3–6秒短预测也不能当真实讲话。之前把MFA文字支持范围外统称非语音区的判断撤回；旧报告/检查点不回写，由最新正式笔记与本报告解释更正。MFA全1200条零OOV结构通过保持，但不等于文字覆盖或边界真值。未改冻结转写、截短音频、重生成TTS或剔除来源。

## Working State
隔离run在Ubuntu /home/wjj/tts-audio/tts-exp/runs/phoneme_direction_20261026_repair02/。300候选WAV和69诊断WAV仍远端；本地24份JSON/报告/源码快照SHA通过。repair01原37份交付SHA再次核对未变。所有本聊天CPU作业结束，无自动化或活跃后台实验；其它聊天生产队列未干预。

最新状态 HOLD_RECONSTRUCTION_ONE_FAILURE_AND_TRANSCRIPT_COVERAGE_RISK，numeric_protocol_frozen=false，dev/confirm false。未估计M/D/J模板、未运行新TFG或SyncNet。未完成新候选全量ASR/事件时序门，因为重建门仍失败。代码未跟踪，仓库大量其它聊天修改不属于本轮；无commit或PR。当前HEAD仅本地验证，GitHub connector核实仓库且当前分支无PR。

## Changed Files
- scripts/experiments/phoneme_direction_20261026/probe_weak_segments.py：23句三个固定候选，原音不旁路。
- scripts/experiments/phoneme_direction_20261026/calibrate_conditioned_fit.py：300句固定三分片峰值候选。
- scripts/experiments/phoneme_direction_20261026/audit_conditioned_fit.py：独立保存WAV/SHA/原门/模型pin读回。
- scripts/experiments/phoneme_direction_20261026/localize_stoi_loss.py：原STOI窗口损失分解。
- tests/test_phoneme_conditioning_20261026.py：可逆增益/静音有限值/分块hop与尾帧位置。
- basic-memory/Research/逐音素差方向算法与验证方案.md、Research/研究主线与证据地图.md、Experiments/逐音素静态动态差的拟合集技术校准与40人验证.md：更新1失败与文字覆盖风险，修正旧解释。
- runs/phoneme_direction_20261026_repair02/report.md与delivery_manifest.json：结果入口。

## Verification
远端固定torch环境17项测试通过。本机14项通过，信号模块因缺numpy/torch明确跳过；对应3项已远端实际通过。compileall与diff空白检查通过。24新交付SHA和37旧交付不变检查通过。实际red复现三句低于0.750，完整300复验仅2句转绿，S0168仍红。正式路线未通过，不称技术校准完成。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- 当前本地HEAD `2b71639c3b555293375d248350999377168d515d`，未验证GitHub可解析。
- 方案与实验笔记见Relations；最新run报告覆盖旧非语音解释。

## Observations
- [result] 原300句重建失败3→1，中位STOI0.900067；S0168仍0.747026低于原0.750，候选未准入。
- [correction] S0168的MFA文本支持范围外不能全称非语音区；ASR两种输入支持额外内容风险，但不是人工真值。
- [decision] 全部原门和全部样本保留，fit-only；开发/确认不动，当前不冻结或自动恢复11臂。
- [blocker] S0168重建与文字覆盖仍HOLD，完整拟合集校准尚未过关。
- [next_step] 确认S0168真实发声与参考完整性，再做弱片段编码器/声码器分离诊断，避免按0.750边界微调或原音残差旁路。

## Relations
- continues [[Codex checkpoint - 2026-10-26T04-00-00Z - Full fit OOV repaired and three reconstruction failures isolated]]
- relates_to [[逐音素差方向算法与验证方案]]
- relates_to [[逐音素静态动态差的拟合集技术校准与40人验证]]
