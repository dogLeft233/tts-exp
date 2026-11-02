---
title: Codex checkpoint - 2026-10-16T10-52-00Z - Training amplitude design and Ubuntu
  history verified
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-16-t10-52-00-z-training-amplitude-design-and-ubuntu-history-verified
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: c6c1e7bd005abeaa4a2cc52991391ef2aaec5223
started: 2026-10-16 10:52:00+00:00
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

# Codex checkpoint - 2026-10-16T10-52-00Z - Training amplitude design and Ubuntu history verified

## Summary

核实 Ubuntu 曾接入 NAS SyncDrift 并复验视频对齐，但未找到 TTS 替轨净化后训练 SyncTalk 的完成证据；下一研究设计转向训练电平与测试电平的交叉控制。

## Story

原目标是有明确贡献、适合投屏的中文 proposal。前次检查点后，proposal 3.1 和研究地图已扩充跨 TFG、跨 TTS 和数据集证据；原生配对分数不直接代表增强。用户已授权将六环幅度实验交给“推进实验”，本会话发送了目标并确认启动：20位新说话人，RAW/FIXED各两次，加四个非零增益档，总160视频，固定原自然音轨评分。原40人机制和15人来源电平结果复用并核验，不重新发明干预；P2停顿修复独立保留。不得把已派发等同完成。

最新用户提出：推理幅度响应是否延伸到训练，是否可把视频换TTS并净化后训练SyncTalk；要求先设计音量实验、选择数据和代码。用户补充 Ubuntu 应有下载代码和会话，本轮已通过现有SSH密钥只读核查192.168.50.2。找到2026-09-24会话、NAS代码和实际10人报告，澄清那是推理后的帧重排，不是新SyncTalk模型训练。

## Working State

Ubuntu代码 /home/wjj/tts-audio/nas-cch-code/voice_def/tools/purify_syncdrift.py，索引 voice_def/cch_419/PURIFICATION_CODE_INDEX.md；项目接入 scripts/purify_syncdrift.py。会话 /home/wjj/.codex/sessions/2026/09/24/rollout-2026-09-24T18-38-52-01a0d2ff-2932-7641-9180-2a089cef30bd.jsonl。旧会话含敏感凭据，后续检索必须过滤，不转存原文。NAS历史路径出现SyncTalk/data_aaai，不等于本项目训练记录。

实际读取 runs/mfa_linear_nas_short_expanded_20260924/report.md：Wav2Lip自然视频C5.940；MFA-linear配自然音轨5.243→NAS后4.259；原TTS视频先均匀定长再配自然音轨1.409→3.200，重复帧42.9%。10人中一边界评分单独标记；去除后TTS8/9提高但仍不支持优于自然。此实验与真实视频换TTS音轨后训练不同。

公开官方SyncTalk支持AVE/DeepSpeech/HuBERT；AVE的utils.py使用80维mel、绝对dB映射并clip[-4,4]，provider.py冻结AudioEncoder并缓存特征。官方提供处理后May数据及训练代码，适合单身份可行性试验，尚未部署测试。HuBERT引用配置do_normalize=true，不能默认波形增益传入编码。provider.py默认test复用val，val只取100帧，正式实验需独立连续段测试与上下文隔离。

拟建议：同初始化和训练预算下比较原电平、固定安全正增益、随机增益，以及固定电平归一化基线；全部在同一自然测试语句的低/原/高电平上比较。归一化部署规则另报告，不能暗换测试链。固定原音频做SyncNet辅评，用真实保留帧的嘴部重建误差作独立参考端点。先一个身份预检，再多身份和多种子；这能回答训练条件效应，不能预先保证质量改善。固定编码器意味着研究的是其表示下的生成器训练，不是重新训练全部音频前端。增强须从波形重新提特征，不能乘缓存编码。

## Changed Files

此前本轮已编辑 basic-memory/Research/tts-tfg-resynthesis-proposal-zh.md 与 研究主线与证据地图.md：3.1现有14个TFG覆盖、四英文TTS配对表和附录证据。当前训练调研未改源码/方案文件、未启动新训练。本地大量继承dirty/untracked工作由推进实验等会话共享，未清理或提交。

## Verification

项目和main记忆全文检索SyncTalk/净化/前置静默；Ubuntu Codex会话、净化代码索引及10人报告实际只读核对。官方仓库README、provider.py、utils.py、HuBERT处理配置和CVPR2025静默捷径论文已读取；这是源码/文献核对，不是训练环境验证。论文是Circumventing Shortcuts in Audio-visual Deepfake Detection Datasets with Unsupervised Learning，前置静默25–30ms，已讨论幅度捷径。分类标签捷径不能直接移作生成训练偏置结论。

Git分支/SHA/root及status实际检查；GitHub API确认canonical repository与该分支PR为空。没有运行新科学实验或源码测试。BM两种身份搜索完成，最新同会话checkpoint是08:43:05并实际核对；10:15:49是推进实验另一session，不接错谱系。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)；当前SHA仅作为本地核验，不构造未验证远端commit链接。
- [SyncTalk官方代码及数据](https://github.com/ZiqiaoPeng/SyncTalk)
- [SyncTalk AVE前处理](https://raw.githubusercontent.com/ZiqiaoPeng/SyncTalk/main/nerf_triplane/utils.py)
- [静默捷径论文](https://arxiv.org/html/2412.00175v2)
- 推进实验 threadId=01a0f26b-a3e7-7f91-b1e8-46813543042f，local；160视频工作与新训练提案分开。

## Observations

- [result] Ubuntu已有NAS净化代码和10人推理对齐结果，所查记录没有完成SyncTalk再训练证据。
- [decision] 新训练方案先研究输入电平依赖与跨电平稳健性；不预设放大越多越好，不把Sync-C改善自动写成质量收益或训练捷径证据。
- [blocker] SyncTalk训练环境、官方May资产下载和真实训练成本尚未实测，无法承诺质量正收益。
- [next_step] 向用户交付历史核查与训练×测试电平方案，明确推荐SyncTalk AVE加官方May作工程pilot及多身份扩展；训练执行前先做实际前端响应和基线训练资格检查。

## Relations

- continues [[Codex checkpoint - 2026-10-16T08-43-05Z - Quality evaluation feasibility and proposal boundary]]
- relates_to [[研究主线与证据地图]]
- relates_to [[NAS 短窗口 10 说话人扩大样本试验 2026-09-24]]