---
title: Codex checkpoint - 2026-09-30T15-40-00Z - Fresh TTS mechanism experiment and
  GPU controls
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-09-30-t15-40-00-z-fresh-tts-mechanism-experiment-and-gpu-controls
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/tts-mechanism-20260930
git_sha: c4a199890bd529928ec34b9be6da86a10a737c97
started: 2026-09-30 15:40:00+00:00
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
trigger: compact
model: gpt-6.1-sol
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-09-30T15-40-00Z - Fresh TTS mechanism experiment and GPU controls

## Summary

已恢复历史研究主线并验证新服务器可运行确定性 Wav2Lip/SyncNet；正式研究转入未参与旧实验的中文配对数据、固定 RMS 四格与校准来源方向干预。

## Story

用户最初要从本地代码、Git 和 BM 恢复旧机器未同步的研究路线，随后授权自主完成 TTS 增强 TFG 的原因研究、重跑实验、使用本地百度盘和网络数据。root 维护理论与因果主线；三个 gpt-6-luna/max 子代理执行代码、数据和实验。用户明确允许所提供的阿里云 key 调用 Qwen TTS。认证已成功，但首次两条中文试跑在声纹注册阶段返回 400 InvalidParameter，local_recovery 正在修正格式；这是工程错误，不是效果阴性。

历史同 PCM Wav2Lip/cloud Qwen 的优势与旧 Ditto codec 混杂已从真实报告区分；9/27 的动态统计仍缺完整 runs。网盘 WORLD holdout74 只有 A 特征，不能冒充拥有 V 和最终统计的实验。为取得直接正向因果证据，openspec/changes/confirm-tts-source-mechanism-20260930/spec.md 固定新 AISHELL 校准20/评估80、生成/评分 RMS 四格、校准拟合的 TTS 512维 z 方向、严格排除测得 level secant 的干预与三个随机剂量对照。全部自然评分音频和时钟保持原始；预设 C 与固定 lag C_anchor 的四项 99% 联合门槛，独立视觉端点另测。抽象来源方向之后还需物理 source/filter 或局部动态范围实验，不能以排除清单结束。

## Working State

服务器准确入口是 root@connect.weste.seetacloud.com，SSH port 12349；不能把旧 docs 的 westc 拼写当成这台服务器。密码只在运行时内存注入，未存笔记、代码或日志。GPU 为 RTX 4080 SUPER，torch CUDA 实际可用；现有 Wav2Lip/SyncNet 与权重均已核验。新产物隔离于 /root/autodl-tmp/tts-research-20260930/。本机 .codex/scratch/remote_bridge.py 的 loopback SSH 进程和认证 token 是易失状态，不是持久访问凭证；恢复时检查存活或由用户消息中的凭证重连。remote_recovery 是唯一 GPU 执行者。

本地 BM 项目在 basic-memory/，Git 忽略；有 Startup Router、实验指令、Task 与来源方向实验笔记。新分支 codex/tts-mechanism-20260930 共享给全部子代理，尚无新提交/PR。runs/recovered_20260930/ 是本地忽略产物。正式 AISHELL 官方 revision bbe295d530192a4cd41644b711c9aecd087df653：校准 S0052–S0061，每人两句；评估 S0062–S0101，每人两句，共100条自然音频，后续100条 Qwen 自克隆。数据挑选不读取效果。50个说话人的下载接近完成，local_recovery 负责冻结 manifest 和小于8MiB的上传分片。两条本地中文 pilot 仅用于工程链路。

## Changed Files

未跟踪的新文件包括 .codex/basic-memory.json、openspec/changes/confirm-tts-source-mechanism-20260930/spec.md、scripts/experiments/qwen_tts_pair_prepare_20260930.py、scripts/experiments/tts_mechanism_visual_20260930.py、其对应 tests/experiments/test_tts_mechanism_visual_20260930.py，以及正在编写的 scripts/experiments/portable_source_axis_20260930/。临时 SSH 桥接/client 不应连同 token 或凭证提交。

## Verification

远端官方例子12.68秒：Wav2Lip重复生成4.78/4.64秒，两次MP4文件和像素完全一致，A/V/距离矩阵差0。官方 SyncNet +5帧音频延迟使 offset −3→−8；guard20固定anchor C 9.502→0.829，而最优lag confidence仍约9.2。自动找峰分数不能作为视频已同步的证据。完整summary/provenance在 runs/recovered_20260930/remote/control_summary.json。修复的 librosa positional mel shim 外置，未修改既有环境；不宣称与旧历史前端逐字一致。

mechanism_audit 报告独立视觉脚本6项 pytest通过、语法/diff检查通过；root要求再与8条件和orth_plus主要比较对齐。Root核验阿里云voice-list HTTP200。Qwen CLI ffmpeg预检成功并绑定Windows二进制hash/version；两个pilot声纹注册400记录在 runs/recovered_20260930/local/qwen_source_axis/pilot_tts/tts_meta.json。正式科学效果尚未测量。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- openspec/changes/confirm-tts-source-mechanism-20260930/spec.md：冻结的研究协议与后续物理机制要求。
- runs/recovered_20260930/remote/control_summary.json：工程控制真实结果。
- runs/recovered_20260930/local/qwen_source_axis/pilot_manifest.json：两条中文pilot；不属于科学校准/评估。

## Observations

- [result] 服务器可执行确定性生成与官方SyncNet评分，已通过重复和已知延迟控制。
- [decision] fresh cohort只用于新确认，旧9/27报告模板中的门槛不是已经核验的正向结果。
- [decision] FLOAT32评分PCM以float64乘32768进入MFCC，不round、不转int16、不clip；避免人为消掉level差异。
- [decision] 严格来源干预仅排除实测level secant，不宣称排除所有非线性响度响应；视觉运动和感知同步分别取证。
- [blocker] Qwen pilot声纹注册400参数错误正在修复；旧9/27完整统计未恢复；独立视觉phone对齐仍需准备。
- [next_step] 修复并完成两条Qwen中文pilot后送远端四格工程检查，同时冻结新100条自然manifest并生成配对TTS，启动校准轴与正式干预。

## Relations

- continues [[Codex checkpoint - 2026-09-30T14-25-18Z - TTS mechanism recovery and remote continuation]]
- relates_to [[解释 TTS 的 TFG 与 SyncNet 优势]]
- relates_to [[TTS 机制研究服务器 weste 12349]]
- relates_to [[TTS 来源方向与声学机制确认]]