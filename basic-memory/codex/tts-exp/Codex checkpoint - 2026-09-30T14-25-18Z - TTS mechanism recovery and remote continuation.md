---
title: Codex checkpoint - 2026-09-30T14-25-18Z - TTS mechanism recovery and remote
  continuation
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-09-30-t14-25-18-z-tts-mechanism-recovery-and-remote-continuation
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: main
git_sha: c4a199890bd529928ec34b9be6da86a10a737c97
started: '2026-09-30T14:25:18Z'
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

# Codex checkpoint - 2026-09-30T14-25-18Z - TTS mechanism recovery and remote continuation

## Summary

本地 Basic Memory 已配置并从 Git 恢复 TTS/TFG 机制研究路线；用户新授权服务器和网盘数据，研究正进入产物恢复与机制实验阶段。

## Story

原任务是回答项目最新进展，并在旧电脑无法启动、记忆未同步时从代码和 Git 找出研究路线。当前目标已扩大为继续完成机制研究：root 维护理论、假设与实验主线，实验和代码交给 gpt-6-luna、max 推理的 subagent；可使用历史数据并重跑实验。用户授权 root@connect.weste.seetacloud.com:12349，要求服务器入口跨上下文压缩保留。

已先读 Startup Router 与实验指令。历史路线从原生 TTS 高分复现转到协议审计、自然音频替换、G/E/I 四格因果干预，再到固定 RMS、mel 前端分区、source-axis/phase 和 WORLD source/filter 分解。CONTEXT 的 codec 修正改变了旧 Ditto 高分的解释；同 PCM 的 Wav2Lip/云 Qwen 优势仍存在。9/27 的若干动态数值需要实际 runs，不能用报告模板代替。优先恢复真实摘要、统计与验证，再决定下一轮正向机制测试。SyncNet 分数、视觉运动和人工感知是分别需要证据的端点。

## Working State

稳定入口为 CONTEXT.md、AGENTS.md、openspec/ 和 scripts/experiments/。本地 HEAD 已通过远端引用和 GitHub commit API 核实是 origin/main 的 c4a1998；没有 main 的开放 PR。CodeGraph 目录存在但没有可用索引，故按规则使用 rg，未自行建立索引。

机器本地状态：新建 BM 项目 tts-exp 指向 basic-memory/；此目录被 Git 忽略。仅 .codex/ 未跟踪，源代码尚未修改。本地初始无 runs/、HANDOFF.md。旧机器仍不可访问，但新服务器可作为替代执行入口，连接尚待完成。CloudDrive B: 已发现 experiment-archive 和 tts-exp-project；sync-20260929 轻量包含部分9/26报告，WORLD holdout74目录含 audio/features，完整9/27数值还在恢复。不要把 holdout74 与 natural holdout80 混用。

已启动三个 Luna max 子任务：remote_recovery 唯一管理 GPU/远端连接，local_recovery 负责网盘及最新源码路线，mechanism_audit 独立审阅因果设计。root 统一写 BM。准备临时 loopback SSH 桥接器，密码只由 root 注入内存，不写文件/笔记/日志；不要依赖桥接进程作为持久凭据。

## Changed Files

.codex/basic-memory.json 为未跟踪的项目配置：primaryProject=tts-exp、mixed、coding、repository=dogLeft233/tts-exp、captureEvents=true、checkpointOnCompact=true。basic-memory/ 下新建 Startup Router、实验指令、Coding Session/Decision/Task schemas、活动 Task 与服务器记忆；都是本机忽略文件，未提交。捕获 hook 状态显示5条待处理 inbox，未 flush，不等于知识笔记已自动写入。

## Verification

BM 项目可读写，Task schema 验证通过（1 pass、0 warning、0 error）；Router 到实验指令 relation 可解析。当前 checkpoint 前搜 coding_session、codex_session 与 legacy codex_session_id，无同会话早期 checkpoint。已读回活动 Task。Git status、branch、root、SHA、origin/main read-only 核验完成；GitHub API 确认仓库 canonical URL 和 SHA，无 main 开放 PR。gh 不存在，改用 API。未运行任何模型实验或源码测试；服务器 CUDA 和实际产物尚未核验。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- CONTEXT.md：9/26 PCM/codec、跨队列、视觉、pre/post mel 与 clock 复核入口。
- scripts/experiments/report_tts_native_spectrum_cross_20260926.py：长期谱匹配的历史记录与实际 summary 路径。
- scripts/experiments/tts_world_source_filter_statistics_20260928.py 与 scoring：草案，不等于正式结果。

## Observations

- [result] 本地 BM 设置完成且项目/全局分离，项目记忆统一写 tts-exp。
- [decision] root 管理论与主线，实验代码由 gpt-6-luna/max 子代理执行，remote_recovery 是唯一 GPU 执行者。
- [decision] 先从真实 run 恢复数字与控制，再冻结下一轮交叉干预，分别讨论 G、E、交互与独立视觉。
- [blocker] 本地原始 runs 缺失；新服务器连接、CUDA 和9/27完整摘要待核验，网盘索引含历史悬空链接。
- [next_step] 建立新服务器内存 SSH 会话，恢复固定 RMS、mel 分区、source-axis/phase 与 WORLD 的实际 run 证据，随后由 root 冻结机制实验并交远端子代理执行。

## Relations

- relates_to [[解释 TTS 的 TFG 与 SyncNet 优势]]
- relates_to [[TTS 机制研究服务器 weste 12349]]