---
title: TTS 机制研究服务器 weste 12349
type: note
permalink: tts-exp/codex/remember/tts-机制研究服务器-weste-12349
tags:
- codex
- manual-capture
---

# TTS 机制研究服务器 weste 12349

用户于 2026-09-30 指定这台服务器继续 TTS 增强 TFG 的机制研究，允许使用历史实验数据、重跑实验和实施新实验。

连接：`ssh -p 12349 root@connect.weste.seetacloud.com`。用户认为已有 Wav2Lip 和 SyncNet；实际路径、GPU 与模型状态须现场核验。不要误用旧文档中的 westc 主机。

密码仅存在用户消息与本次临时内存连接中，遵守 AGENTS.md 不写入笔记、代码、日志。压缩后先恢复此服务器入口和活动任务，再检查临时 SSH 桥接进程是否仍存活。

## Observations

- [server] root@connect.weste.seetacloud.com:12349，由用户授权用于本项目研究。
- [workflow] root 维护理论和实验主线；实验与代码交给 gpt-6-luna、max 推理的 subagent。
- [data] 本机 CloudDrive B:\experiment-archive 与 B:\tts-exp-project 是可使用的历史实验数据入口。

## Relations

- relates_to [[解释 TTS 的 TFG 与 SyncNet 优势]]