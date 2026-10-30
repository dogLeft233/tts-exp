---
title: MFA 本地安装修复状态
type: report
permalink: tts-exp/docs/experiments/mfa-本地安装修复状态
date: '2026-08-13'
tags:
- mfa
- tooling
- alignment
- installation
- aishell
---

# MFA 本地安装修复状态

## Context

本项目的 MFA 用于中文自然语音与 TTS 的 MFA-linear 对齐，当前生产基线要求使用 `mandarin_mfa` 词典和声学模型。2026-08-13 检查本机安装后确认：问题不在正在使用的 Conda 环境，而在另一套损坏的 pip/venv 安装。

## 修复结果

已验证可用环境为 `[redacted-local-path]`：

- MFA 2.2.17
- Python 3.10.20
- `joblib 1.3.2`（满足 `<1.4`）
- `setuptools 80.10.2`（满足 `<81`）
- PostgreSQL 依赖可用
- `mandarin_mfa` acoustic 与 dictionary 模型均已安装

原 `[redacted-local-path]` 是 MFA 3.4.1 + Python 3.12，启动时报 `ModuleNotFoundError: No module named '_kalpy'`，不能通过普通 pip 安装可靠修复。该目录已保留为可回滚备份 `[redacted-local-path]`。原路径现在是兼容入口，调用已验证的 Conda MFA，并显式把 `[redacted-local-path]` 加入 PATH；`[redacted-local-path]` 也已指向该入口。

## 验证

- `mfa version` 返回 `2.2.17`
- `mfa model list acoustic` 返回 `['mandarin_mfa']`
- 通过原兼容路径对一条中文 16 kHz 音频完成真实 MFA 对齐，成功导出 TextGrid，耗时约 30 秒
- 本机硬件为 32 CPU、31 GiB RAM、Tesla V100 16 GiB；对齐主要使用 CPU，脚本默认 `--num_jobs 4` 保持稳定
- 本机代理 `[redacted-ip]:7890` 可用；本次修复未重复下载模型，未修改已有 MFA 数据与对齐结果

## 后续使用

优先使用：

```bash
conda activate mfa
mfa version
```

项目脚本可继续使用默认路径 `[redacted-local-path]`，或使用已修复的 `[redacted-local-path]` 兼容路径。不要恢复或继续使用备份中的 MFA 3.4.1 pip 环境。

## Observations

- [problem] pip/venv MFA 3.4.1 缺少 `_kalpy`，启动即失败 #mfa #tooling
- [solution] 保留 Conda MFA 2.2.17 作为唯一工作安装，并为旧路径提供带 PATH 注入的兼容入口 #mfa #conda
- [decision] 不删除损坏 venv，保留 `[redacted-local-path]` 以便回滚 #reversibility
- [learning] MFA 2.2.17 需要 Python 3.10、`joblib<1.4`、`setuptools<81`、PostgreSQL 依赖，且直接调用时必须包含环境 `bin` 在 PATH 中 #mfa #tooling
- [verification] 真实中文单文件对齐成功，当前 MFA 安装可用于 AISHELL cohort 工作 #validation

## Relations

- relates_to [[多样域中文试点：MFA-linear 在自发对话域失败]]
- relates_to [[MFA-linear 冻结 VC 重合成与下游验证]]
- supports [[推进 TTS 节奏因素实验]]