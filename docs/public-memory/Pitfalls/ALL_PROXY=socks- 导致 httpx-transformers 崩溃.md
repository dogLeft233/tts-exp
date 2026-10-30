---
title: ALL_PROXY=socks:// 导致 httpx/transformers 崩溃
type: pitfall
permalink: tts-exp/pitfalls/all-proxy-socks-导致-httpx-transformers-崩溃
tags:
- pitfall
- proxy
- httpx
- transformers
---

# ALL_PROXY=socks:// 导致 httpx/transformers 崩溃

## 现象

环境变量 `ALL_PROXY=socks://[redacted-ip]:7890/` 时,httpx 抛 `ValueError: Unknown scheme for proxy URL URL('socks://...')`,任何基于 httpx 的工具(如 Basic Memory MCP、litellm)启动即崩;transformers/huggingface_hub 也出现下载失败。

## 原因

httpx 不认 `socks://` scheme,只接受 `socks5://`/`socks5h://` 或 `http://`。系统里同时有 `http_proxy` 和 `ALL_PROXY`(socks)时,httpx 优先用 ALL_PROXY。

## 修复

- MCP 配置(MCP 的 environment 块)覆盖: `ALL_PROXY: http://[redacted-ip]:7890/`, `all_proxy: http://[redacted-ip]:7890/`
- 命令行跑模型:取消代理 `unset ALL_PROXY all_proxy` 或用 `HF_HUB_OFFLINE=1`

## Observations
- [area] env
- [symptom] httpx ValueError: Unknown scheme for proxy URL
- [cause] ALL_PROXY 用了 socks:// scheme,httpx 不兼容
- [fix] 覆盖为 http:// 或 unset

## Relations
- relates_to [[tts-exp]]
- relates_to [[Basic Memory 技能与连接验证]]
