---
title: Codex checkpoint - 2026-10-23T12-02-57Z - Qwen17 same10 raw norm completed
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-23-t12-02-57-z-qwen17-same10-raw-norm-completed
status: open
project: tts-exp
cwd: E:/Documents/tts-exp
repo_root: E:/Documents/tts-exp
repository: dogLeft233/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-23T12:02:57.119Z'
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a16144-7517-77a1-b372-027fba42da62
model: gpt-6.1-sol
trigger: compact
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-23T12-02-57Z - Qwen17 same10 raw norm completed

## Summary
同一云端10人上的完整1.7B与raw/norm比较已完成、独立复核并封存；1.7B对0.6B的评分增益保留，但相对自然脆弱、相对云端未明确。

## Story
原研究希望解释TTS原生Sync-C优势，用户最新要求用已部署1.7B在原云端10条来源上复用自然/0.6B/云端结果，保持下游一致、比较归一化前后且不训练。按原参考、文本、seed及实际解码参数生成30条1.7B；复用70旧norm，补130新Wav2Lip视频和官方评分，总200格。以人等权、TTS三draw先平均，三个norm主比较用98.333%调整bootstrap区间。

1.7B−0.6B raw/norm +0.663/+0.656，norm区间[+0.199,+1.213]、9/10提高；剔除旧0.6B两停止异常仍+0.418。1.7B−自然norm+0.409但严格9人区间跨零；1.7B−云端−0.154区间跨零。不把checkpoint配置差异归因纯参数量或学习任务，也不把Sync-C当独立人类同步质量。

## Working State
权威结果在runs/qwen17_level_comparison10_20261023_attempt02/report.md及analysis.json，生成/评分/CPU后处理均已停止，无待执行科学单元。远端Ubuntu wjj@192.168.50.2，E仅小镜像。完整C盘tar 1643427840字节，SHA256 4ebe82b42e2a7576e6674417d9b1a876fb70a3013039c8d271612637cb341b7a，路径C:/Users/dogle/.codex/tts-exp-artifacts/qwen17-level-comparison10-20261023/qwen17_level_comparison10_20261023_attempt02.tar。44897清单成员逐SHA与200CSV行读回PASS；archive_receipt.json为tar外回执；模型权重外置，无网盘备份声明。

attempt01合成actual0、音频门控1，原失败保留；attempt02复用30合成，仅旧0.6B/S0016/draw3 raw的8点PCM饱和例外。ROOT沿用此前保留完整来源+严格子集偏好并事前披露，无针对8点的新逐点授权回复。严格9人补充排除整个人。audio/render/official actual0；原父包装在CPU索引前结束，原SSH25151已不可恢复，父退出码/stderr/原因未知。仅补冻结CPU index actual0，未重跑forward，见execution_completion_addendum.json及original_transport_termination.json。

工作区已有大量无关dirty/untracked，不曾提交或清理。当前HEAD为本地2b71639c3b555293375d248350999377168d515d，GitHub查询422无法解析该SHA，故不构造远程提交链接；该分支GitHub PR查询返回空。脚本和测试仍untracked，须依封存source_snapshot与归档恢复，不能仅依赖git。

## Changed Files
新增scripts/experiments/qwen17_level_comparison10_20261023/执行包，以及qwen17_level_*20261023.py分析、音频/评分审计、补充分析、报告、封存与读回脚本；对应tests/test_qwen17_level_*20261023.py。更新HANDOFF.md当前完成状态、basic-memory/docs/deployment/qwen3-tts-1p7b.md批量显存事实。通过MCP创建最终实验笔记并更新研究主线与证据地图、SYNC-C-SCORES，均实际读回。

## Verification
100原波形到200PCM逐样点独立回放PASS，70旧norm桥接精确一致，30新生成参考codes/seed/解码/EOS审计PASS。200评分/track、90主+42补充统计独立复算PASS，最大误差3.55e-15。5个相关测试文件共9项通过；root_final_test_receipt.json保存实际命令与结果。统计PNG实际查看PASS。归档全成员SHA与CSV绑定实际读回PASS，readback_source_sha256 123fe2555209cfd6a29e1727aacd26cac656ddca87ec964ff7310b39f27ffac3。无训练；单部署smoke显存不能代表批量峰值，本轮allocated9.33GiB/reserved14.80GiB。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- runs/qwen17_level_comparison10_20261023_attempt02/report.md
- runs/qwen17_level_comparison10_20261023_attempt02/archive_receipt.json
- HANDOFF.md

## Observations
- [result] 1.7B对0.6B增益在raw/norm、资格补充与严格9人中保留；自然优势不稳健，云端差异未明确。
- [decision] 全draw作为主结果，处理后资格及严格9人为补充；保留前次失败和父包装未知退出凭据。
- [decision] 暂不训练、不扩样；本次执行完成，旧训练与head路线继续暂停。
- [next_step] 向用户交付报告和模型/归一化对照；后续研究方向须基于这些证据另行确定，不自动重跑。

## Relations
- summarizes [[同一10人上的Qwen1.7B与音量归一化对照2026-10-23]]
- relates_to [[研究主线与证据地图]]
