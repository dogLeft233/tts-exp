---
title: Codex checkpoint - 2026-10-30T00-55-00Z - Chinese20 full experiment and audible
  normalization
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-30-t00-55-00-z-chinese20-full-experiment-and-audible-normalization
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-30T00:55:00Z'
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a1880e-73d5-7863-a38b-20bcdf7d5ebc
model: gpt-6.1-sol
trigger: compact
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-30T00-55-00Z - Chinese20 full experiment and audible normalization

## Summary
用户已授权中文20人完整FlashHead生成、九评分器与配对比较，并追加要求将自然和TTS音量归一化至可听水平；生产运行尚未启动。

## Story
先前在192个人目录完成FlashHead Pro与九评分器实际单片验收，部署凭据和源码保持冻结。异步答复随后明确只使用中文20源20N+60既有TTS，active_cohort.json已排除英文；原评估机192.168.50.2也经只读SSH确认在线。最新请求从部署扩展为完整实验，不应停在计划或单片验收。先按server-192 skill重读飞书1048手册，核对个人目录、nohup/tmux、共享conda与GPU索引规定，再检查实时资源、原音频和科学协议。新增音量处理必须另存派生音频并贯穿生成与评分，不能覆盖规范原件或只给试听视频加增益。

## Working State
权威部署指南 [[server192-nine-scorers]]，本地 results/soulx_flashhead_pro/server192_nine_scorers_20261030/README.md 与active_cohort.json。远端TFG /home/data2/wangjiajun/SoulX-FlashHead-Pro；评分根 /home/data2/wangjiajun/av-evaluation。私有SSH运行器 C:/Users/dogle/.codex/tmp/server192_run.py 用本机DPAPI，凭证不复制。既有九项全部部署PASS不需重装。仓库大量原有dirty/untracked文件未清理，本轮尚未修改代码。

## Verification
本轮重新SSH确认wangjiajun/HOME=/home/data2/wangjiajun，共享个人盘剩2.6T。8卡均有他人任务；A6000约8.5–9.8GiB占用，需单UUID和资源限制，不能杀其他任务。全部80条音频只读测量：RMS约0.0376838（-28.477dBFS），时长2.48–7.919秒。部分低于官方5秒分类器和4.75秒SyncReward中心窗，因此应明确UNMEASURABLE和配对完整来源数，禁止静音填充冒充同协议结果。TTS三draw多为完全相同波形，统计必须以20来源为单位，记录重复哈希而非60独立样本。尚未选择/执行具体新响度目标、生成或评分。
Git branch/SHA/status及用户主机已读取，GitHub只读确认仓库canonical URL，当前head PR_COUNT=0。CodeGraph查询已执行（新脚本可能未索引）。没有新测试结果。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- 本地SHA `2b71639c3b555293375d248350999377168d515d`，未验证远程可解析。
- [1048手册](https://gcnadw6o1ien.feishu.cn/wiki/Vq5gwxt51i2DVdkDhobcP4iqnkh?fromScene=spaceOverview)

## Observations
- [decision] 本轮只跑中文20来源，20N+60既有TTS；每来源汇总三draw再做配对统计。
- [decision] 原始规范音频冻结；音量处理另建派生输入，生成和评分必须绑定同一PCM。
- [next_step] 确定不削波的统一音量目标并保存协议、制作派生音频及批量控制器，单UUID后台生成，再跑九评分器、核验并汇总来源级配对结果。

## Relations
- continues [[Codex checkpoint - 2026-10-29T17-21-00Z - FlashHead nine scorer deployment verified]]
- documents [[server192-nine-scorers]]
