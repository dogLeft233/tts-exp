---
title: Codex checkpoint - 2026-10-30T03-05-00Z - EchoMimicV3 Flash Chinese20 experiment
  authorized
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-30-t03-05-00-z-echo-mimic-v3-flash-chinese20-experiment-authorized
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-30T03:05:00Z'
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

# Codex checkpoint - 2026-10-30T03-05-00Z - EchoMimicV3 Flash Chinese20 experiment authorized

## Summary
FlashHead 中文20人九评分器实验已完成；用户现已授权在192个人目录部署 EchoMimicV3-Flash 并用相同静态图、音频和评分协议完成独立实验。

## Story
原始目标是比较自然语音与TTS驱动TFG的评分差异。权威已完成结果是 [[76-flashhead-chinese20-nine-scorers]]，不能重跑或覆盖冻结证据。用户要求追加业内常用、24GB可推理模型，先建议LatentSync1.6，核对其官方CLI后确认只支持视频。用户明确要求静态图，最终选 EchoMimicV3-Flash，最新请求已授权部署和完整实验，不再停留于候选推荐。

计划复用已有20自然+60TTS、41唯一音频、同一3.png，RMS−24dBFS派生PCM直接校验复用，保留12同句/8扩展来源分层、三TTS来源均值、九评分器原协议与支持集。新模型仅生成新视频和新分数，不拼接历史评分。需核验Flash官方长音频生成及帧数支持；不能把run_flash.sh默认81帧当作覆盖所有完整音频。

## Working State
已重新读取 server-192/SKILL.md、references/personal-deployment.md、remote-gpu-deploy。已当次通过浏览器读取飞书手册：个人区/home/data2/wangjiajun，共享Anaconda只建独立环境，程序tmux/nohup，下载先查公共权重，GPU索引存在映射差异。IAB滚动接口报错后改用已打开Edge手册，读取GPU索引和注意事项成功。

本轮SSH只读确认wangjiajun、HOME及目录所有者；个人盘余2.5T、公共119G。公共model有限maxdepth3查Echo/Wan/W2V目录无输出，不表示全机不存在。所有GPU有他人驻留进程，A6000多数约5.8GB占用，必须启动前重新检查，只用单UUID、保守CPU/显存；不终止他人进程。未部署EchoMimic、未启动新GPU作业。

完成FlashHead远端 /home/data2/wangjiajun/av-evaluation/jobs/flashhead_chinese20_rms24_20261030_attempt01；本地 results/soulx_flashhead_pro/chinese20_rms24_20261030_attempt01/ 为交付入口，evidence.tar.gz SHA256 7cf9916be529e8f0f4099bd346a4cae168bb590bd254af9266cfee213ee7a700。复用输入和scorer环境要验证实际路径/哈希。

机器本地脆弱入口 C:/Users/dogle/.codex/tmp/server192_run.py，stdin bash及--upload/--download，凭证运行时DPAPI解密、固定SSH指纹；不得打印凭证。仓库原有大量dirty/untracked，不清理。

## Changed Files
本轮尚无实验代码改动。前轮 scripts/experiments/flashhead_chinese20_20261030/ 和 tests/test_flashhead_chinese20_20261030.py 为未提交可复用来源。CodeGraph已先调用，但未索引当前FlashHead目录，误匹配其他controller；之后可直接读目标实际文件，勿以误匹配代码为当前实验入口。

## Verification
本轮服务器账号/磁盘/GPU只读检查通过，手册通过Edge读取。官方README和run_flash.sh已联网核验：Flash8步、中文wav2vec2、原生图片和音频、官方12GB宣称；run_flash默认768²、81帧、25fps、CFG6/AudioCFG3、seed43。尚无本机Echo推理、依赖安装、测试或九评分器结果，12GB仅上游声明。
前轮FlashHead80媒体及2234证据文件哈希PASS、25组独立统计读回PASS、4测试/11编译PASS，具体边界见报告76和上一checkpoint。本轮未重做这些验证。
GitHub只读get_repo确认dogLeft233/tts-exp；当前head PR搜索为空。当前SHA仅本地核验，不构造远程commit链接。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- [EchoMimicV3 official](https://github.com/antgroup/echomimic_v3)
- [Flash entry](https://github.com/antgroup/echomimic_v3/blob/main/run_flash.sh)
- [Server manual](https://gcnadw6o1ien.feishu.cn/wiki/Vq5gwxt51i2DVdkDhobcP4iqnkh?fromScene=spaceOverview)
- 本地SHA `2b71639c3b555293375d248350999377168d515d`，未验证远程可解析。

## Observations
- [decision] 用户最终选择EchoMimicV3-Flash原生单图输入，已授权完整中文20人九评分器实验。
- [decision] 沿用冻结3.png、RMS−24dBFS PCM和来源配对协议；新生成器独立目录，不改FlashHead已完成证据。
- [result] 当次手册读取和SSH资源检查成功；新模型部署尚未开始。
- [next_step] 获取并审查官方infer_flash长音频路径与权重清单，在个人目录建立独立环境和校验下载，完成单样本后运行80逻辑视频及九评分器。

## Relations
- continues [[Codex checkpoint - 2026-10-30T02-47-10Z - Completed Chinese20 evidence and next TFG candidates]]
- documents [[76-flashhead-chinese20-nine-scorers]]
