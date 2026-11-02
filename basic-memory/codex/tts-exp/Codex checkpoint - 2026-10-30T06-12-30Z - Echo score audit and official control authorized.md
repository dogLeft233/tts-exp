---
title: Codex checkpoint - 2026-10-30T06-12-30Z - Echo score audit and official control
  authorized
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-30-t06-12-30-z-echo-score-audit-and-official-control-authorized
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-30T06:12:30Z'
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

# Codex checkpoint - 2026-10-30T06-12-30Z - Echo score audit and official control authorized

## Summary
冻结的Echo中文20人实验已完成；历史模型身份和分数已核实，用户现授权官方配置对照，尚未启动新GPU实验。

## Story
原目标是比较同一静态图和中文20人自然/TTS在九评分器的来源配对差异。报告77及attempt01已经归档并交付：Sync-C自然4.945、TTS4.755、差−0.191，九项Holm均不显著。相同队列FlashHead自然6.143、TTS6.355。用户怀疑Echo分数低，已离线核对两个模型80输入逐字节一致、SyncNet核心代码一致、41唯一视频原始矩阵归约一致，未发现汇总或大范围偏移问题，尚未证明根因。

历史BM检索确认早期13样本是EchoMimic V2（自然2.980、TTS3.165、差+0.185），目前未找到更早的V3-Flash实测；不同版本、数据和音频处理不可当成同模型方向逆转。

用户最新明确授权按EchoMimicV3-Flash官方默认做对照，重点音量归一化与推理参数，遵守192规则、不覆盖既有结果。先读取当次手册，再新目录做单变量及联合对照；不得直接重跑硬编码attempt01包装器。

## Working State
权威结果：results/echomimic_v3_flash/chinese20_rms24_20261030_attempt01/；归档SHA256 ca8cd17cb6974016e69e3470c9cd01fb9927bb2d9d0b8eefca2f51ff3f09380b。远端个人模型/home/data2/wangjiajun/EchoMimicV3-Flash，评估/home/data2/wangjiajun/av-evaluation。只在个人目录写新实验，不改共享环境或他人任务。

当前RMS−24绕过官方−23LUFS，实际ChineseW2V do_normalize=False；原推理512²/实际61–197帧/seed42，对应官方run_flash 768²上限/81帧/seed43。默认sample_size不保证将512图放大，需核实实际尺寸。8步、CFG6、audio3、shift5、TeaCache .1/skip5已一致。须共同片段评分，分清驱动波形改变与评分音轨改变，避免截断/幅度混杂。

本机临时SSH入口C:/Users/dogle/.codex/tmp/server192_run.py运行时DPAPI解密、固定指纹。离线审计脚本C:/Users/dogle/.codex/tmp/echomimic_v3_score_diagnosis_20261030.py，输出同名目录offline_audit.json。这些是机器本地脆弱状态。

仓库1355行dirty/untracked，本轮未修改实验代码或提交。服务器尚未访问；网页工具当前无法读飞书，已找到Edge已登录手册标签，浏览器读取待继续。

## Changed Files
先前完整实现：scripts/experiments/echomimic_v3_chinese20_20261030/、对应tests，报告77和部署指南及SYNC-C-SCORES。上两次问答只读项目证据；新对照尚无实现。

## Verification
离线审计实际执行验证80输入跨模型相同、两套各41矩阵重新归约，FP64与原FP32最大误差Echo1.8521e−6、Flash2.4485e−6。Echo41唯一视频偏移均在±1帧；不能由此断言嘴型正确。新对照未运行、无新分数。历史实验验证边界见上个checkpoint及报告77。

本次git只读确认branch、SHA、状态；GitHub get_repo确认仓库，当前head PR搜索为空。当前SHA仅本地核验。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- 本地SHA `2b71639c3b555293375d248350999377168d515d`
- [Official Flash entry](https://github.com/antgroup/echomimic_v3/blob/main/run_flash.sh)
- [Server manual](https://gcnadw6o1ien.feishu.cn/wiki/Vq5gwxt51i2DVdkDhobcP4iqnkh?fromScene=spaceOverview)

## Observations
- [result] 初步离线核对未发现输入串换或Sync-C汇总错误，Echo低于Flash的现象尚待配置对照解释。
- [decision] 保留旧run与归档不可变；新对照独立目录，音量和推理参数分别干预并固定共同评分支持。
- [next_step] 浏览器读取当次服务器手册，检查个人资源与已部署代码，实施官方默认及单变量对照并汇总Sync-C。

## Relations
- continues [[Codex checkpoint - 2026-10-30T05-58-32Z - EchoMimicV3 Flash Chinese20 complete]]
- relates_to [[77-echomimic-v3-chinese20-nine-scorers]]
