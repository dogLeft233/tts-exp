---
title: Codex checkpoint - 2026-10-26T03-03-47Z - Phoneme direction design confirmed
  and implementation authorized
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-26-t03-03-47-z-phoneme-direction-design-confirmed-and-implementation-authorized
status: open
project: tts-exp
cwd: E:/Documents/tts-exp
repo_root: E:/Documents/tts-exp
repository: dogLeft233/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-26T03:03:47.5686754Z'
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a175f1-7164-7510-ae4b-f5af8e9d0fde
trigger: compact
model: gpt-6.1-sol
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-26T03-03-47Z - Phoneme direction design confirmed and implementation authorized

## Summary
逐音素静态、动态与联合差迁移方案已确认；用户现在授权先拟合集校准并冻结参数，再运行40人完整11臂自动评价，60人确认集保持封存。

## Story
目标是从TTS与自然音频的编码差异获得可迁移、可解释的生成端证据，同时拆开评分音轨响应。权威方案是 [[逐音素差方向算法与验证方案]]，术语在 [[GLOSSARY]]。Q1–Q11全部接受：冻结WavLM-Large L6→prematched HiFi-GAN→Wav2Lip，100人拟合模板，40人各预定一句，完整11臂、剂量±0.25，联合正向相对原自然音轨驱动为唯一主检验。静态/动态用共同支持与相位网格，动态迁移回自然帧后再加窗中心化；不复制TTS时长。

此前仅授权设计，方案与词表已更新并读回。最新用户明确授权实施， supersedes方案最后一段“仅更新不执行”的历史边界。先技术校准：MFA模型/音素位置配对、上游真实L6和帧时钟、prematched声码器重建、内容与时序、共同支持/随机控制/统计及自动口部可测性。未通过则记录并停止，不靠删句、降低门槛或换模型救回结果。全部数值在开发效果前冻结，才进入40人自动评价；开发结果也不自动解封确认。

## Working State
本地工作区E:/Documents/tts-exp，当前分支和SHA见frontmatter。Git存在大量先前修改与未跟踪文件；本聊天此前只改方案和词表，不混入或撤回其它工作。GitHub查询确认仓库，当前分支未搜到PR；gh未安装，改用GitHub connector。

Ubuntu wjj@192.168.50.2可用，实际repo /home/wjj/tts-audio/tts-exp。2026-10-26约11:06北京时间：磁盘约55GB空闲，V100 16GB中约13GB被既有数据准备占用。不得中断其它聊天的数据队列；先清单与CPU工作。此前10:39已验100fit/300source/1200 N+L17 units齐备；当前总队列进度须刷新。现有SyncNet/字级FunASR缓存不是WavLM/MFA。

MFA3.4.1在 /home/wjj/miniconda3/envs/mfa3，Chinese专用工作目录mfa3_chinese_rerun_20260815；旧2.2与3.x词典不兼容曾造成spn/假覆盖，不能复用。适配器scripts/wavlm_knn_vc_adapter.py冻结kNN-VC revision c616845c4e309e24d5927f15adbdf277a3d65358、hop320、16k、1024D、prematch_g_02500000.pt。历史普通g_02500000.pt重建STOI .928不验证prematched版本。

## Changed Files
- basic-memory/Research/逐音素差方向算法与验证方案.md：完整11臂、联合唯一主检验、相位估计和自然时钟动态迁移、失败处理。
- basic-memory/docs/reference/GLOSSARY.md：静态/动态/联合差与评分/生成响应术语。

## Verification
此前方案全文KB读回与本地检查通过：11臂、440逻辑视频、唯一主检验、自然时钟迁移和失败边界均存在，无过时待用户选择正文。文档修改未运行代码测试。当前回合只读核验SSH、磁盘、GPU和代码适配器，尚未新对齐、提取、生成或评分。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)；当前SHA仅本地证据，未验证GitHub可解析，不构造commit链接。
- 权威设计与词表见Relations。

## Observations
- [decision] 采用完整11臂；联合正向对原自然驱动为唯一主检验，次要拆解保持探索性。
- [decision] 最新授权实施：拟合集技术校准→参数冻结→40人开发自动评价；确认集不动。
- [blocker] 技术校准未完成；GPU当前被既有数据队列占用，不能直接启动竞争性重建任务。
- [next_step] 刷新fit资产清单和运行环境，创建隔离run与fit-only校准入口，先检查MFA和重建实际合同。

## Relations
- relates_to [[逐音素差方向算法与验证方案]]
- relates_to [[GLOSSARY]]
