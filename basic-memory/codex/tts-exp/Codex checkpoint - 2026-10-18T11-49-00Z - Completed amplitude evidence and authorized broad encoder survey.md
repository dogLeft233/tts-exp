---
title: Codex checkpoint - 2026-10-18T11-49-00Z - Completed amplitude evidence and
  authorized broad encoder survey
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-18-t11-49-00-z-completed-amplitude-evidence-and-authorized-broad-encoder-survey
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: 2026-10-18 11:49:00+00:00
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

# Codex checkpoint - 2026-10-18T11-49-00Z - Completed amplitude evidence and authorized broad encoder survey

## Summary
报告46的幅度验证已经完成；用户现在授权ROOT直接扩展主流语音编码器的方向共性实验，使用网盘解决空间，不再委派子代理。

## Story
项目研究从TTS原生高分走向自然PCM保持下的生成机制。报告46完成20人160视频：固定自然self评价，FIXED主guard20 ΔC+0.049、95%[-0.005,+0.106]，14/20正；预定+3dB +0.067、四剂量同时95%[+0.013,+0.121]，15/20正。独立嘴部开度和速度确实变化，但不是准确性或感知改善证明。既有完整320距离矩阵与160轨迹验证已完成，不重跑。

本轮先遵循Router读取最新研究地图，发现其他会话已补算方向共性并完成留出生成。Wav2Lip CAL20配对音素中TTS跨人改变量cos从RAW0.329降至统一RMS0.015，纯+3dB为0.884；另20人纯增益队列cos0.812。留出10人共同位移生成+0.055、真实增益+0.122，未通过±0.05等效门。原HANDOFF停留10月16日，不能用它否定更新的地图和正式实验。

最新用户要求主流编码器扩展；计划在同一CAL20/10人配对音素上分开RAW TTS、双方FIXED TTS和FIXED±3dB，按编码器原生前处理、各层坐标、真实时间中心计算跨人及留一人几何。第一阶段只有编码前向，不声称TFG或Sync-C收益，不恢复暂停增强头或P2。

## Working State
Ubuntu wjj@192.168.50.2根盘543MiB、SHM148MiB，SyncTalk Python PID816719占3418MiB；保留其他会话训练。本机RTX4060约2.7GiB占用，计划单模型串行。新隔离依赖target位于F:/Datasets/tts-exp/encoder_survey_20261018/deps，继承仅已安装torch的独立解释器C:/Users/dogle/.codex/tts-exp-artifacts/gain-common-direction-20261018/env/Scripts/python.exe，安装仍进行中，不能称编码器已运行。

空间已按用户授权处理：逐片验证9个May完成实验传输分片、并核对保留完整archive SHA后，仅删除冗余part文件，释放4301034008字节；F盘空闲约5.34GB。完整tar.gz和展开remote科学目录都保留，已验证云分片也保留。精确receipt为F:/Datasets/tts-exp/synctalk_may_training_pilot_20261017/redundant_transport_cleanup_20261018.json；不修改CloudDrive配置或旧拒绝删除对象。

## Verification
读取Router/实验指令/最新地图及两份方向报告；BM限定本chat identity搜索确认最新前驱为10月16日11:21:59，10月18日其他checkpoint不是本chat lineage。Git实查本地SHA/branch及九项继承tracked dirty；GitHub只读API返回canonical仓库、当前head没有PR。直接检查本机GPU及磁盘、Ubuntu当前进程；没有实现测试或新编码前向。旧报告46实际QA以正式报告为准，不将历史执行重复计作本轮测试。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- 当前SHA `2b71639c3b555293375d248350999377168d515d`为本地提交，未核验GitHub可解析，未构造远程commit链接。
- `basic-memory/docs/experiments/46-amplitude-gain-independent-validation.md`
- `runs/tts_direction_commonality_20261018/{analyze.py,result.json,qa.json}`
- `runs/gain_shared_direction_20261018/`及其外部C盘完整产物指针。

## Observations
- [decision] 用户最新明确不用子代理，ROOT亲自编写、执行和汇总本次编码器实验。
- [decision] 统一原始语料和统计定义，保留模型原生归一化；不把方向几何推广成生成质量。
- [blocker] 多个物理盘空间紧张；已清理明确冗余传输副本，继续采用串行模型和可重下载缓存策略。
- [next_step] 固定代表性编码器名单和同音素协议，完成隔离依赖与官方权重下载，运行首个编码器并检查实际产物。

## Relations
- continues [[Codex checkpoint - 2026-10-16T11-21-59Z - Amplitude six-link validation completed with responder coverage]]
- relates_to [[研究主线与证据地图]]
- relates_to [[TTS 编码方向共性与电平混淆的配对音素补算]]
- relates_to [[增益共同方向与残差的留出说话人生成验证]]