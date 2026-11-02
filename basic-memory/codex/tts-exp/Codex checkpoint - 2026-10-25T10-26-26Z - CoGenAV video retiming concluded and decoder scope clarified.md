---
title: Codex checkpoint - 2026-10-25T10-26-26Z - CoGenAV video retiming concluded
  and decoder scope clarified
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-25-t10-26-26-z-co-gen-av-video-retiming-concluded-and-decoder-scope-clarified
status: open
project: tts-exp
cwd: E:/Documents/tts-exp
repo_root: E:/Documents/tts-exp
repository: dogLeft233/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-25T10:26:26Z'
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a16cf2-be54-7ba0-bcfb-be8433abd951
model: gpt-6.1-sol
trigger: compact
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-25T10-26-26Z - CoGenAV video retiming concluded and decoder scope clarified

## Summary
CoGenAV视频重定时先导已经完成；最新澄清区分视频重定时、历史音频重合成，以及论文文字解码器。

## Story
用户希望解释TTS相对自然语音的Sync-C优势，依次完成官方CoGenAV权重检查、20来源评分对照、固定视频EQ、已知时间变形DTW、变形自然音频原生驱动，以及将TTS驱动视频重定时回自然音频。最新六来源视频实验固定自然PCM，只按CoGenAV跨模态特征路径重复/跳过TTS视频帧，官方SyncNet均未超过自然原生基线。这条路线没有合成对齐后的TTS波形。

随后用户问是否做过先对齐TTS音频再驱动视频：历史确有MFA-linear/WavLM L6特征重采样、冻结HiFi-GAN重合成、Wav2Lip及SyncNet。早期单说话人n15相对自然ΔC+0.964；5说话人n25复核ΔC−0.069，区间跨0，不能称稳定优势。近期CoGenAV+DTW仅编码器特征及路径；尚未进行CoGenAV路径生成新音频再驱动视频。最新问题是论文是否有音频解码器，本次核对原论文2.2.2/2.3.2：主体冻结Whisper ASR头生成文字，Generative指文字生成；下游AVSS/AVSE接AV-SepFormer分离/增强头，不能据此认为主体有任意CoGenAV特征到语音的通用解码器。不要把“没有波形解码器”误写成“没有任何解码器”。

## Working State
视频实验权威报告 runs/cogenav_video_retime_20261025_attempt01/parent_report.md。平均C：自然6.649、同编码自然6.660、原生TTS7.448、CoGenAV跨模态DTW5.814、音频DTW5.360、MFCC5.146、线性2.183；四重定时条件均0/6超过自然。六来源按历史T>N筛选，仅先导证据，不代表感知同步真值。
完整远端 /home/wjj/tts-audio/tts-exp/runs/cogenav_video_retime_20261025_attempt01；完整本机C:/Users/dogle/.codex/tts-exp-artifacts/cogenav-video-retime-20261025，非百度网盘。E仅小元数据，空间约1.2GB属于上次测量，本轮未重查。无新增实验授权或在途工作。当前大量继承dirty/untracked，不触及其他chat训练工作；无提交推送。用户级BM配置不存在，采用项目.codex/basic-memory.json。

## Changed Files
前序本chat新增未跟踪 scripts/experiments/cogenav_video_retime_20261025.py、cogenav_video_retime_parent_audit_20261025.py及对应tests/test_cogenav_video_retime*。更早DTW/native实验脚本与报告仍在对应run。当前问答未修改实验代码，仅新增本交接笔记。

## Verification
前序已完成独立18个cost矩阵和DP重建、30视频帧/PTS/PCM/窗口支持/官方距离标量核验，DP代价最大差7.105e-14；2执行器+3父侧直接契约断言PASS，未pytest。8201归档成员逐SHA通过；主manifest SHA44547bd7e5ead16c34ff21b88a85a25749d1b3f5cc547e1bbf60861d24b27fbb，父侧7成员addendum独立封存。科学计算本轮未重跑。
本轮重新读取论文及git状态/分支/SHA，GitHub REST验证canonical仓库及当前分支open PR为空；本地SHA远程可解析性未确认，不构造commit链接。上一同chat checkpoint的agent/session_id已全文核实；其他WIMHF/source200 checkpoint属于其他chat，不接续。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- [Paper sections 2.2.2 and 2.3.2](https://arxiv.org/html/2505.03186v2)

## Observations
- [result] 固定CoGenAV视频重定时未将原生TTS评分优势转移到自然音频评分。
- [decision] 区分主体文字解码器、下游分离增强头、历史外接声码器，不把自定义DTW称官方对齐算法。
- [next_step] 回答论文解码器问题；等待用户下一项实验要求，不自行启动CoGenAV音频重合成实验。

## Relations
- continues [[Codex checkpoint - 2026-10-25T03-05-00Z - Native warp comparison concluded and natural audio video alignment authorized]]
- relates_to [[CoGenAV特征重定时TTS视频并换回自然音频的六来源先导 2026-10-25]]
- relates_to [[MFA-linear 冻结 VC 重合成与下游验证]]
