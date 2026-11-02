---
title: Codex checkpoint - 2026-10-25T02-39-41Z - Known warp DTW concluded and native
  video comparison authorized
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-25-t02-39-41-z-known-warp-dtw-concluded-and-native-video-comparison-authorized
status: open
project: tts-exp
cwd: E:/Documents/tts-exp
repo_root: E:/Documents/tts-exp
repository: dogLeft233/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-25T02:39:41Z'
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a16cf2-be54-7ba0-bcfb-be8433abd951
trigger: compact
model: gpt-6.1-sol
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-25T02-39-41Z - Known warp DTW concluded and native video comparison authorized

## Summary
已知时间变化DTW先导完成且独立核验；用户刚授权这6来源原始自然、三种变形自然、原生TTS各自驱动视频并比较Sync-C，执行器正在核对生成配置，尚无新视频结果。

## Story
起点是怀疑Sync-C偏好TTS，前序CoGenAV对照和固定视频谱形EQ均已结论化，见上一checkpoint。最新已完成实验用人为已知局部拉伸及320ms停顿，比CoGenAV/MFCC/Mel+相同DTW找回时间的精度。六来源按历史TTS C>N筛选，12原音频×四条件，CoGenAV48次真实前向；三种DTW均远好于全局线性，但CoGenAV均值7.433ms、MFCC6.513ms、Mel9.537ms，CoGenAV只1/6来源更好。该验证没有生成视频，不能回答原生驱动优势。

用户现明确要求实际下游实验。因此新工作范围是同六来源N、N_stretch、N_pause、N_combined、所选原生T五条件各自驱动、各自音频评分，冻结相同TFG/头像/设置并核实旧视频可比性。计划优先新生成匹配30视频，最多32含重复诊断，无新TTS/训练/API，不按分数调整变形。ARS experiment-agent显式code_runner路由已派发native_warp_sync_runner；根代理随后独立核验。不能沿用先前“未授权TFG”作为阻断，现已授权。

## Working State
旧DTW权威报告runs/cogenav_dtw_pilot_20261025_attempt01/parent_report.md，完整C:/Users/dogle/.codex/tts-exp-artifacts/cogenav-dtw-pilot-20261025及远端同名run。主manifest215成员33031600bytes，SHA568b039f6e168b32f1550fb87e7791ab32be05b6155ddf3ad417d5598f707c76，父侧四文件addendum不改主清单。C是本机备份，不是网盘。

新run预定runs/native_warp_sync_pilot_20261025_attempt01，完整备份C:/Users/dogle/.codex/tts-exp-artifacts/native-warp-sync-pilot-20261025。SSH wjj@192.168.50.2，远端repo /home/wjj/tts-audio/tts-exp。本次实时读GPU322MiB/16384MiB、0%，磁盘16GB空闲；E余1207599104bytes，保留>1GiB仅小元数据，C余82.3GB。现存大量继承dirty/untracked，未提交或推送，不动其他chat工作。用户级BM配置不存在，采用项目配置。最新GitHub只读查canonical仓库成功、分支open PR空，本地SHA未确认远程可解析，不造commit链接。

## Changed Files
上一实验新增scripts/experiments/cogenav_dtw_pilot_20261025.py、cogenav_dtw_parent_audit_20261025.py及对应tests/test_cogenav_dtw_*。均本机未提交状态。新执行器预定scripts/experiments/native_warp_sync_pilot_20261025.py及对应测试，派发时尚未实施，不声称已完成。

## Verification
旧DTW父侧不导入执行器重建全部时钟、掩码、144路径投影及聚合，误差0；独立DP最优代价差max1.421e-14，18真实N/T探索路径结构通过。3执行器+2父侧契约函数直接断言PASS，未pytest。215归档成员SHA全部独立核验。父审计初版浮点边界运算顺序失败后仅修复审计算术，未改科学数据；最初tar目录变化rc1保留，最终外部打包exit0。实际N/T路径无真值，时间分歧不是准确度。最新下游实验尚未评分。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)

## Observations
- [result] 已知人工变形中CoGenAV没有相对MFCC的额外精度证据，真实N/T音素对齐仍无真值。
- [decision] 新native实验使用已保存变形波形且冻结设置，不调参最大化Sync-C；主指标完整原生时长含插入停顿，保留D与lag解释支持变化。
- [decision] 历史T>N筛选决定先导性质，不能据新T>N作总体确认；Sync-C提升也不直接证明感知同步改善。
- [next_step] 完成五条件匹配TFG生成及官方SyncNet评分，根侧独立核验每来源配对差值与归档后向用户报告。

## Relations
- continues [[Codex checkpoint - 2026-10-25T02-18-03Z - CoGenAV comparison and fixed PCM EQ concluded]]
- relates_to [[CoGenAV与MFCC及Mel的已知时间变形对齐先导 2026-10-25]]
