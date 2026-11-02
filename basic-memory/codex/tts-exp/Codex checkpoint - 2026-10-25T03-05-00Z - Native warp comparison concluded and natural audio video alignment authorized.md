---
title: Codex checkpoint - 2026-10-25T03-05-00Z - Native warp comparison concluded
  and natural audio video alignment authorized
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-25-t03-05-00-z-native-warp-comparison-concluded-and-natural-audio-video-alignment-authorized
status: open
project: tts-exp
cwd: E:/Documents/tts-exp
repo_root: E:/Documents/tts-exp
repository: dogLeft233/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-25T03:05:00Z'
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

# Codex checkpoint - 2026-10-25T03-05-00Z - Native warp comparison concluded and natural audio video alignment authorized

## Summary
六来源五条件原生驱动对照已完成并独立核验；最新用户要求将TTS等增强驱动视频对齐回自然音频，比较自然音频原生Sync-C。

## Story
用户怀疑Sync-C对TTS偏好。CoGenAV官方权重及Eq1对照、固定视频谱形干预、已知时间变形DTW先导均已完成。最新完成的native实验重新生成同六来源N/stretch/pause/combined/T五条件30视频及2条N重复，用官方Wav2Lip GAN和官方SyncNet各自驱动各自评分。N平均6.649，stretch6.113(Δ−.535)，pause6.323(−.325)，combined5.905(−.744)，T7.448(+.800)。任意时间变形没有复现TTS平均优势；三种变形分别0/6、1/6、0/6提分，T6/6。来源按历史T>N筛选，这不是随机总体确认。

用户现明确授权后续：用TTS或有原生增强的音频驱动视频，将视频与自然音频对齐，再比较自然原生评分。尚未实现或评分；先核对论文是否有可直接使用的推理对齐算法。此前核对CoGenAV论文仅同步训练对/同帧余弦，并无现成异语速DTW，不能把自定义重定时称作论文官方算法。可复用刚完成fresh T视频与fresh N对照，固定自然音频PCM，只重定时视频，避免新合成或按Sync-C调参。

## Working State
权威报告runs/native_warp_sync_pilot_20261025_attempt01/parent_report.md；对应实验笔记见Relations。完整远端/home/wjj/tts-audio/tts-exp/runs/native_warp_sync_pilot_20261025_attempt01，C:/Users/dogle/.codex/tts-exp-artifacts/native-warp-sync-pilot-20261025完整本机备份，非百度网盘；E仅小元数据，余约1.2GB，保留>1GiB。最新终态GPU322MiB/0%，新任务尚未实时探测。SSH wjj@192.168.50.2。大量继承dirty/untracked，未提交推送，当前SHA仅本地验证，不构造远程commit链接。

## Changed Files
新增未跟踪scripts/experiments/native_warp_sync_pilot_20261025.py、native_warp_sync_parent_audit_20261025.py及tests/test_native_warp_sync_*。主run8907文件272888789bytes，manifest SHA0cfe0d189ebbb8d0c3bf55251cf6bba9201bcfb40aa5f82d0d8361203762baea；父侧4文件addendum SHA9aaab07548e648983175df19a8106b7a2ef655123eb302a979211c3fb1642f77，保留冻结清单。

## Verification
执行器2与父侧2契约函数直接断言通过，未pytest。根独立核验全部32源/PCM/样本数、mux帧精确保持、唯一face轨frame0、官方activesd距离、实际评分PCM与日志；按官方连续[31,windows]float32重算，执行器旧布局最大差2.384e-6，主表用父canonical。2条N重复视频字节及C完全一致。fresh对历史批次最大C差.025，主比较全部fresh。主8907成员C备份父/子各逐SHA通过；父addendum远端/C4/4通过。GitHub REST本次确认canonical仓库，当前分支open PR空。新任务尚无实验结果。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- [CoGenAV paper](https://arxiv.org/html/2505.03186v2)

## Observations
- [result] 已知任意局部拉伸与停顿未复现原生TTS平均Sync-C优势；不能据此排除真实TTS节奏作用或广义评分偏好。
- [decision] 保存科学失败与冻结输入，不以Sync-C挑选对齐参数；Sync-C改善不自动等于感知同步改善。
- [next_step] 核对论文与官方代码，实施同六来源增强视频对齐自然音频的有界先导，保持自然PCM固定并与fresh自然原生及线性对照评分。

## Relations
- continues [[Codex checkpoint - 2026-10-25T02-39-41Z - Known warp DTW concluded and native video comparison authorized]]
- relates_to [[自然音频已知时间变形的原生驱动SyncNet对照 2026-10-25]]
