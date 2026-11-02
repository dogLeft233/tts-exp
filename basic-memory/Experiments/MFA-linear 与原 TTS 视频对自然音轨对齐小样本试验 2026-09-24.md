---
title: MFA-linear 与原 TTS 视频对自然音轨对齐小样本试验 2026-09-24
type: experiment
permalink: tts-exp/experiments/mfa-linear-与原-tts-视频对自然音轨对齐小样本试验-2026-09-24
status: concluded
date: '2026-09-24'
result: 2/2 样本中，对齐后的 MFA-linear/TTS 视频使用原自然音轨评分均未超过自然音频驱动视频直接评分。
conclusion: NAS window SyncDrift 对这两条样本没有逆转自然视频的 Sync-C 优势；不外推到其他样本或音素级 TTS 视频对齐。
report: runs/mfa_linear_natural_track_probe_20260924/report.md
tags:
- mfa-linear
- tts
- syncnet
- video-retiming
- exploratory
---

# MFA-linear 与原 TTS 视频对自然音轨对齐小样本试验

AISHELL-1 clean-MFA 配对样本 1、201；肖像 3；Wav2Lip GAN 生成 N/M/T 视频。T 先均匀匹配 N 帧数，M/T 再用 NAS window SyncDrift 默认参数对齐自然音轨。所有最终评分使用相同 SyncNet V2 官方管线，min_track=100，重新封装原始自然 PCM16 WAV；逐 cell 音频 PCM 完全一致。

## Observations

- [status] concluded
- [progress] 探索性 2 条语音、1 肖像；非泛化结论。
- [result] 样本 1：自然 N/N C=6.870；M/N 对齐前/后 5.958/5.858；T/N 均匀定长后/再对齐后 2.104/1.970。样本 201：自然 7.034；M 6.530/6.696；T 3.585/3.703。对齐后的 M、T 在 2/2 样本均未超过自然基线。
- [validation] 所有 cell 的原自然 PCM16 WAV 解码后逐样本一致、视频像素和 PTS 在最终 mux 前后不变、每样本各 cell 人脸轨长度一致。自然视频经过与 NAS 输出相同 MP4V 编码但零位移的对照 C=7.144、7.305，结论方向不变。
- [diagnosis] NAS Blinken attacked 41.6 秒案例有 38 个窗口、多种位移，本地复现 C 3.019→7.681；本试验每臂只有 2–3 个窗口且估计位移恒定，实际仅是固定帧移/恒等。M 两条 offset −2→0 仍低于自然；T 样本201原 offset=0 仍低 C，表明整体偏移不是唯一原因。T 样本1均匀定长后，39个匹配音素中13个中心与N相差>240ms；零位移转码使自然C上升约0.27，因此小幅前后差不能单归因于时间修复。
- [short_window] 固定 2 秒窗口/0.5 秒步长、±240ms 搜索的诊断：样本1/201的 T/N C 分别由5秒窗口的1.970/3.703提升为3.018/3.977，仍低于自然6.870/7.034；M/N均与5秒结果相同5.858/6.696。T短窗估计变为随时间变化，表明方法可用于短片；同一SyncNet搜索和评分，属探索性证据。
- [nas_short_preset] 原样复现 NAS 短窗净化参数 0.5秒窗口/0.1秒步长/±1000ms、batch20，在相同4条输入上官方评分：样本1 M=4.804、T=6.387（自然6.870）；样本201 M=5.965、T=4.985（自然7.034）。T1提升明显但重复/跳帧13.7%/12.2%；其余最高重复27.8%、跳帧11.1%。四条均未超过自然。最终评分音轨PCM逐样本一致；同一SyncNet搜索与评分，画质未人工验证。详情见 `runs/mfa_linear_natural_track_probe_20260924/nas_w05_s01_a1000/summary.json`。
- [limit] T 先做均匀时长匹配而非音素级时间映射；NAS 对齐和最终评分使用同一 SyncNet 权重；没有画质人工评价。样本 101 的 T/N 时长比超出预设 20% 窗口而排除。样本 201 在原数据 train split。
- [outputs] [报告](runs/mfa_linear_natural_track_probe_20260924/report.md)；[汇总](runs/mfa_linear_natural_track_probe_20260924/summary.json)；脚本 `scripts/experiments/mfa_linear_natural_track_probe.py`。

## Relations

- relates_to [[MFA-linear 视频重定时与自然音频同步 Implementation Spec]]
- relates_to [[MFA-linear 视频重定时 Sync-C 约束工程 smoke 结果 2026-09-23]]

## Changelog

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 完成两条同文本三臂视频生成、NAS 对齐、统一自然音轨 SyncNet 评分和编码敏感性对照 | September 24, 2026 | user（实验请求）；agent（执行与验证） |
| 对照 NAS 时间窗和原试验音素边界，补充低 C 未恢复的诊断 | September 24, 2026 | user（追问原因）；agent（核验） |
| 固定短窗敏感性试验：T 分数局部改善但仍低于自然，证明非长视频专用 | September 24, 2026 | user（迁移可行性追问）；agent（执行与验证） |
| 按 NAS 0.5秒/0.1秒/±1000ms 短窗参数复测四条视频并核验 PCM、重复跳帧与 Sync-C | September 24, 2026 | user（NAS短视频配置试跑）；agent（执行与验证） |
