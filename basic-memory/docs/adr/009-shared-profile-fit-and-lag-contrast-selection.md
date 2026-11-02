---
title: 009-shared-profile-fit-and-lag-contrast-selection
type: decision
permalink: tts-exp/docs/adr/009-shared-profile-fit-and-lag-contrast-selection
git_sha: cc7d42b9884bec49b0fa05459e201fa8eaf3ad80
decided: '2026-10-03T19:12:17.1697574+00:00'
project: tts-exp
status: open
branch: codex/cross-tfg-audio-head-20261003
repository: dogLeft233/tts-exp
tags:
- audio-head
- shared-profile
- selection
---

# 009-shared-profile-fit-and-lag-contrast-selection

## Decision

并行推进低维共享频谱控制器：同一组12个接收者局部参数用于Wav2Lip、MuseTalk、Ditto，不针对模型改变波形参数。仅在已暴露历史fit8句/4人/单seed的三模型360行上筛选最多3个候选，再进入固定已暴露dev20句/10人/3seed；候选与配置最终冻结后才读取40人确认集与heldout模型效果。全程增强音只驱动生成，恢复原自然PCM评分和播放，不改变音频/视频时钟。

## Context and rationale

实际Wav2Lip360开发中固定零时刻训练v3较RMS使D0下降0.264、Canchor提高0.153，但主Cbest下降0.105/B下降0.111。降低零时刻距离本身不足以保持对错误滞后的区分。v3独立嘴部轨迹重算显示开口标准差与速度RMS略增，因此不能把分数下降解释为简单开口运动变平。局部压缩较RMS Cbest提高0.033，是有用的接收者局部控制线索；这些是开发结果，还不是跨模型或确认结论。

报告25的供体均值/轨迹移植说明接收者相容性重要。新family以局部log幅度减高斯平滑谱的对比、轻微log频率倾斜和可选ratio0.90局部压缩进行有界扰动，不搬运供体谱形。正实幅度mask保持谱阶段工作波形的STFT相位；局部压缩发生在STFT前，不宣称原始波形相位完全不变。mask每frame零均值并缩放至±3dB，固定RMS0.0376838172675746/统一peak0.98，峰衰减实际激活时记录达成RMS，样本数完全保持。此family尚无实际跨模型分数，属于可运行候选。

## Selection contract

所有15方法使用同一模型/clip/seed的完整31lag公共query支持；缺失或失败保持规划分母，不删方法扩大支持。独立consumer输出canonical float32曲线/指标，selector验证B=median、Dmin=min、D0=lag0、Cbest=float32(B−Dmin)、Canchor=float32(B−D0)、first argmin和official offset=−bestlag、完整支持摘要及fit roster。

模型m的描述性fit utility为ΔCbest+min(0,ΔCbest)−max(0,ΔD0)−max(0,−ΔCanchor)+0.25Δfar，其中far=mean(Dlag−D0)，|lag|≥4。先clip等权到speaker，再speaker等权到model；跨model等权utility并加最差模型的负utility惩罚。远滞后项是辅助加权目标，不是tie-breaker；D0−Dmin是对齐代价，不称时间区分度。完整输出各模型全部指标/lag，不以B单独奖励、不重定时。fit选择没有CI，也不宣称方法成功；最终原音Cbest/超RMS/≥3前端/未读确认的冻结门槛仍按总spec。

## Alternatives and consequences

- 固定D0/unit-cos单目标作为旧v3保留：已有开发主Cbest下降，下一迭代需额外保护错误滞后区分和主端点。
- median-minus-softmin/B奖励作为旧fit24/v2保留：更长训练未超越RMS，范数/方向和代理路径诊断继续，不单靠增加步数。
- 直接复制TTS供体平均谱/动态轨迹不作为部署方案：先前相容性干预已显示伤害风险；共享family保持接收者局部结构。
- 后端各自选择参数可作为单独校准模式，但不得把其结果称为一个共享头。训练代理修正与黑盒共享family并行，失败候选不结束任务。

这一选择增加三模型实际生成成本，但可直接选择在实际原生前端/裁剪/原音评分下工作的波形方法，避免只在可微代理上拟合。当前12候选未实跑，不得写成已验证增强。若历史8句不稳定，可在预分配历史fit说话人中扩充；开发和确认身份仍分开。最终交付包括固定配置/CLI、原音换轨、ablation、时延、内容与独立运动检查和完整可恢复产物。

## Affected work and references

- `scripts/audio_enhancement/shared_frequency_profile_v1.py` SHA fd7303f513fdbada80da50e5dd395e9441142f1658544765d7523f1618f55253；CLI v1 SHA8eabe9fa038f5a5831383f247638c9784197bb149aba0f90b8f508fbbfe7a6c4。
- `scripts/audio_enhancement_eval/shared_profile_selection_v3.py` SHA1e329d4b1d2f5918230e5cdcf2d3c3c91256daf7a4536e7a7424ad33ff609a6d；protocol v4 SHAe79d33ce4e3338fafdc78ad52761817123fdac9070dace810fd8a10928be3cd2。
- `openspec/changes/develop-cross-tfg-audio-head-20261003/spec.md`与`transfer-iteration.md`；`.codex/scratch/root_v3_motion_temporal60_independent_readback_attempt01.json` SHA3ce1cd1bb503b4aa640646df9fe45e8c8d47cd6b3d4ac5864fad35f9e6b6a95c。
- [Repository](https://github.com/dogLeft233/tts-exp)，当前HEAD为本地未推送；旧已执行fit/dev源码和失败入口保留不改。

## Observations

- [decision] 同12局部频谱候选、同参数跨三前端历史fit筛选，最多3候选进入固定dev，再冻结未读确认。
- [rationale] v3固定匹配改善与主Cbest下降同时存在，需保护原音主端点、固定匹配和远滞后区分。
- [consequence] 参数选择仅fit/dev，完整支持/失败分母/原PCM/时钟保持；共享模式与模型校准模式明确标注。
- [alternative] 单独B奖励、固定D0奖励、直接供体谱移植均保留既有结果且不作为本轮唯一选择依据。

## Relations

- relates_to [[008-fixed-lag-and-rms-reference-head-objective]]
- affects [[开发跨 TFG 的自然音频增强头]]
- affects [[跨 TFG 自然音频增强头的训练与原音换轨验证]]