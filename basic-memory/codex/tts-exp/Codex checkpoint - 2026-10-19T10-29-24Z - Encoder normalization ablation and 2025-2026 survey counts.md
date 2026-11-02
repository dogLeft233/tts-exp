---
title: Codex checkpoint - 2026-10-19T10-29-24Z - Encoder normalization ablation and
  2025-2026 survey counts
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-19-t10-29-24-z-encoder-normalization-ablation-and-2025-2026-survey-counts
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: 2026-10-19 10:29:24.024871+00:00
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

# Codex checkpoint - 2026-10-19T10-29-24Z - Encoder normalization ablation and 2025-2026 survey counts

## Summary
同权重输入归一化开关实验53与2025历史源码调查已完成；两年23个受查版本中5条保留前端整体增益、17条抑制、1条未知，这不是视频或Sync-C响应比例。

## Story
项目原目标是解释TTS驱动同步优势并形成增强头；用户已暂停增强头，改由ROOT直接研究编码器增益响应及主流TFG归一化路径，不再委派子代理。先用52共同cohort完成同权重输入norm开关实验，再按2025年末前官方commit重建13个公开版本的处理路径，与已有2026十条源码调查合并。当前用户只问两年调查的受影响比例，因此复用现有源码和BM，不新增模型推理或训练。纯整体增益的前端可达性、encoder方向响应和生成视频/Sync-C效果分开报告。

## Working State
报告53及探索笔记为科学结果入口：8个相同checkpoint、同CAL20十男性二十句、同自然PCM/RMS/时钟，512 encoder forward；7/8移除输入norm后恢复方向响应。W2vBERT去CMVN仍近似不敏感，实际projection前LayerNorm消除log-power常数平移；60 CPU前端hash重放一致。移除norm也改变基线表示，data2vec off/on cosine约0.673，不能称语义改善。HTTP429的两个零forward失败保留，官方LFS同SHA仅恢复剩余三模型，首五未重跑。所有worker已完成，权重缓存回收。正式产物F:/Datasets/tts-exp/encoder_normalization_ablation_20261019，236成员/1198785208B，manifest SHA 0a29a6e3956fa20bbc5c323cc8e53c0f5fbaf0d1d2b112a8267dff21181145ab。

2025调查为3 LUFS、6波形标准化、4 Whisper原生前端保留电平。2026为4 LUFS、4波形标准化、LiveAct固定乘3保留电平、AVTR图内未知。合计17/23=73.9%前端抑制、5/23=21.7%保留、1/23=4.3%未知；排除未知为5/22=22.7%。五条是Sonic、HunyuanVideo-Avatar、MuseTalk1.5、LatentSync1.6、SoulX-LiveAct。计数单位是所选公开模型/版本，LongCat v1与v1.5各一条，不是市场份额或独立家族数。正常可测无裁剪语音的纯正增益限定不推广局部能量/EQ；内部norm仍可能消除前端变化。

2025源码158成员/1535436B，runs/tfg_2025_loudness_survey_20261019/artifact_manifest.json SHA d2bdd35a24e2460dc22e07b3cb1da4852fb13a052cb3700e322849d2f77e32a6。2026来源见对应reference与runs/tfg_2026_loudness_survey_20261018。2025固定年末前官方源及历史HF配置，避免把2026新增处理回写2025。未下载模型或运行TFG。

增强头及旧训练/中心/Dev/Confirm继续暂停。Ubuntu的Patrick RMS任务属于别的chat，不触碰。机器本地BM/runs和F盘科学产物尚无云同步声明；源码、BM、HANDOFF未提交。九个原tracked dirty保留，未commit/push，当前SHA在前轮GitHub查询不可解析，按本地SHA记录。仓库canonical及当前分支无PR本轮只读查询确认。

## Changed Files
新实现位于scripts/experiments/encoder_survey_20261018/normalization_ablation.py、normalization_recovery.py、normalization_analysis.py、normalization_report.py、normalization_w2vbert_diagnostic.py（未跟踪）。BM新增53正式报告/探索笔记和2025 reference，已维护研究主线与证据地图、HANDOFF及CONTEXT（99行）。本次只在地图已有2025段补两年合并分母与边界；没有改实现。

## Verification
53的24次norm-on重放对旧帧逐值一致，160新NPZ原数组重建：RMS差0，unit方向max误差2.98e-8。W2vBERT前端相对变化0.049201，经无仿射LN核心约7.91e-7；无额外encoder。2025历史源/配置实际body与hash保存并读回；本次两年reference和地图直接BM读取，算术重新计算。未运行实现tests、pytest、compileall、新GPU/TFG/SyncNet。新checkpoint沿exact agent/session搜索并直接读取前驱frontmatter，不链接别的chat的Patrick checkpoint。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- basic-memory/docs/experiments/53-encoder-input-normalization-ablation.md
- basic-memory/docs/reference/2025年TFG的响度与输入尺度处理.md
- basic-memory/docs/reference/2026年TFG的响度与输入尺度处理.md
- basic-memory/Research/研究主线与证据地图.md

## Observations
- [result] 两年23条受查公开模型/版本中5条前端保留整体电平、17条自适应处理抑制、1条未知；已知条目中保留比例22.7%。
- [result] 同权重8编码器输入norm消融完成，7条恢复方向响应；W2vBERT内部逐帧LN是明确反例。
- [decision] 不用源码暴露比例冒充实际视频或Sync-C受影响比例，不用消融后的7/8代表现有TFG分布。
- [decision] 保持用户暂停增强头和ROOT直接操作的范围，不自动扩大实验。
- [blocker] 未完成这23条TFG的配对整体gain视频/评分实验，真实输出受影响率尚无本调查的实测分母。
- [next_step] 向用户交付两年分项比例、五条前端保留模型名单及源码与实际效果的界限。

## Relations
- continues [[Codex checkpoint - 2026-10-18T13-00-13Z - Normalization source surveys and loudness literature boundary]]
- relates_to [[研究主线与证据地图]]
- relates_to [[53-encoder-input-normalization-ablation]]
- relates_to [[2025年TFG的响度与输入尺度处理]]
- relates_to [[2026年TFG的响度与输入尺度处理]]