---
title: Codex checkpoint - 2026-10-18T13-00-13Z - Normalization source surveys and
  loudness literature boundary
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-18-t13-00-13-z-normalization-source-surveys-and-loudness-literature-boundary
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: 2026-10-18 13:00:13.426384+00:00
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

# Codex checkpoint - 2026-10-18T13-00-13Z - Normalization source surveys and loudness literature boundary

## Summary
主流编码器结果已完成；已核查历史TFG与2026公开实现的归一化入口，当前正在区分音量—Sync-C直接文献、处理链敏感性与归一化的原始动机。

## Story
项目从恢复TTS的同步优势机制发展到自然音频增强头；用户现已暂停增强头，并要求ROOT直接执行编码器调查、不用子代理。报告52完成后，用户追问归一化是否解释增益响应以及现有论文是否已研究音量与Sync-C。当前采用小源码/配置与一手论文调查，不启动TFG、训练或下载大权重。不能把编码方向响应换成生成收益，也不能把内部LayerNorm当作逐音频响度归一化。

## Working State
报告52为11新checkpoint/1331正式forward及缓存Wav2Lip，CAL20/10男性/663共同音素，原数组与统计已复核；权重缓存回收，科学产物在F:/Datasets/tts-exp/encoder_survey_20261018。8弱响应路径均有输入标准化/CMVN；4强响应路径保留整体增益，但这不是同权重归一化开关的因果消融。

已写basic-memory/docs/reference/已测TFG的音频归一化调用链.md，16历史/补充模型：8输入标准化，6无显式逐输入标准化或固定缩放，2内部/闭源未知。Ditto实际固定affine保留增益；Joy自定义路径绕过HF processor。已写2026年TFG的响度与输入尺度处理.md：10代表实现中4显式−23LUFS、4波形标准化、LiveAct固定×3保留电平、AVTR内部未知。Tencent中文wav2vec2-base实际do_normalize=False，已纠正此前聊天和地图，不与中文HuBERT混淆。公开源码不等于所有历史容器已恢复。

两源码调查产物分别在runs/tfg_audio_normalization_audit_20261018与runs/tfg_2026_loudness_survey_20261018，manifest分别e297469d87c27d08e36ae7e085d4cd9835a89bfd9a0cb172fcc04e457d32668f与171f5348e19575f8d233801a1cbedd830a4d147dae6b6fb78406d92566fe4720。研究地图、短HANDOFF和CONTEXT已维护；BM/runs/新脚本为机器本地/未提交状态。原9个tracked dirty完整保留。无PR；当前SHA本地，GitHub此前不能解析，不构造提交链接。

当前文献：MOVA arXiv2602.08794v2 §4.3/6.3将LUFS用于避免CFG响度爆炸，Phase2 LSE-C升高但同时改noise schedule/dropout/数据，不是volume-only消融。THEval arXiv2511.04520v2 AppendixC报告mp4a/mpga改变LSE-C/D平均绝对约0.4、个别1.2，是codec而非纯gain；其§3.3/B.3另用独立min-max的嘴开合与局部RMS指标。Yaman ECCV2024研究SyncNet不稳定，尚未找到纯音量扫掠。检索未找到不等于不存在/首创。EBU R128是节目感知响度一致；CMVN经典Viikki/Laurila1998是减少噪声条件训练/测试失配；BN/LN/GN分别围绕优化、样本内稳定、小batch训练。下一参考文献笔记尚未写。

## Changed Files
本段维护两个reference笔记、Research/研究主线与证据地图.md、HANDOFF.md和CONTEXT.md；源码快照位于runs。当前文献阶段无实现代码修改。

## Verification
实际公开源码调用路径/config/revision与manifest已核查；当前已读MOVA、THEval和EBU原文、BN/LN/GN原始摘要。CMVN原文publisher摘要检索命中，PDF/DOI open失败保留来源边界。未运行实现tests、pytest、GPU实验。Startup Router/实验指令及same-chat typed/legacy当日搜索已读，前驱frontmatter exact agent/session确认。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- [MOVA](https://arxiv.org/html/2602.08794v2)
- [THEval](https://arxiv.org/html/2511.04520v2)
- [EBU R128](https://tech.ebu.ch/docs/r/r128.pdf)

## Observations
- [result] 公开归一化入口调查完成；原编码器强弱响应与是否消除输入增益相符，但尚非归一化开关因果对照。
- [decision] 继续暂停通用增强头与旧训练/中心/Dev/Confirm；只完成当前文献问题，不占其他chat的Ubuntu任务。
- [decision] 将纯增益、codec、联合训练改动和局部能量—口部指标分开记载；不冒称首篇音量研究。
- [blocker] 当前未找到固定视频纯全局gain→Sync-C的明确论文；这仅限定检索结果。
- [next_step] 完成音量—Sync-C及归一化原始动机的参考笔记和地图链接，向用户给出带一手来源的比较表。

## Relations
- continues [[Codex checkpoint - 2026-10-18T12-24-36Z - Gain direction interpretation and transfer boundary]]
- relates_to [[研究主线与证据地图]]
- relates_to [[52-encoder-direction-commonality]]
- relates_to [[已测TFG的音频归一化调用链]]
- relates_to [[2026年TFG的响度与输入尺度处理]]