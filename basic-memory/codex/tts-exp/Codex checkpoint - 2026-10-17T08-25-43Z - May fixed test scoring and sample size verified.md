---
title: Codex checkpoint - 2026-10-17T08-25-43Z - May fixed test scoring and sample
  size verified
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-17-t08-25-43-z-may-fixed-test-scoring-and-sample-size-verified
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: 2026-10-17 08:25:43+00:00
username: laptop-e133gsle\dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a1445a-5cf8-7243-9764-adae839515e6
trigger: compact
model: gpt-6.1-sol
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-17T08-25-43Z - May fixed test scoring and sample size verified

## Summary
已从正式报告、实际评分源码及持久scores.json核对May先导的固定测试口径，区分44秒完整测试与40秒SyncNet评分，并明确独立来源仅一人。

## Story
用户先要求May训练电平−6/0/+6 dB保持共同初始化和预算，随后九格测试统一原音频评分。训练、评估和持久交付在上一checkpoint已完成；当前只追问如何评分、是否固定测试集和测试规模。本轮只读核验，不重训、不重评、不扩数据。

官方May自然录制重新按时间划分为train4150帧/166秒、val550帧/22秒、test1100帧/44秒（25fps），不是官方默认划分。test监督source frame为[4925,6025)。同一固定最终60000步EMA模型在三档测试驱动上渲染全部1100帧，所有九格共享时间支持和GT条件。SyncNet实际读取准备阶段manifest中预定四个250帧/10秒clip，source起点4929/5179/5429/5679；合计1000个不同源帧/40秒。剩余100帧仍进入全测试重建误差。每格四clip分别评分再等权平均，共36个生成评分实例，不能解释为36个独立自然样本或四个独立身份。

三档测试gain只改变驱动，正式评分使用完全相同的原始16kHz PCM16 channel0及原时钟；代码逐字节核对mux音频，未用增益后的音轨评分。FFV1/BGR0正式媒体避免观看副本的有损压缩；GT-owned S3FD轨迹平滑后的裁剪几何共享九格。SyncNet V2严格原权重、FP32，5帧视觉/20 MFCC窗口，±15帧lag搜索。每clip245滑窗，主guard20去两端20后205窗；full245仅敏感性。每lag先在有效窗取平均距离，C为31lag距离中位数B减最小值Dmin，D为Dmin，同时保留零lag D0和最佳offset。较高C不自动表示原时钟更准或感知质量更好。

这是已暴露May英语单身份、单自然录制、单seed42的时间留出工程先导。新生成器没有训练val/test监督帧，但部分test时间曾用于旧预训练推理，且继承官方全序列pose/背景预处理；不能称完全未见无污染确认集。四clip相关，重复电平条件不增加独立样本量。本轮回答应明确规模小，仅支持本素材条件响应，不证明数据充分或普遍最佳训练音量。

## Working State
权威报告[[50-synctalk-may-gain-training-pilot]]、done Task[[训练 May 电平交叉先导]]保持不变。F:/Datasets/tts-exp/synctalk_may_training_pilot_20261017/remote/scores/scores.json是本轮实际读回的评分记录；完整F盘及百度盘独立读回证据已在前一checkpoint保存，无需重复传输。Ubuntu /dev/shm工作副本易失，不当备份。没有新GPU任务或新增科学结果。

Git分支和SHA未变，共享工作树大量既有dirty/untracked文件未清理、stage或commit。本轮唯一新本地文件是.codex/scratch/checkpoint_may_scoring_explanation_20261017.py（写记忆辅助，可丢弃），没有修改实验执行源码、协议、报告或scores。GitHub只读API确认仓库canonical URL、当前head查询PR为空；当前SHA查询422，仅保留本地SHA，不生成commit链接。gh CLI缺失，已改用只读REST完成仓库核验。

## Verification
先读Startup Router、研究地图、实验指令，项目全文搜索命中正式50报告并读全文。CodeGraph返回score_test.py当前完整源码，确认manifest固定四clip、原PCM校验、共享GT裁剪、FP32权重、guard20/full和四clip平均实现。持久scores.json读回9个cell，每个恰4clip；四起点一致，primary_windows均205/full均245。此前运行的9900测试帧、36矩阵273420距离QA见正式报告，本轮没有重新运行这些GPU核验。未运行新测试，因本轮仅解释既有结果。

同chat checkpoint搜索覆盖coding_session/codex_session及legacy codex_session_id，结果无后续分页；直接读最新08:14:27Z的frontmatter确认agent/session_id精确相同，再建立continues。

## References
- [项目仓库](https://github.com/dogLeft233/tts-exp)
- 本地SHA：`2b71639c3b555293375d248350999377168d515d`
- basic-memory/docs/experiments/50-synctalk-may-gain-training-pilot.md
- scripts/experiments/synctalk_may_training_pilot_20261017/score_test.py
- F:/Datasets/tts-exp/synctalk_may_training_pilot_20261017/remote/scores/scores.json

## Observations
- [result] 正式SyncNet评分支持是预定四个10秒clip共40秒，九格全测试渲染支持各44秒；每clip205主窗口，四clip等权平均。
- [decision] 统计规模报告为一身份/一录制/单seed；36次评分是重复条件，不作为独立样本分母。
- [decision] 仅驱动音频施加增益，所有正式评分保持同原PCM和原时钟，Sync-C不能替代感知同步端点。
- [next_step] 若用户继续要求确认训练电平排序，先在新run做第二共同seed复现；当前只回答评分与测试规模，不自动扩训。

## Relations
- continues [[Codex checkpoint - 2026-10-17T08-14-27Z - May gain training pilot completed and cloud verified]]
- relates_to [[50-synctalk-may-gain-training-pilot]]
- relates_to [[训练 May 电平交叉先导]]