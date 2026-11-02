---
title: Codex checkpoint - 2026-10-30T02-47-10Z - Completed Chinese20 evidence and
  next TFG candidates
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-30-t02-47-10-z-completed-chinese20-evidence-and-next-tfg-candidates
status: completed
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-30T02:47:10Z'
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a1880e-73d5-7863-a38b-20bcdf7d5ebc
model: gpt-6.1-sol
trigger: compact
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-30T02-47-10Z - Completed Chinese20 evidence and next TFG candidates

## Summary
中文20人FlashHead Pro完整九评分器实验已经完成并独立复核，当前转入截至2026年10月30日的新TFG候选选择。

## Story
用户要求遵守192服务器条例，在个人目录部署SoulX-FlashHead Pro与九评分器，使用中文20人生成自然语音及既有TTS对应视频，并追加可听音量归一化。完成的实验将20自然和60既有TTS音频统一派生到RMS−24dBFS（约+4.48dB，无削波和时长编辑），同一PCM驱动生成与评分。41种唯一输入真实生成，完全相同输入/图片/seed显式复用，80逻辑条目保留。统计先平均每来源三个TTS，再与自然语音配对，保持同句参考12人与扩展参考8人分层。

权威结论与产物入口是[[76-flashhead-chinese20-nine-scorers]]。同句参考12人九指标Holm9均不显著；全20人SyncNet差+.212，CI[−.399,+.873]，仅PEAVS偏自然（差−.452971，Holm p=.04131）。这些是评分器证据，不等于感知同步结论。两个中心5秒分类器仅7人完整配对，其他七项20人；不插补、不合并总分。SyncReward使用官方内部短片补齐并另报完整4.75秒8人敏感性，PEAVS保留64+16主批次并用相同特征重放batch20。

用户最新询问还有哪些SOTA TFG值得测试。本轮已读历史跨模型报告、部署总览和2026候选调查，确认AVTR-1、AvatarForcing、LeapTalk、StableAvatar有历史结果，避免重复当成新候选。联网核验官方仓库和论文，优先建议LongCat-Video-Avatar1.5（Whisper-Large-v3条件、8步）、AptAvatar（中文wav2vec2、2步）；Wan2.2-S2V可补大模型基线但官方示例需80GB。Kandinsky S2V于2026-10-13提交论文，Qwen2.5-Omni条件有机制研究价值，但本轮官方项目页链接GitHub404、HF集合401，不能称为已可部署。此处只是候选建议，未下载、安装或启动新模型。

## Working State
完成远端任务位于 /home/data2/wangjiajun/av-evaluation/jobs/flashhead_chinese20_rms24_20261030_attempt01。前轮实际ps与nvidia-smi核验本人推理和后处理均退出；本轮未重新登录服务器或启动任务。代码、环境、缓存、权重和结果均在个人目录，凭证通过本机DPAPI运行时解密，未复制明文。原评估机已在线，但未用于本次执行。

本地交付 results/soulx_flashhead_pro/chinese20_rms24_20261030_attempt01/ 含report.md、summary.json、逐视频与来源配对CSV、video_index.json、图和evidence/。证据包309272893字节，SHA256为7cf9916be529e8f0f4099bd346a4cae168bb590bd254af9266cfee213ee7a700，包含2234内容文件和manifest。标准播放视频与评分无损ALAC分开保留，原始规范音频不覆盖。

仓库存在大量先前dirty/untracked文件；本轮只读git状态，未清理、提交或创建PR。当前SHA为本地已验证状态，未确认GitHub可解析该commit。先前HANDOFF含其他任务，恢复优先权威报告76与本checkpoint。

## Changed Files
前轮新增未提交 scripts/experiments/flashhead_chinese20_20261030/（11个Python文件与README）和tests/test_flashhead_chinese20_20261030.py；CONTEXT.md更新报告76入口且仍100行。项目部署记录、实验笔记与Sync-C表已同步BM。全局server-192/references/personal-deployment.md补充已验证的SyncNet参数、官方短片处理、重复输入复用和归一化工作流。本轮未再修改实验代码和结果。

## Verification
前轮4项归一化/时序/来源配对/Holm测试及11脚本编译通过，PNG实际查看；80条PCM与媒体身份核验、七评分器原始矩阵/embedding/logit归约通过，SyncNet独立FP64对照最大差2.4485e−6。PEAVS/SyncReward是实际模型输出和身份覆盖检查，未声称独立重实现。六项真实偏移控制通过；SyncNet/PEAVS/SyncReward本队列未重做控制，历史控制失败边界保留。全局skill quick_validate通过。

本轮直接读回local_readback.json：PASS、2234文件、80导出视频、证据包SHA匹配。读回local_statistics_audit.json：PASS、25非空分层、来源均值/10000bootstrap/100000signflip/Holm9/胜负计数最大误差0。本轮没有重复运行GPU实验或pytest。GitHub只读API重新确认仓库dogLeft233/tts-exp，当前head PR搜索为空。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- 本地SHA：`2b71639c3b555293375d248350999377168d515d`；不构造未验证远程commit链接。
- [权威报告76](../../docs/experiments/76-flashhead-chinese20-nine-scorers.md)
- [LongCat官方代码](https://github.com/meituan-longcat/LongCat-Video)
- [AptAvatar官方代码](https://github.com/TaoLiveAIGC/AptAvatar)
- [Wan2.2官方代码](https://github.com/Wan-Video/Wan2.2)
- [Kandinsky S2V论文](https://arxiv.org/abs/2610.19290)

## Observations
- [result] 中文20来源80视频及九评分器完整实验已交付，本地内容和统计读回PASS。
- [decision] 保留RMS−24dBFS、来源配对、参考12/8分层与每评分器支持集，不将跨模型历史分数直接拼接。
- [decision] 下一轮优先覆盖音频编码与大模型底座差异；历史已经测过的模型不算新增候选。
- [blocker] Kandinsky官方代码/权重链接本轮不能访问；Wan官方80GB示例不证明单A6000可运行，其他大模型也需资源smoke核验。
- [next_step] 向用户给出截至2026-10-30的候选优先级和部署证据边界；选择后再沿用冻结中文20人与九评分器协议。

## Relations
- continues [[Codex checkpoint - 2026-10-30T02-05-11Z - Chinese20 FlashHead nine scorer experiment complete]]
- documents [[76-flashhead-chinese20-nine-scorers]]
