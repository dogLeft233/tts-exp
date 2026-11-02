---
title: Codex checkpoint - 2026-10-29T09-20-44Z - Chinese Sync-C threshold verified
  and current talking head shortlist
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-29-t09-20-44-z-chinese-sync-c-threshold-verified-and-current-talking-head-shortlist
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-29T09:20:44Z'
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a18566-b890-7a41-9d5a-4b9e2ca55c9d
trigger: compact
model: gpt-6.1-sol
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-29T09-20-44Z - Chinese Sync-C threshold verified and current talking head shortlist

## Summary
中文云端20来源60配对中，TTS−自然Sync-C严格大于0.7为31对（51.7%）；当前转向截至2026-10-29可复现的先进TFG/THG选型。

## Story
原任务EARS40自然/TTS的Wav2Lip九评分器实验已全部交付，报告75为权威结果，不再等待或重跑回归器。其Sync-C平均差+.859、34/40提高，九重Holm后SyncNet/SyncReward/Synchformer正向；控制失败、天花板和单文本限制保留，不推断感知真值。
用户随后观看英文大分差与中文不同文本高分样本；六段中文调响视频已交付于results/wav2lip/chinese_highscore_previews_20261029，来源及音量播放处理详见上一checkpoint。中文最新阈值问题按配对delta.SyncC > .7计算，31/60；每source三draw先平均后为11/20=55%。60包含同来源重复和重复TTS字节，不能当60独立说话人。
用户现要求找几个当日SOTA TFG/THG。已检索论文和官方仓库，确认LongCat-Video-Avatar1.5（2026-05-21代码权重，Whisper-Large-v3/8步）、SoulX-FlashHead（2026-02，1.3B，Pro/Lite权重）、EchoMimicV3-Flash（2026-01-22，1.3B/8步）、LiveAvatar（ECCV2026 Spotlight，14B/4步）、InfiniteTalk与LatentSync1.6均有推理与权重入口。将区分完整头/身体生成与视频嘴部编辑，标明作者宣称和不同benchmark不可直接排名。未部署或启动新实验。

## Working State
权威中文输入runs/cloud_av_preference20_20261027_attempt05/paired_scores.json与protocol.json；英文runs/ears40_w2l_nine_scorers_20261029_attempt01及basic-memory/docs/experiments/75-ears40-english-wav2lip-nine-scorers.md。播放副本只用于查看，原评分不重算。
本机E:/Documents/tts-exp、远端Ubuntu仓库/home/wjj/tts-audio/tts-exp。已有1351行git短状态，继承大量其他任务修改；本轮未改代码、未commit/push，当前分支GitHub搜索无PR。外部训练/暂停任务不动。

## Verification
本轮PowerShell重新读取原paired_scores.json，实际得到pairs60、over31、source20、meanOver11。前序30测试通过与全九评分独立复算/SHA验收是已完成历史检查，本轮未重复。新选型通过官方GitHub README和arXiv核查，未实测性能或显存；官方速度必须带硬件配置。
Git只读确认root、branch、HEAD与1351条状态；GitHub connector get_repo确认repository id1291624588及canonical clone URL，search_prs当前head返回空。HEAD仅本地确认，未验证可在GitHub解析。

## References
[Repository](https://github.com/dogLeft233/tts-exp)，local HEAD `2b71639c3b555293375d248350999377168d515d`。
[LongCat官方](https://github.com/meituan-longcat/LongCat-Video)、[技术报告](https://arxiv.org/abs/2605.26486)；
[SoulX官方](https://github.com/Soul-AILab/SoulX-FlashHead)、[技术报告](https://arxiv.org/abs/2602.07449)；
[EchoMimicV3官方](https://github.com/antgroup/echomimic_v3)；
[LiveAvatar官方](https://github.com/Alibaba-Quark/LiveAvatar)；
[InfiniteTalk官方](https://github.com/MeiGen-AI/InfiniteTalk)；
[LatentSync官方](https://github.com/ByteDance/LatentSync)。

## Observations
- [result] 中文此批T−N>.7为31/60=51.7%，以speaker三draw均值计11/20=55%。
- [decision] SOTA候选按任务与实际可用性比较，不按异源论文Sync-C硬排序；本轮仅调研，不部署。
- [next_step] 向用户交付六个模型的官方链接、日期/类别/适用性及优先补测建议。

## Relations
- continues [[Codex checkpoint - 2026-10-29T09-02-29Z - EARS40 results complete and Chinese high score playback pairs]]
- relates_to [[75-ears40-english-wav2lip-nine-scorers]]
