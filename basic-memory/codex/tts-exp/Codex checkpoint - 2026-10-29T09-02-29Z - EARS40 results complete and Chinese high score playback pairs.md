---
title: Codex checkpoint - 2026-10-29T09-02-29Z - EARS40 results complete and Chinese
  high score playback pairs
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-29-t09-02-29-z-ears40-results-complete-and-chinese-high-score-playback-pairs
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-29T09:02:29Z'
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

# Codex checkpoint - 2026-10-29T09-02-29Z - EARS40 results complete and Chinese high score playback pairs

## Summary
EARS40九评分器完整实验与报告75已交付；最新按用户改为中文高分案例的要求，制作三对不同文本自然/TTS的可听播放副本。

## Story
用户原要求复用40人EARS自然及既有云Qwen3.1 TTS，沿用Wav2Lip并用九评分器判断英文优势。冻结40人20女20男、统一RMS0.0376838172675746、原时钟与固定头像，完成80视频、720原生评分、720真实±400ms控制。权威报告为 basic-memory/docs/experiments/75-ears40-english-wav2lip-nine-scorers.md；原来等待最后两项回归器的checkpoint已过时，全部完成，不要再次运行。

Sync-C自然7.327/TTS8.186，配对差+0.859、95%[+.624,+1.092]，34/40提高。九项Holm后SyncNet、SyncReward、Synchformer正向，其余六项未确认均值差。Sync-C控制降分15/80但相对offset80/80；SyncReward33/80与PEAVS41/80降分门失败；SparseSync存在天花板。结论只到本条件评分优势，未证明人类嘴型准确或语言因果效应。40人共享同段sentences_16_regular，不能研究跨文本效应。80评估音频，另40独立克隆参考，共120（排除格式副本）。

用户随后要求大音量前三个Sync-C差距案例；p046、p039、p105六段可听副本已在results/wav2lip/ears40_louder_previews_20261029交付。T−N>.7为24/40=60%，绝对差>.7为25/40=62.5%。用户再问内容与分数关系及不同文本高分样本；先找到历史LRS3英文库并生成临时预览，但用户中断改为此前中文实验。当前以20261027云Qwen Audio3.1 Plus中文20来源/60配对库，按TTS Sync-C降序、每来源最多一对，选择S0031/C2、S0013/C1、S0006/C1。分数N/T依次8.584/9.517、7.931/9.473、7.602/9.434。仅音量播放处理，不重新评分或生成。

## Working State
持久实验run：runs/ears40_w2l_nine_scorers_20261029_attempt01；parent_analysis含summary、360行配对CSV、原始端点/控制复核、中央支持和百万次MC敏感性、Material Passport ANALYZED及完整ledger。3585文件675359638bytes交付清单SHA6483624c09ceb31e333bc65efba3991e05c0a38e1aec0871413fa8860f3ae0d8。80视频在results/wav2lip/{natural_raw,tts_raw}/<speaker>/ears40_20261029.mp4；index为results/wav2lip/ears40_20261029_index.md。报告、研究地图、评分表与实验索引均已更新，CONTEXT.md正好100行。

最新六段中文副本：results/wav2lip/chinese_highscore_previews_20261029/{BAC009S0031W0121,BAC009S0013W0121,BAC009S0006W0121}_{N,T}_audible.mp4，preview_manifest.json记录源视频/音轨SHA、原评分、选择规则与响度。两遍loudnorm目标−16LUFS，AAC48k mono160k；实测−16.15至−17.37LUFS。自然视频来自既有selected_inputs，T来自cloud attempt05，均核原始视频/规范音频SHA后生成。

机器局部未提交：六个ears40脚本及六个对应tests、此次.codex/scratch/content_highscore_20261029/脚本。大量继承脏文件不归本次；本次只读git status保存该scratch/git_status.txt（1351行）。远端wjj@192.168.50.2仓库/home/wjj/tts-audio/tts-exp。英文预览脚本已结束，无仍需监控的本任务GPU进程；不要触动其它训练或暂停任务。无commit/push/PR；read-only GitHub确认当前分支无PR。

## Changed Files
scripts/experiments/ears40_{w2l_nine_scorers,cogenav_scorer,extra_scorers,retrieval_scorers,regressor_scorers,results}_20261029.py及对应tests为本轮新增未提交源码。报告75及ADR052在Basic Memory；ADR保留整数帧预取窗修复及旧缓存不可变凭据，不重写历史中文冻结结果。最新播放脚本为.codex/scratch/content_highscore_20261029/make_chinese_previews.py。

## Verification
前序完整执行：ROOT六个针对测试30passed/0.54s；独立CPU原始数组/统计与3585文件SHA/size通过，720原生及720控制身份/数值复核通过。未重复九模型神经前向，故ANALYZED而非完整复现VERIFIED。百万次MC敏感性九项判定不变。
本次实际执行六个中文视频/规范音轨源SHA核验、ffmpeg两遍响度处理、视频压缩码流SHA不变核验；回传六文件本地SHA与manifest全部PASS。没有新TTS、TFG、评分或训练。

## References
[Repository](https://github.com/dogLeft233/tts-exp)，本次read-only connector确认；Git SHA仅本地验证为 `2b71639c3b555293375d248350999377168d515d`，不宣称本次新增源码已推送。
中文权威来源：runs/cloud_av_preference20_20261027_attempt05/{paired_scores.json,native_cloud_report_20261027.md,protocol.json}；自然身份为runs/cogenav_sync_full20_20261024_attempt01/selected_inputs.json。

## Observations
- [result] 当前英语EARS单文本设置具有Sync-C评分优势，但不提供跨文本泛化或感知同步真值。
- [decision] 所有失败控制、全部40对与旧修复凭据保留；不据分数自动重开训练。
- [decision] 最新中文展示是事后最高TTS分案例，每来源一次，不能当总体效果或最大分差抽样；播放音量不进入评分。
- [next_step] 向用户展示三对中文可听视频、文字与N/T分数，等待其观看反馈；无授权的新实验待执行。

## Relations
- continues [[Codex checkpoint - 2026-10-29T06-14-30Z - EARS40 Wav2Lip nine scorer experiment awaiting final regressors]]
- relates_to [[75-ears40-english-wav2lip-nine-scorers]]
- relates_to [[052-integer-frame-native-window-preselection]]
