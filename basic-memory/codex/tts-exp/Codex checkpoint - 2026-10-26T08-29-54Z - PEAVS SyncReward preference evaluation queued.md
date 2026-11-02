---
title: Codex checkpoint - 2026-10-26T08-29-54Z - PEAVS SyncReward preference evaluation
  queued
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-26-t08-29-54-z-peavs-sync-reward-preference-evaluation-queued
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-26T08:29:54Z'
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

# Codex checkpoint - 2026-10-26T08-29-54Z - PEAVS SyncReward preference evaluation queued

## Summary
已把此前20来源CoGenAV队列准备为PEAVS与SyncReward的650格评测，CPU校验完成，服务器任务等待既有audio200生产流水线结束；尚无新模型评分结果。

## Story
本聊天原目标是检验TTS原生Sync-C优势是否是评价器偏好，并探索将TTS口型收益迁移到自然音轨。音频/视频DTW与STFT幅度移植已尝试失败，不包装成新路径。上次固定视频交叉自然/TTS音轨对照完成：两模型大多偏自身驱动音轨；跨音轨的发音时钟错配使其不能单独证明或排除声学偏好。

最新用户说刚部署PEAVS和SyncReward，要求复做自然/TTS偏好对照，区分SyncNet偏好与真实口型改善。已读两部署笔记并实际读取Ubuntu入口。两者使用已部署官方评分代码；SyncReward是V100 FP32兼容入口、Stage2严格加载，中心4.75秒；PEAVS整段官方I3D/PWC、VGGish及评分头。不改官方源文件。时间支持不同，不能声称与旧SyncNet/CoGenAV同窗口复现。多指标一致只增加评价器证据，仍不能替代本项目中文域盲评。

## Working State
Ubuntu实际有效入口wjj@192.168.50.2，repo /home/wjj/tts-audio/tts-exp。Windows唯一入口E:/Documents/tts-exp。权威run为 runs/av_metric_preference_20261026_attempt02/，其prepared.json确认650格、源音视频276文件SHA通过，全部音频ALAC解码逐样本相等，原生视频压缩包逐包相等，交叉视频无损重编码后解码帧hash逐帧相等。138原生格、118自然/TTS配对、472四格交叉单元、40个自然视频±400ms固定长度零填充控制；继承原2条CoGenAV >30秒未测排除，不按新模型表现筛样本。

交叉支持取四格共同最短音频/视频长度并向下取整至25fps，保持起始前缀，不做DTW或最佳时移。原生保留完整原视频流和原源PCM。20来源等权，L06/L17分开，报告两个冲突方向和来源bootstrap描述区间。控制低于32/40降低则标CONTROL_FAILED；不把失败当科学阴性。

后台queue进程PID1447117，已独立session脱离SSH，runner.pid/runner.log记录；最近实际status为WAITING_GPU/native000、existing_production_active=true。当前audio200_nl17_controller_20261025.py PID320414及batch041执行器仍活跃，GPU在生成/评分阶段间有短暂空闲。新任务明确等待该既有controller退出、无外来GPU计算进程且显存<1024MiB/util<15连续3次检查，避免抢占空档造成后续TTS显存冲突；不终止既有任务。不设定时通知、不声称评分已启动。任务会先完成两模型原生评分，再做时移控制和交叉评分；每24格断点保存。

本地已有report.md、summary.json、status.json、protocol.json、analysis_protocol.json、cells.json、prepared.json、input_pins.json快照，均明确未完成。650媒体和逐格preparation收据仍只在Ubuntu，尚未归档/复制本地；未来状态以远程为准。

## Changed Files
新建未跟踪 scripts/experiments/av_metric_preference_20261026.py 与 tests/test_av_metric_preference_20261026.py，已复制Ubuntu。脚本SHA256 308ad4901f436cb40ff36b6adfc79e9d0dd81366a2591ca69da0361355537f36。原工作树有大量继承dirty/untracked，不归本轮、不清理。未commit/push，Git根/分支/SHA实际核验；GitHub连接器确认dogLeft233/tts-exp，当前分支open PR查询为空。本地SHA未确认已推送，不构造commit链接。

## Verification
首次attempt01原生138格准备通过，第一交叉格stream-copy裁切预期113帧实际115帧失败，未评分，完整保留。修正为交叉格lossless H264精确帧数并逐帧核验，使用新attempt02，不改旧冻结协议。attempt02的650格全部prepare成功。
三项测试函数直接执行通过：固定视频四格完整性/排除缺失draw，方向与单来源区间，不等draw下来源等权及冲突方向并忽略raw评分中间文件。py_compile成功。远端解释器无pytest，pytest命令失败，不称pytest通过。SyncReward官方load_av实际成功解码ALAC原生输入，147帧25fps、48k音频287952样本；没有GPU前向。
analysis_protocol.json记录代码、清单、协议、两评分头权重SHA。实际重算Stage2权重SHA为6c09263ac41d59baf78a181f17e4116128a337d70a03637886de3734450a32a2，与部署记录一致。协议SHA6c146c9e5bdfad2e83c633ee8a197233d75ce8b417aae554f32117dfbafada4a，input_pins SHA5b71f87b77a6c1fc837fe1c8af5b9e5fef3154b1d950e90d3bf0264c7e3e0f2f。尚无新指标结果或独立数值审计。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- [SyncReward official source](https://github.com/WeChatCV/SyncReward)
- [PEAVS official source](https://github.com/amazon-science/avgen-eval-toolkit)

## Observations
- [result] 650个评测媒体准备与无损校验完成，新评分尚未开始。
- [decision] 复用20来源完整可测队列与既有排除条件，不按旧Sync-C或新模型分数挑有利样本。
- [decision] 原生比较、固定视频换轨、已知时移控制分别解释；多指标一致不等于感知真值。
- [blocker] 既有audio200生产controller仍活跃，新评测排队等待资源；不得中断既有生产。
- [next_step] 待远端任务推进后读取status/runner.log及summary，复核完整性、时移控制、来源配对和双向冲突，再向用户报告真实数值；若runner失败则修复并断点继续。

## Relations
- continues [[Codex checkpoint - 2026-10-26T06-59-10Z - Fixed video crossed soundtrack comparison]]
- relates_to [[peavs]]
- relates_to [[syncreward]]
