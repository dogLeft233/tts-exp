---
title: Codex checkpoint - 2026-10-26T06-59-10Z - Fixed video crossed soundtrack comparison
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-26-t06-59-10-z-fixed-video-crossed-soundtrack-comparison
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-26T06:59:10Z'
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

# Codex checkpoint - 2026-10-26T06-59-10Z - Fixed video crossed soundtrack comparison

## Summary
固定视频交叉音轨的20来源对照完成，两模型均更偏视频自身驱动音轨，未见固定视频普遍TTS偏爱。

## Story
本聊天围绕TTS原生Sync-C优势、CoGenAV指标分歧和自然音轨迁移。用户纠正先前建议：音频预对齐和视频后对齐都已尝试失败；不能再当新路径。官方AVSE是视觉条件下的去噪/分离头，不是驱动增强器。之前提出小幅平滑STFT掩码作为自然音频处理方案，用户指出已有幅度移植音质差、无原生增益；现已撤回优先推荐。保留相位、样本数和小幅度不能保证瞬态/内部时序或TTS收益。生成器运动层视觉条件仅为未验证远期方向，不自动恢复旧失败head路线。

最新用户要求实际比较SyncNet与CoGenAV，异步澄清为同一视频分别配自然/TTS，而非音频间相似。已通过CPU重组138条核验的独立编码器缓存，形成118自然–TTS配对、每对两种固定视频、472格评分；无新TFG、训练、媒体重封装或模型前向。每配对四格取共同有效起始前缀，不进行DTW。官方SyncNet calc_pdist与median-min；CoGenAV论文Eq1自写实现，max为自定义诊断。

## Working State
权威产物本地与Ubuntu同相对目录 runs/cogenav_fixed_video_cross_20261026_attempt01/：report.md、protocol.json、rows.json、summary.json、cache_pins.json、identity_checks.json、independent_audit.json。20来源、L06/L17重复draw不当独立样本；原2条>30s无CoGenAV继续排除。自然视频配N/T：C7.032/2.831、Eq1 .321323/.299272；TTS视频配N/T：C2.835/7.133、Eq1 .300423/.312852。TTS高分配对分别C1/118和117/118，Eq1 26/118和87/118。两组Sync-C来源平均20/20偏自身驱动音轨。max分别.467232/.358845和.369053/.475502。全部仅描述性结果，无人评。

原20来源native各自视频评分分歧、六来源重定时失败和五小时解码器低保真结果仍有效，见前检查点与实验笔记。新结果不是原生两模型一致性统计的替代。时间结构错配可解释交叉换轨损失，不能单独排除时间匹配后TTS声学偏好。共同前缀不等同全长原生基线。官方编码器缓存重放不称新全官方媒体流程，CoGenAV没有官方完整同步benchmark。

## Changed Files
新建未跟踪 scripts/experiments/cogenav_fixed_video_cross_20261026.py 与 tests/test_cogenav_fixed_video_cross_20261026.py，另有上述run报告和JSON。已复制源码/测试到Ubuntu当前repo并在临时源码执行。继承工作树极多dirty/untracked，不归本轮所有，不清理。未commit/push。Git根/分支/SHA已实际确认；GitHub连接器确认仓库，当前分支open PR查询为空。本地SHA未远程核验，不构造commit链接。

## Verification
138缓存逐SHA核验，原生C重放误差0、Eq1小于1e-6。全部472格独立NumPy计算最大误差1.049041748046875e-05小于2e-5。两个测试函数直接执行通过；pytest不可用未运行。无GPU使用。Ubuntussh实际连接成功。
summary.json SHA256 0374341aa3a5163c9c7fce0cc90aaf339213a49cfe7c242be3636817e427f31b，rows.json SHA256 74b915cc4f090f7e745ec2d080b9607fe37780d8751dee96a188670525929ed0。本地已回传报告数据，未逐文件远端本地全SHA比对，不能称归档资格完成。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- [CoGenAV official implementation](https://github.com/HumanMLLM/CoGenAV)

## Observations
- [result] 固定视频下Sync-C两组来源均20/20偏自身驱动音轨，CoGenAV Eq1大多同向。
- [decision] 撤回STFT幅度引导作为优先新迁移路线，小幅修改不保证音质或原生增益。
- [decision] 音频预对齐与视频后对齐已失败，不重复包装为新建议。
- [blocker] 缺少保留自然时间结构且有真实口型优势的监督，当前没有经验证的迁移算法。
- [next_step] 向用户呈现固定视频四格结果、实现边界及时间错配解释，不自动启动新训练。

## Relations
- continues [[Codex checkpoint - 2026-10-26T03-58-09Z - CoGenAV transfer options and directional metric disagreement]]
- relates_to [[固定视频交叉自然与TTS音轨的双同步指标对照 2026-10-26]]
