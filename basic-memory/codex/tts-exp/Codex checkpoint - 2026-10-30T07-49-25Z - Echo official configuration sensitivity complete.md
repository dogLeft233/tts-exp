---
title: Codex checkpoint - 2026-10-30T07-49-25Z - Echo official configuration sensitivity
  complete
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-30-t07-49-25-z-echo-official-configuration-sensitivity-complete
status: completed
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-30T07:49:25Z'
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

# Codex checkpoint - 2026-10-30T07-49-25Z - Echo official configuration sensitivity complete

## Summary
EchoMimicV3-Flash的6来源官方配置诊断已完成并独立复核：LUFS影响较小，81帧上限及seed显著改变点估计，原20来源结果保持冻结。

## Story
原目标是自然语音与TTS驱动TFG的配对多评分器实验，报告77已经完成。用户怀疑Echo分数较低，最新授权按官方默认做对照，重点检查音量归一化与推理参数，遵守192规则且不能覆盖旧结果。已当次通过Edge读取飞书1048手册，重新核验个人目录与资源；全程个人环境、单GPU、CPU4，无共享环境或他人进程改动。

预先按自然长度在同句12与扩展8两层各选最短/下中位/最长来源，各取自然与既有C1，合计6对。单独改变整段LUFS-23、81帧上限、seed43，另测官方组合及只改变评分音量。所有条件按来源共同前65/81帧评分，固定RMS24 PCM；不是全20、三TTS或九评分器重测。官方sample_size768不会放大同一512图，仍保留BF16 loader、mask语义SDPA和CPU offload兼容适配，不宣称无修改官方CLI或论文benchmark复现。

权威报告 [[78-echomimic-v3-official-control]] 已写入并读回。旧配置自然4.887/TTS4.556/Δ−0.332；仅LUFS4.818/4.610/Δ−0.208；仅81帧4.753/5.494/Δ+0.741；仅seed43为3.895/4.622/Δ+0.727；官方组合4.128/5.149/Δ+1.021，5/6来源TTS更高。官方自然下降0.759、TTS增加0.594，不能说整体绝对分普遍变高。6来源探索性区间不是全20确认性证据；长度干预还共同改变特征上下文与latent时序形状，机制尚未分解。后续全20/多seed是另一个实验，当前不自动启动。

## Working State
全部GPU工作已结束。本地入口results/echomimic_v3_flash/official_control_20261030_attempt01/，有paired_comparison.csv/json、scores.json、comparison.png/pdf、contact_sheet.png、local_readback.json、postrun_resource_check.json及evidence/。图和接触表已实际查看，仅作工程QA。

远端run /home/data2/wangjiajun/av-evaluation/jobs/echomimic_v3_official_control_20261030_attempt01；代码experiments/echomimic_v3_official_control_20261030。固定run入口和带reservation的修复启动器已执行，不可盲目重跑。50生成（2重放+48干预）、74评分及4时移控制完成。

新归档76292497字节、1281成员/1280内容文件，SHA256 9c87ccf21c39b60a5f231fbc9859a3c6d9b0c45402e6e5d5831243d027b23588；manifest697c21c15d70ae9a6ccc4ec77b04c233c6dbf5c1c62aaa392e55fce23093a10f。模型视频、基线视频、PCM、原始矩阵、像素审计和执行修复源码全部保留；可无损重建的评分前缀容器及解码缓存仅留哈希清单。归档保持不可变，资源回查是归档后的独立补充。

旧报告77归档SHA ca8cd17cb6974016e69e3470c9cd01fb9927bb2d9d0b8eefca2f51ff3f09380b已再次确认不变。不能把本6对Δ+1.021与旧20来源Δ−0.191当作相同cohort/时窗/三TTS定义的直接对比。

机器本地临时SSH助手C:/Users/dogle/.codex/tmp/server192_retry.py与echo_control_collect_stream.py使用运行时DPAPI解密、固定主机指纹。SSH间歇性中断后，用户明确要求多试；单通道分块传输及原子文件核验最终成功，未改代理/SSH服务。临时助手不是持久项目接口，最终采集进程已退出，不需继续后台监控。

## Changed Files
- 新未跟踪scripts/experiments/echomimic_v3_official_control_20261030/：control、score_repair、shift_control、finalize、deliver及各执行包装器/README。
- 新未跟踪tests/test_echomimic_v3_official_control_20261030.py，3项测试。
- CONTEXT.md追加报告78入口，仍100行。
- Basic Memory报告78新建；SYNC-C-SCORES与echomimic-v3-flash部署说明追加独立诊断小节，均读回确认。
- C:/Users/dogle/.codex/skills/server-192/references/personal-deployment.md追加精确前缀、特征/评分音量拆分、UTF8/LF和原子传输/独立执行修复经验；主SKILL及手册约束未改。

仓库1357行dirty/untracked，继承原1355行；没有清理、提交或修改他人工作。当前源文件未提交，应视为机器本地工作状态；服务器实际执行源码随证据包保存。

## Verification
所有50媒体检查、两条baseline解码像素重放及分数复现通过。帧数未实际变化的四条cap_no_op控制，像素和Sync-C一致。74原始矩阵FP64独立重建最大误差2.963202e−6；本地来源均值与全部bootstrap区间独立复算通过；1280内容文件大小/SHA全通过。4个真实±400ms控制准确使offset反向移动±10帧；不要求搜索时延后的Sync-C一定下降。SyncNet pip-freeze集合与旧实验相同。

首次评分预检查捕获H264包复制截81帧遗漏3.20秒并带入3.24秒，尚未产生评分。保留syncnet_preflight_failed_attempt01和原control.py，独立score_repair.py按ADR003绑定旧计划/新源码，使用解码后截帧及无损x264 CRF0/r25/bf0，逐片验证像素、PTS和PCM。trim/passthrough试验02仅80帧失败也保留；试验03验证正确81帧和像素哈希。post_attempt02 Windows CRLF在评分前失败，保留源和日志，LF-only post_attempt03继续，仅评分重做，没有重跑生成。执行修复证据在run/execution_repair_attempt02.json及code_repair/。

3项本地测试、5Python编译通过。skill validator初次因本地缺PyYAML失败，经uv临时提供PyYAML后quick_validate通过，未给服务器安装依赖。图和联系表已查看。生成累计43.20分钟，不含加载，PyTorch峰值allocated13.260GiB/reserved14.467GiB，5秒NVIDIA采样峰值15174MiB；是单A6000/22GiB allocator实测，非物理24GB卡测试。最后回查本人实验进程为空，GPU驻留4MiB/空闲48666MiB/利用率0%，未操作他人任务。

Git根、branch、SHA、状态刚核验；GitHub get_repo确认canonical dogLeft233/tts-exp，当前head PR搜索为空。SHA仅本地核验，不构造未验证的远程commit链接。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- 本地SHA `2b71639c3b555293375d248350999377168d515d`
- [Official Flash configuration](https://github.com/antgroup/echomimic_v3/blob/7e89489ca51c0d008fc1963ec6c03fc5bd0b9397/run_flash.sh)
- basic-memory/docs/experiments/78-echomimic-v3-official-control.md
- results/echomimic_v3_flash/official_control_20261030_attempt01/local_readback.json

## Observations
- [result] 官方组合在6对共同前缀将ΔSync-C从−0.332变为+1.021；主要敏感因素为帧数/seed，驱动LUFS影响较小。
- [decision] 保持原20来源结果与本6对探索性对照分别冻结；不把评分变化称作感知质量变化。
- [verification] 50生成、74评分、4时移控制、1280文件及全部bootstrap读回通过，旧归档不变，服务器任务结束。
- [next_step] 向用户交付报告78、6来源对照表与适用限制；本次工作无需再启动GPU。

## Relations
- continues [[Codex checkpoint - 2026-10-30T06-12-30Z - Echo score audit and official control authorized]]
- documents [[78-echomimic-v3-official-control]]
- relates_to [[77-echomimic-v3-chinese20-nine-scorers]]
- follows [[003-immutable-execution-repair-provenance]]
