---
title: Codex checkpoint - 2026-10-26T03-17-02Z - Phoneme calibration held on reconstruction
  and OOV
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-26-t03-17-02-z-phoneme-calibration-held-on-reconstruction-and-oov
status: open
project: tts-exp
cwd: E:/Documents/tts-exp
repo_root: E:/Documents/tts-exp
repository: dogLeft233/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-26T03:17:02.0019560Z'
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a175f1-7164-7510-ae4b-f5af8e9d0fde
trigger: compact
model: gpt-6.1-sol
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-26T03-17-02Z - Phoneme calibration held on reconstruction and OOV

## Summary
已实际运行拟合集首轮校准，但prematched声码器重建与MFA词典OOV未过技术门；未冻结开发协议，40人11臂和确认集均未启动。

## Story
用户从解释TTS优势的讨论推进到实施：先100fit技术校准，合格冻结后40dev各首句完整11臂，联合正向相对原自然驱动为唯一主检验，60confirm不动。设计权威入口 [[逐音素差方向算法与验证方案]]，执行与结果见 [[逐音素静态动态差的拟合集技术校准与40人验证]]。

本轮建立隔离run和fit-only入口，校验完整100人/300来源/1200自然与L17音频的canonical WAV SHA、16k单声道与长度。按speaker排序0,5,…95，各选source ID最小的一句，形成固定20人80音频技术子集；没有按分数选句。CPU完成20自然重建与80条MFA，未占GPU或修改其它聊天的生产任务。

固定kNN-VC实际commit c616845c4e309e24d5927f15adbdf277a3d65358且clean；fast path真实output_layer=6，no VAD/no input level normalization。实际conv RF400、hop320、中心199.5样本。候选输入右补零至ceil(N/320)帧，解码右裁回N后匹配自然RMS；该合同尚未被最终准入。WavLM SHA6fb4b3c3e6aa567f0a997b30855859cb81528ee8078802af439f7b2da0bf100f；prematched SHA f924c7632c6eaf99004386d62293e124419e33582573552a6ae976eb88ed2dd5。

20句重建STOI中位0.809533500449477，未达到重建前写入的本轮工程底线0.85；S0200为0.7023285405348269低于单句0.75。相同长度/RMS、finite和无削波通过；全局包络10ms-grid均0ms，独立1ms-grid范围−2至+2ms、中位0，统一延迟补偿0样本。不是明显共同延迟可解释的问题。0.85/0.75只是本轮预写工程准入标准，不是STOI通用阈值或人类质量真值。

MFA3.4.1首次因进程PATH缺已有fstcompile失败，原日志保留；独立repair脚本仅加mfa3/bin后80/80成功。涯/撰/膛在自然与三个TTS中均词典OOV，共12spn。初始alignment_audit简单phone key计数误含unknown；独立alignment_readback已按同文字位置+字内完整phone序列匹配、排除unknown且保留声调，修正为2514有效配对phone。不是应用相位/类别门后的动态覆盖率，也没有真实边界真值保证。

因此按既定失败策略停止推进。没有全fit模板/稳定性、ASR/事件/模板保留性后续门、随机控制、重复TFG/口部校准、协议冻结、新TFG视频或SyncNet评价。不能将本轮依赖失败写成差方向无效；也不能静默换普通声码器、降门、删句或补词典后宣称原协议通过。

## Working State
权威产物在Ubuntu /home/wjj/tts-audio/tts-exp/runs/phoneme_direction_20261026_attempt01/。本地同名run有16份报告/JSON凭据且逐SHA读回一致；重建WAV、特征、TextGrid与完整MFA运行日志仍在Ubuntu。技术作业已结束；原数据生产控制器独立继续，其当前进度需另查。约55GB空间是启动时读数，不是后续保证。

本地分支仍codex/cross-tfg-audio-head-20261003、HEAD未变；四个新脚本与一个测试未跟踪。仓库大量既有脏文件与本轮无关，不能撤回或打包混提交。GitHub connector已核仓库，无当前分支PR；未验证GitHub可解析此SHA。

## Changed Files
- scripts/experiments/phoneme_direction_20261026/calibrate.py：fit-only准备、重建、质量、MFA和初检。
- scripts/experiments/phoneme_direction_20261026/repair_mfa_path.py：保留原失败的PATH修复启动。
- scripts/experiments/phoneme_direction_20261026/audit_reconstruction.py：独立WAV与STOI读回、细分包络延迟。
- scripts/experiments/phoneme_direction_20261026/audit_alignment_readback.py：独立TextGrid核对与unknown排除。
- tests/test_phoneme_direction_calibration_20261026.py：隔离/确定性选择/身份/unknown/声调序列检查。
- basic-memory/Research/逐音素差方向算法与验证方案.md：授权与technical_hold边界。
- basic-memory/Experiments/逐音素静态动态差的拟合集技术校准与40人验证.md：本次执行合同和技术结果。
- basic-memory/Research/研究主线与证据地图.md：添加依赖失败入口。
- runs/phoneme_direction_20261026_attempt01/report.md：可读技术报告与对应JSON凭据。

## Verification
本机unittest6项通过，compileall通过；16份本地交付逐SHA一致。远端100fit1200源字节与格式通过；20重建读回STOI差0；80TextGrid独立读取一致。Git diff --check未见问题（新文件仍未跟踪）。文档/最终状态本地读回确认technical_hold、未冻结、dev/confirm false。没有跑全仓测试，无新TFG或评分。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- 本地HEAD `2b71639c3b555293375d248350999377168d515d`，仅本地证据。
- run报告与JSON为当前结果权威，旧MFA失败和初检错误记录保留，由readback解释修正。

## Observations
- [result] 首轮技术校准未过；STOI中位0.809534，三个OOV字共12spn，40人评价未启动。
- [decision] 保留完整11臂、联合唯一主检验和确认封存；不为通过技术门降低标准、换样本或自动换模型。
- [blocker] 当前固定prematched路线重建质量未过本轮工程底线；OOV策略也未解决，开发协议不能冻结。
- [next_step] 先基于保留产物解释prematched重建损伤并明确OOV处理，再提出独立校准修订；不自动更换模型、降门或运行开发集。

## Relations
- continues [[Codex checkpoint - 2026-10-26T03-03-47Z - Phoneme direction design confirmed and implementation authorized]]
- relates_to [[逐音素差方向算法与验证方案]]
- relates_to [[逐音素静态动态差的拟合集技术校准与40人验证]]
