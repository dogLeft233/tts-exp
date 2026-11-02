---
title: 015-same-gpu-deterministic-controls
type: decision
permalink: tts-exp/docs/adr/015-same-gpu-deterministic-controls
repository: dogLeft233/tts-exp
project: tts-exp
decided: '2026-10-07T13:28:38.3474600+00:00'
git_sha: b5bcd8b1759a902d4e154fe183dfb9effbd7ca5d
status: open
tags:
- audio-head
- determinism
- evaluation
---

# 015-same-gpu-deterministic-controls

## 决定

在迁移后的同一物理GPU和确定性设置下生成增强头候选及完整自然/RMS对照。新Dev540使用180候选加新生成360对照；旧900/540归档保留为历史结果与输入来源，不作为新候选的配对控制。训练context、Fit40 producer、零头和120步训练也须匹配同一执行设置。

## 背景与理由

用户要求产出能跨TFG使用的自然音频增强头，方法通过门槛包括相对RMS平均额外+0.050。服务器2026-10-07迁移到westd:18625，GPU UUID从旧机变为GPU-d09622d0-3535-e662-bb45-3df2b049fcea。40原始WAV和模型仍完整，旧SHM缓存丢失。

实际旧pilot QC05使用相同PCM、portrait、seed和模型，identity/repeat Cbest为6.311/6.417，逐query138×31矩阵全部不同，repeat差+0.106已超过方法门槛。根已下载27项media、matrix、crop、tracks、日志并核验原字节，保留该控制失败。QC06增加固定单线程、严格torch确定性、TF32关闭及进程启动种子后，实际108.797秒完成三臂、rc0；identity/repeat生成MP4、elementary stream、裁剪、track/bbox和逐query矩阵完全相同。自然PCM仍92225样本且全SHA相同。

QC06 identity/repeat Cbest6.339759826660156，RMS6.397294998168945，RMS−identity为+0.057535171508789；QC05相同端点差+0.178926467895508。该曝光pilot只用于工程资格，不参与方法选择、Fit/Dev/Confirm效果。确定性组合消除了重复波动；没有把其中一个设置单独认定为差异原因。对照与候选必须执行同一设置，避免把执行差异混入增益。

## 备选与代价

继续复用旧360控制最节省生成成本，但实际同输入波动与设置改变足以影响+0.050门槛，不能建立合格配对。只在一个pilot上计算校正值也不合适：它没有覆盖20条Dev、三个模型或非线性交互。因此重新生成完整360控制，保持原20clip/10spk、3seeds、3backends、自然原PCM/clock、完整失败分母与共同支持，不用Dev调头参数。

新执行增加360次生成/换轨/评分与相应存储，按当前cgroup/SHM/overlay容量准入并保留严格备份。新控制计划、collector/archive及join adapter使用实际来源晚绑定，不硬套旧archive SHA。旧08训练部署不覆盖；需要新context/source绑定时使用新的producer05和training09版本。Confirm80/40spk与LatentSync仍待最终方法冻结。

## 权威证据与影响

QC06实际终态 `.codex/scratch/qc06_packet_attempt01/05_root_read_terminal_status_attempt02.json.response.json`，11631B/SHA `0b81ea4667243cdce94050d59058423ac0e3aea6850db1385c05c2892eb05136`。全部27项完整读回位于 `.codex/scratch/qc06_readback_root_attempt01/`，根字节/SHA核验receipt为`root_whole_file_readback_qc_attempt01.json`。manifest53551B/SHA `f09cc4f339c43e96ea532cdaac4f9818fc42caec6ba7b212dcfe7ae4dc2c5436`；terminal10195B/SHA `a4fa225dc299f1b89c5cd8fc20b915df7f613c51b491280624895eef052f7269`。

确定性源 `scripts/experiments/cross_tfg_audio_head_20261003/engineering_qc_oldpilot_attempt06.py` 31789B/SHA `cf4d34832ee93bf5a4d2e4be349dad15469e423888c0a1c41bbbdf53f2a4726d`，提交于本地未发布b5bcd8b。受影响：新context导出、Fit40 producer/训练绑定、Dev360/180执行、540联合分析和最终报告26。

## Observations

- [decision] 新候选与自然/RMS控制在同一新GPU和确定性策略下生成；重新生成完整360对照，旧归档保留历史来源。
- [result] QC05同输入repeat Cbest差+0.106；QC06三臂rc0且五项repeat全部exact，实际RMS差由+0.179变为+0.058。
- [rationale] 执行波动足以改变额外+0.050门槛，不能用旧机结果或单pilot校正建立新配对。
- [alternative] 复用旧360对照成本低，但无法满足实际同设置配对要求；完整重跑已获用户授权。
- [consequence] 增加360次对照生成及新来源绑定，保持原Dev队列/种子/统计与Confirm封存。

## Relations

- affects [[开发跨 TFG 的自然音频增强头]]
- affects [[跨 TFG 自然音频增强头的训练与原音换轨验证]]