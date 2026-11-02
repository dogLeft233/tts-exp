---
title: 同一10人上的Qwen1.7B与音量归一化对照2026-10-23
type: note
permalink: tts-exp/experiments/同一-10-人上的-qwen1.7-b-与音量归一化对照-2026-10-23
status: concluded
run_id: qwen17_level_comparison10_20261023_attempt02
branch: codex/cross-tfg-audio-head-20261003
base_commit: 2b71639c3b555293375d248350999377168d515d
tags:
- qwen
- tts
- amplitude
- no-training
---

# 同一10人上的Qwen1.7B与音量归一化对照2026-10-23

> 路径说明：`${USER_HOME}`代指原记录对应的个人主目录，不表示接收者已有该文件。定位归档时继续核对资产映射与原运行凭据；示例中的相对仓库路径从仓库根解释。

用户希望判断刚部署的完整1.7B是否改变此前本地0.6B与云端TTS的Sync-C结论，并对照音量归一化前后。实验复用云端原10人及自然、0.6B和云端的70个归一化评分，仅新增30条1.7B合成和130个下游单元；未训练、未新增云端API调用。

## 设计与材料
最终run：`runs/qwen17_level_comparison10_20261023_attempt02`，报告`report.md`，逐单元`all_draw_scores.csv`，完整统计`analysis.json`与`reliability_sensitivity.json`。每位来源3个TTS draw先平均，统计n=10人；自然每人1条。四组N/L06/L17/CLOUD各raw与norm，总200评分。沿用原face3.png、Wav2Lip GAN、nosmooth/batch4/25fps与官方run_pipeline→run_syncnet，保留其原MP3中间链。1.7B沿用0.6B原文字、自克隆参考、seed及实际解码参数。

| 组 | raw Sync-C | norm Sync-C | norm−raw |
|---|---:|---:|---:|
| 自然 |6.737|7.108|+0.371|
| 0.6B |6.604|6.861|+0.257|
| 1.7B |7.266|7.517|+0.251|
| 云端Plus |7.522|7.670|+0.149|

## 判断
norm三主比较使用Bonferroni调整98.333%来源bootstrap区间：
- 1.7B−0.6B +0.656 [+0.199,+1.213]，9/10提高；raw +0.663。排除旧0.6B两条无EOS/时长异常draw，norm仍+0.418，描述性95%[+0.217,+0.621]。严格9人norm +0.532，95%[+0.190,+0.982]。
- 1.7B−自然 +0.409 [+0.001,+0.767]，8/10提高；严格9人+0.361，95%[-0.004,+0.692]。相对自然的证据脆弱，不写成稳健普遍的TTS优势。
- 1.7B−云端 -0.154 [-0.421,+0.138]，4/10提高，尚未明确；跨零不是等效。

归一化提高四组平均绝对分数，TTS相对自然的点差缩小，但缩小量区间跨零或接近零。1.7B对0.6B优势归一化前后几乎相同，整体RMS或两条停止异常均不能充分解释。模型checkpoint/配置联合差异不等于纯参数量因果；云端Plus不视为同一开源1.7B权重。归一化同时作用驱动和评分音频，未分离生成端和评价端；Sync-C不能代替独立人类嘴型准确度。

## 执行与核验
attempt01合成30/30 actual exit0、全部EOS，音频门控因旧0.6B/S0016/draw3原始音频8点超限退出1，0视频/评分。原失败保留。ROOT沿用此前用户保留完整来源并报告严格子集的偏好、对话先披露，冻结attempt02仅此8点标准PCM16饱和例外；无针对8点的新用户逐点确认。另报排除整个S0016的严格9人，不按评分挑样本。音频RMS变动约-0.000186dB（不含普通量化）。

attempt02音频、130视频、130官方评分均actual exit0。原父包装在索引前终止，SSH session25151不能恢复，父退出码/stderr/原因未知，不认定BrokenPipe或原父完整成功。ROOT仅补冻结CPU索引actual0，随后CPU分析/补充/审计/报告均actual0，无新forward。

100原始波形→200PCM逐样点独立重放PASS；70旧norm桥接精确一致；30新draw参考codes/seed/解码/EOS审计PASS；200评分与track、90主统计+42补充独立复算PASS，最大误差3.55e-15；9项测试通过，PNG实际查看PASS。1.7B30条合成671.887s，PyTorch峰值allocated9.33GiB/reserved14.80GiB，不能把单条部署smoke当批量显存上限。

协议SHA `3f1b29a3305065e03578127db1a3d6e11afcc51b60eff94bb9852a4e32a6ceb8`；analysis SHA `d38af3aa963496444c8561ba98bece051583e1380f2d84b8d95d60ff90c84acb`。归档包含新旧证据与失败attempt01，权重正文仍为外部已部署依赖；本机归档和读回以run内archive_receipt.json为准，不声称网盘备份。

## Observations
- [result] 1.7B相对0.6B的Sync-C增益在raw/norm、资格补充及严格9人中均保留；相对自然不够稳健，相对云端未明确。
- [decision] 保留全部draw为主结果，EOS/时长资格与严格9人仅作补充，不训练、不扩样。
- [limitation] 固定10人观察队列、单下游人脸与Wav2Lip；不外推普遍人类同步质量。

## Relations
- informs [[研究交接总览]]

## 完整归档读回

本机C盘完整tar：`${USER_HOME}/.codex/tts-exp-artifacts/qwen17-level-comparison10-20261023/qwen17_level_comparison10_20261023_attempt02.tar`，1,643,427,840字节，SHA256 `4ebe82b42e2a7576e6674417d9b1a876fb70a3013039c8d271612637cb341b7a`。44,897清单成员逐SHA读回PASS，200行评分CSV对档内JSON/音频绑定PASS；读回凭据位于同目录local_readback_receipt.json及run镜像archive_receipt.json。权重正文不在归档，无网盘备份声明。封存后不修改档内结果；后加运输回执位于tar外。
