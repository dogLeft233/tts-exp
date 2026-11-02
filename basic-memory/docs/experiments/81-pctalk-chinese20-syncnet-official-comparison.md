---
title: 81-pctalk-chinese20-syncnet-official-comparison
type: experiment
permalink: tts-exp/docs/experiments/81-pctalk-chinese20-syncnet-official-comparison
status: concluded
date: '2026-10-31'
scope: SyncNet checkpoint only
---

# 81 — PC-Talk 中文20来源Sync-C与官方I2V分数对照

✅ CONCLUDED（仅SyncNet阶段，评价器观察性证据），2026-10-31。HDTF TalkLikeYou维持用户暂停；其余八评分器未启动。

20来源、20N+60T逻辑媒体，三TTS先来源内平均；41唯一推理与评分。固定3.png、RMS−24、16k/25fps、person192/lip1/无EMC/seed42。

|支持集|来源数|自然C|TTSC|T−N|95%来源bootstrap CI|T较高|
|---|---:|---:|---:|---:|---|---:|
|same_reference12|12|7.182|7.639|+0.457|[+0.020,+0.855]|9/12|
|extended_reference8|8|7.393|7.932|+0.539|[-0.313,+1.316]|5/8|
|all20|20|7.266|7.756|+0.490|[+0.074,+0.884]|14/20|

本轮只完成SyncNet；未完成九评分器家族，没有Holm9显著性结论。区间为描述性边际区间。重复TTS draw不是独立合成重复。

官方论文Table1 I2V HDTF LSE-C9.370、MEAD-Neutral8.190。本轮自然相差-2.104/-0.924，TTS相差-1.614/-0.434。官方基准是自然音频，TTS差距仅背景，不是同测试集复现误差。

官方来源：https://arxiv.org/html/2503.14295v3#S4.SS1 。论文为HDTF/MEAD、10秒跨身份、多测试人物；本轮中文AISHELL短句、同固定脸、统一电平，数据/语言/脸/长度不同。公开代码无评分脚本，论文具体style/lip参数、crop与SyncNet权重SHA未齐全；CLI默认lip1与README示例lip.6加happy控制也有差异，均未经本轮按分调参。不能以绝对差直接认定评分错误或模型失效。

评分核验：官方joonson/syncnet_python，FP32，syncnet_v2.model、vshift15/crop_scale.4；项目min_track25（上游100）。完整输入PCM重新无损封装，PTS/PCM逐值校验，视频原生尾部不足保留，不补帧/移音。原始31shift距离矩阵独立FP64归约与官方stdout一致，源/权重前后SHA不变。该核验排除这几类实现/绑定错误，不能验证作者完整协议或人类感知同步真值。

内部处理补充：官方run_pipeline.py默认FFmpeg AVI转码后的音轨codec实测为['mp3']；本轮按官方流程保留，并非从入口到SyncNet模型始终无损。实际41唯一人脸轨迹的长度范围59–195帧；min_track100会改变短片纳入资格，不能把本轮min_track25结果称完整论文默认协议。

预先选定smoke首对的只读音轨核查：自然和TTS内部音频相对原PCM的最大相关均位于+1105样点（+69.0625ms），不是逐值相同波形；首个自然片原生145帧而官方处理后轨迹147帧。该内部转码行为属于本轮保留的官方入口实现；SyncNet本身搜索±15帧。本轮没有进行PCM强制内部转码对照，不能量化其对C差值的贡献，也不能说69ms解释了论文分差。

## 实际实现与来源

PC-Talk官方commit `a0392744cbac857d96db1b99ceab70cf7b5f6c83`，权重revision `e6220816118bd8714ff57dc83e5906122b4e9168`；100源码文件与12权重SHA前后通过，模型源码未改。person192为推荐preset，lip1为CLI默认，无额外EMC/表情；face-index1避免固定3.png背景误检。与TalkLikeYou/FlashHead/Echo此前80WAV及图逐SHA相同；没有新云ASR/TTS或训练，没有按分数选择seed、幅度或风格。

评分器为joonson/syncnet_python revision `907c0b579c2e2d83f0eae1b2ac9e720cde4e5623`，官方gitblob通过；syncnet_v2.model SHA `961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442`。18源码/权重/guard文件钉住，eval-base包集合与FlashHead冻结记录相同。41唯一track平均框位置/大小均覆盖固定图主体脸，首条实际裁剪已目视，最佳offset全−2帧；轨迹长度59–195，14/41的现有轨迹长度≤100，不满足原默认严格>100帧轨迹门限（不重跑默认门限）。offset现象和首对MP3延迟不能单独解释平均C差距。

生成原片57–193帧，原生尾部少3106–3792采样点，不人为补帧/拉伸/校正offset；官方SyncNet内部转码另有上文记录的尾帧变化。自然/tts评分入口均绑定完整原始PCM；播放AAC为作者原生输出，音频尾部支持可能短于完整WAV。无人工盲评，不能把Sync-C当人类同步真值。

## 核验与交付

本机6478内容文件全SHA、41原始距离矩阵独立归约、三个来源分层均值/10000次bootstrap重算及80播放媒体导出PASS；服务器FP64归约与主FP32最大差2.563889e−6，全部与官方stdout三位小数吻合。测试2PASS，源码compileall通过。

来源级CI为边际bootstrap，未运行其余八项，没有Holm9结果，不根据CI声明九评分器显著效应。全20 14/20来源TTS均值高；主要同句参考12为9/12。官方自然基准与本次自然更相近；TTS没有对应的官方论文TTS测试，差距仅作背景。

远端run：`/home/data2/wangjiajun/av-evaluation/jobs/pctalk_chinese20_rms24_20261031_attempt01`。本机`results/pctalk/chinese20_rms24_20261031_attempt01/`含`report.md`、`summary.json`、`paired_scores.csv`、`per_video_scores.csv`、`index.html`和`video_index.json`；80播放原片在`results/pctalk/{natural_raw,tts_raw}/<source>/`。

证据包133628381字节，SHA256 `3604e0a57c20f4b5c5a680d4bd9f50a19059fb3c8cbd2fd4cfc9692b242b8c98`。`local_audit.json`与单列`terminal_receipt.json`为验收入口。旧generation_status RUNNING80仅历史状态，完成凭据权威，不用于重新launch。所有本实验PID已退出，GPU727终局2MiB/0%；不操作他人任务。服务器本轮通过飞书MCP应用身份读取1048手册后执行，个人data2/nohup/单UUID727/CPU4与affinity0–3，生成Torch45%与评分guard60%上限，635.197秒为41生成+封装校验累计而非纯推理性能，NVIDIA PID采样峰2350MiB。无网盘归档声明。

源码`scripts/experiments/pctalk_chinese20_20261031/`，测试`tests/test_pctalk_chinese20_20261031.py`；新增未提交，继承dirty未动。初次权重清单路径前缀解析失败在生成前修复；测试发现上游浮点取整边界并在生成前匹配原运算顺序，无科学前向重跑。

## Relations
- uses [[PC-Talk 192服务器部署]]
- completes [[PC-Talk中文20来源Sync-C与官方对照2026-10-31]]
- compares_with [[79-talklikeyou-chinese20-nine-scorers]]
- relates_to [[80-cross-tfg-nine-scorer-comparison]]
