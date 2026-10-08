---
title: 37-mfa-linear-intraphone-time-warp
type: experiment
permalink: tts-exp/docs/experiments/37-mfa-linear-intraphone-time-warp
status: concluded
date: '2026-09-25'
cohort_size: 27
artifact: runs/mfa_linear_intraphone_time_20260925/analysis.json
tags:
- mfa-linear
- intraphone
- time-warp
- wav2lip
- syncnet
---

# MFA-linear 音素内部非线性时间映射实验

## 问题和预先固定的设计

接续 [[36-mfa-linear-heldout-speech-residual]]：缺失停顿修复后，MFA-linear 在纯发声区仍与自然语音有明显 mel 和口型差距。这里检验一个更窄的解释：**音素起止点已经对齐，但音素内部的声学事件发生时刻不同**。如果如此，在不改音素边界、停顿、长度和声码器的情况下，有界的音素内部非线性重采样应改善以自然音轨评估的 Wav2Lip 视频。

名单先于本批评分冻结：从 AISHELL-1 n100 中排除前两批 10+17 条样本，再取自然音频至少 4.8 秒的全部 27 条，涉及 10 位说话人。样本 ID 与前两批不重复，**说话人并不独立**；n100 已无新的说话人可用。未用本批 SyncNet 分数选样，但部分原始 n100 样本可能在其他项目实验中被评分，因此不称作完全未见验证集。冻结清单、输入哈希与参数见 `runs/mfa_linear_intraphone_time_20260925/protocol.json`。

五路驱动视频均使用相同脸视频、Wav2Lip GAN、25 fps、`--nosmooth`、冻结 WavLM-L6/HiFi-GAN，音频均裁成自然语音的精确采样长度。五路**评价音轨都是同一条自然语音**，并逐条检查官方预处理后解码 PCM 完全相同。

| 条件 | 驱动音频 | 时间映射 |
|---|---|---|
| N | 原始自然语音 | 自然基线 |
| F | 已修复静音/停顿的 MFA-linear | 音素内线性映射基线 |
| T | F 的 WavLM 特征 | 按自然/TTS 匹配音素内部谱通量事件设一个锚点，端点固定，分段线性重采样 |
| P | F 的 WavLM 特征 | 同一锚点反方向移动，对照时间方向 |
| O | F 的 WavLM 特征 | 在自然/TTS 的 40-bin log-mel 上选择每个音素的有界单锚点，诊断性声学引导时间映射 |

T/P 只考虑至少 120 ms 的匹配音素，事件须位于音素中间 25%–75%；源锚点移动 15–60 ms，最大相位移动 0.18，音素首尾与总音频长度保持不变。51 个 T 锚点分布在 23/27 条，绝对移动中位数 28.8 ms；4 条无锚点。O 允许 0–60 ms 移动，且 40-bin mel 形状误差至少改善 3% 才接受，共 235 个锚点、覆盖 27 条。O **读取了自然语音声学信息来选择映射**，不是可部署算法；它也只是单锚点候选族，不是所有非线性映射的上界。O 的协议在查看本批 SyncNet 分数前冻结于 `oracle_protocol.json`。

主终点预先定为 27 条 T−F 自然音轨整句 Sync-C；另看 Sync-D、反向 P、自然基线 N、声学引导 O，以及自然 N 最佳偏移处的纯发声/目标窗口局部 SyncNet 距离。局部距离越低越好，但**不是 Sync-C，也不是音素级分数**。说话人等权均值的 95% 区间用固定种子、按 10 位说话人重抽样 20,000 次计算；这些区间是描述性不确定性，并非新说话人泛化保证。

## 渲染与评分核验

GPU 曾两次失联并伴随服务器重启，先前 CPU/混合设备的部分产物单独存档。最终五路视频统一采用 CUDA 上的 Wav2Lip 与同一份 132 帧人脸检测框；26 个已完成的同协议 GPU 视频复用，另 109 个由单 GPU 进程生成，均核验 CUDA 日志。**135/135 个最终视频通过目标帧数检查**，没有混入 CPU 渲染。过程见 `render_protocol.json`、`face_boxes_132.json`。

官方 `run_pipeline.py --min_track 100` 在每条 N 视频上做人脸追踪一次。F/T/P/O 使用该条 N 的轨迹，复现官方 ffmpeg/JPEG/XVID 裁剪和各自自然音轨解码。这是**共用人脸几何的 SyncNet V2 配对协议**，不是五路各自独立人脸追踪。当前 GPU 运行的 a1_003 上，用 N 轨迹重做 N 裁剪与官方 N 裁剪的 182 帧解码画面逐像素完全相同。此前独立 F 与共用轨迹 F 的校准差为 Sync-C +0.039；方法详情见 `crop_protocol.json`。最终 135/135 裁剪成功，27 条五路自然 PCM 全部逐样本相同，SyncNet 搜索 offset 均未触及边界。

## 整句结果

所有分数为 27 条等权平均。ΔC、ΔD 均相对 F；Sync-C 越高越好，Sync-D 越低越好。

| 驱动视频 | 平均 Sync-C ↑ | ΔC vs F | 平均 Sync-D ↓ | ΔD vs F |
|---|---:|---:|---:|---:|
| N 自然 | 5.948 | +0.454 | 7.075 | −0.815 |
| F 线性基线 | 5.494 | 0.000 | 7.890 | 0.000 |
| T 事件锚点 | 5.481 | −0.013 | 7.890 | +0.000 |
| P 反向锚点 | 5.469 | −0.025 | 7.906 | +0.016 |
| O 声学引导 | 5.518 | +0.023 | 7.858 | −0.031 |

T−F 的说话人等权平均 ΔC 为 −0.016，95% bootstrap 区间 **[−0.065, +0.030]**；12/27 条 T 高于 F。只看确实设了锚点的 23 条，T−F 为 −0.010，11/23 上升，区间 [−0.081, +0.032]。P−F 全体均值 −0.025，O−F +0.023，后者区间 [−0.037, +0.065]；这些小幅变化应结合下述渲染波动解读。自然 N−F 仍为 +0.454，22/27 条自然更高，10/10 位说话人的组内均值均为正；说话人等权区间 [0.287, 0.644]。五路整句最佳 offset 都集中在 −2/−3 帧，没有出现映射后统一偏移的改善。

## 声学与局部窗口

纯发声区 Wav2Lip 实际 80×16 mel 块相对 N 的平均 MAE：F/T/P/O = **0.7643/0.7645/0.7657/0.7522**。T−F 为 +0.0002，只有 11/27 条 mel 更近；O−F 为 −0.0121，23/27 条更近，说话人等权区间 [−0.0263, −0.0050]。因此这个单锚点候选族可以改善部分声学接近度，却没有同步带来可靠的 Sync-C 提升。

在 T 预先指定的事件目标窗口，23 条有锚点样本的 T−F 固定 N 偏移 SyncNet 距离平均 **+0.043**，说话人等权区间 [−0.069, +0.195]；正值表示更远。全发声区 T−F 距离为 +0.004。O 在自身目标窗口的 O−F 距离为 −0.026，区间 [−0.116, +0.196]。这些局部结果未显示有界单锚点映射在真正改变的位置稳定改善自然音轨对应。

4 条无事件样本的 F/T/P 驱动音频 SHA256 完全相同，但分别运行 Wav2Lip 后，T−F Sync-C 为 −0.042、−0.079、+0.008、−0.007；绝对差中位数 0.025、最大 0.079。原始视频解码画面也并非全同，说明独立渲染/编码带来可测波动。预先定的 27 条主分析保持原值；事后将这 4 条“无干预”视作严格相同视频的敏感性均值为 T−F **−0.008**，有锚点 23 条的均值为 −0.010，均不改变方向判断。T 的平均变化量小于本次空干预视频差异尺度；不能把 P 的小幅下降断言为可靠的时间方向效应。逐条控制见 `sensitivity.json`。

## 结论与边界

在这批 27 条、单脸 Wav2Lip、自然音轨 SyncNet 协议下，**按音素内部谱通量事件做 15–60 ms 单锚点非线性时间映射，没有恢复 MFA-linear 相对自然语音的约 0.45 Sync-C 差距**。声学引导的单锚点 O 虽减少了 mel 误差，也未可靠提升 Sync-C。证据不支持把当前剩余差距主要归因于这个有界单锚点时序误差；同时间声学形状、音素内容/共发音及 Wav2Lip 对输入的响应仍需分开检验。

这**不排除**更密集或更大幅度的非线性映射、多音素跨边界错位；O 也不是全局最优时间 oracle。样本说话人与既有实验重合，部分原始样本可能在别处被评分，不能称完全独立验证。单脸与单模型限制外推；同音频多次渲染的波动限制了对小于约 0.08 Sync-C 的逐条变化作归因。下一步若继续时间路径，应先让同音频视频渲染可重复，或在同一视频的固定表示上做对照；随后把局部声学形状替换与时间变换在独立样本中分别评估。

## 复现产物

- `runs/mfa_linear_intraphone_time_20260925/experiment.py`、`oracle.py`：固定样本、事件/反向/声学引导单锚点、音频生成；`manifest.json` 保存逐条音频哈希。
- `inference_cached.py`、`render_fast_shard.py`、`face_boxes_132.json`、`render_protocol.json`：GPU Wav2Lip 与人脸检测框来源；`video/` 保存最终 135 个视频。
- `common_crop_shard.py`、`crop_replay.py`、`crop_protocol.json`：官方 N 轨迹及配对共几何裁剪；`pipeline/`、`common_pipeline/` 保存 135 个裁剪。
- `acoustic.py`、`score.py`、`analyze.py`、`sensitivity.json`：mel、PCM 核验、SyncNet V2 分数及局部窗口；原始统计在 `acoustic.json`、`scores.json`、`analysis.json`、`matrices/`。
- `plots/summary.png`、`plots/summary.pdf`：配对 Sync-C 与 mel 变化图。图中浅灰带为这批无干预样本观察到的最大绝对 Sync-C 差 0.079，仅用于展示渲染波动量级。

## Observations

- [status] concluded
- [result] 27 条自然音轨 Sync-C N/F/T/P/O=5.948/5.494/5.481/5.469/5.518；主终点 T−F=−0.013，10 说话人等权 95% CI [−0.065,+0.030]。#syncnet
- [result] O 的纯发声 Wav2Lip mel MAE 比 F 低 0.0121（23/27 更近），但 Sync-C 仅 +0.023，区间跨零。#wav2lip
- [limit] 4 条零锚点同音频视频仍有绝对 Sync-C 差中位数 0.025、最大 0.079；说话人与旧批重合。#reproducibility
- [conclusion] 当前有界单锚点音素内部时间映射不足以解释或修复 MFA-linear 的主要自然音轨差距。

## Relations

- extends [[36-mfa-linear-heldout-speech-residual]]
- relates_to [[35-mfa-linear-speech-residual-alignment]]
- relates_to [[32-mfa-linear-vocoder-wav2lip-split]]

## Changelog

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 完成 27 条五条件音频、GPU 视频/裁剪/评分、局部时间与声学分析、空干预渲染波动核验 | September 25, 2026 | user（音素内部非线性时间差实验及 GPU 继续请求）；agent（执行） |
