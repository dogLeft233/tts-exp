---
title: 固定视频 SyncNet 监督 Gate B 结果
type: experiment
permalink: tts-exp/experiments/syncnet-supervision/固定视频-sync-net-监督-gate-b-结果
status: no-go
gate: B
sample_count: 15
gradient_nonzero_count: 75
optimization_total: 75
optimization_within_one_frame_count: 14
next_step: independent_critic_or_stop
tags:
- syncnet
- fixed-video
- gradient
- no-go
- mfa-linear
- literature
initial_optimization_within_one_frame_count: 2
lag_aware_loss_implemented: true
fixed_coordinate_loss_implemented: true
---

# 固定视频 SyncNet 监督 Gate B 结果

## Context

本线程验证 frozen SyncNet 是否能为 MFA-linear residual phone-duration warp 提供可靠的 waveform-level 时间监督。实验使用 AISHELL-1 中文固定脸协议，固定视频、候选音频可微变形、SyncNet audio branch 保留梯度；Wav2Lip 和文件级 SyncNet 只用于独立下游评估。

## Project validation

固定视频验证使用 15 条样本、15 个 speaker。已有固定视频曲线对偏移敏感，但 15/15 样本的最佳参考位置都约为 -2 个 SyncNet frame（-80 ms），说明当前视频/音频窗口协议有系统性偏移。

原始同索引 differentiable Gate B：

- 梯度非零：30/30
- MFCC finite/parity：通过
- 一帧内恢复：2/30
- 部分样本优化后 cosine 下降

lag-aware loss 之后：

- 在 ±15 个 SyncNet frame 的重叠窗口上计算 cosine；
- 中心 lag=+2 frame，temperature=0.05，lag penalty=0.05；
- soft aggregation 参与反向传播，hard selected lag 只作诊断；
- 梯度非零：30/30；一帧内恢复：10/30；
- `-160 ms` 已知偏移：0/15 恢复，平均只学习约 17.4 ms，而期望修正为 +160 ms。

lag-aware loss 提高了局部窗口错位的鲁棒性，但允许 lag 搜索吸收全局时移，因此不能单独作为绝对 phone-duration supervision。

### 固定坐标矩阵监督 A

为避免一次叠加多个设计，首先只实现固定坐标版本：

```text
visual[t] ↔ audio[t + δ0]
```

其中 `δ0=+2` frame 为一次性校准得到的协议偏移。候选音频经过完整 differentiable MFCC 和 frozen SyncNet audio branch，构造完整 visual/audio cosine matrix，并对固定 diagonal 使用 cross-entropy；没有加入 expected-lag regression、lag search 或独立 activity anchor。

15 条固定视频、5 种已知 shift（-160/-80/0/+80/+160 ms）、每项 40 步的结果：

- 梯度非零：75/75；
- candidate waveform 到 SyncNet audio branch 的 finite gradient：通过；
- 一帧内恢复：14/75（18.7%）；
- `-160 ms`：0/15；`-80 ms`：3/15；`0 ms`：10/15；`+80 ms`：1/15；`+160 ms`：0/15；
- Gate B：`NO_GO`。

固定坐标矩阵确实对已知 shift 有响应，但 waveform gradient 经常被错误的局部内容匹配吸收，不能可靠恢复全局绝对时移。因此没有继续叠加 expected-lag regression。

结果文件：

- `runs/aishell1_syncnet_supervision_validation_full/summary.json`
- `runs/aishell1_syncnet_supervision_differentiable_full/summary.json`
- `runs/aishell1_syncnet_supervision_lagaware_gateb_20260818/summary.json`
- `runs/aishell1_syncnet_supervision_fixed_coordinate_gateb_20260819_full/summary.json`
- `scripts/experiments/syncnet_supervision/online_syncnet_objective.py`
- `scripts/experiments/syncnet_supervision/validate_fixed_coordinate_syncnet_signal.py`

### 独立活动 anchor 预诊断

在不改训练代码的情况下，使用 crop video 下半脸帧间运动与 40 ms 音频能量做独立坐标相关性诊断：15 条样本最佳 lag 的中位数为 3 帧，分布为 -8 到 +8 帧，相关系数整体较弱，多个样本接近零或为负。因此不能把简单嘴部像素运动–音频能量差异直接当成可靠 absolute-lag teacher。

### 独立 learned lag critic 方案

由于 raw lag-score 曲线在 15 条 Gate B 样本上呈现稳定的人工 shift 映射（-160/-80/0/+80/+160 ms 对应 -2/0/+2/+4/+6 frame），训练了一个只接收 17 维 lag-score curve 的小型绝对 lag 分类 critic。训练数据来自 35 条非 Gate-B 的 AISHELL-1 natural/Wav2Lip fixed-face pair，8 条独立 validation pair，13 个 shift 类别（-240 到 +240 ms）；Gate B 的 15 条样本完全排除。训练/validation 分类准确率均为 100%。

随后冻结该 critic，仅使用 zero-shift class cross-entropy 对候选 waveform 做 waveform-level 监督。完整 15 条 Gate B、5 种已知 shift、75 次优化结果：

- SyncNet 参数和 critic 参数均保持冻结；
- candidate waveform 到 SyncNet audio branch 的梯度非零：75/75；
- 一帧内恢复：17/75（22.7%）；
- `-160 ms`：0/15；`-80 ms`：3/15；`0 ms`：9/15；`+80 ms`：5/15；`+160 ms`：0/15；
- Gate B：`NO_GO`。

留出样本上的 critic 输出出现明显域偏移，例如单样本未移位音频被预测为约 +73 ms。说明在 synthetic shift feature 上达到高分类准确率，不代表 critic 对新的固定视频/候选 waveform 能提供可靠的绝对时间梯度。

结果文件：

- `runs/aishell1_absolute_lag_critic_20260819/summary.json`
- `runs/aishell1_absolute_lag_critic_20260819/critic.pt`
- `runs/aishell1_absolute_lag_critic_gateb_20260819_full/summary.json`
- `scripts/experiments/syncnet_supervision/absolute_lag_critic.py`
- `scripts/experiments/syncnet_supervision/train_absolute_lag_critic.py`
- `scripts/experiments/syncnet_supervision/validate_absolute_lag_critic_signal.py`

### paired audio-video 的定义

这里的 **paired audio-video** 指同一次真实录制中的视频帧和音频轨道：

- 视频中的说话人、嘴部运动和音频中的发声属于同一条 utterance；
- 视频帧和音频样本共享同一条真实时间轴，具有明确的 frame rate、sample rate 和起始时间关系；
- 音频不是事后替换进固定人脸视频的轨道，视频也不是由该音频驱动 Wav2Lip/Ditto 后得到的反事实画面；
- 可以对音频人为施加已知 `±40/±80/±160 ms` 时移来生成训练标签，但原始未移位样本必须保留真实 AV 配对关系。

例如，LRS3 原始视频文件中的原生画面与原生音轨，属于 paired audio-video。当前 AISHELL-1 Gate B 使用的 fixed-face/Wav2Lip crop 主要是受控评测视频，不等同于真实 paired video；learned lag critic 虽然在这些样本的 synthetic shift 分类上达到 100% 内部验证准确率，但在留出 fixed-video Gate B 上只有 17/75 恢复，说明这种数据域不能支撑可靠的绝对时间泛化。

如果后续训练 timestamp critic，最低数据契约应是：真实 paired audio-video → 一次性 AV offset 校准 → 只对 audio 做已知人工 shift → 以 speaker/sample 分组留出验证。该 critic 仍不需要 MFA 音素边界、音素时长或 pause 标签；它学习的是音频相对于视频的绝对时间位置。

### LRS3 原始 paired audio-video 方案

LRS3 online 数据集确实提供了真实 paired audio-video：原始视频文件与同一视频的 natural audio 轨道共享时间轴，不是 fixed-face/Wav2Lip 反事实视频。数据契约为 500 条、43 个 source groups、25 FPS、16 kHz mono；train/validation/test 分别为 347/69/84 条。该数据集明确声明 `natural_mfa_used=false`、`candidate_audio_embeddings_cached=false`、`similarity_matrix_cached=false`。

使用 LRS3 原始 paired 数据训练 learned lag-curve critic：

- 每个 train source group 固定抽 2 条，共 60 条；
- validation source groups 固定抽 2 条，共 12 条；
- 7 个 test source groups 保留未见，Gate B 先抽 14 条；
- 13 个 synthetic audio shift 类别，-240 到 +240 ms；
- critic train/validation 分类准确率：92.4%。

在未见 LRS3 test groups 上冻结 critic，使用 zero-shift class cross-entropy 做 waveform-level Gate B：

- 优化总数：70；
- candidate waveform → differentiable MFCC → frozen SyncNet audio branch 梯度非零：70/70；
- 一帧内恢复：15/70（21.4%）；
- `-160 ms`：0/14；`-80 ms`：0/14；`0 ms`：12/14；`+80 ms`：3/14；`+160 ms`：0/14；
- Gate B：`NO_GO`。

有趣但不能改变结论的是：critic 的初始 shift 预测在 LRS3 test 上相当准确，平均约为 -155/-79/-1/+80/+159 ms；但通过 waveform 反向优化 correction 后，预测只小幅变化，极端 shift 完全不能恢复。把 loss 改成 expected-shift regression 的小样本诊断也没有得到稳定恢复。因此问题已经从“是否有 paired audio-video”缩小为“候选 waveform 到时移估计器的局部梯度是否可靠”。

LRS3 产物：

- `runs/lrs3_qwen_cloud_n500_20260818/03_rhythm_data_online_syncnet/dataset.json`
- `runs/lrs3_absolute_lag_critic_20260819/summary.json`
- `runs/lrs3_absolute_lag_critic_gateb_20260819/summary.json`
- `scripts/experiments/syncnet_supervision/train_lrs3_absolute_lag_critic.py`
- `scripts/experiments/syncnet_supervision/validate_lrs3_absolute_lag_critic_signal.py`

LRS3 结果也不能支持直接启动 RhythmWarp residual training。当前路线实质性卡壳：需要重新设计可微、平滑且经过 waveform-level recovery 验证的 timestamp critic/resampling operator，而不是继续叠加 SyncNet lag loss。

### LRS3 后续可微算子尝试

在 LRS3 learned lag critic 之后又按单变量原则测试了三种方案：

1. **Smooth Fourier waveform shift**：保留 zero-shift CE、只替换人工 correction 算子为 zero-padded Fourier band-limited shift。14 个 test samples、70 次优化中梯度非零 70/70，但一帧内恢复仅 16/70（22.9%），`NO_GO`。
2. **Differentiable-MFCC timestamp critic**：不再使用 SyncNet audio embedding，直接用 waveform → differentiable MFCC → learned audio/visual timestamp critic。critic validation accuracy 97.4%；Gate B 梯度和 candidate waveform gradient 均 70/70 非零，但一帧内恢复仅 12/70（17.1%），`NO_GO`。
3. **MFCC-coordinate correction**：将 correction 放到 MFCC frame coordinate，以连续 frame interpolation 采样窗口。40 步时恢复 53/70（75.7%），是目前最好的版本，但仍未达到严格 Gate B；160 步/lr=2 后降至 47/70（67.1%），说明极端 ±160 ms 在不同 test group 上出现梯度饱和，增加优化步数不能解决。

这些结果共同说明：LRS3 的真实 paired audio-video 数据、SyncNet-derived critic、learned MFCC critic 和平滑/帧级 resampling 都能产生非零 waveform gradient，但目前没有一个能稳定恢复绝对时移。问题不是缺少 paired data，也不是单纯的 sample-level resampler；是候选 waveform 到绝对时间判别器之间的梯度方向/跨样本泛化不可靠。

本轮未启动 LRS3 TTS residual training，也未将任何失败 critic 接入 RhythmWarp。相关独立产物：

- `runs/lrs3_absolute_lag_critic_gateb_20260819/summary.json`
- `runs/lrs3_absolute_lag_critic_smooth_gateb_20260819/summary.json`
- `runs/lrs3_differentiable_timestamp_critic_20260819/summary.json`
- `runs/lrs3_differentiable_timestamp_critic_gateb_20260819/summary.json`
- `runs/lrs3_mfcc_coordinate_gateb_20260819/summary.json`
- `runs/lrs3_mfcc_coordinate_gateb_20260819_steps160_lr2/summary.json`

当前结论是停止继续叠加 waveform-level timing loss；如果项目必须继续，需要新的建模假设或外部可验证的时间标注，而不是再调当前 critic。

## Literature verification

### Original SyncNet

原始 SyncNet 的核心用途是学习音频/嘴部视觉 embedding，并在推理或评估时估计音视频 offset。offset search 是对相对时间位置的评分机制，不等价于一个已经校准好的绝对 waveform 时钟。官方项目和实现也把去除 audio-visual lag 作为使用场景。

### Wav2Lip

Wav2Lip 训练一个 lip-sync expert，随后冻结该 expert，用生成的连续 5 帧视频和对应音频窗口计算同步损失。论文没有报告 generator loss 内的 vshift 搜索，也没有报告独立的全局 offset calibration；方法依赖训练数据和窗口预处理已经正确对齐。该设定与“固定视频、反向修改音频”的任务不同，因此不能据此假设 absolute waveform gradient 一定可靠。

### VideoReTalking、Diff2Lip、StyleSync

这些方法都采用固定的连续视频窗口与对应音频窗口进行 SyncNet 或 SyncNet 类监督，通常为 5 帧/约 0.2 秒；论文没有报告把 vshift-style offset search 直接作为训练 loss，也没有系统报告全局 offset calibration。

### LatentSync

LatentSync 明确报告了与本项目相同类型的风险：in-the-wild 视频含有 audio-visual offsets；在送入 SyncNet 前必须把 offset 调整到零；不做 offset adjustment 会显著损害 SyncNet/模型收敛；affine transformation 前后的 offset adjustment 也会影响结果。该工作还指出 SyncNet 本身可能难以收敛，并据此研究 StableSyncNet、batch size、输入帧数和预处理。

这说明 SyncNet offset 并非本项目特有的实现错误，而是已有 lip-sync 研究明确遇到的数据协议和训练稳定性问题。

### 当前判断：时间信息是否丢失

不是原始数据中的时间信息消失了。LRS3 原始 paired audio-video 仍然共享真实时间轴，人工施加的 `±40/±80/±160 ms` 偏移也是已知的；raw lag curve 在初始状态下通常能正确判断全局偏移。因此“数据没有时间信息”不是当前问题。

真正丢失的是**可用于反向优化的稳定绝对时间坐标**：

```text
已知真实时间轴
→ 音频窗口/MFCC/SyncNet embedding
→ 局部相似度或 pooled lag curve
→ waveform correction 的梯度
```

在中间表示和目标函数中，多个不同的 waveform 时移或局部内容变化可能得到相近的分数。于是 critic 的初始预测可以是正确的，但对 correction 求导时，梯度只告诉优化器如何改变局部相似度，不一定告诉它“向全局时间轴的零点移动”。这表现为 candidate waveform gradient 始终非零，却经常方向错误、幅度不足或在约 ±115 ms 饱和。

因此更准确的说法是：**时间信息在数据和前向诊断中存在，但在当前 differentiable objective 中没有形成可靠的、可泛化的绝对时间监督信号。** 这属于表示/目标的可辨识性和梯度可靠性问题，不是单纯的音频时间戳被删除或数据集没有配对关系。

## Decision
Gate B 仍为 `NO_GO`。固定坐标矩阵、raw expected-lag anchor、learned lag critic、smooth waveform shift、differentiable-MFCC critic 和 MFCC-coordinate correction 都能保留非零 candidate waveform gradient，但没有在留出样本和极端 ±160 ms 时移上稳定恢复绝对时间。

这不表示 LRS3 或 paired audio-video 缺少时间信息：原始音频和视频仍共享真实时间轴，初始 lag 估计通常也正确。问题是当前窗口化 embedding、局部相似度和 pooled lag curve 在反向传播时不能把这个绝对坐标稳定传回 waveform correction；局部匹配、边界裁剪和跨 group 域差异使梯度方向不可靠。因此当前方案不能作为 MFA-linear residual 的 phone-duration 学习目标。

既有 MFA-linear、RhythmWarp checkpoint、数据集和下游评测结果保持不变；所有失败实验写入独立 run 目录。
## Next
如果继续，只能引入新的、独立的绝对时间信息源，例如真实 timestamp 标注、可校准的 frame-level temporal critic，或设计具有明确单调时间坐标的中间表示，并先重新通过 waveform-level recovery Gate B。若没有这类新信息，不再继续调当前 SyncNet/MFCC lag loss，也不启动完整 RhythmWarp residual training；保留 MFA-linear 作为当前可用基线。
## Observations

- [decision] 不把 vshift/lag-aware max 单独当作 absolute phone-duration training target #syncnet
- [decision] 固定坐标矩阵监督 A 在 75 个受控时移优化中仅 14 个一帧内恢复，不能进入 residual training #absolute-coordinate
- [insight] SyncNet 可以提供局部音视频相关性，但 offset search 或错误的局部 diagonal 都会吸收全局 waveform 时移 #lag-aware
- [problem] 固定视频协议存在约 -80 ms 系统性窗口偏移，校准后梯度方向仍不稳定 #offset
- [evidence] LatentSync 论文明确报告 offset adjustment 对 SyncNet 收敛至关重要 #literature
- [problem] 简单嘴部运动–音频能量 anchor 的 lag 分布不稳定且相关性弱 #independent-anchor
- [tradeoff] Wav2Lip 类模型依赖预对齐窗口；这适合视频生成器匹配音频，不等于适合固定视频反向学习音频时间轴 #wav2lip

## References

- Wav2Lip: https://arxiv.org/abs/2008.10010
- Original SyncNet paper: https://doi.org/10.1007/978-3-319-54181-5_16
- SyncNet project: https://www.robots.ox.ac.uk/~vgg/software/lipsync/
- SyncNet Python implementation: https://github.com/joonson/syncnet_python
- VideoReTalking: https://arxiv.org/abs/2211.14758
- Diff2Lip: https://arxiv.org/abs/2308.09716
- StyleSync: https://arxiv.org/abs/2305.05445
- LatentSync: https://arxiv.org/pdf/2412.09262

## Relations

- relates_to [[建立可学习音频对齐模型]]
- relates_to [[MFA-linear 原样基线实验]]
- relates_to [[LRS3 SyncNet RhythmWarp 训练]]
