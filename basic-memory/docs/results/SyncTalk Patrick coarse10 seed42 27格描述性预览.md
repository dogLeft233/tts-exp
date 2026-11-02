---
title: SyncTalk Patrick coarse10 seed42 27格描述性预览
type: experiment
permalink: tts-exp/docs/results/sync-talk-patrick-coarse10-seed42-27-格描述性预览
status: CONTROL_FAILED
execution_status: PREVIEW_COMPLETE
project: tts-exp
created: '2026-10-18T10:38:43.549356+00:00'
---

# SyncTalk Patrick coarse10 seed42 27格描述性预览

PatrickLeahy 的三训练档 × 三驱动档 × 三评分档共27格粗阶段预览已完成并通过独立数值与持久化复核。训练实际成功更新不等，科学控制状态仍为CONTROL_FAILED；本页描述实际模型，不能据此声称训练音量的受控因果效应。

## 固定范围与解释边界
单身份PatrickLeahy、seed42，固定coarse epoch10的EMA参数，尚非fine17精训练完成。每个训练档60000次尝试，−6/0/+6 dB实际成功59985/59985/59984次，AMP跳步15/15/16次，无终止性非有限loss。没有按测试分数挑checkpoint、删片段、改训练档或补训练步。

原六段各250帧测试片段（source_start_frame=50、300、550、800、1050、1300），六片等权；每段主端点guard20使用205个窗口，次端点full使用245个窗口。六片是同一个人的片段，不能当成六个独立身份。

训练和渲染保持FP16，评分为SyncNet FP32、TF32关闭。评分0dB输入原PCM16；±6dB在float64原PCM单位下缩放，避免整数截断/归一化混入。source-owned ROI跨条件固定。正式评分来自FFV1无损RGB与原PCM音轨，H264仅为观看预览。

## 本次观察
1. 在九种相同驱动/评分设置下，三个实际模型的guard20 Sync-C均为训练−6档最高、0档其次、+6档最低。这是一致的描述排序；更新数不等及单身份单seed限制，不能将排序推广为最佳训练音量或感知唇形改善。
2. 驱动音量的分数变化随实际模型而异。评分0dB固定时，驱动−6→+6dB的Sync-C变化，三个训练档分别为+0.053、+0.289、+0.658；−6训练模型在驱动0dB为8.213、驱动+6dB为8.158，并非所有模型都随驱动音量升高而增分。
3. 同一渲染视频仅改变评分输入音量也会改分：九条视频的scorer−6→+6dB引起Sync-C变化范围为−0.131至+0.039。这个轴没有改变视频，因此不能把它直接解释成生成嘴型变化。

## 主结果：guard20 Sync-C（越高越好）

### 评分音量 -6 dB

| 训练音量 / 驱动音量 | −6 dB | 0 dB | +6 dB |
|---|---:|---:|---:|
| -6 dB | 8.088 | 8.168 | 8.137 |
| +0 dB | 7.836 | 8.059 | 8.054 |
| +6 dB | 7.389 | 7.784 | 7.986 |

### 评分音量 +0 dB

| 训练音量 / 驱动音量 | −6 dB | 0 dB | +6 dB |
|---|---:|---:|---:|
| -6 dB | 8.106 | 8.213 | 8.158 |
| +0 dB | 7.797 | 8.068 | 8.086 |
| +6 dB | 7.336 | 7.770 | 7.994 |

### 评分音量 +6 dB

| 训练音量 / 驱动音量 | −6 dB | 0 dB | +6 dB |
|---|---:|---:|---:|
| -6 dB | 8.045 | 8.208 | 8.138 |
| +0 dB | 7.719 | 8.024 | 8.083 |
| +6 dB | 7.259 | 7.705 | 7.971 |

## 主结果：guard20 Sync-D（越低越好）

### 评分音量 -6 dB

| 训练音量 / 驱动音量 | −6 dB | 0 dB | +6 dB |
|---|---:|---:|---:|
| -6 dB | 6.812 | 6.781 | 6.823 |
| +0 dB | 6.929 | 6.779 | 6.825 |
| +6 dB | 7.321 | 6.966 | 6.870 |

### 评分音量 +0 dB

| 训练音量 / 驱动音量 | −6 dB | 0 dB | +6 dB |
|---|---:|---:|---:|
| -6 dB | 6.872 | 6.790 | 6.822 |
| +0 dB | 7.051 | 6.840 | 6.837 |
| +6 dB | 7.485 | 7.077 | 6.927 |

### 评分音量 +6 dB

| 训练音量 / 驱动音量 | −6 dB | 0 dB | +6 dB |
|---|---:|---:|---:|
| -6 dB | 6.959 | 6.819 | 6.847 |
| +0 dB | 7.196 | 6.924 | 6.866 |
| +6 dB | 7.656 | 7.205 | 6.995 |

## 次结果：full Sync-C

### 评分音量 -6 dB

| 训练音量 / 驱动音量 | −6 dB | 0 dB | +6 dB |
|---|---:|---:|---:|
| -6 dB | 7.782 | 7.908 | 7.905 |
| +0 dB | 7.554 | 7.798 | 7.820 |
| +6 dB | 7.139 | 7.516 | 7.680 |

### 评分音量 +0 dB

| 训练音量 / 驱动音量 | −6 dB | 0 dB | +6 dB |
|---|---:|---:|---:|
| -6 dB | 7.771 | 7.904 | 7.951 |
| +0 dB | 7.518 | 7.797 | 7.836 |
| +6 dB | 7.085 | 7.493 | 7.700 |

### 评分音量 +6 dB

| 训练音量 / 驱动音量 | −6 dB | 0 dB | +6 dB |
|---|---:|---:|---:|
| -6 dB | 7.743 | 7.853 | 7.921 |
| +0 dB | 7.439 | 7.759 | 7.830 |
| +6 dB | 7.009 | 7.444 | 7.686 |

## 核验与可复现材料
9条视频各1500帧，执行端已逐帧FFV1 RGB及PCM读回精确一致；基线首250帧重复渲染exact。9份评分归档225个成员本轮重新逐SHA验证，与保留本地数组相同。采用另一按列实现的NumPy float64距离公式，从保存的visual features和原canonical audio features重新计算全部162个245×31矩阵，最大绝对差5.24194470941e-06；端点最大差2.78232345963e-06，均小于原3e-5阈值。所有27格的六片等权均值与offset列表逐项核对。另检27对保留PNG/原head数组像素hash及9条PCM；完整电影和云payload本轮未重复解码/下载，采用之前全部像素读回和独立云整档读回证据，范围明确区分。

9个原lossless媒体archive已F全成员验证与独立Linux云端整包读回，随后按规则退休F重复tar和remote媒体cache；cloud_storage_locator.json给出显式恢复位置。F保留每条件H264预览、PNG、raw head、PCM、评分原矩阵与完整receipt。任何未来lossless电影读取须显式恢复。

- [27格汇总图](F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017/trials/fp16_other_identities_trial_v1/early_score_preview_v2/delivery_v1/patrick_27_grid.png)
- [完整主/次端点CSV](F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017/trials/fp16_other_identities_trial_v1/early_score_preview_v2/delivery_v1/scores_27_cells.csv)
- [平均距离曲线CSV](F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017/trials/fp16_other_identities_trial_v1/early_score_preview_v2/delivery_v1/mean_distance_curves.csv)
- [27格原始逐clip JSON](F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017/trials/fp16_other_identities_trial_v1/early_score_preview_v2/all27_preview.json)，SHA `0c727888b1fbeb09e573ab26bc7129d453aa03b45de345b460d66973ac6fac1a`。
- [独立最终核验receipt](F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017/trials/fp16_other_identities_trial_v1/early_score_preview_v2/delivery_v1/independent_review.json)，包含9份score/render archive SHA及云读回receipt绑定。
- [训练0/驱动0观看预览](F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017/trials/fp16_other_identities_trial_v1/early_score_preview_v2/renders/train_+0/driver_+0/local/preview.mp4)（H264，非评分源）。
- 原试跑contract SHA `787d917d259eeca89311016737893348578335e3623f217858797926311893bd`。

图表首尝试因本机缺matplotlib而失败，数值核验已先完成；随后只加载已有独立绘图库完成图表，未重跑训练/渲染/评分。图表标签和数值已查看核验。失败记录plot_attempt01_failure.json保留。

## Observations
- [result] 单身份单seed固定粗10 EMA的27格描述预览执行完成并通过工程QA，训练控制失败状态保留。
- [decision] 预览不用于改预算、选择训练条件或checkpoint，原Brown失败和原完整实验版本全部保留。
- [limitation] 非fine17完整模型、非24完整模型结果；未完成跨身份/seed汇总、未作人类感知或数据充分性推断。
- [next_step] 按已授权调度继续Chris、Cathy的独立FP16粗训练，未来正式完整模型结果另列。

## Relations
- implements [[046-synctalk-first-identity-early-score-preview]]
- relates_to [[试跑 Patrick Chris Cathy 的 FP16 粗训练]]
- relates_to [[SyncTalk 四新身份官方完整训练电平实验 Spec]]