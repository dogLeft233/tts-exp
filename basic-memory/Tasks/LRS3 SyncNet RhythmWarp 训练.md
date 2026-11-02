---
title: LRS3 SyncNet RhythmWarp 训练
type: task
permalink: tts-exp/tasks/lrs3-sync-net-rhythm-warp-训练
status: done
completed: '2026-08-17'
priority: high
current_step: 6
assigned_to: claude
steps:
- 建立 source_group 无泄漏划分并运行自然/Qwen 音频 MFA、视频帧和 SyncNet embedding 缓存
- 实现 bounded monotonic RhythmWarp 和 phone-duration renderer，加入 offline SyncNet/边界/暂停/平滑约束
- 用少量 source group 完成 smoke 训练、单元测试和训练曲线检查
- 扩展到全量 train groups，保存 checkpoint、history 和 provenance
- 只在 held-out source groups 上比较 raw Qwen、global resample、MFA-linear 和 RhythmWarp，执行
  Go/No-Go 判断
started: '2026-08-17'
tags:
- task
- lrs3
- syncnet
- rhythmwarp
- online-supervision
---

# LRS3 SyncNet RhythmWarp 训练

## Context

本项目研究在保留 TTS 音素内容和说话人条件的前提下，学习调整 TTS 音频的音素时长、节奏和停顿，使其在视频口型同步评测中更接近自然音频。实验使用约 500 条 LRS3 音频/视频配对和云端 Qwen TTS。LRS3 本地 manifest 没有可靠 speaker metadata，因此结论按 source_group 划分解释，不能声称 speaker-disjoint generalization。

## 已确定的设计边界

- natural MFA phone boundary、duration 和 pause duration 不能作为 online RhythmWarp 的训练目标。
- TTS MFA token sequence、TTS source-relative phone spans、TTS waveform segmentation 可以作为输入和可微变形的分段依据。
- SyncNet 必须以冻结参数直接接入训练：SyncNet 参数 `requires_grad=False` 且处于 `eval()`，但候选 waveform → 可微 MFCC → `forward_aud()` 的梯度必须保留。
- Wav2Lip 和文件级 `SyncNetInstance.evaluate()` 只能用于训练后的独立 downstream evaluation，不能进入训练图。
- 不缓存 candidate audio embedding、`tts_similarity`，也不把候选视频送回 SyncNet 形成训练循环。

## 先前 full-objective 路线

早期 full objective 的主要 bug 是 paired TTS token 同时包含 TTS source 时间和映射到 natural audio 的 target 时间，但旧 cache 用 source TTS 时间在自然视频上取 `phone_video_context`。6,806 个 validation phones 的实际 context 与 target-time context 平均 cosine 只有 `0.1516`，平均 phone 中点错位约 `1.2184s`，83.98% 的视觉窗完全无重叠。

已在 `scripts/prepare_lrs3_rhythm_dataset.py` 增加 target-aware `phone_context_from_visual()` 并通过 regression test。修复后独立数据集为 `runs/lrs3_qwen_cloud_n500_20260818/03_rhythm_data_mfa_target_context/`，context 与 target-time 重算结果平均 cosine=`0.99999997`。修复后 full-objective 重训 checkpoint 为 `runs/lrs3_qwen_cloud_n500_20260818/04_rhythm_train_mfa_target_context_retry/checkpoint.pt`，best step=`300`；validation path loss `0.932157→0.931140`，boundary MAE `0.424066→0.415243s`，只改善约 2.08%，低于预设 5% gate。

修复后 69 条 validation 的 downstream 均值仍未超过自然音频：

| arm | mean Sync-C | mean Sync-D |
|---|---:|---:|
| natural audio → Wav2Lip | 6.9000 | 7.5345 |
| RhythmWarp audio → Wav2Lip | 6.1575 | 8.5082 |
| RhythmWarp video + natural audio | 2.0720 | 12.4749 |

结论：视觉上下文 bug 已修复，但 full objective 仍为 No-Go；剩余主要问题是 pseudo Sync boundary 信号和 full objective，而不是 waveform 长度、声码器 NaN 或 replacement mux。

## 当前 online frozen-SyncNet 路线

正式代码：

```text
scripts/online_syncnet_objective.py
scripts/prepare_lrs3_rhythm_online_dataset.py
scripts/train_lrs3_rhythm_warp.py
scripts/eval_lrs3_validation_rhythmwarp_wav2lip_syncnet.py
```

online loss 路径：

```text
TTS waveform
→ sigmoid soft interval piecewise time warp
→ differentiable MFCC
→ frozen SyncNet audio branch
→ natural video visual embedding cosine loss
```

协议为 16 kHz、25 FPS、5-frame visual window、20-frame MFCC window、4-frame MFCC step、`[W,1,13,20]` audio input、1024 维 embedding，MFCC 前使用 `32767.0` amplitude scale。初版 hard `searchsorted()` warp 的 boundary 梯度不稳定，已改为 sigmoid soft interval partition；当前 warp 保证正 duration、严格单调 boundary、精确 endpoint 和精确输出长度。短 phone sequence 的空 second-difference 也已修复。

online 数据集：

```text
runs/lrs3_qwen_cloud_n500_20260818/03_rhythm_data_online_syncnet/
```

统计为 500 条、43 个 source groups、347 train、69 validation、84 test、0 failures。`natural_mfa_used=false`、`candidate_audio_embeddings_cached=false`、`similarity_matrix_cached=false`。online 分支只读取 TTS MFA/source spans、TTS waveform、natural total duration 和 detached natural-video visual embeddings。

## online full training

训练目录：

```text
runs/lrs3_qwen_cloud_n500_20260818/04_rhythm_train_online_syncnet/
```

配置为 online SyncNet objective、800 steps、learning rate `2e-5`、每 100 steps validation、early stopping patience 5。最佳 checkpoint 为 step 600，best validation loss=`0.9192197789`。validation score 从 global-resample baseline 的 `0.0800843035` 到模型的 `0.0807802244`，delta=`+0.0006959209`；这是小幅 online baseline 改善，不等同于 downstream lip-sync 成功。全部 history finite；69 条 validation 的 duration 为正、boundary 单调，最大 endpoint error 为 `3.814697265625e-06s`。

## 当前 downstream 状态
online checkpoint 的 validation 音频位于：

```text
runs/lrs3_qwen_cloud_n500_20260818/05_rhythm_eval_online_syncnet/
```

三臂为：

1. `natural_audio`：natural audio → Wav2Lip → SyncNet；
2. `rhythmwarp_audio`：RhythmWarp audio → Wav2Lip → SyncNet；
3. `rhythmwarp_video_natural_audio`：保留 RhythmWarp Wav2Lip video stream，只替换 natural audio track → SyncNet。

最终结构化报告已经生成：

```text
runs/lrs3_qwen_cloud_n500_20260818/05_rhythm_eval_online_syncnet/video_summary.json
runs/lrs3_qwen_cloud_n500_20260818/05_rhythm_eval_online_syncnet/analysis.json
```

完整性检查：207/207 score cells、三臂各 69 条、0 failures、所有 score finite；69/69 replacement 的 video stream 保持，69/69 replacement 的 natural audio PCM 保持。报告记录了 sample、paired key、source group、split、音频/视频/log hashes、Wav2Lip/SyncNet hashes、min-track、fallback box 和 replacement integrity。

最终 aggregate（Sync-C 越高越好，Sync-D 越低越好）：

| arm | n | mean Sync-C | SD Sync-C | mean Sync-D | SD Sync-D | mean AV offset |
|---|---:|---:|---:|---:|---:|---:|
| natural audio → Wav2Lip | 69 | 6.9150 | 1.3461 | 7.5484 | 0.9164 | -2.0000 |
| RhythmWarp audio → Wav2Lip | 69 | 6.1816 | 0.9879 | 8.5229 | 0.7082 | -2.1014 |
| RhythmWarp video + natural audio | 69 | 2.0966 | 0.7530 | 12.4818 | 0.8722 | -0.9420 |

RhythmWarp 相对 natural audio：mean ΔC=`-0.7333`、mean ΔD=`+0.9744`；Sync-C 改善 13/69，Sync-D 改善 5/69，joint improvement 3/69。replacement 相对 RhythmWarp audio：mean ΔC=`-4.0850`、mean ΔD=`+3.9590`；C/D/joint 改善均为 0/69。

Wav2Lip 和 SyncNet 批处理使用临时脚本 `/tmp/wav2lip_batch.py`、`/tmp/syncnet_batch.py`，但结果已用样本 `lrs3_6VWPHKABRQA_00003` 与官方流程逐项交叉验证。正式 evaluator 新增了 `--recover-existing-logs`，可从已完成日志恢复结构化报告而不重新推理；同时修复了代码重排后的直接执行 import path 问题和日志字面量 `\\t` 解析问题。
## 当前结论
旧 full-objective 路线在修复视觉上下文后仍未达到 downstream 自然音频基线。online frozen-SyncNet 路线已经完成 TTS-only 数据准备、可微 waveform/SyncNet 训练、early stopping、69 条 validation 音频生成以及完整的三臂 207-cell downstream 评测。

最终结果显示 natural audio 明显优于 RhythmWarp audio：RhythmWarp 的 mean Sync-C 低 `0.7333`，mean Sync-D 高 `0.9744`，joint improvement 只有 3/69。因此当前 online checkpoint 的 downstream lip-sync 结论仍为 No-Go，不能声称已经实现可用的同步增强。训练阶段相对 global-resample 的小幅 online loss 改善没有转化为 validation Wav2Lip/SyncNet 改善。

replacement arm 在所有 69 条样本上都比 RhythmWarp audio 更差，说明生成视频对其生成音频条件存在明显依赖；但该 arm 不能被解释为自然音频质量分数。

对三个 source group 的固定首条 validation 样本（`lrs3_6VWPHKABRQA_00003`、`lrs3_6XS8TA4RBog_00002`、`lrs3_6YKYo00mFAg_00003`）进行了 Sight 定性抽样。自然、RhythmWarp 和 replacement 视频均被观察到整体可理解、没有明显灾难性错位；个别样本有轻微闭唇不足、词尾滞后或连读抖动。Sight 的主观判断有时认为 replacement 表现良好，与正式 SyncNet 结果（replacement 相对 RhythmWarp 在 C/D 上 0/69 改善）不一致，因此只作为辅助可视化证据，不能推翻 69 条定量矩阵的 No-Go。

下一步若继续研究，应优先重审 phone-level timing 参数化、SyncNet visual target 和在线训练信号强度，而不是继续扩大当前 checkpoint 的训练步数。
## Observations

- [decision] online objective 不使用 natural MFA timing target，只使用 TTS MFA/source spans 作为输入分段依据 #online-syncnet
- [decision] SyncNet 参数冻结但 candidate waveform audio branch 保留梯度 #differentiable-supervision
- [solution] 用 sigmoid soft interval partition 替代 hard searchsorted warp，解决 boundary 梯度离散问题 #waveform-warp
- [insight] target-aware visual context 修复消除了严重时间错位，但没有单独带来 downstream 提升 #diagnosis
- [insight] online validation score 仅小幅优于 global-resample baseline，必须以独立 Wav2Lip/SyncNet 结果为准 #evaluation
- [problem] 207 个 online downstream score logs 已完成，但正式 `video_summary.json`/`analysis.json` 尚未生成 #pending
- [status] done
- [tradeoff] 采用临时单进程批处理避免重复加载模型；批处理流程已与官方流程单样本交叉验证 #runtime
- [constraint] LRS3 本地 metadata 不支持 speaker-disjoint generalization 声明 #data-split

## Relations

- relates_to [[syncnet]]
- relates_to [[MFA-linear 冻结 VC 重合成与下游验证]]
