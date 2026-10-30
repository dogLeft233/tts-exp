---
title: LRS3 ASR 错误与局部同步低分相关性实验
type: report
permalink: tts-exp/docs/experiments/lrs3-asr-错误与局部同步低分相关性实验
thread_id: fcd4c077-21ff-4445-a06c-07696af24afb
status: complete
scope: project
run_dir: runs/lrs3_asr_sync_error_correlation_20260831
engineering_decision: GO
scientific_overall: SUPPORTED_NATURAL_ONLY
tags:
- lrs3
- asr
- syncnet
- correlation
- experiment
- project-note
---

# LRS3 ASR 错误与局部同步低分相关性实验

## Context

本线程完成了一个 fit-only、可复现的 LRS3 实验，用未经语言模型修正的 Wav2Vec2-CTC greedy ASR 定位识别错误时间段，并将其与同一音频上从完整 SyncNet V2 distance matrix 派生的局部同步置信度 `local_c` 对齐。实验分别分析 natural audio 与 TTS audio，目标是判断 ASR 错误是否更常发生于局部唇形同步较差的区间。

实验严格使用 frozen `fresh_confirmation` cohort：24 个 sample、24 个 source group、natural/TTS 共 48 个 arm-record；未访问 internal-dev、validation 或 test sealed media。

## Frozen protocol

- ASR 模型：`facebook/wav2vec2-large-960h-lv60-self`
- immutable revision：`54074b1c16f4de6a5ad59affb4caa8f2ea03a119`
- 解码：逐帧 argmax + 标准 CTC collapse；无 beam search、语言模型、lexicon、spell correction 或 reference-conditioned decoding
- reference timing：每个 arm 独立执行 CTC forced alignment，仅用于定位 reference words，不修改 greedy ASR 输出
- SyncNet：使用完整 `[T,31]` distance matrix、whole-track offset 和 vendor-equivalent PyTorch reduction/median semantics 派生 `local_c`，不是独立五帧 clip 的 standalone Sync-C
- 低同步阈值：每个 arm 独立使用 `mean(local_c) - population_std(local_c)`，严格 `<`
- 统计单位：whole sample；`PCG64(20260831)`，10,000 次 bootstrap；不把 25 Hz cells 当成独立样本

## Completed implementation and run

代码位于 `scripts/experiments/asr_sync_error_correlation/`，测试位于 `tests/experiments/asr_sync_error_correlation/`，OpenSpec change 为 `add-asr-sync-error-correlation`。完整运行位于：

`runs/lrs3_asr_sync_error_correlation_20260831/`

最终工程状态为 `GO`：

- preflight、locked offline model、CUDA smoke、forced-alignment smoke 均通过
- 48/48 strict mux、ASR、SyncNet、alignment cells 均通过
- 所有 SyncNet global offset/Sync-C parity 与 paired track metadata 检查通过
- 24 个 paired diagnostic plots 已生成
- 独立 artifact audit 通过：48 records、192 markers、144 NPZ families、24 plots；JSON/NPZ 全部 finite，marker hashes 全匹配
- focused tests：26 passed
- `openspec validate add-asr-sync-error-correlation --strict` 通过

运行过程中修复了几个影响正确性的细节：LRS3 数字和 `{LG}` timing token 的 normalization；forced-alignment repeated token span 消费；vendor `tracks.pckl` wrapper schema；standalone adapter JSON newline；以及 SyncNet PyTorch stack/reduction/median 的精确数值语义。历史失败 ledger 保留 79 次已修复的重试记录，不代表最终缺失 cell。

## Scientific result

整体科学状态为 `SUPPORTED_NATURAL_ONLY`。

### Natural arm — SUPPORT

- defined records：20
- error cells：655
- source groups：20
- median Spearman(error mask, `-local_c`)：`0.1448`
- 95% bootstrap CI：`[0.0555, 0.1799]`
- median badness contrast：`1.0914`
- 95% bootstrap CI：`[0.1640, 1.4841]`
- micro IoU：`0.0943`
- error recall：`0.2321`
- low-sync precision：`0.1371`

Natural arm 的两个预声明 bootstrap lower bounds 均大于 0，因此达到 SUPPORT 门槛。

### TTS arm — NO_SUPPORT

- defined records：19
- error cells：409
- source groups：19
- median Spearman：`0.0899`
- 95% bootstrap CI：`[-0.0283, 0.1595]`
- median badness contrast：`1.1738`
- 95% bootstrap CI：`[-0.1315, 1.7042]`
- micro IoU：`0.0419`
- error recall：`0.1491`
- low-sync precision：`0.0551`

TTS arm 的两个置信区间均跨过 0，因此不能声称存在稳定正相关。

## Interpretation of overlap

“存在正相关”不等于低 `local_c` 能准确预测 ASR 错误。对应的条件比例为：

- Natural：低同步 cells 中 `152/1109 = 13.71%` 位于 ASR 错误区间
- TTS：低同步 cells 中 `61/1107 = 5.51%` 位于 ASR 错误区间
- 合计：`213/2216 = 9.61%`

反方向，即 ASR 错误 cells 中有多少也是低同步：

- Natural：`152/655 = 23.21%`
- TTS：`61/409 = 14.91%`

因此 natural arm 有统计上的弱正关联，但绝对重合比例低；大多数低同步区间没有 ASR 错误，而且 TTS 上的关联更弱且不满足支持门。

## Observations

- [decision] ASR 识别序列只允许 uncorrected CTC greedy decoding，forced alignment 仅提供 reference timing。 #asr #protocol
- [decision] 局部同步序列命名为 `local_c`，必须从完整 SyncNet distance matrix 和 whole-track offset 派生。 #syncnet #protocol
- [insight] Natural audio 的 ASR 错误与局部同步 badness 有可重复的弱正关联，但该关联不具有高预测精度。 #result
- [insight] TTS audio 未通过预声明科学支持门，不能把 natural-only 结果推广到 TTS。 #tts #result
- [insight] Low-sync precision 比相关系数更直观地揭示实际重合有限：Natural 13.71%，TTS 5.51%。 #interpretation
- [pattern] 对 25 Hz autocorrelated cells 的不确定性分析应以 whole sample 为 bootstrap 单位，不能报告 frame-independent p-values。 #statistics
- [solution] 精确复现 SyncNet parity 需要匹配 vendor 的 PyTorch tensor layout、reduction 和 `torch.median` 数值语义，而非仅复现代数公式。 #syncnet #reproducibility

## Relations

- relates_to [[tts-exp|Wav2Lip replacement is primary objective]]
- relates_to [[tts-exp|Natural reference audio available at inference]]
- relates_to [[LRS3 MFA-linear TFG native/replacement 2026-09-13|LRS3 MFA-linear replacement NO-GO]]
- implements [[ASR-Sync Error Correlation Experiment]]
