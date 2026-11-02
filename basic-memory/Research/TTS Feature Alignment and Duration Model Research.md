---
title: TTS Feature Alignment and Duration Model Research
type: research_topic
permalink: tts-exp/research/tts-feature-alignment-and-duration-model-research-1
---

# TTS Feature Alignment and Duration Model Research

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| Initial Document | August 12, 2026 | user |
| Restructured to Research/ with schema observations | August 12, 2026 | user |

## 研究问题

> natural 输入音频与 TTS 音频在 duration、pause、phone timing 不一致时,如何构造可靠的监督 target,并选择能够让输入特征向 TTS 特征靠近的简单模型?

重点不是寻找更大的 waveform generator,而是区分四个独立问题:

1. boundary/alignment:哪些 source frame 与哪些 target frame 或 phone 对应;
2. duration/occupancy:每个 phone 或 pause 应占多少输出 frame;
3. feature mapping:natural representation 如何向 TTS-like representation 移动;
4. waveform rendering:目标 feature 如何还原为可用 waveform,以及长度如何处理。

相关项目记录:[[16-phone-local-warp-validation]]、[[17-tts-feature-rhythm-alignment-research]]、[[22-tts-feature-supervision-and-mvp]]、[[29-two-stage-feature-targeted-waveform-model]]。

## 核心结论

### 1. 当前方案只部分处理了时序不一致

当前 two-stage pipeline 的 target 构造是:

```text
natural/TTS 各自做 MFA
→ matched phone span 按 label/order 匹配
→ 在 natural HuBERT frame centers 上把 TTS L6 feature 局部线性映射/插值
→ unmatched pause/frame 不参与 target loss
→ Stage 2 在 natural sample grid 上输出 same-N residual
```

这解决了直接逐帧比较 natural[t] 与 TTS[t] 的最严重全局错位,但没有显式学习 phone occupancy matrix、TTS phone duration、pause 插入/删除、phrase-level rhythm、natural-only 推理时的 target duration。因此当前模型应称为 **feature-targeted same-N waveform enhancer/renderer**,不能称为 duration-aware TTS rhythm transfer model。

### 2. same-N 和 rhythm transfer 不是同一能力

同长 residual TCN/U-Net 的 output shape contract 只保证 N_out = N_input。它不会自动重复、删除或重新分配 phone/frame。当 natural 与 TTS 总时长不同,以下目标通常不能同时满足:保留 natural 绝对时间、复制 TTS 绝对时间、输出严格等于 natural sample 数。如果 TTS rhythm 是硬目标,应允许 variable-length output 或显式 duration/length regulator。

### 3. fixed-K phone-local SSL warp 不是成熟标准范式

公开文献中没有找到成熟方案明确完成 forced-alignment phone boundaries → 每 phone 独立重采样到固定 K。成熟近邻是 phone-level mean/attention pooling、MFA/CTC boundary supervision、FastSpeech-style duration expansion、phoneme-constrained/global DTW、whole-utterance SSL soft-DTW、latent duration warp。因此当前 fixed-K/source-to-target feature-grid warp 应定义为 research prototype。

### 4. 最重要的第一步不是扩大模型,而是替换 target geometry

最有判别力的 target 对照:A. 当前 MFA-linear target; B. phone-level HuBERT pooling + duration expansion; C. phone-anchored hard DTW target; D. constrained soft-DTW loss; E. duration predictor + length regulator。应在同一 K=1 样本、相同 HuBERT、相同 loss 和相同 renderer/direct upper-bound 下比较。

## 三条主 alignment 主线

- **A. Phone-local hard DTW**: 同序 matched phone spans 内独立 DP,不跨 phone。建议加 duration-ratio scaled band、Sakoe-Chiba/Itakura slope constraint、max run、duration penalty、explicit sil state、duplicate frame aggregation 规则。hard path 不可微,作 offline stop-gradient pseudo target。
- **B. Constrained soft-DTW**: 可微 soft-min 聚合,需 phone anchor、sil mask、scaled band、slope/run constraint、temporal regularizer;单独使用会 collapse。LASER 是最近的 HuBERT/WavLM SSL soft-DTW 近邻,但复现成本高。
- **C. Differentiable duration/warping alignment**: FastSpeech2 的 duration predictor + length regulator 是最适合"TTS rhythm 是目标 + 推理有 transcript"的路线;natural waveform-only 无 transcript 时不能可靠决定 TTS phone duration。

## 模型路线建议

1. **Feature residual predictor**: identity-initialized,隔离 alignment/feature mapping 与 waveform decoder(首选诊断)
2. **TCN waveform residual**: same-N、identity init、实现简单;当前 RF 253 samples ≈ 15.8ms 太短,先扩 dilation 1..512
3. **Residual U-Net**: 第二阶段,只有 TCN bottleneck 被证明是 temporal context 问题时才值得

## 建议的最小实验矩阵

- Target geometry: MFA-linear / phone pool+duration / hard DTW / soft-DTW loss / variable-length
- Renderer: direct waveform UB / feature residual predictor / TCN 1..32 / 1..128 / 1..512 / residual U-Net
- 指标: masked HuBERT gap, gap reduction, per-phone error, duration ratio, pause presence, CER/PER/WER, speaker similarity, Ditto/SyncNet 只最后评估

## 停止条件

- target 不能在 K=1 direct upper bound 中改善 → 先修 target/alignment
- feature predictor 能、waveform renderer 不能 → 修 decoder/renderer
- 只有训练 loss 降而 validation 不改善 → 停止增加容量
- DTW 路径跨 phone、长 run、依赖极低 confidence → 停止信任该 target
- U-Net 只改善 STFT 不改善 content/pause → 不继续

## 必须避免的表述

不要写:"MFA 自动完成了 phone-local warp";"HuBERT frame index 就是 phone label";"20ms hop 就是 HuBERT 完整局部窗口";"phone_local_warp 是 DTW";"soft-DTW 自动生成唯一 frame target";"same-N renderer 已完成 TTS rhythm transfer";"feature proximity 已证明 SyncNet/TFG 因果效果"。

## Observations
- [status] active
- [question] natural 与 TTS 音频 duration/pause/phone timing 不一致时,如何构造可靠监督 target 并选择简单模型?
- [decision] 优先实现 phone-level pooling + explicit duration 与 phone-anchored hard DTW 对照,再决定是否扩大 renderer

## Relations
- relates_to [[增强头下游评估 (Ditto + SyncNet)]]
- relates_to [[Ralph Step 1 - Controlled Baseline]]
- relates_to [[Ralph Step 2 - Phone Target Geometry Screening]]
- relates_to [[docs/experiments/16-phone-local-warp-validation]]
- relates_to [[docs/experiments/17-tts-feature-rhythm-alignment-research]]
- relates_to [[docs/experiments/22-tts-feature-supervision-and-mvp]]
- relates_to [[docs/experiments/29-two-stage-feature-targeted-waveform-model]]
- relates_to [[TTS 增强 TFG 的原因：证据、排除项与因果分叉]]