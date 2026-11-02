---
title: 让两个音频共享 SyncNet 时间坐标
type: concept
permalink: tts-exp/concepts/让两个音频共享-sync-net-时间坐标
status: active
date: '2026-08-22'
tags:
- syncnet
- wav2lip
- audio-alignment
- replacement
- mfa
- dtw
---

# 让两个音频共享 SyncNet 时间坐标

## 核心概念

SyncNet 并不直接比较两条音频；它比较视频嘴形与一条音频是否同步。在 replacement 链路中，真正需要的是：由 candidate audio 驱动 Wav2Lip 得到的视频，在换回 original natural audio 后仍与 natural audio 同步。

设 natural audio 为 `N`、candidate audio 为 `C`、生成视频为 `V_C = Wav2Lip(face, C)`，目标不是只有 `SyncNet(V_C, C)` 高，而是 `SyncNet(V_C, N)` 也高。这要求 `C` 与 `N` 共享局部发音事件的时间坐标，而不仅是总长度或全局 offset 相同。

## 简单方法的优先级

### 1. Natural driver oracle

最简单可靠的方法是直接使用 original natural audio 驱动 Wav2Lip，再将同一 natural audio mux 回生成视频：

```text
natural audio → Wav2Lip → generated video + natural audio
```

natural audio 在当前项目的推理阶段可用，因此没有必要先制造一条独立 TTS timing clock，再要求它生成的嘴形兼容 natural。该方法应作为可达上界和首个 protocol gate。

### 2. Polarity-inversion sanity control

构造 `C(t) = -N(t)`。两条 PCM 波形逐样本不同，但极性翻转保持短时幅度谱、mel/MFCC 时间结构基本不变，听感通常也不变。因此它们应被 Wav2Lip/SyncNet 视为几乎相同的时间驱动信号，除非 preprocessing、clipping 或编码过程引入非线性差异。

该控制只能证明管线与特征等价性，不能产生真正不同的说话风格。

### 3. Frame-synchronous identity reconstruction / voice conversion
如果第二条音频需要听起来不同，frame-synchronous VC 在理论上比独立 TTS 更接近目标，但当前项目没有证据证明它能保留 replacement 增益。相反，最容易的 identity reconstruction 已失败：natural 经 WavLM-L6 + frozen HiFi-GAN 后，驱动视频更偏好 reconstructed audio；换回 natural 时 Sync-C/D 退化。若连 identity reconstruction 都不可替换，增加音色转换通常只会扩大局部 mel、onset、F0/formant 和相位差异。

因此 VC 目前只能视为未验证假设，不是推荐下一步。只有新的高保真 codec 能先通过严格 identity replacement gate，才值得继续增加音色变换。
### 4. MFA phone-clock alignment for independent TTS

如果必须使用独立 TTS：分别对 natural/TTS 做 MFA，取得相同 transcript 的 word、phone、silence 边界，把每个 TTS phone 区间单调拉伸到 natural 对应区间。MFA 只提供边界；实际 warp 应由特征插值、phase vocoder 或 Rubber Band 完成。

phone boundary 对齐是必要但不充分的，因为 phone 内部的起音、辅音释放、能量轨迹和 coarticulation 仍可能不同。

### 5. MFA-constrained phone-local DTW
项目已经比较过 MFA-linear、phone mean pooling 和 MFA-constrained hard DTW。100-step 及多样本 equal-budget 结果中，hard-DTW 与 MFA-linear 几乎相同，没有稳定优势；phone mean pooling 明显更差。波形级 phone-local warp 还会引入 phase、transient 和拼接边界问题。

因此 phone-local DTW 目前只保留为 audit/control，不应被描述为很可能解决 replacement 的方案，也不应优先于更直接的 Wav2Lip-control decoupling。
## 对现有 CEM 方案的判断

Global-shift CEM 适合诊断固定编码延迟，但不能解决局部 phone duration 和发音事件差异。LRS3 direct audit 已有 proxy/official AV-offset agreement 100%，但 replacement 仍退化；这说明当前主要失败点不太可能只是一个全局 offset。

因此，在升级 CEM、local knots 或 RL 前，应先验证 natural-driver、polarity-inversion 和高保真 identity reconstruction 三个更简单的控制。

## 推荐最小实验

在同一固定 n=15 cohort 上进行四臂 official replacement audit：

1. `natural → Wav2Lip → mux natural`：真实 oracle；
2. `-natural → Wav2Lip → mux natural`：特征等价控制；
3. `high-fidelity identity reconstruction → Wav2Lip → mux natural`：编码器/声码器容差；
4. `MFA-linear TTS → Wav2Lip → mux natural`：独立 TTS clock 对照。

判断：

- Arm 1 失败：先修复 Wav2Lip、素材或 baseline protocol；
- Arm 1/2 通过但 3 失败：主要是 codec/representation 损失，不应先训练 timing adapter；
- Arm 1/2/3 通过但 4 失败：主要是独立 TTS clock，再做 MFA + phone-local DTW；
- global offset 已一致但 replacement 仍失败：不要把问题简化为整段平移。

## 项目证据

现有 zero-residual direct protocol 使用 natural WavLM features 经 frozen HiFi-GAN 重合成。其 candidate-video official score略有改善，但换回 original natural audio 后下降：mean `ΔSync-C = -0.114`、mean `ΔSync-D = +0.149`。这表明 exact length 和近似时间轴不足以保证 replacement，局部 Wav2Lip 驱动特征仍需更接近 natural。

现有 MFA-linear 结果说明 phone-clock 对齐通常优于 raw TTS，但仍未稳定解决 replacement mismatch，因此 phone 边界一致也不是充分条件。

## Polarity-inversion official audit（2026-08-22）

LRS3 balanced n=15 使用 sample-exact `C=-N` 驱动 official Wav2Lip，再换回 original natural audio：

- natural/inverted Wav2Lip mel：MAE `0.0`，max abs `0.0`（15/15）；
- candidate/replacement AV offset 与 historical natural baseline：15/15 相同；
- replacement integrity：视频流和 original PCM 15/15 保持；
- inverted candidate vs historical natural：mean `ΔSync-C +0.0107`，mean `ΔSync-D +0.0059`，实质 parity；
- replaced vs historical natural：mean `ΔSync-C +0.0534`，mean `ΔSync-D -0.0375`，属于很小的 protocol/container-level 差异，不能解释为新方法增益；
- 与历史 natural Wav2Lip 视频的 elementary stream 只有 5/15 bit-identical，尽管 mel 完全相同，说明独立 Wav2Lip rerun/历史生成条件不应仅靠 stream hash 判定特征等价。

结论：polarity inversion 验证了 magnitude-feature invariance，但它与 natural driver 在 Wav2Lip 看来是同一个控制信号，不是有意义的第二种语音。它不能支持 VC/DTW 会成功。

Artifacts：

- smoke：`runs/lrs3_qwen_cloud_n500_20260818/13_lrs3_polarity_inversion_audit_smoke/`
- n15：`runs/lrs3_qwen_cloud_n500_20260818/14_lrs3_polarity_inversion_audit_n15/`

## Observations
- [definition] SyncNet 是 audio-video 同步判别器，不是 audio-audio 对齐器。
- [requirement] replacement 成功要求 candidate 与 natural 在 Wav2Lip/SyncNet 相关的局部短时事件上共享时间坐标。
- [control] `C=-N` 在 n=15 中得到逐样本完全相同的 Wav2Lip mel 和 15/15 AV-offset agreement，验证 magnitude-feature invariance，但不是有意义的第二种语音。
- [evidence] identity reconstruction 已在严格 2×2 audit 中表现出 crossed preference，不能被视为 VC 会成功的支持证据。
- [evidence] MFA-constrained hard DTW 与 MFA-linear 在 equal-budget screening 中无稳定差异，不应优先升级 DTW。
- [constraint] 总长度、MFA phone boundaries 或 global offset 相同均不是充分条件。
- [decision] natural driver 只是 oracle；若最终必须保留 natural audio 且 natural 在推理时可用，优先考虑直接解耦 Wav2Lip control，而不是继续押注 VC/DTW。
- [source] Researched and experimentally checked on 2026-08-22.
## Relations

- relates_to [[Wav2Lip proxy-to-official parity gate]]
- relates_to [[MFA-linear 冻结 VC 重合成与下游验证]]
- relates_to [[固定视频 SyncNet 监督 Gate B 结果]]
- depends_on [[tts-exp|Natural reference audio available at inference]]

## Sources

- https://www.robots.ox.ac.uk/~vgg/software/lipsync/
- https://github.com/joonson/syncnet_python
- https://arxiv.org/abs/2008.10010
- https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/workflows/alignment.html
- https://breakfastquay.com/rubberband/
