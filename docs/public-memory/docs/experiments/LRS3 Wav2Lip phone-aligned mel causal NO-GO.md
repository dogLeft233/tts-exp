---
title: LRS3 Wav2Lip phone-aligned mel causal NO-GO
type: experiment
permalink: tts-exp/docs/experiments/lrs3-wav2-lip-phone-aligned-mel-causal-no-go
capture_scope: lrs3-wav2lip-mel-causal-no-go
status: no-go
thread_id: 596d4aad-dd9d-48b1-8585-6ca75a768641
tags:
- lrs3
- wav2lip
- syncnet
- replacement
- mel-intervention
- causal-audit
- no-go
---

# LRS3 Wav2Lip phone-aligned mel causal NO-GO

## Context

项目的正式目标不是让 candidate audio 与 candidate-generated video 在 diagonal 上取得高分，而是让 TTS-like candidate driver `C` 生成的视频在换回 untouched original natural audio `N` 后，优于 natural driver baseline：

```text
Baseline = mux(Wav2Lip(face, N), N)
Target   = mux(Wav2Lip(face, C), N)
```

此前 60 clips / 30 train source groups 的 strict raw-TTS 2×2 已经表明 `Wav2Lip(face,T)+T` 的 diagonal 可以较高，但正式 replacement cell `Wav2Lip(face,T)+N` 相对 baseline 为极强 NO-GO。phone-aligned mel intervention 是本地 Wav2Lip TTS-transfer 路线最后一个预注册 existence audit：它不复制 TTS waveform，而是在冻结 official Wav2Lip 的真实 mel→generator seam 上，把 TTS mel 映射到 natural phone clock，再只注入小幅 residual。

本次正式 artifact：

`runs/lrs3_motion_teacher_20260823/12b_wav2lip_mel_intervention_causal_retry4/`

运行完整覆盖 30 条 train-only records、30 个互异 source groups；validation/test 均未访问，`test_media_opened=false`。固定 `alpha=0.25`、exact-label matched non-pause/non-`spn` phone mask、shared face boxes、official file-level SyncNet、untouched 16 kHz mono PCM replacement。没有 sweep alpha、mask、样本或阈值。

## Intervention and controls

固定四个 mel arms：

```text
identity = natural mel
 target  = natural mel + 0.25 × phone_mask × (phone-aligned TTS mel − natural mel)
shuffle  = natural mel + active-speech-only equal-norm shuffled residual
raw      = globally mapped TTS mel (diagnostic only)
```

所有 arm 都把生成视频重新 mux 回同一条 untouched natural PCM。ordinary official natural-audio path 与显式 identity mel path 同次运行，用于逐样本 interface parity；所有 arm 使用同一 face-box trajectory。

MFA alignment coverage 不是明显失败源：natural phone matched fraction 均值 `0.9267`、中位数 `0.9300`、范围 `[0.8261, 0.9841]`；masked mel fraction 均值 `0.7270`、中位数 `0.7363`。target residual 没有 clipping（30/30 `target_clip_fraction=0`）。两个全局 MFA failures 不在冻结的 60-record teacher subset 内，teacher subset provenance 完整。

## Official result

### Target versus identity

预注册主 gate 的八项检查全部失败，而且效应不是接近零的 marginal miss，而是稳定朝错误方向：

- mean ΔSync-C `-0.10233`，median `-0.10650`；cluster-bootstrap 95% CI `[-0.13610, -0.06610]`；仅 `5/30` C 改善；
- mean ΔSync-D `+0.06883`，median `+0.07850`；95% CI `[+0.02443, +0.11217]`；仅 `7/30` D 改善；
- C/D joint better 仅 `3/30 = 10%`，而 gate 要求 ≥60%；positive groups `3/30`，而 gate 要求 ≥20/30；
- target 与 identity 的 AV offset 为 `30/30` 相同，排除 global offset 变化作为主要解释。

对应 mean absolute scores：identity `C=7.3201, D=7.2904`；target `C=7.2178, D=7.3593`。

### Controls

- shuffle 也为 NO-GO：mean ΔC `-0.16153`、mean ΔD `+0.13227`、joint `3/30`；符合“非结构化 residual 不应通过”这一 specificity control。
- raw arm 是灾难性 NO-GO：mean ΔC `-5.38563`、mean ΔD `+5.37040`、joint `0/30`，复现 strict raw-TTS off-diagonal incompatibility。
- phone-aligned target 相对 shuffle 有一些结构性缓解：target−shuffle mean ΔC `+0.0592`，mean ΔD `-0.0634`；target 在 16/30 上有更高 C、19/30 上有更低 D、12/30 两指标同时优于 shuffle。这个结果只说明 phone alignment 比随机时序更少伤害，不能转化为相对 identity 的 replacement gain。

## Why this is a scientific NO-GO rather than an infrastructure failure

Gate 2A 与本次 30 条 full run 的基础设施检查都通过：

- identity parity `30/30` 通过；pre-H264 raw AVI frame MAE mean `0.00314`、max `0.00830`，远低于固定 threshold `1.0`；
- ordinary↔identity official score absolute drift mean约 `0.034`，最大 C `0.114`、最大 D `0.099`，均在预注册 `0.15` tolerance 内；AV offset `30/30` 一致；
- 所有 ordinary/identity/target/shuffle/raw cell 的 video payload、video PTS timeline、original PCM SHA/sample count 与 A/V relative start integrity 全通过；
- checkpoint、official inference、helper、teacher manifest、MFA target-context dataset、strict 2×2 parent 和 parity parent hashes 都被冻结并绑定；
- 30 groups 是预先固定的 train-only selection，没有按 score 删样、选样或调参。

因此不能再把 NO-GO 归因于 mel override 接错、face detection 不一致、H.264 parity confound、mux 编码、PCM 改写、A/V offset、MFA subset 缺失或 test leakage。

## Mechanism interpretation

当前最符合全部历史证据的解释仍是 diagonal co-adaptation，而不是一个可独立迁移的“TTS motion quality”变量。

Phone boundaries 只对齐了粗粒度发音事件的时钟，不能让 TTS 与 natural 在 Wav2Lip/SyncNet 所使用的局部声学轨迹上等价。注入 residual 仍携带 TTS-specific 的 phone 内 onset/release、coarticulation、能量、F0/formant、频谱及说话人实现差异。Wav2Lip 会把这些差异变成口部运动；最终换回 natural PCM 后，这些运动与 natural 的细粒度音频 embedding 不再匹配。自然 identity driver 本身已是与最终 `N` 最兼容的条件，因此 TTS residual 即使按 phone 对齐，仍系统性降低 replacement compatibility。

Target 比 shuffle 少伤害说明 phone-aligned residual 不是纯噪声，并且 temporal structure 确实保留了一部分相关信息；但它没有提供正 absolute effect。该信号不能被解读成“再训练一些”或“只需稍微调 alpha”就会成功。固定 alpha=0.25 的方向具有显著反向 CI；更小 alpha 可能只把伤害收缩到 identity 的零效应，更大或其他 alpha 是否反转没有证据。事后 sweep 会改变预注册问题，只能作为全新假设，不能挽救本 gate。

## Decision boundary

本结果关闭当前 **Wav2Lip-specific TTS-motion / phone-aligned TTS-mel transfer** 主线：

- 不进入 sample-exact waveform reachability oracle；该 gate 只有 mel causal GO 时才允许；
- 不训练 waveform/audio head，不扩大到 validation/test，不调 alpha/mask/threshold，不通过更多 steps、landmark loss、output-pixel teacher 或 proxy reward绕过正式 NO-GO；
- 不把 target 相对 shuffle 的“少伤害”称为 replacement enhancement；
- 不再把 candidate diagonal `Wav2Lip(face,C)+C` 当作成功证据。

这个结论不证明所有可能的 audio head、所有 TFG family 或所有 natural-conditioned control representation 都不可行。它严格否定的是：在当前 LRS3、冻结 Wav2Lip、official SyncNet 和预注册 phone-aligned TTS mel residual 构造下，存在稳定 replacement-compatible causal benefit。若未来重开方向，应提出不同的 teacher/estimand，例如直接学习与 `N` 兼容的 natural-conditioned control representation，而不是继续把 paired TTS mel residual 当作 motion-quality teacher；任何新路线都必须从新 protocol 开始，不能沿用本次 partial positives 调参。

## Observations

- [decision] LRS3 Wav2Lip phone-aligned TTS mel causal audit is a complete NO-GO; waveform reachability and audio-head training are not allowed downstream of this gate. #replacement #wav2lip
- [evidence] Target versus identity is significantly harmful in both official metrics: ΔC `-0.10233` with CI fully below zero and ΔD `+0.06883` with CI fully above zero. #syncnet
- [evidence] Only `3/30` clips/source groups jointly improve versus the required `18/30` clips and `20/30` groups. #heldout-protocol
- [integrity] Identity parity, all mux/PCM/video/timing checks, frozen provenance and train-only selection pass, so the NO-GO is scientific rather than an interface failure. #integrity
- [insight] Phone alignment reduces harm relative to shuffled residual but does not beat natural identity; coarse phone-clock equivalence is insufficient for replacement compatibility. #alignment
- [insight] TTS mel residual retains within-phone acoustic realization differences that Wav2Lip converts into motion incompatible with the final natural audio. #coadaptation
- [boundary] The result closes this Wav2Lip TTS-transfer construction, not every conceivable natural-conditioned audio head or every TFG architecture. #scope

## Relations

- extends [[tts_tfg_mechanism_report]]
- follows [[Raw TTS strict replacement NO-GO]]
- relates_to [[让两个音频共享 SyncNet 时间坐标]]
- relates_to [[tts-exp|Wav2Lip replacement is primary objective]]
