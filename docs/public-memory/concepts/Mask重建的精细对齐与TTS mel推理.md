---
title: Mask重建的精细对齐与TTS mel推理
type: note
permalink: tts-exp/concepts/mask-重建的精细对齐与-tts-mel-推理
tags:
- masked-reconstruction
- tts
- alignment
- supervision
- soft-dtw
- hard-dtw
- replacement
- experiment
---

# Mask重建的精细对齐与TTS mel推理

## Context

本线程评估在每个 MFA 音素内部使用 natural 与 paired TTS 连续特征进行 Soft-DTW/硬 DTW，再将 TTS acoustic representation 重采样到 natural 时间轴的路线。项目目标不是 candidate audio 与其生成视频的自洽分数，而是 candidate 驱动的 frozen Wav2Lip/TFG 视频在替换 untouched natural PCM 后仍能获得稳定官方 SyncNet 收益。

因此必须分开判断 natural reconstruction、paired TTS trajectory preservation、candidate-driven TFG gain 和 replacement-safe gain。首轮只做 path-only comparison，不加入 residual 或 Mask Transformer。

## 当前模型与方案边界

当前 masked-natural reconstruction 模型的 TTS 输入是按 MFA 音素边界截取、再用 `phone_phase_linear` 拉伸到 natural core 长度的 WavLM-L6；输出是 80-bin natural mel，没有 `aligned_TTS + residual` bypass。它证明 TTS modality 有助于 natural reconstruction，不证明 TTS acoustic identity 或正确 paired instance trajectory 被保留。

若 `S_TTS` 是原始 TTS representation、`P` 是 TTS 到 natural 的单调时间映射，则 `B=P(S_TTS)` 应直接计算；已知 linear mapping 不应训练模型复现。只有未知 correspondence、路径蒸馏/加速或有限 seam correction 才需要训练。

## 历史判断

- AISHELL hard-DTW 与 MFA-linear 的 equal-budget feature reachability 对照没有稳定 DTW 优势；这削弱 nonlinear-path 预期，但不能作为 Soft-DTW 直接反例。
- AISHELL 多说话人 n=25 未复现早期单说话人 MFA-linear 正结果；LRS3 MFA-linear WavLM + HiFi-GAN strict replacement 的 C/D benefit 均显著为负。
- LRS3 n=30 的 `natural mel + 0.25 * (aligned TTS mel - natural mel)` 具体 residual 显著为负，但它不是完整 `B=P(S_TTS_mel)`。
- masked reconstruction scale-up 证明 TTS modality 有用，但 paired TTS 相对 same-phone wrong-instance 未稳定优于；within-phone reversed 结果支持 phone 内顺序有作用，但未证明正确 paired instance 是必要条件。
- direct WavLM resynthesis 的 2×2 identity control 显示 audio-video co-adaptation 是 candidate replacement 的主要系统性风险。
- 未经 leakage probe，HuBERT/WavLM 只能称 frozen SSL matching feature，不能称 speaker-invariant。

## MFA-linear 与 MFA-hard-DTW LRS3 对比结果

原始 hard-DTW protocol 在 Stage 01 发现固定 center-time ownership 对 11 个短 matched phone 不可行，保持为独立的 engineering BLOCKED / scientific decision not available 证据，没有用 122 条子集做 outcome 结论。

随后建立并严格校验了独立 protocol revision：

`openspec/changes/resolve-lrs3-mfa-dtw-short-phone-ownership/`

修订规则是：每个 WavLM frame 使用以 nominal frame center 为中心的半 stride support 区间；如果 support 与精确 MFA3 phone interval 有正交叠，则该 frame 可进入该 phone 的 source set。相邻 phone 可以共享边界 frame；不允许非相邻、不同 instance 或不重叠 phone 的 frame 泄漏；超出最终 nominal feature support 的 tail 不外推。natural frame rows、silence policy、band、tie order、TTS-only conditioning 和 downstream gates 均未改变。

主实验 run root：

`runs/lrs3_mfa_dtw_replacement_short_phone_20260904/`

- Stage 00：133/133 records，23 source groups，ordered cohort hash `61f8c982041cdfdded8daf8850d31e127943386ba2cb7035aa742e94cea9a973`。
- Stage 01：133/133 hard-DTW candidates 全部通过；所有候选只从 TTS WavLM values 构造，`natural_values_in_conditioning=false`。
- Stage 02：133/133 frozen Wav2Lip diagonal renders 和 official SyncNet V2 scores；133/133 mux 的 decoded PCM exact match 与 video stream-copy 校验通过。
- Stage 03：23 source groups、10,000 draws、seed `20260903` 的 cluster bootstrap。
- 注册 Stage 04：因 promotion gate 未通过而保持 `sealed_not_run`。

candidate-driven diagonal 配对指标为：

```text
delta_C = SyncC_DTW - SyncC_MFA_linear
delta_D = SyncD_MFA_linear - SyncD_DTW
```

结果：

- MFA-linear：Sync-C mean `6.8747`，Sync-D mean `7.5003`；
- MFA-hard-DTW：Sync-C mean `6.8661`，Sync-D mean `7.4931`；
- `delta_C` mean `-0.0087`，95% CI `[-0.0624, 0.0316]`；
- `delta_D` mean `0.0072`，95% CI `[-0.0598, 0.0356]`；
- DTW 在 C 上胜出 63/133，在 D 上胜出 60/133，同时胜出 49/133；
- final decision：`NO_DTW_TFG_ADVANTAGE`。

两个 confidence-interval lower bound 都不严格大于零，因此注册 replacement gate 未授权。

## Exploratory replacement 结果

用户随后明确要求直接比较 replacement 表现，因此另行执行了 descriptive-only exploratory run，没有修改注册实验的 Stage 04 或 final decision：

`runs/lrs3_mfa_dtw_exploratory_replacement_20260904/`

该 run 复用已生成的 133 个 DTW diagonal video，替换为 untouched natural PCM，使用同一 strict mux 和 official SyncNet V2；全部 133 条通过 PCM exact match 与 video stream-copy 校验。MFA-linear replacement 使用绑定历史 manifest 的 `G_M_E_N` cell。

绝对分数：

- MFA-linear replacement：Sync-C mean `6.0093`，Sync-D mean `8.3679`；
- MFA-DTW replacement：Sync-C mean `5.9446`，Sync-D mean `8.4329`。

DTW replacement 相对 MFA-linear replacement 的 paired benefit 定义仍是 `delta_C = DTW C - linear C`、`delta_D = linear D - DTW D`：

- `delta_C` mean `-0.0647`，95% CI `[-0.1420, -0.0270]`；
- `delta_D` mean `-0.0649`，95% CI `[-0.1340, -0.0396]`；
- DTW 在 C 上胜出 50/133，在 D 上胜出 57/133，同时胜出 41/133；
- 两个平均 benefit 都为负，且 bootstrap CI 都完全位于零以下，表示 DTW replacement 相对 MFA-linear replacement 在 C、D 两项均更差。

两种 replacement 相对 untouched natural baseline 也都明显下降：

- DTW vs natural：`delta_C` mean `-1.2654`，`delta_D` mean `-1.1262`；
- MFA-linear vs natural：`delta_C` mean `-1.2007`，`delta_D` mean `-1.0613`。

这组 replacement 数值的解释范围是 descriptive-only；它不能把注册实验升级为 replacement-safe，也不能改变已注册的 `NO_DTW_TFG_ADVANTAGE`。注册 replacement gate 仍保持 sealed。

## TTS 来源

本轮 MFA-linear 和 MFA-hard-DTW 使用的是同一批云端 Qwen 音频，不是两套不同 TTS。绑定来源为：

- provider：`dashscope_vc`；
- model：`qwen3-tts-vc-2026-01-22`；
- manifest：`runs/lrs3_qwen_cloud_n500_20260817/02_tts/tts_meta.json`。

## 方案组件判定

| 组件 | 判定 |
|---|---|
| 逐音素 Soft-DTW | 未测试，开放但负先验明显 |
| MFA-constrained hard-DTW | 完整 133-record diagonal 和 exploratory replacement 均已完成；未优于 MFA-linear，replacement 上进一步落后 |
| speaker-invariant matching feature | 未验证；HuBERT/WavLM 只能称 frozen SSL matching feature |
| natural 只决定 path、TTS 提供值 | 结构上实现，但没有带来 downstream advantage |
| 完整 `B_linear` / `B_hard` / `B_soft` TTS-mel warp | `B_hard` 已完成 LRS3 diagonal/replacement diagnostic，未优于 MFA-linear；Soft-DTW 尚未形成 frozen downstream arm |
| `natural + 0.25 * aligned-TTS residual` | LRS3 n=30 具体版本已否定 |
| 直接训练模型复现已知 linear warp | 不必要 |
| bounded natural-conditioned residual | 保持关闭 |

## 下一步决策

本轮 hard-DTW 已在完整 frozen cohort、固定下游模型和预注册 cluster-bootstrap gate 下未建立 candidate-driven TFG 优势；独立 exploratory replacement 也显示 DTW 相对 MFA-linear 在两个 replacement 指标上都更差。不要继续扩展 hard-DTW、放宽 band、改做子集 outcome 或直接训练 residual。

若继续研究 Soft-DTW，必须另起独立 protocol，预先固定 temperature、regularization、phone-local support 和 downstream gate；不能把 hard-DTW 结果直接当作 Soft-DTW 的形式否定。waveform decoder gate 继续 `CLOSED`。

## Observations

- [decision] Soft-DTW full-TTS-mel 不能被历史结果直接否定，但应视为低先验、严格筛选路线。 #soft-dtw
- [insight] nominal-support/shared-boundary revision 解决了 hard-DTW 的完整 cohort feasibility blocker，但没有产生 frozen-TFG advantage。 #alignment
- [result] 完整 133-record diagonal 得到 `NO_DTW_TFG_ADVANTAGE`，C/D bootstrap lower bounds 都不严格大于零。 #experiment
- [result] 独立 exploratory replacement 中，DTW 相对 MFA-linear 的 Sync-C 与 Sync-D paired benefits 分别为 `-0.0647` 和 `-0.0649`，两个 95% CI 都完全低于零。 #replacement
- [constraint] 不能未经 leakage probe 把 HuBERT/WavLM 称为 speaker-invariant。 #representation
- [decision] 首轮仍只做 path-only comparison，不加入 residual 或 Mask Transformer。 #experiment
- [decision] waveform decoder gate 继续 CLOSED；注册 replacement gate 因 Stage 03 未授权而保持 sealed。 #gate

## Relations

- relates_to [[natural-slot-oracle-result]]
- relates_to [[Direct waveform upper bound]]
- relates_to [[LRS3 masked TTS trajectory-specificity diagnosis]]
- relates_to [[LRS3 masked TTS retention scale-up preflight]]
- relates_to [[LRS3 MFA-linear TFG native/replacement 2026-09-13|LRS3 MFA-linear replacement NO-GO]]