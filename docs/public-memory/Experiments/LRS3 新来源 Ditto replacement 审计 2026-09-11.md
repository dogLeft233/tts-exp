---
title: LRS3 新来源 Ditto replacement 审计 2026-09-11
type: experiment
permalink: tts-exp/experiments/lrs3-新来源-ditto-replacement-审计-2026-09-11
status: concluded
tags:
- lrs3
- ditto
- replacement
- syncnet
- new-source
- cross-generator
- exploratory
- negative-result
---

# LRS3 新来源 Ditto replacement 审计 2026-09-11

## 结论

在冻结的 14 个新来源 LRS3 source groups 上完成 Ditto TRT Ampere_Plus 两臂生成（natural_raw / direct_raw），并完成严格 untouched-natural-audio replacement 与 candidate-audio self diagnostic。两条端点都没有建立新的 Ditto 收益；本结果是探索性审计，不是确认性功效结论。

## 资源与执行

- 服务器：NVIDIA GeForce RTX 4080 SUPER 32 GB，driver 580.105.08。
- Ditto 环境：Python 3.10，torch 2.5.1+cu121，TensorRT 8.6.1，CUDA 可见。
- 生成：28/28 MP4 完成并通过 H.264/帧数/时长检查。
- mux：28/28 严格 Matroska、视频流 copy、16 kHz mono PCM16；视频 elementary stream 与 natural PCM 均验证一致。
- SyncNet V2：replacement 和 self 各 28/28，均有可解析 C/D/offset。

## Replacement 主端点

定义 `delta_C = direct - natural`，`D_benefit = natural_D - direct_D`：

- n=14；mean delta_C = -0.059，median = -0.148，bootstrap 99% CI [-0.172, +0.073]。
- mean D_benefit = -0.124，median = -0.136，bootstrap 99% CI [-0.299, +0.059]。
- C positive 5/14，D positive 5/14，joint positive 5/14。

## Self diagnostic

- mean delta_C = -0.048，median = -0.066，bootstrap 99% CI [-0.224, +0.112]。
- mean D_benefit = -0.213，median = -0.136，bootstrap 99% CI [-0.487, -0.004]。
- C positive 6/14，D positive 4/14，joint positive 3/14。

Self diagnostic 仅用于检查 candidate-video/candidate-audio 自洽，不能代替 replacement 门槛。

## 产物

- `runs/fresh_source_cross_generator_20260911_unblock_r4/run/remote_ditto_summary.json`
- `runs/fresh_source_cross_generator_20260911_unblock_r4/run/replacement/ditto/replacement_manifest.json`
- `runs/fresh_source_cross_generator_20260911_unblock_r4/run/replacement/ditto/syncnet_eval/scores.json`
- `runs/fresh_source_cross_generator_20260911_unblock_r4/run/replacement/ditto/analysis.json`
- `runs/fresh_source_cross_generator_20260911_unblock_r4/run/self/ditto/syncnet_eval/scores.json`
- `runs/fresh_source_cross_generator_20260911_unblock_r4/run/self/ditto/analysis.json`

下一步按既定边界转向另一生成器和独立视觉验证，不在同批样本上继续 Ditto 小幅参数调整。服务器任务完成后已关机。

## Observations

- [status] concluded
- [result] 在 14 个新来源 LRS3 source groups 上完成 Ditto replacement 与 candidate-audio self diagnostic，replacement 主端点未建立新的 Ditto 收益。
- [boundary] 这是探索性审计，不是确认性功效结论；下一步转向另一生成器和独立视觉验证。

## Relations

- follows [[新来源跨生成器与视觉验证 2026-09-10]]
- relates_to [[LRS3 visual dynamic specificity 2026-09-10]]
