---
title: 保守 acoustic envelope correction 与内容保持验证
type: task
permalink: tts-exp/tasks/保守-acoustic-envelope-correction-与内容保持验证
current_step: 5
priority: high
status: done
tags:
- task
- syncnet
- acoustic
- content-preservation
completed: '2026-08-20'
---

# 保守 acoustic envelope correction 与内容保持验证

## Observations
- [description] 在 fixed real video + frozen SyncNet 上，验证 natural-derived energy-envelope correction 的小强度版本是否能在不破坏 TTS 内容与说话人特征的情况下保留 SyncNet gain；该实验仍是 oracle diagnosis，不代表部署 policy。
- [status] done
- [assigned_to] claude
- [completed] 2026-08-20
- [current_step] 5
- [dependency_check] 当前环境没有 transformers/torchaudio/librosa/speechbrain/pyworld，因此 PER/ASR/speaker verification/HuBERT content 未运行；artifact 明确标记为 proxy-only。
- [envelope_full] 69 条 conservative envelope full 已通过：alpha=0.25 median gain=0.0116、positive=82.61%、bootstrap CI=[0.0091,0.0172]、spectral-shape cosine min=0.999997、content proxy gate=100%。
- [timing_full] 在 alpha=0.25 envelope warm start 后，69 条 global timing CEM full 通过：median additional gain=0.3215、positive=79.71%、bootstrap CI=[0.2240,0.4175]、6/6 source-group medians positive、repeat difference=0 ms、boundary hit=0%。
- [local_timing_boundary] local 4-knot synthetic timing route 仍然 NO-GO；global timing 可以可靠增加收益，但不能据此扩展到 local knots。
- [final_decision] 当前最稳妥的实验组合是 conservative energy envelope（alpha=0.25, ratio=[0.8,1.25]）+ global timing residual；但 correction 仍 natural-derived oracle，且缺少正式 PER/ASR/speaker verification，因此不能声称已有可部署泛化 policy。

## Steps
1. [x] 核对内容指标可用性；PER/ASR/speaker verification/HuBERT 运行依赖不可用，改用 log-mel spectral-shape 和 active-frame proxy，并明确不是正式 PER。
2. [x] 实现 conservative envelope-strength ablation：强度 α、ratio clamp、exact-length/finite 检查；不加入 spectral tilt 或 timing warp。
3. [x] 在 12 条 pilot 上比较 SyncNet gain 与内容保持指标；四个强度的 content proxy 均通过。
4. [x] 运行 69 条 conservative envelope full；alpha=0.25 作为最保守通过配置。
5. [x] 完成 timing-after-envelope 的 69 条 full；global timing 通过，local knots 保持停止；当前卡点是 natural-derived oracle 和缺少正式内容/说话人指标。

## Artifacts
- conservative envelope pilot: `runs/lrs3_cem_fixed_video_20260820/07_conservative_envelope_content_pilot/summary.json`
- conservative envelope full: `runs/lrs3_cem_fixed_video_20260820/08_conservative_envelope_content_full/summary.json`
- timing pilot: `runs/lrs3_cem_fixed_video_20260820/09_timing_after_conservative_envelope_pilot/summary.json`
- timing full: `runs/lrs3_cem_fixed_video_20260820/10_timing_after_conservative_envelope_full/summary.json`

## Context
- Gate 3A local timing 未通过 promotion：30 ms recovery=83.3%、evaluation reward recovery=83.3%。
- Gate 4A full：50 ms natural-derived energy-envelope oracle median gain=0.0989，positive=72.46%，bootstrap CI=[0.0679,0.1378]，6/6 group medians positive；global gain/spectral tilt 未通过。
- Gate 4B combination pilot 未通过，未增加 spectral tilt 或 RL。
- 既有 energy oracle 使用 ratio clamp [0.5,2.0]；当前安全配置是 ratio [0.8,1.25]、strength alpha=0.25，要求不改变时间轴。Timing full 只搜索 global residual；timing action 的 frame-wise spectral proxy 不用于判断内容保持，因为 intentional timing shift 会改变同一绝对 frame 的比较。

## Relations
- project [[固定真实视频上的 CEM residual 对齐方案]]
