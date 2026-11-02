---
title: 016-source-clock-roi-and-full-delivery-pcm
type: decision
permalink: tts-exp/docs/adr/016-source-clock-roi-and-full-delivery-pcm
scope: artifact PCM validation; frozen scientific contract unchanged
status: open
decided: '2026-10-07T19:20:45.2786677+00:00'
repository: dogLeft233/tts-exp
project: tts-exp
---

# 016-source-clock-roi-and-full-delivery-pcm

## Decision

按产物职责核验原音：完整成片和完整自然视频 mux 必须与完整原音逐样本一致；selected-track 分析 mux 必须与原音的确切源帧区间逐样本一致，并保存区间与绝对时钟映射。分析切片不能标为完整成片，不能把切片音轨当完整自然输入。保留 producer05 几何、缓存和已经结束的 Fit40 训练；新审计明确核验这两种契约。

## Context and rationale

rawdecode03 全4,979成员与已验证归档一致，但错误把 selected-track mux 要求为完整原音，第一行失败。实际只读 probe02 从原 WAV 与 MKV 解码：原音62,639样本，selected源帧[0,97)应为62,080样本；解码62,080，PCM SHA64ea76d743cdfe9201c2e6b17d3c52ec04ef49022f0448552c658288d17ffa91，与源切片逐样本一致，末尾559样本是区间之外的源尾部。97视频帧25fps、PTS0..3.84，音频从0到3.88，未用-shortest。全自然mux另有62,639样本逐样本PASS。

训练与固定末步评分从完整源 WAV 及完整自然音频缓存计算，不消费 selected mux；子 agent 已审计对应读取路径。中间ROI遵循既有007的直接原音切片定义。无需通过改权重、重训、补零或覆盖原归档来处理审计器的契约错误。

## Alternatives and consequences

把所有ROI重新合成完整音轨，会混淆 cropped-video局部零时钟和绝对源时钟；放宽最终完整PCM要求，会改变用户要求。两种方式均不采用。新 rawdecode04 保留全库存/源SHA/精确时钟/解码检查，分别对完整原音和源区间切片核验。Dev/Confirm成片完整原音门槛、common-query/GUARD20、RMS对照、阈值和失败分母保持冻结。当前只有一行实际ROI诊断通过，全40仍待执行。

## References

- `.codex/scratch/fit40_native_rms_geometry_attempt05_mux_probe_attempt02/00_readonly_selected_track_mux_diagnostic_attempt02_request.json.response.json`，raw SHA c668f89c8090b1cb3dc9bbc26886e99be322069b5db1b3e685966c13cdf61a19。
- `.codex/scratch/fit40_native_rms_geometry_attempt05_root_rawdecode_only_attempt03/07_readonly_status_attempt03_request.json.response.json`，原失败及全库存匹配保留。
- `scripts/experiments/cross_tfg_audio_head_20261003/run_fit40_native_transition_head_v3_attempt11.py`，完整自然音训练读取；producer05几何SHA efa5b0bfba926e6769bcc2d1b2fae2300a7c7e50fdb7dcb5ded56e58552c9208。

## Observations

- [decision] 完整成片按全原音、分析ROI按明确源区间核验精确PCM，二者保存可追踪的源时间映射。
- [rationale] 实际解码证实第一行selected mux完全等于声明的源切片；原fullaudio训练路线不使用它。
- [consequence] 修正审计器的产物契约，保留完整原音成片、训练和旧失败证据；全40核验仍必须实际完成。

## Relations

- refines [[007-original-pcm-scoring-for-waveform-head]]
- affects [[开发跨 TFG 的自然音频增强头]]
- affects [[跨 TFG 自然音频增强头的训练与原音换轨验证]]