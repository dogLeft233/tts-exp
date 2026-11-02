---
title: AISHELL-1 natural-anchor alpha blend audio experiment
type: task
permalink: tts-exp/tasks/aishell-1-natural-anchor-alpha-blend-audio-experiment
current_step: 5
priority: high
status: done
steps:
- 检查 GPU/输入 provenance，冻结 speaker/sample selection 和 alpha 定义。
- 新增独立 alpha-blend 生成脚本；验证 natural/TTS feature shape、MFA tokens、model hashes 和输出目录隔离。
- 生成 15 × 5 条音频，检查 16 kHz、exact natural length、finite、peak/clipping、hash 和 alpha=1
  与已有 MFA-linear 的一致性/差异。
- 提取 alpha 对比特征：peak、crest、F0/energy、spectral、speaker/phonetic metadata；写出 summary/report。
- 更新项目实验笔记，判断较小 alpha 是否缓解 S0901/S0912 的局部变质，再决定是否进入下游评测。
tags:
- task
- mfa-linear
- natural-anchor
- alpha-blend
- aishell1
completed: '2026-08-14'
---

# AISHELL-1 natural-anchor alpha blend audio experiment

## Observations
- [description] 对固定 speaker `S0765`、`S0901`、`S0912` 的 15 条 paired utterances 生成 natural-anchor MFA residual 五档音频：`alpha=0/0.25/0.5/0.75/1`。
- [status] done
- [assigned_to] claude
- [completed] 2026-08-14
- [result] 75/75 音频生成成功，0 failures；所有输出 16 kHz、exact natural length、finite、0 clipped samples。GPU 使用 `cuda`，未终止任何既有进程。
- [result] Alpha=1 与已有 resample-poly MFA-linear 参考逐条数值等价：15/15 的最大绝对波形差 ≤1e-5；worst max diff `5.94e-6`，worst RMSE `8.23e-8`。SHA 不同是 CUDA/浮点重算造成，不是输入协议混用。
- [result] S0901 speaker mean peak/crest：alpha 0 `0.3240/10.977`，0.25 `0.2944/11.009`，0.5 `0.2994/12.189`，0.75 `0.3199/12.801`，1 `0.3313/12.449`。较小 alpha 没有稳定降低 crest；alpha 0.5–0.75 反而更高。
- [result] S0912 speaker mean peak/crest：alpha 0 `0.5724/9.834`，0.25 `0.5463/9.726`，0.5 `0.5165/9.375`，0.75 `0.4927/8.762`，1 `0.4991/8.647`。S0912 的 peak/crest 随 alpha 整体下降，但 alpha=0 的 natural direct resynthesis 本身也有高峰值。
- [result] 重点样本 S0901/103：alpha 0/0.25/0.5/0.75/1 peak=`0.2618/0.1846/0.2217/0.2427/0.2312`，crest=`9.932/9.715/12.487/13.503/11.529`；S0912/203 peak=`0.7543/0.7086/0.6675/0.6687/0.6226`，crest=`12.244/11.988/11.482/11.592/10.758`。
- [decision] alpha blend 只改变 feature replacement 强度，不能作为可靠修复；下一步若继续，应先加 residual norm/velocity constraint、phone-boundary smoothing、silence anchor 或 natural F0/energy conditioning，再做 audio-only QC。

## Steps
1. [x] 检查 GPU/输入 provenance，冻结 speaker/sample selection 和 alpha 定义。
2. [x] 新增独立 alpha-blend 生成脚本；验证 natural/TTS feature shape、MFA tokens、model hashes 和输出目录隔离。
3. [x] 生成 15 × 5 条音频，检查 16 kHz、exact natural length、finite、peak/clipping、hash 和 alpha=1 与已有 MFA-linear 的一致性/差异。
4. [x] 提取 alpha 对比特征：peak、crest、F0/energy、spectral、speaker/phonetic metadata；写出 summary/report。
5. [x] 更新项目实验笔记；结论是较小 alpha 对 S0901/S0912 没有一致修复效果，暂不进入下游评测。

## Context
### Inputs
- Frozen cohort: `/home/wjj/tts-audio/tts-exp/runs/aishell1_mfa_linear_n25_20260813/data_boundary/aishell1_n25_speaker_balanced_cohort.json`
- Resample-poly TTS metadata: `/home/wjj/tts-audio/tts-exp/runs/aishell1_mfa_linear_n25_resample_poly_20260813/tts_meta.json`
- Resample-poly TTS WAV directory: `/home/wjj/tts-audio/tts-exp/runs/aishell1_mfa_linear_n25_resample_poly_20260813/tts_resample_poly_16k/`
- Frozen MFA tokens: `/home/wjj/tts-audio/tts-exp/runs/aishell1_mfa_linear_n25_20260813/mfa/tokens.json`
- Existing MFA-linear reference: `/home/wjj/tts-audio/tts-exp/runs/aishell1_mfa_linear_n25_resample_poly_20260813/mfa_linear/summary.json`

### Outputs
- Alpha audio and metadata: `/home/wjj/tts-audio/tts-exp/runs/aishell1_mfa_linear_n25_resample_poly_20260813/natural_anchor_alpha_20260814/`
- Report: `natural_anchor_alpha_20260814/report.md`
- Summary: `natural_anchor_alpha_20260814/summary.json`
- Alpha arms: `audio/alpha_0p00`, `alpha_0p25`, `alpha_0p50`, `alpha_0p75`, `alpha_1p00`

### Models and code
- Generator: `/home/wjj/tts-audio/tts-exp/scripts/run_natural_anchor_alpha_blend.py`
- Alpha unit tests: `/home/wjj/tts-audio/tts-exp/tests/test_natural_anchor_alpha_blend.py`
- Mapping: `/home/wjj/tts-audio/tts-exp/scripts/knn_vc_retrieval.py:mfa_linear_target`
- Tail handling: `/home/wjj/tts-audio/tts-exp/scripts/pilot_generate_mfa_linear.py:extend_tokens_for_feature_tail`
- Audio QC: `/home/wjj/tts-audio/tts-exp/scripts/audit_aishell1_n25_audio_artifacts.py:extract_audio_features`
- Frozen `bshall/knn-vc` revision `c616845c4e309e24d5927f15adbdf277a3d65358`; WavLM-Large L6 1024-d, prematched frozen HiFi-GAN; device `cuda`.

### Definition
`z_alpha = z_natural + alpha * (z_tts_warp - z_natural)` on natural WavLM-L6 frames. `alpha=0` is natural-feature direct resynthesis; `alpha=1` is full MFA-linear. No video evaluation was run.

## Relations
- project [[MFA-linear 冻结 VC 重合成与下游验证]]
