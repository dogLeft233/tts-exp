---
title: AISHELL-1 natural silence and phone-boundary residual experiment
type: task
permalink: tts-exp/tasks/aishell-1-natural-silence-and-phone-boundary-residual-experiment
current_step: 5
priority: high
status: done
steps:
- 固定输入、model、speaker/sample selection 和条件定义，核对 GPU 不终止既有进程。
- 新增 silence-mask/boundary-smoothing 脚本和测试，复用 frozen MFA-linear target。
- 生成 baseline、silence anchor、boundary smoothing、组合条件音频并完成 QC。
- 比较 S0901/S0912 的峰值、crest、局部瞬态候选、silence 区域和 feature cosine。
- 更新项目笔记；只有 audio gate 通过的条件才考虑统一音频轨道协议下的下游评测。
tags:
- task
- mfa-linear
- natural-anchor
- silence
- boundary
- aishell1
completed: '2026-08-14'
---

# AISHELL-1 natural silence and phone-boundary residual experiment

## Observations
- [description] 固定 alpha=1，在 S0765/S0901/S0912 各 5 条 paired utterance 上测试 natural silence anchor 与 phone-boundary residual smoothing，针对 MFA-linear 局部重合成伪影。
- [status] done
- [assigned_to] claude
- [completed] 2026-08-14
- [result] 90/90 音频生成成功（15 samples × 6 conditions），0 failures；全部 16 kHz、exact natural length、finite、0 clipped samples；相关回归测试 22 passed。
- [result] 条件：`unconstrained`、`silence_anchor`、`boundary_smooth_r2`、`boundary_smooth_r3`、`silence_anchor_boundary_smooth_r2`、`silence_anchor_boundary_smooth_r3`。Alpha 固定为 1。
- [result] S0901 mean peak/crest：unconstrained `0.3313/12.449`；silence anchor `0.3380/12.649`；boundary r2 `0.3315/12.272`；boundary r3 `0.3585/13.257`；combined r2 `0.3357/12.314`；combined r3 `0.3618/13.275`。没有稳定修复，r3 变差。
- [result] S0912 mean peak/crest：unconstrained `0.4991/8.647`；silence anchor `0.4810/8.463`；boundary r2 `0.4932/8.545`；boundary r3 `0.4990/8.617`；combined r2 `0.4854/8.576`；combined r3 `0.4942/8.705`。silence anchor 有小幅改善，r2 组合次之。
- [result] Silence anchor 显著提高 output→natural feature cosine：S0901 `0.6983→0.8094`，S0912 `0.7441→0.7952`；但 feature proximity 不等价于音质或 lip-sync 改善。
- [result] S0901/103 silence anchor peak 从 `0.2312` 升至 `0.2618`，说明该策略不能修复该重点样本；S0912/203 silence anchor peak 从 `0.6226` 降至 `0.6173`。
- [decision] 暂不推进 boundary/silence 条件到下游；若继续，优先做 phone-context-aware transition 或 natural F0/energy conditioning，并保留 audio gate。
- [protocol] 未来 TFG/SyncNet 必须固定视频音频轨道的 sample rate、编码和 mux；已有 audio-track replacement 实验表明仅替换视频音频轨会改变 SyncNet 分数，不能把该差异直接解释为嘴型质量。

## Steps
1. [x] 固定输入、model、speaker/sample selection 和条件定义，核对 GPU 不终止既有进程。
2. [x] 新增 silence-mask/boundary-smoothing 脚本和测试，复用 frozen MFA-linear target。
3. [x] 生成 baseline、silence anchor、boundary smoothing、组合条件音频并完成 QC。
4. [x] 比较 S0901/S0912 的峰值、crest、局部瞬态候选、silence 区域和 feature cosine。
5. [x] 更新项目笔记；当前条件没有稳定 audio-level 修复，暂不进行下游评测。

## Context
### Inputs
- Cohort: `/home/wjj/tts-audio/tts-exp/runs/aishell1_mfa_linear_n25_20260813/data_boundary/aishell1_n25_speaker_balanced_cohort.json`
- Resample-poly TTS metadata: `/home/wjj/tts-audio/tts-exp/runs/aishell1_mfa_linear_n25_resample_poly_20260813/tts_meta.json`
- TTS WAV: `/home/wjj/tts-audio/tts-exp/runs/aishell1_mfa_linear_n25_resample_poly_20260813/tts_resample_poly_16k/`
- Frozen MFA tokens: `/home/wjj/tts-audio/tts-exp/runs/aishell1_mfa_linear_n25_20260813/mfa/tokens.json`
- Existing MFA-linear reference: `/home/wjj/tts-audio/tts-exp/runs/aishell1_mfa_linear_n25_resample_poly_20260813/mfa_linear/summary.json`

### Outputs/code
- Output: `/home/wjj/tts-audio/tts-exp/runs/aishell1_mfa_linear_n25_resample_poly_20260814/natural_anchor_boundary_20260814/`
- Summary/report: `natural_anchor_boundary_20260814/summary.json`, `report.md`
- Generator: `/home/wjj/tts-audio/tts-exp/scripts/run_natural_anchor_boundary_silence.py`
- Tests: `/home/wjj/tts-audio/tts-exp/tests/test_natural_anchor_boundary_silence.py`
- Mapping: `/home/wjj/tts-audio/tts-exp/scripts/knn_vc_retrieval.py:mfa_linear_target`
- Frozen `bshall/knn-vc` revision `c616845c4e309e24d5927f15adbdf277a3d65358`; WavLM-Large L6 1024-d; prematched frozen HiFi-GAN; CUDA.

### Definition
Natural silence anchor zeros `d = z_tts_warp-z_natural` on natural MFA silence/non-speech frames. Boundary smoothing replaces residual within ±2 or ±3 WavLM frames of natural token boundaries with a 3-frame moving-average trajectory. No F0/energy conditioning, learned decoder, or video evaluation was used.

## Relations
- project [[MFA-linear 冻结 VC 重合成与下游验证]]
