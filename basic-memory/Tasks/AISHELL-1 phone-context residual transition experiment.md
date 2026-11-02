---
title: AISHELL-1 phone-context residual transition experiment
type: task
permalink: tts-exp/tasks/aishell-1-phone-context-residual-transition-experiment
current_step: 5
priority: high
status: done
steps:
- 固定输入、模型、speaker/sample selection 和 crossfade 条件；记录 GPU/协议状态。
- 新增 context-crossfade 生成脚本和测试，验证只修改 boundary windows、silence mask 行为和 exact length。
- 生成 unconstrained、crossfade r2/r3、silence-anchor crossfade r2/r3 音频并完成 QC。
- 比较 S0901/S0912 peak、crest、spectral flux、feature cosine 和重点样本。
- 更新项目笔记；仅当 audio gate 稳定时才进入统一音频轨道协议的下游评测。
tags:
- task
- mfa-linear
- natural-anchor
- phone-context
- crossfade
- aishell1
completed: '2026-08-14'
---

# AISHELL-1 phone-context residual transition experiment

## Observations
- [description] 固定 alpha=1，在 S0765/S0901/S0912 各 5 条 paired utterance 上测试 phone-boundary context-aware residual crossfade。
- [status] done
- [assigned_to] claude
- [current_step] 5
- [result] 75/75 输出成功；unconstrained alpha=1 与既有 MFA-linear 15/15 数值等价（max abs <= 1e-5）；相关完整回归 24 passed。
- [decision] radius=2 仅有轻微、speaker-dependent 变化，radius=3 明显恶化；不进入 Wav2Lip/SyncNet。
- [boundary] peak、crest 或 output-to-natural cosine 变化不能证明音质、音素保真或 lip-sync 改善。

## Steps
1. [x] 固定输入、模型、speaker/sample selection 和 crossfade 条件。
2. [x] 新增 context-crossfade 生成脚本和测试。
3. [x] 生成五种条件音频并完成 exact-length/finite/clipping QC。
4. [x] 比较 speaker aggregate 与重点样本。
5. [x] 完成回归并记录结论；audio gate 未通过，不运行下游。

## Inputs and model
- Cohort: `/home/wjj/tts-audio/tts-exp/runs/aishell1_mfa_linear_n25_20260813/data_boundary/aishell1_n25_speaker_balanced_cohort.json`
- Resample-poly TTS metadata: `/home/wjj/tts-audio/tts-exp/runs/aishell1_mfa_linear_n25_resample_poly_20260813/tts_meta.json`
- Frozen tokens: `/home/wjj/tts-audio/tts-exp/runs/aishell1_mfa_linear_n25_20260813/mfa/tokens.json`
- MFA-linear reference: `/home/wjj/tts-audio/tts-exp/runs/aishell1_mfa_linear_n25_resample_poly_20260813/mfa_linear/summary.json`
- Model: frozen `bshall/knn-vc` revision `c616845c4e309e24d5927f15adbdf277a3d65358`, WavLM-Large layer 6 (1024-d) + prematched HiFi-GAN.

## Outputs and code
- Output: `/home/wjj/tts-audio/tts-exp/runs/aishell1_mfa_linear_n25_resample_poly_20260814/natural_anchor_context_20260814/`
- Generator: `scripts/run_natural_anchor_context_crossfade.py`
- Tests: `tests/test_natural_anchor_context_crossfade.py`
- Report: `runs/aishell1_mfa_linear_n25_resample_poly_20260814/natural_anchor_context_20260814/report.md`

## Results
- S0901 unconstrained peak/crest `0.3313/12.449`; r2 `0.3369/11.955`; r3 `0.3618/11.917`.
- S0912 unconstrained `0.4991/8.647`; r2 `0.4970/8.429`; silence+r2 `0.4847/8.346`; r3 `0.5875/10.053`.
- S0901/103 未被 r2 稳定修复；S0912/203 r3 peak 从 `0.6226` 增至 `0.6970`。
- r2 常覆盖约一半以上 frames，r3 可达约 60–73%，并不是真正稀疏的 boundary edit；扩大窗口会抹除真实 phone transition。

## Conclusion
冻结 feature-space 的手工 residual 约束路线（alpha、norm/velocity、silence、moving-average boundary、context crossfade）已经显示相同模式：对 S0912 有时略改善，但不能修复 S0901，且较强修改容易恶化。后续应转向可学习的 confidence/gating 或显式分离 content/prosody/speaker 的模型，而不是继续扩大手工平滑网格。

related_to [[MFA-linear 冻结 VC 重合成与下游验证]]