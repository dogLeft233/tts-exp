---
title: AISHELL-1 n100 conditional MFA adapter 2026-08-16
type: experiment
permalink: tts-exp/experiments/aishell-1-n100-conditional-mfa-adapter-2026-08-16
tags:
- aishell1
- mfa-linear
- conditional-adapter
- speaker-disjoint
- experiment
---

# AISHELL-1 n100 conditional MFA adapter 2026-08-16

## Context

在已完成的 AISHELL-1 100 条云端 Qwen TTS、Mandarin MFA 3.4.1 与 MFA-linear 扩展上，验证一个不把 MFA-linear 结果作为输入的双分支特征适配器。目标是让自然音频提供说话人/风格条件，让云端 Qwen 特征提供内容条件，并把 Qwen 内容轨迹映射到自然音频时间轴。

## Model and protocol

- Natural branch：自然音频 WavLM-Large layer 6 特征，承担 speaker/style 条件。
- Content branch：同一配对的云端 Qwen WavLM-Large layer 6 特征，仅做全局线性重采样到 natural frame count；不输入 MFA-linear 结果。
- Target：MFA-linear 生成的 Qwen 特征轨迹，位于 natural clock 上。
- Frozen representation/vocoder：pinned `bshall/knn-vc` revision、WavLM layer 6、1024 维、16 kHz、320-sample frame stride、prematched HiFi-GAN。
- Speaker split：12 个训练 speaker、80 条训练样本；S0914/S0915/S0916 为未见验证 speaker、20 条验证样本。
- 输出：冻结 HiFi-GAN 重合成，严格匹配 natural sample count。

## Feature-level results

Smoke run 使用 4 train + 2 eval、5 steps、16 hidden channels，成功完成 checkpoint 和 2 条渲染：验证 feature loss `0.432147 → 0.430211`，改善 `0.001937`。

完整 run 使用 80/20 speaker-disjoint split、1000 steps、128 hidden channels：

- natural identity baseline held-out feature loss：`0.405501`
- conditional adapter held-out feature loss：`0.384662`
- improvement：`0.020839`
- S0914：`0.439341 → 0.413623`，改善 `0.025718`
- S0915：`0.378648 → 0.362160`，改善 `0.016488`
- S0916：`0.397351 → 0.377127`，改善 `0.020224`
- 20/20 held-out 音频完成渲染；全部 exact natural length、finite、未削波，最大 peak `0.676541`。

## Held-out Wav2Lip/SyncNet result

使用三个未见 speaker 的 20 条样本、同一 fixed face、同一 Wav2Lip/SyncNet checkpoint、`min_track=50`，四臂共 80/80 cell 完成。该结果仍是 fixed-face exploratory，不是 speaker-matched 或真实视频泛化结论。

| arm | mean Sync-C | mean Sync-D |
|---|---:|---:|
| natural raw | 5.8355 | 7.12405 |
| raw Qwen TTS | 6.5186 | 7.12625 |
| MFA-linear | 6.46415 | 7.00620 |
| conditional adapter | 6.00325 | 7.24090 |

- conditional − natural：Sync-C `+0.16775`，cluster bootstrap CI `[-0.54771, +1.03843]`，11/20 改善；Sync-D `+0.11685`（变差方向），CI `[+0.05971, +0.22143]`；joint improvement `7/20`。
- conditional − raw Qwen：Sync-C `-0.51535`，paired p=`0.0375`；Sync-D `+0.11465`；joint improvement `4/20`。
- conditional − MFA-linear：Sync-C `-0.46090`，cluster bootstrap CI `[-0.72550, -0.17029]`；Sync-D `+0.23470`，CI `[+0.15386, +0.34617]`；conditional 在两个方向上都较差，joint improvement `3/20`。
- 三个 speaker 的 conditional − natural Sync-C 均值分别为：S0914 `+1.03843`、S0915 `-0.54771`、S0916 `-0.01333`，说明 feature-level 改善没有稳定转化为下游收益。

## Interpretation and decision

双分支模型成功学习了接近 MFA-linear teacher target 的特征修正，但该改善没有转化为 fixed-face SyncNet 优势。conditional 相对 natural 只有不稳定的 Sync-C 小幅正向，Sync-D 反而变差；相对 MFA-linear 和 raw Qwen 均明显落后。因此当前 checkpoint 不应作为“优于 MFA-linear”的模型，也不应继续仅依据 feature loss 扩大训练或调参。

MFA-linear 是教师生成的伪标签，不是独立的理想 hybrid waveform ground truth；feature loss 改善不能证明 speaker identity、发音或自然度改善。当前最合理的结论是：双分支条件输入链可运行且具有 feature-level 可学习性，但监督目标与下游音频质量之间存在明显不一致，下一版需要加入独立音频/边界/听感约束，而不是简单增加容量。

## Artifacts and tests

- Implementation: `scripts/train_aishell1_n100_conditional_mfa_adapter.py`
- Downstream evaluator: `scripts/eval_aishell1_conditional_adapter_heldout_wav2lip_syncnet.py`
- Unit tests: `tests/test_aishell1_n100_conditional_mfa_adapter.py`
- Evaluator tests: `tests/test_aishell1_conditional_adapter_heldout_wav2lip_syncnet.py`
- Smoke: `runs/aishell1_qwen_mfa_linear_n100_20260816/09_conditional_adapter_smoke/`
- Full checkpoint and summary: `runs/aishell1_qwen_mfa_linear_n100_20260816/10_conditional_adapter_n100/`
- Held-out evaluation: `runs/aishell1_qwen_mfa_linear_n100_20260816/12_conditional_adapter_heldout_wav2lip_syncnet/`
- Checkpoint: `runs/aishell1_qwen_mfa_linear_n100_20260816/10_conditional_adapter_n100/adapter.pt`
- Tests: `6 passed` for the adapter and evaluator together via temporary uv pytest overlay; Conda environment was not modified.

## Observations

- [decision] 采用 natural-style + Qwen-content 双分支，避免把 MFA-linear target 泄漏到输入。
- [result] 三个 speaker-disjoint held-out speaker 均有 feature loss 改善，但 80/80 下游评分没有支持 conditional 优于 MFA-linear。
- [learning] 全局重采样内容输入结合自然特征条件，能学习 teacher trajectory 的修正，但该 proxy target 不足以保证可听或可同步的 waveform 质量。
- [boundary] 当前证据支持 feature-level teacher-distillation 可行，不支持把模型作为下游部署模型或 SyncNet 改善模型。

## Relations

- relates_to [[MFA-linear 冻结 VC 重合成与下游验证]]
- relates_to [[Explicit duration-pause-conditioned acoustic rendering]]
- supersedes [[Train learned phone trajectory adapter]]