---
title: Ralph Step 1 - Controlled Baseline
type: experiment
permalink: tts-exp/research/ralph-step-1-controlled-baseline
---

# Ralph Step 1 - Controlled Baseline

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| Initial Document | August 12, 2026 | user |
| Restructured to Research/ with schema observations | August 12, 2026 | user |

本步建立了后续探索的可控基线,没有启动大规模训练,也没有触碰 heldout 数据。

## 环境

- Python: `.venv/bin/python`
- PyTorch: `2.5.1+cu121`
- GPU: Tesla V100-SXM2 16GB
- 基线执行时 GPU memory used 约 322 MiB,GPU utilization 0%,没有发现资源压力。

## 验证命令

```text
uv run --with pytest --python .venv/bin/python -- python -m pytest -q \
  tests/test_hubert_feature_alignment.py \
  tests/test_train_feature_targeted_waveform_renderer.py \
  tests/test_direct_waveform_hubert_upper_bound.py
```

结果:`14 passed in 4.39s`。

## Preflight

- contract: `direct_raw_tts_features_train_valid_only`
- train: 290 pairs
- valid: 50 pairs
- heldout_loaded: `False`
- train speakers: `S0901,S0906,S0912,S0913,S0914,S0915`
- valid speaker: `S0765`
- 没有把 heldout 用于 checkpoint、参数或 alignment 选择。

## K=1 direct waveform baseline

样本:`aishell1_test_400__BAC009S0901W0122`

- natural frame count: 416
- matched target frames: 232
- mask coverage: 0.557692
- natural → aligned raw-TTS L6 combined gap: `0.3813196588`
- exact-length phone-local waveform warp gap: `0.6508054212`
- phone warp output sample count: 133328,exact sample count 为 true

direct waveform optimization(100 steps):

| mode | residual scale | initial gap | final gap | gap reduction |
|---|---:|---:|---:|---:|
| bounded | 0.05 | 0.3813 | 0.0394 | 89.7% |
| bounded | 0.20 | 0.3813 | 0.0335 | 91.2% |
| unbounded | 1.0 | 0.3813 | 0.0438 | 88.5% |

bound=0.20 的最终 residual peak 约 0.1280、无 clipping。

## 结论

1. 当前 alignment、HuBERT gradient 和 direct upper-bound 路径可稳定复现。
2. K=1 上 aligned L6 target 具有较强 waveform feature 可达性,不能首先归因于 target 完全不可达。
3. exact-length phone-local waveform warp 比 natural baseline 更差,说明 naïve waveform warp 不是可靠 ground truth;可能包含 phase、transient、boundary splice 和 contextual SSL geometry 问题。
4. 后续实验应先替换 target geometry(phone pooling/duration、constrained hard DTW),不要先扩大数据或盲目加大 renderer。
5. 资源策略:保持单样本、短步骤、GPU memory 监控;暂不安装 Wav2Lip 或运行 SyncNet,直到 feature target/alignment 对照完成。

## 下一步

在同一 K=1 样本和同一 frozen HuBERT 上,增加 phone-level pooling/duration target 与 phone-anchored hard DTW 的离线对照,先只比较 target gap 和 direct waveform reachability。

## Observations
- [status] concluded
- [result] K=1 direct upper bound: bound=0.20 达 gap 0.0335 (91.2% reduction)
- [conclusion] aligned L6 target 可达;naïve phone-local warp 不可靠;先替换 target geometry 再考虑扩大模型
- [report] 记忆笔记, 研究步骤报告

## Relations
- part_of [[TTS Feature Alignment and Duration Model Research]]
- relates_to [[Ralph Step 2 - Phone Target Geometry Screening]]
