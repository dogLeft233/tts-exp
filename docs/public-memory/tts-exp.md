---
title: tts-exp
type: project
permalink: tts-exp
---

# tts-exp 项目

TTS-TFG 实验项目：利用“TTS 特征音频有时比原始自然音频更能驱动 TFG 生成高同步视频”的现象，设计一个可跨 TFG 模型即插即用的音频增强头。增强头输出只用于驱动视频生成；最终视频必须换回 original natural audio，并且 replacement 后仍显著优于直接用 natural audio 驱动同一 TFG 的 baseline。

- 仓库: [redacted-local-path] (git)
- 入口文档: CONTEXT.md (项目状态, ≤100 行), HANDOFF.md (当前快照), AGENTS.md (agent 配置)
- 实验报告: basic-memory/docs/experiments/NN-<slug>.md；实验记忆: basic-memory/Experiments/；当前索引见 basic-memory/docs/experiments/README.md
- 核心现象: 中文 AISHELL-1 n=100 的 Ditto ΔSync-C≈+1.02；LRS3 英文队列也有优势，LibriSpeech/HDTF 旧队列未复现。效应依赖语料、生成器及评价协议；Sync-C 增益本身不证明视觉口型更准确。
- 流水线: scripts/00_datacheck → 01_asr → 02_tts → 03_ditto → 04_eval → 05_report (run_all.sh)
- TTS 引擎: faster_qwen3 (0.6B ICL 自克隆); TFG 模型: Ditto 等 9 个
- 评估: SyncNet V2, Sync-C (越高越好) / Sync-D (越低越好) / AV offset
- 运行环境: ditto conda (步骤 00–03, 05); syncnet 独立 env (04_eval)

## 最终目的与成功判据

### 目标链路

对同一原始自然音频 `N`、同一人脸/源视频 `F` 和同一冻结 TFG 模型 `M`：

```text
C = audio_enhancement_head(N, optional TTS conditioning)
V_C = M(F, C)
Final = mux(V_C, N)
```

增强头输出 `C` 应保留或引入有利于 TFG 的 TTS 特性，只作为视频生成控制信号；交付视频的最终音轨仍必须是 original natural audio `N`。

### 唯一权威比较

最终结果必须与直接 natural-driver baseline 做同条件配对比较：

```text
Baseline = mux(M(F, N), N)
Target   = mux(M(F, C), N)
```

成功要求 `Target` 在 official file-level SyncNet 中相对 `Baseline` 保留真实增强，而不是仅做到不退化。优先要求 Sync-C 提升、Sync-D 降低，并在 held-out clips/source groups 上稳定成立。

以下结果不能单独算成功：

- `M(F, C) + C` 自洽分数较高；
- candidate audio 与 natural/TTS 的特征距离改善；
- 固定真实视频上的 proxy/SyncNet loss 改善；
- natural driver、`-natural` 或 identity replacement 通过；
- 仅 differentiable proxy 提升而 official replacement 不提升。

### 产品化目标：即插即用音频增强头

最终希望增强头位于标准 waveform/audio-feature 输入与 TFG 之间，而不是绑定某个 TFG 的内部网络：

```text
original audio → enhancement head → enhanced driver audio → frozen TFG
```

理想契约：

- TFG 模型保持冻结，不要求修改或重新训练其主体；
- 输出为标准音频接口，能够接入 Wav2Lip、Ditto 及更多 speech-driven TFG；
- 尽量不依赖某一 TFG 的私有梯度、hidden states 或专用 crop/proxy；
- 保持 transcript/content、精确长度和可替换性；
- 在多个 TFG、数据域和 held-out source groups 上验证，而不是只对单模型/单 cohort 过拟合。

可以使用某个 TFG/SyncNet 做研发期监督或筛选，但最终主张必须来自跨模型 official replacement audit。若方法只能对单一 TFG 通过，应标记为 model-specific adapter，不能称为通用即插即用增强头。

### 当前研究含义

natural driver 和 `-natural` 只能证明 identity/feature-invariance，不实现“TTS 特性带来额外视频增益”。VC、MFA、DTW、cross-attention、CEM 或其他方法都只是候选机制；是否保留取决于它们能否通过上述 `Target > Baseline` 的最终链路，而不是方法本身是否看起来合理。

## 本地环境 (2026-08-11 搭建)

- `.venv/` (gitignored): torch 2.5.1+cu121 (Tesla V100 16GB), transformers 4.51.3, numpy/soundfile/scipy, paramiko
- HuBERT: facebook/hubert-base-ls960 已缓存 (~/.cache/huggingface), 离线可用 (HF_HUB_OFFLINE=1)
- 注意: 环境有 socks 代理 (ALL_PROXY=socks://[redacted-ip]:7890), httpx/transformers 不兼容 → 跑模型前取消代理或用 HF_HUB_OFFLINE=1
- 注意: transformers 5.x 强制 torch≥2.6, 而 cu121 最高 torch 2.5.1 → 用 transformers 4.x
- SSH: scripts/sshr.py (paramiko), 凭据走环境变量 TTS_SSH_PASS, 不落文件

## 增强头现状 (2026-08-11)

- 29 号两阶段模型: HuBERT L6 特征上 enhanced 比 natural 接近 TTS 1.38% (50/50 closer), 但身份漂移极小
- 下游评估 (Ditto+SyncNet, 50 valid): ΔSync-C = +0.062, **无下游改善** (负结果)
- 特征空间移动不能穿透 Ditto 原生 1024-D frontend

## Observations

- [objective] 利用 TTS 特性音频对 TFG 的驱动优势生成更好的视频，同时最终换回 original natural audio 后仍优于 natural-driver baseline。 #primary-objective
- [baseline] 唯一主比较是 `mux(TFG(face, enhanced_driver), natural)` 对 `mux(TFG(face, natural), natural)` 的同条件 paired official audit。
- [requirement] replacement 后必须保留正增益，不只是“不下降”，也不能用 candidate-video/candidate-audio 自洽分数代替。
- [product] 最终产物是跨 TFG 模型的即插即用音频增强头，以标准音频接口连接冻结 TFG。
- [generalization] 通用增强头必须在多个 TFG、数据域及 held-out source groups 上成立；单模型成功只能称 model-specific adapter。
- [non-goal] natural driver、`-natural` 和 identity replacement 只是控制实验，不实现 TTS 特性所带来的额外增强。

## Relations
- relates_to [[Startup Router]]
- relates_to [[增强头下游评估 (Ditto + SyncNet)]]
- relates_to [[TTS Feature Alignment and Duration Model Research]]
- relates_to [[Basic Memory 技能与连接验证]]
- relates_to [[TTS 增强 TFG 的原因：证据、排除项与因果分叉]]

- relates_to [[跨数据集 TFG 测评（5×50 multiset）]]
- relates_to [[论文实验数据整理与写作主线]]
