---
title: 006-portable-waveform-enhancement-head
type: decision
permalink: tts-exp/docs/adr/006-portable-waveform-enhancement-head
repository: dogLeft233/tts-exp
project: tts-exp
decided: '2026-10-03T02:18:59.5639185+00:00'
status: open
---

# 006-portable-waveform-enhancement-head

## Context

用户在机制研究结案后明确提出新的交付目标：适配较多TFG的自然音频增强头，增强音生成视频再换原音，并持续迭代形成完备方法。报告25已定位增益正向路径和局部谱形相容性；单TFG latent干预、原生TTS自身评分与直接供体移植不能替代此目标。

## Choice and rationale

以模型外部的等长波形h为默认产品接口，最终使用原自然PCM。16k mono同samplecount无需依赖特定TFG隐层，能经不同mel/Hubert/Whisper前端接入。先复现真实RMS驱动阳性对照，再学习幅度受限、时序保持的局部谱残差，并用多模型实际生成及未优化模型确认选择。共享head和backend校准版分别报告，保留使用者对接成本。

## Alternatives and consequences

特定Wav2Lip latent/mel adapter可作为训练代理，但其接口不覆盖其它TFG；直接用TTS整句或时间warp会改变自然音配对时钟，已有replacement失败提示其相容性成本；仅提高原生TTS分数不满足原音换轨目标。它们可用于对照或后续分支，不充当默认交付。

新Task与Experiment管理未完成研究，原DONE任务和报告25保持结案。最终验收必须含真实成片/原PCM mux/fresh评分、speaker-disjoint数据、至少3模型及未优化模型、自然/RMS基线、消融/失败/适用边界及可使用代码权重。模型前端可能消除全局gain，因此预设局部残差及模型校准分支，按失败原因迭代。

## Observations

- [decision] 新交付采用TFG外部等长波形接口与固定原自然音最终评分。
- [rationale] 共同波形入口能覆盖不同音频前端，增益阳性基准与局部相容性给出可检验的改进路线。
- [alternative] 特定隐层adapter作为训练代理，整句TTS替换和时间warp作为对照，不替代主要波形接口。
- [consequence] 需要新的split/多模型真实render-mux-score和产品封装，不能用已有单模型缓存正向结果宣布交付完成。

## Relations

- affects [[开发跨 TFG 的自然音频增强头]]
- affects [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- builds_on [[25-tts-tfg-mechanism]]
- relates_to [[005-separate-score-motion-and-native-estimands]]