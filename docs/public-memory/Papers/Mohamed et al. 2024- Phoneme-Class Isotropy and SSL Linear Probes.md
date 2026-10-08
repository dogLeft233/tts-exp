---
title: 'Mohamed et al. 2024: Phoneme-Class Isotropy and SSL Linear Probes'
type: paper
permalink: tts-exp/papers/mohamed-et-al.-2024-phoneme-class-isotropy-and-ssl-linear-probes
---

# Mohamed et al. 2024: Phoneme-Class Isotropy / SSL Geometry and Linear Probes

Interspeech 2024, arXiv 2406.09200。音素保真度分析的核心几何框架,直接支撑项目的 intra/inter-class dist、fisher ratio、silhouette + 线性探针指标组合。

音素质心各向同性(Ph\Ph)是 phone 线性探针准确率的最强几何预测:pooled Spearman ρ=0.94,六模型逐模型 0.69–0.90,全部 p<0.05。说话人/音素子空间正交性(Ph\Spk)与 probe 显著相关(Transformer 族 ρ=0.54–0.78,pooled 0.54)。

## Observations
- [status] cited
- [authors] Mohamed et al.
- [url] https://arxiv.org/abs/2406.09200
- [key_insights] 项目 SSL 几何指标组合(Ph\Ph、Ph\Spk、fisher ratio、silhouette + GroupKFold 探针)已被文献验证
- [key_insights] TTS-vs-natural 比较应控制说话人/音素解纠缠(B5 解纠缠实验的依据)

## Relations
- part_of [[TTS Feature Alignment and Duration Model Research]]
