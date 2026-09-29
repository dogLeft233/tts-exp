## Why

英文 TTS 也有明确的下游优势。2026-09-15 根据 multiset manifest 和原始逐条评分重算，LRS3 50 对的原生 ΔSync-C：Ditto +1.124（45/50 正向），LeapTalk +1.389（46/50 正向）。此前 LibriSpeech/HDTF 的阴性结果不能概括为“英文无优势”。原生增益、相对弱基线的 TTS conditioning 增益、严格 natural replacement 增益必须分开。

本轮研究问题是：**这些已观察到的 Sync-C 优势，在评分计算上来自最佳匹配距离下降，还是来自错位距离背景上升；去掉边界 padding、限制全局 offset 搜索后，优势如何变化？** 这是机制研究的第一步，不预设音质、语言、语速或评估器偏好是原因。

## What Changes

- CPU 复用两个模型各 50 对评分，精确分解 C 增益，按真实 source group 推断。
- 固定 12 个不同来源的 LRS3 记录，只对已有 LeapTalk 视频补算完整 SyncNet 距离矩阵。24 个主 cell、2 个重复、2 个延迟控制，最多 28 个新评分 cell，零新 TTS、零新 TFG。
- 分别报告历史官方全支持、去 padding 的内部支持、等窗口数量敏感性；不把重评分当成历史同环境复现。
- 生成可复算表格、距离曲线图和匿名人工评估包。人工尚未返回时，自动分析完整交付，感知结论明确 unavailable。

## Capabilities

### New Capabilities

- `lrs3-tts-gain-mechanism`: LRS3 原生 TTS 增益的分数分解、完整曲线诊断与有界评估控制。

### Modified Capabilities

无。历史实验结论和 replacement 门槛不改写。

## Impact

下游实现 `scripts/experiments/lrs3_tts_gain_mechanism/` 和对应测试；输出 `runs/lrs3_tts_gain_mechanism_<run_id>/`。入口、固定输入和全部计算契约见 README、input-bindings.json、design.md。本次交付 spec 和输入绑定，不执行模型。

## Non-goals

不训练增强头、不重新跑多模型四格矩阵、不扫描声学特征、不做 MFA/DTW 或新波形干预；不将分数代数分解称作声学因果分解，也不把固定 lag 距离称作真实对齐误差。后续声学干预应由本轮定位出的现象单独设计。
