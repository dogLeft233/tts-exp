---
title: LRS3 English TTS Phoneme Separability Transfer n100 2026-09-21
type: experiment
permalink: tts-exp/experiments/lrs3-english-tts-phoneme-separability-transfer-n100-2026-09-21
status: concluded
date: '2026-09-21'
tags:
- lrs3
- english
- tts
- phoneme-separability
- condition-transfer
---

# LRS3 English TTS Phoneme Separability Transfer n100 2026-09-21

本实验把既有 AISHELL-1 中文 n=100 音素可区分度分析迁移到同一批 100 条 LRS3 English 配对语音，用于比较自然音频、既有云端 Qwen TTS、本地 Qwen TTS、IndexTTS-2 和 SiliconFlow CosyVoice2-0.5B。迁移的对象是指标定义、配对统计和结果组织方式，不是中文实验的数值、最佳 SSL 层、音素规则或 MFA 边界。英文每个音频条件都必须使用独立的英语 MFA 对齐，不能把自然音频或旧 cloud-Qwen 的 TextGrid 复制给新 TTS。

## Observations
- [status] planned；实验 cohort 固定为现有 LRS3 配对数据的前 100 条，五个条件使用相同 sample_id 和 transcript。
- [reuse] 推荐复用既有 frame→phone pooling、intra/inter/Fisher/silhouette、分组线性 probe、boundary sharpness、segment stability、duration JSD、per-phone KLD/AUC 及 paired permutation/bootstrap/BH-FDR 的实现与报告结构。
- [transfer_guard] 英文不剥离中文声调；保留英语 MFA phone label，并固定英文 phone/viseme 映射与归并规则。
- [transfer_guard] 自然、cloud-Qwen、local-Qwen、IndexTTS-2、CosyVoice2 各自使用自己的 MFA TextGrid；只有在音频 hash 与当前 cohort 一致时才复用既有 natural/cloud 对齐产物。
- [transfer_guard] HuBERT/XLS-R 层位只作为候选层；中文 AISHELL-1 的最佳层不能直接当作 LRS3 English 的事实，需在英文 cohort 上重新记录 layer curve。
- [protocol] 所有 TTS 与 natural 的主比较使用 paired key，并报告 TTS−natural 方向、bootstrap CI、paired permutation p 值及 BH-FDR；self-LOO 与 cross-condition phone recognition 分开命名，不能混写。
- [protocol] per-phone KLD/AUC 仅使用两臂均达到最小支持数的 phone，并按配对 utterance 内 phone occurrence 顺序匹配；每个条件的 MFA span 独立计算。
- [boundary] 不能因为生成音频较短或 MFA coverage 不足而退回均匀切分；失败样本、phone support 和 coverage 需要单独统计并保留。
- [limitation] LRS3 当前 manifest 没有可直接用于 speaker-disentangled 推断的完整 speaker_id；paired sample 是主要阻断混淆手段，不能把结果表述为 speaker-controlled。
- [limitation] 该实验只回答语音表征/音素可区分度差异，不能直接推出 SyncNet 或 TFG 因果收益；若后续联结视频分数，需另做按样本的关联分析。
- [artifact] 计划运行目录为 `runs/lrs3_english_phoneme_transfer_n100_20260921/`，按 inputs、MFA、embedding、metrics、report 分阶段保存 provenance、hash 和失败清单。

## Relations
- extends [[AISHELL-1 n100 phoneme extension]]
- relates_to [[Per-phoneme KLD and grouped AUC]]
- relates_to [[MDC English representation audit]]
- relates_to [[LRS3 phone rules implementation]]


## Results

- [status] concluded；100 条 LRS3 English 五臂输入均已生成并绑定到同一 sample_id/transcript。natural 与 cloud-Qwen 复用前序资产前做了音频 SHA-256 校验；local-Qwen、IndexTTS-2、CosyVoice2 使用各自独立的 English MFA。
- [alignment] MFA 使用 `english_us_mfa + english_mfa`，统一高 beam 配置 `beam=100, retry_beam=400`。对齐覆盖 natural=100、qwen_cloud=100、qwen_local=100、index_tts2=100、cosyvoice2=97，共 497/500 个 audio arms。CosyVoice2 缺失的 3 条音频只有 3.80/5.76/4.24 秒，却对应 61/72/77 个词；高 beam 仍无法对齐，未使用均匀 span fallback。
- [feature_reuse] HuBERT/XLS-R pooled feature 提取完成；旧 cache 仅在 hash 可核验时复用：natural 38 条、qwen_cloud 34 条，其余为 fresh frozen SSL extraction。最终比较含 192 个 paired tests（2 model families、候选层、phoneme/viseme、4 个 TTS conditions、6 个核心指标），2,000 次 sign-flip permutation、5,000 次 paired bootstrap 和全局 BH-FDR。
- [metric_scope] 保留 intra/inter-class distance、Fisher ratio、cosine silhouette、boundary sharpness、segment stability、duration JSD、HuBERT layer-6 per-phone symmetric KLD/AUC。历史 768 维全 token logistic probe 因计算代价过高，本次未运行；JSON 保留兼容字段并显式标记 `linear_probe.status=not_run`。
- [result] HuBERT layer 6 phoneme pooled：qwen_cloud 的 Fisher=27.9642、IndexTTS-2=25.4706，高于 natural=20.2741；qwen_local=22.8273，CosyVoice2=14.5554。相对 natural 的配对结果中，qwen_cloud 与 IndexTTS-2 的 inter/Fisher/silhouette 整体为正，CosyVoice2 的 inter-class 和 silhouette 为负；qwen_local 变化较小，方向混合。
- [per_phone] HuBERT layer 6 平均 symmetric KLD/AUC：qwen_cloud 1.6098/0.6249，qwen_local 1.0636/0.5504，IndexTTS-2 0.9169/0.5677，CosyVoice2 1.7690/0.6062。支持数不足的 phone 显式排除；每个 TTS arm 使用自己的 MFA span，并在 paired utterance 内按同标签 occurrence 顺序匹配。
- [duration] rate-normalized duration mean JSD：qwen_cloud 0.0225、qwen_local 0.0199、IndexTTS-2 0.0195、CosyVoice2 0.0202；这些是英文 LRS3 cohort 内结果，不能与 AISHELL-1 中文数值直接比较。
- [claim_boundary] 本实验只给出 LRS3 English 的语音表征/音素可区分度关联，不推出 Sync-C 或 TFG 因果；也不作 speaker-controlled 结论。中文实验的最佳层、方向和绝对数值均未迁移。

## Artifacts

- 主报告：`runs/lrs3_english_phoneme_transfer_n100_20260921/03_metrics/report.md`
- 全量指标：`runs/lrs3_english_phoneme_transfer_n100_20260921/03_metrics/metrics.json`
- MFA 审计：`runs/lrs3_english_phoneme_transfer_n100_20260921/01_mfa/alignment_manifest.json`
- 特征审计：`runs/lrs3_english_phoneme_transfer_n100_20260921/02_embeddings/summary.json`
- 协议与 hash：`runs/lrs3_english_phoneme_transfer_n100_20260921/protocol.json`
- runner/test：`scripts/experiments/lrs3_english_phoneme_transfer.py`、`tests/test_lrs3_english_phoneme_transfer.py`

## Changelog

- 2026-09-21：完成 LRS3 English n=100 条件迁移；记录独立 MFA、497/500 对齐覆盖、hash-verified cache reuse、核心指标与限制。