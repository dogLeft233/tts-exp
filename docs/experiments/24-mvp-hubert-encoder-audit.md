# 24 — MVP waveform HuBERT 编码器特征审计（samples 51–100）

## 目的与范围

实验 21 已确认 MVP 与严格 original natural cohort 可以进行 waveform 级配对，但 independent Mandarin MFA 和逐样本 MVP/natural SyncNet ledger 仍不可用。本实验先回答一个不依赖 MFA 的问题：

> MVP waveform 重新经过同一个冻结 HuBERT encoder 后，表征更接近 natural 还是 raw TTS？

对象为严格 paired 的 samples 51–100：

- natural：`results/rhythm_style_500/aishell1_test_400/natural/{sid:04d}.wav`
- raw TTS：`results/rhythm_style_500/aishell1_test_400/tts/{sid:04d}.wav`
- MVP：`results/rhythm_style_500/aishell1_test_400/test_controls_best/mvp/conditions/mvp/{sid}.wav`

本实验是 frame-level encoder geometry audit，不是 MFA phone-level 分析，不是 Ditto-native 1024-D frontend 分析，也不把 encoder distance 解释为 lip-sync 因果指标。

## 重要的模型/接口区分

MVP 训练时使用的 `natural_content` 是 768-D HuBERT 内容特征，训练结构为：

```text
natural waveform + natural HuBERT content
+ TTS style/prosody + rhythm features
→ waveform residual
→ MVP waveform
```

因此，训练输入是 HuBERT feature，并不意味着 MVP 输出 waveform 重新经过 HuBERT 后就会进入 TTS representation space。必须对三个 waveform 条件重新运行同一 encoder 才能比较输出表征。

此外，Ditto 使用的是独立的 native streaming HuBERT frontend：1024-D、约 25 fps，并非本实验使用的 HuggingFace HuBERT Base 768-D。该接口差异意味着本结果不能直接替代 Ditto-native feature audit。

## 方法

- encoder：`facebook/hubert-base-ls960`，本地缓存，冻结推理；
- sample：50 个严格 paired samples 51–100；
- layers：0、6、11、12；
- layer 0 代表较早的 acoustic representation，layer 6 代表中间 phonetic representation，layer 11/12 代表较晚层；
- 所有 waveform 转为 16 kHz mono；
- 每个 sample 的 frame sequence 线性归一化到 128 个相对时间点后计算 mean frame-wise cosine distance；
- 同时计算整句 mean-pooled embedding 的 cosine distance；
- 距离越低表示两个 waveform 在该 encoder 的表征几何上越接近。

这一步没有使用 TextGrid、均匀切分、character boundary 或 source-TTS alignment。

## 结果一：时间归一化 frame-level distance

| HuBERT layer | natural–MVP | natural–TTS | MVP–TTS |
|---|---:|---:|---:|
| 0 | 0.1227 | 0.7124 | 0.7168 |
| 6 | 0.1388 | 0.7425 | 0.7530 |
| 11 | 0.1182 | 0.7524 | 0.7612 |
| 12 | 0.0938 | 0.7193 | 0.7352 |

MVP 与 natural 的 frame-level distance 明显低于 natural 与 TTS，也低于 MVP 与 TTS。换言之，MVP 的时序表征轨迹仍然贴近 natural，而没有表现为 TTS-like trajectory。

按“谁更接近 natural”统计，MVP 在四层均为 50/50 比 raw TTS 更接近 natural。按“谁更接近 TTS”统计，MVP 只在 19/50（layer 0）、9/50（layer 6）、12/50（layer 11）、4/50（layer 12）个 sample 上优于 natural；均值上不存在稳定的 TTS-side shift。

## 结果二：整句 pooled embedding distance

| HuBERT layer | natural–MVP | natural–TTS | MVP–TTS | MVP 更接近 natural |
|---|---:|---:|---:|---:|
| 0 | 0.0671 | 0.1049 | 0.1656 | 37/50（74%）|
| 6 | 0.0469 | 0.0771 | 0.1273 | 41/50（82%）|
| 11 | 0.0519 | 0.0963 | 0.1520 | 47/50（94%）|
| 12 | 0.0393 | 0.0711 | 0.1180 | 43/50（86%）|

整句 pooling 的结论与 frame-level 结论一致，且在 HuBERT layer 11 最明显：47/50 个样本中，MVP 比 raw TTS 更接近 natural。

## 与已有 natural/TTS 表征实验的关系

此前 natural-vs-TTS 实验已经显示，TTS 在多个 HuBERT/XLS-R 层的 phone/viseme 表征上更紧凑或更易迁移识别。例如：

- AISHELL-100 HuBERT viseme probe：
  - layer 0：natural `0.849`，TTS `0.863`；
  - layer 6：natural `0.939`，TTS `0.953`；
  - layer 11：natural `0.932`，TTS `0.948`。
- MDC English 50 对中，TTS 在多个层的 phone/viseme probe、boundary 和 segment stability 指标整体更强，但这些仍是 representation-level 证据。

本实验尚未对 MVP 做 MFA phone-conditioned probe，因此不能声称 MVP 的 phone separability、boundary sharpness 或 phone PER 高于或低于 natural/TTS。当前能支持的结论仅是：MVP waveform 的整体 HuBERT geometry 没有复现 raw TTS 的表征偏移。

## 与 downstream SyncNet 的关系

同一 MVP run 的历史 paired 视频汇总为：

| 条件 | Mean Sync-C | Mean Sync-D | Mean AV offset |
|---|---:|---:|---:|
| natural | 5.04526 | 7.90412 | 0.700 |
| MVP | 4.16242 | 7.93172 | 0.080 |
| MVP − natural | −0.88284 | +0.02760 | −0.620 |

因此，“MVP 更接近 natural 的 HuggingFace HuBERT geometry”并没有带来更高的 Sync-C。这个结果不矛盾：

1. HuBERT 768-D frame geometry 不是 Ditto 的 native 1024-D frontend；
2. encoder distance 是描述性指标，不等于 phone/viseme 可分性；
3. waveform 可能在 Ditto frontend 的局部时间结构、音频-运动映射或 evaluator interaction 上出现问题；
4. 既有实验 15 也显示，单个表征/韵律指标不能在 FDR 后预测逐样本 Sync-C 变化。

## 对 MVP 失败机制的当前解释

当前证据更支持以下谨慎表述：

> MVP 没有把 natural waveform 的 HuBERT 表征整体推向 TTS；它主要保留了 natural-side encoder geometry，同时在 waveform/STFT/mel 等低层目标上产生变化。这个变化足以让 MVP 的 downstream Sync-C 低于 natural，但目前无法定位是 HuggingFace HuBERT 表征、Ditto-native frontend、phone timing、局部 prosody 还是音频-视频管线交互导致。

不能据此声称：

- MVP 的 phonetic content 一定损坏；
- MVP 的 HuBERT 表征一定比 TTS 差；
- HuBERT distance 是 Sync-C 下降的原因；
- MVP 已经完成了 TTS feature transfer；
- natural-like encoder geometry 对 Ditto 一定更优。

## 下一步

优先对同一 51–100 cohort 提取 Ditto native `hubert_streaming_fix_kv.onnx` 的 1024-D、25 fps 特征，并比较：

1. natural/MVP/raw TTS 的 native feature distance；
2. phone-conditioned centroid distance（需要可用 Mandarin MFA）；
3. frame smoothness、boundary sharpness 和 voiced-region coverage；
4. native feature delta 与逐样本 Sync-C decline 的关联（需要恢复 paired score ledger）。

在上述两个 blocker 解决前，不应把本 audit 扩写成 phone-level failure diagnosis 或 causal claim。

## 相关文件

- MVP 训练结构：`scripts/rhythm_style_generator.py`
- MVP 数据构建：`scripts/prepare_rhythm_style_dataset.py`
- MVP 训练日志：`results/rhythm_style_500/aishell1_test_400/training_runs/full_mvp/train.log`
- MVP 局部音频指标：`results/rhythm_style_500/aishell1_test_400/test_controls_best/local_audio_metrics.json`
- HuBERT/XLS-R 提取脚本：`scripts/15_extract_ssl_embeddings.py`
- HuBERT 审计临时汇总：`/tmp/mvp_hubert_encoder_audit_51_100.json`
- natural/TTS 表征基线：`docs/experiments/09-aishell100-phoneme-extension.md`
- TFG 关联边界：`docs/experiments/15-tfg-link.md`
- MVP waveform forensics：`docs/experiments/21-mvp-audio-forensics.md`
- Ditto native interface：`docs/experiments/23-ditto-native-feature-head-feasibility.md`
