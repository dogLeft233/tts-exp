# 21 — MVP waveform 失败诊断与 provenance 审查（AISHELL-1 samples 51–100）

## 目的与范围

实验 16–18 已显示：raw TTS 的 Sync-C 优势不能由简单 phone-local waveform warp、duration interpolation、响度、能量或长时 spectral control 单独复现。本实验进一步审查当前 rhythm-style MVP waveform，目标是定位其相对 natural 音频表现下降的可能原因。

本报告记录的是诊断阶段结果，不把观察性声学差异写成因果结论。严格 paired 音频对象为 MVP 生成时 sidecar 所记录的 original natural cohort：`results/rhythm_style_500/aishell1_test_400/natural/`，而不是当前 `data/aishell1_100_zh/audio/`。历史 MVP/natural 视频的 SyncNet 汇总只作为背景；在本地缺少逐样本 ledger 时不重新推断相关性。

## 已知视频评测背景

此前固定同一批 samples 51–100、人脸循环映射、Ditto seed=42、TRT online 和 SyncNet 协议完成了 paired 评测：

| 指标 | natural | MVP | MVP − natural |
|---|---:|---:|---:|
| Mean Sync-C | 5.04526 | 4.16242 | −0.88284 |
| Mean Sync-D | 7.90412 | 7.93172 | +0.02760 |
| Mean AV offset | 0.700 | 0.080 | −0.620 |

配对覆盖为 50/50。MVP 的 AV offset 反而更好，但 Sync-C 下降，因此不能把失败简单解释为整体音视频延迟增加。Sync-C 下降更接近嘴型可辨识度、音频内容表达、局部韵律或声学表征与 Ditto/SyncNet 的交互问题。

## 诊断实现

新增脚本：

```text
scripts/26_mvp_natural_audio_forensics.py
```

脚本提供：

1. 固定 sample ID 集合并拒绝缺失、重复或额外输入；
2. 逐文件记录 SHA-256、native sample rate、channel、sample count、duration、subtype、peak、finite 和 clipping；
3. 复用 `scripts/08_audio_features_v2.py::extract_features_v2`，以 16 kHz mono 提取 duration、RMS、peak、LUFS、LRA、energy envelope、silence、F0、spectral、onset、formant、HF/LF ratio 和 MFCC 特征；
4. 读取 MVP 每样本 JSON sidecar，验证 sample ID、输出文件、采样率、sample count、finite、clipping 和 exact-length metadata；
5. 接收显式 SyncNet score ledger，不自动扫描或替代其他历史 run；
6. 支持 natural/MVP TextGrid 的 phone duration、pause 和 speech-span 指标，但不把 source-TTS TextGrid 当作 MVP alignment；
7. 支持逐样本 feature delta、Spearman 关联和 BH-FDR；
8. 在 score ledger 缺失时显式输出 unavailable，而不是生成伪相关性。

测试文件：

```text
tests/test_mvp_natural_audio_forensics.py
```

回归测试共 14 个，覆盖 ID 集合、duplicate/missing、score delta 方向、汇总复现、WAV metadata、TextGrid pause merge、确定性统计、feature extraction、严格 sidecar provenance、named-tier parser、MFA staging/probe、phone sequence mismatch、Sync-C decline 和 local prosody smoke。测试全部通过；librosa/audioread 仅报告 Python 3.13 的 `aifc`/`sunau` deprecation warnings。

## 严格 51–100 partial scan 结果

完整 samples 51–100 partial run 使用以下严格输入：

- natural：`results/rhythm_style_500/aishell1_test_400/natural/{sid:04d}.wav`
- MVP：`results/rhythm_style_500/aishell1_test_400/test_controls_best/mvp/conditions/mvp/{sid}.wav`
- transcript：`results/rhythm_style_500/aishell1_test_400/transcripts/{sid:04d}.txt`
- natural MFA：`results/rhythm_style_500/aishell1_test_400/mfa/natural/{sid:04d}.TextGrid`

结果为：

| 阶段 | 状态 | 覆盖/说明 |
|---|---|---|
| corpus/provenance | complete | 50/50 transcript 与 MVP sidecar 对应的 natural sample count/hash 通过 |
| natural named-tier alignment | complete | 50/50 natural TextGrid 通过 words/phones、边界和 transcript gate |
| MVP independent MFA | unavailable/blocked | 本机 MFA 启动失败：`ModuleNotFoundError: No module named '_kalpy'`；当前可用环境也未提供 Mandarin dictionary/acoustic model |
| phone/pause/prosody comparison | unavailable | 没有 MVP TextGrid，因此不生成 positional deltas |
| per-sample SyncNet ledger | unavailable | 未找到对应 natural-vs-MVP ledger；历史均值不展开为 sample-level 数据 |
| correlations | unavailable | 没有 paired alignment features 和 score ledger，未计算相关性 |
| overall | partial | 仅 global waveform 与 natural-side alignment 可用 |

partial 输出保存在 `/tmp/mvp_forensics_51_100/`，包括 50 条 natural-side `phone_intervals.csv`、`pause_intervals.csv` 和 `prosody_intervals.csv`，但所有 paired delta 文件与 correlations 为空。这是 fail-closed 行为，不是“没有差异”或“没有关联”的结论。

## 本地声学扫描结果

本地 global waveform 扫描成功处理严格 paired natural/MVP 各 50 条 waveform，并生成：

```text
results/rhythm_style_500/aishell1_test_400/forensics/mvp_vs_natural_51_100/
```

包括：

- `paired_audio_features.csv`
- `paired_alignment_features.csv`
- `paired_scores.csv`
- `correlations.csv`
- `summary.json`
- `config_and_provenance.json`
- `report.md`

当前 score ledger 不可用，因此 `paired_scores.csv` 和 `correlations.csv` 为空是有意行为。远程 SSH 在本轮尝试时返回 `Connection refused`，没有从其他 run 猜测或复制结果。

在**严格 original cohort 已通过 provenance 配对审查**的前提下，全局声学差异可作为 paired diagnostic description，但不能在缺少 MVP MFA 和逐样本 score ledger 时判断这些差异是否与 Sync-C 逐样本下降相关。

| 特征差值 MVP − natural | mean | median |
|---|---:|---:|
| duration | +0.7165 s | +0.5585 s |
| LUFS | −8.4602 dB | −7.5050 dB |
| RMS | −0.0163 | −0.0101 |
| peak | −0.1646 | −0.1058 |
| F0 mean | +103.2 Hz | +104.4 Hz |
| F0 std | +18.3 Hz | +16.6 Hz |
| F0 range | +98.8 Hz | +97.9 Hz |
| energy envelope std | −0.0106 | −0.0054 |
| spectral centroid | −143.1 Hz | −149.0 Hz |
| spectral bandwidth | −104.0 Hz | −96.0 Hz |
| spectral flux | −0.0654 | −0.0441 |
| silence ratio | 0.0000 | 0.0000 |

这些结果说明 MVP waveform 的幅度、响度、F0、频谱和时长与严格 paired original natural 文件存在明显差异，但不能在缺少 MVP phone alignment 和逐样本 score ledger 时判断这些差异是否与 Sync-C 逐样本下降相关。

## 旧 natural 目录的 provenance 误配与修正

早期扫描曾把 `data/aishell1_100_zh/audio/` 当作 natural cohort；sample 51 的 sample count 为 113859，而 MVP sidecar 声明的生成输入为 75232，因此该早期比较被拒绝，不能用于解释 acoustic delta。随后定位到正确的 MVP-generation cohort：`results/rhythm_style_500/aishell1_test_400/natural/`。samples 51–100 的 sample count 与 SHA-256 均与 sidecar 一致，且 transcripts 与 natural named-tier TextGrid 通过 strict gate。当前结论基于修正后的 cohort；旧目录 mismatch 不再是当前 blocker。

MVP sidecar 中的旧目录误配示例为：

```text
sidecar natural_sample_count = 75232
old data/aishell1_100_zh/audio/51.wav = 113859 samples
```

将旧目录作为输入时，脚本会严格拒绝：

```text
ERROR: MVP sidecar natural provenance mismatch for sample 51:
sidecar=75232 supplied=113859
```

该拒绝验证了 provenance gate 的 fail-closed 行为；它不适用于已通过校验的 `results/rhythm_style_500/aishell1_test_400/natural/` cohort。MVP sidecar 的 `exact_natural_length=true` 只证明 MVP 当时匹配了它所使用的 natural waveform。

```text
target_type = tts_phone_local_warp_weak_supervision
weak_target_warning = phone-local warp is supervision only, not a validated disentangled target
```

因此 MVP 的训练目标本身也不能被解释为已经验证的 disentangled TTS-style target。

## 当前证据等级

### 已被当前证据支持

1. 历史 paired 视频评测中，MVP 的 Sync-C 低于 natural，但 AV offset 更好；失败不是简单的全局延迟问题。
2. 严格 original natural cohort 与 MVP sidecar 的 samples 51–100 provenance 一致，transcript 与 natural named-tier alignment 均为 50/50 通过。
3. 当前 MVP 输出本身通过 sidecar 记录的 finite、无 clipping、exact-length 等内部检查。
4. MVP waveform 与严格 paired original natural 文件在 duration、LUFS、RMS、F0 和 spectral 特征上存在系统差异。
5. MVP 的训练 target 是 weak phone-local warp supervision，而不是经过验证的单属性解耦目标。

### 仍然只是候选假设

- phone duration、pause mask 和 phrase-level rhythm 不匹配；
- 连续 F0、energy envelope 或 coarticulation 被破坏；
- phonetic/content representation 与 Ditto mouth-motion mapping 不匹配；
- 低响度、F0 偏移或频谱差异参与了 Sync-C 下降；
- waveform 处理与 Ditto/SyncNet evaluator 存在交互；
- 多个声学因素耦合导致单一 feature control 无法恢复 raw-TTS 优势。

### 当前不能声称

- 不能声称 duration、LUFS、F0 或 spectral centroid 中任何一个是已证实的单一原因；
- 不能在缺少 MVP alignment 和逐样本 score ledger 时解释 feature-SyncNet 相关性；
- 不能用 source-TTS MFA TextGrid 代表 MVP waveform 的 phone timing；
- 不能把全局声学差异当作 human naturalness 或 speech intelligibility 结论；
- 不能排除 Ditto、PCM、视频、tracking 或 SyncNet preprocessing 的影响。

## 下一步决策

1. 获取对应 natural-vs-MVP paired 评测的逐样本 SyncNet JSON/CSV ledger，并校验汇总为 Sync-C `5.04526 → 4.16242`、Sync-D `7.90412 → 7.93172`、AV offset `0.700 → 0.080`；
2. 准备兼容的 Mandarin MFA runtime、dictionary 和 acoustic model，对 MVP waveform 运行独立 MFA，禁止复用 source-TTS TextGrid；
3. 在 MVP MFA 与 score ledger 都通过后，再计算 phone duration、pause、boundary、局部 F0/energy、spectral 与 Sync-C/Sync-D 的 paired correlation；
4. 最后运行 identity/no-op control，区分 audio effect 与 Ditto/SyncNet pipeline drift。

## 状态

⚠️ **诊断阶段完成；严格 original cohort 与 natural alignment 已通过，但 MVP Mandarin MFA 和逐样本 SyncNet ledger 仍阻塞。**

本实验的实现、严格 51–100 provenance/natural alignment 检查、global acoustic scan 和测试已经完成；MVP timing/pause/boundary/local prosody 与逐样本 SyncNet 关联尚未达到可解释条件。

## 相关文件

- 诊断脚本：`scripts/26_mvp_natural_audio_forensics.py`
- 测试：`tests/test_mvp_natural_audio_forensics.py`
- 输出目录：`results/rhythm_style_500/aishell1_test_400/forensics/mvp_vs_natural_51_100/`
- MVP sidecar 示例：`results/rhythm_style_500/aishell1_test_400/test_controls_best/mvp/conditions/mvp/51.json`
- 声学特征复用：`scripts/08_audio_features_v2.py`
- 韵律/phone 分析参考：`scripts/36_duration_jsd.py`
- 前序 phone-local warp：`docs/experiments/16-phone-local-warp-validation.md`
- 前序 feature controls：`docs/experiments/18-tts-feature-control-validation.md`
