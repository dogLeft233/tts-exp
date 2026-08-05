# Dataset Reference

## 项目使用的数据集总览

| 数据集 | 语言 | 用途 | 来源 | 许可 |
|--------|------|------|------|------|
| AISHELL-1 | 中文 | TTS 克隆源音频 | OpenSLR | Apache-2.0 |
| HDTF | 英文 | 视频口型同步数据 | Synology NAS | 项目内部 |
| LibriSpeech (test-clean) | 英文 | 英文对比实验 | OpenSLR | CC-BY-4.0 |
| CSS10-Multilingual-LJSpeech | 中/日/俄等 | 旧版多语言 TTS 参考 | HuggingFace | Apache-2.0；LibriVox-derived，严格 unseen 禁用 |
| CML-TTS | 德/意/葡/西/法等 | 旧版多语言 TTS 参考 | HuggingFace | CC-BY-4.0；MLS-derived，严格 unseen 禁用 |
| KSS | 韩语 | 旧版韩语 TTS 参考 | HuggingFace | CC-BY-NC-SA-4.0；仅 B 风险 |
| MDC multilingual set | 英/德/意/西/韩 | 新多语言 TTS 参考（当前 5×20 条） | MDC | 按数据集许可 |
| GSV-TTS | 中文 | 实验生成音频 | 本地 TTS 推理 | — |

---

## AISHELL-1

- **来源**: <https://www.openslr.org/33/>
- **语言**: 普通话 (Mandarin Chinese)
- **规模**: ~170 小时，340 说话人
- **原始用途**: 中文语音识别 (ASR) 基准数据集
- **本项目用途**: TTS 语音克隆的参考音频；从中选取 13 条音频作为 Qwen3-TTS 的声音克隆源，再合成为 TTS 音频用于 Ditto 推理
- **采样率**: 16 kHz
- **许可**: Apache-2.0
- **本地位置**: `data/test.csv`（阿里云 OSS 外链），`data/data/audio/`（本地 WAV 副本）

## HDTF (High-Definition Talking Face Dataset)

- **来源**: Synology NAS (`10.108.21.182:/dhg_mm/HDTF`)
- **语言**: 英文
- **规模**: 676 原始视频，692 原始音频
- **原始用途**: 说话人脸视频生成（Talking Face Generation）
- **本项目用途**: Ditto 等多个 TFG 模型的视频数据源；从中提取 13 帧画面作为 TFG 输入
- **NAS 路径**: `/dhg_mm/HDTF/_audio_raw`, `/dhg_mm/HDTF/_videos_raw`
- **凭证**: `NAS_DHG_MM_PASSWORD`（运行时注入，不留文件）

## LibriSpeech

- **来源**: <https://www.openslr.org/12/>
- **语言**: 英文
- **规模**: ~1000 小时，多说话人
- **原始用途**: 英文语音识别 (ASR)
- **本项目用途**: 英文对比实验集；选取 `test-clean` 子集 13 条样本（时长匹配 AISHELL-1 中文样本），作为 Ditto 的英文自然语音参考
- **采样率**: 16 kHz（原始 FLAC），本地转为 WAV
- **许可**: CC-BY-4.0
- **本地位置**: `data/data/audio_en/manifest.json`, `data/data/audio_en/*.wav`

## CSS10-Multilingual-LJSpeech

- **来源**: <https://huggingface.co/datasets/davidguzmanr/CSS10-Multilingual-LJSpeech>
- **上游**: CSS10（Kyubyong Park, Thomas Mulc） + LJSpeech
- **CSS10 原始数据来源**: LibriVox 公有领域有声书
- **语言**: Chinese, Dutch, English, Finnish, French, German, Greek, Hungarian, Japanese, Russian, Spanish
- **规模**: ~64K 音频，~140 小时
- **原始用途**: 单说话人 TTS 训练
- **本项目用途**: 旧版多语言 TTS 参考音频（中文/日文/俄文）；不作为严格 unseen 结果
- **采样率**: 22.05 kHz
- **许可**: Apache-2.0（衍生仓库）
- **本项目选取**: `Chinese/test`, `Japanese/test`, `Russian/test`（过滤 5-10 秒，各 20 条）
- **本地位置**: `data/multilingual_tts/audio/{zh,ja,ru}/`，索引见 `data/multilingual_tts/manifest.json`

## CML-TTS（旧版，严格 unseen 禁用）

- **来源**: <https://huggingface.co/datasets/ylacombe/cml-tts>
- **原始数据来源**: Multilingual LibriSpeech (MLS)；底层为公开有声书数据
- **开发方**: 戈亚斯联邦大学 (UFG) 人工智能卓越中心 (CEIA)
- **论文**: Oliveira et al., "CML-TTS A Multilingual Dataset for Speech Synthesis in Low-Resource Languages", arXiv:2306.10097
- **语言**: Dutch, French, German, Italian, Polish, Portuguese, Spanish
- **规模**: 多说话人，多语言
- **原始用途**: 低资源语言 TTS 训练
- **本项目用途**: 旧版多语言 TTS 参考音频（德/意/葡/西/法）；不作为严格 unseen 结果
- **采样率**: 24 kHz
- **许可**: CC-BY-4.0
- **本项目选取**: `german/test`, `italian/test`, `portuguese/test`, `spanish/test`, `french/test`（过滤 5-20 秒，各 20 条）
- **本地位置**: `data/multilingual_tts/audio/{de,it,pt,es,fr}/`，索引见 `data/multilingual_tts/manifest.json`

## KSS (Korean Single Speaker Speech Dataset)

- **来源**: <https://huggingface.co/datasets/Bingsu/KSS_Dataset>
- **作者**: Kyubyong Park，2018
- **语言**: 韩语
- **规模**: 12,853 条音频，~12 小时，单说话人（专业女声优）
- **原始用途**: 韩语 TTS 训练
- **本项目用途**: 韩语 TTS 参考音频
- **采样率**: 44.1 kHz
- **许可**: CC-BY-NC-SA-4.0（非商业用途）
- **本项目选取**: 训练集（过滤 5-12 秒，20 条）
- **本地位置**: `data/multilingual_tts/audio/ko/`，索引见 `data/multilingual_tts/manifest.json`

## GSV-TTS

- **来源**: 本地 TTS 引擎推理生成
- **语言**: 中文
- **规模**: 13 条 TTS 合成音频
- **用途**: 音高/韵律扰动实验的合成音频（`data/gsv-tts/`）
- **TTS 引擎**: `faster_qwen3`（Qwen3-TTS-12Hz-0.6B-Base 本地 ICL 推理）

---

## 数据集关系图

```
AISHELL-1 (中文 170h ASR)
  ├── 13 条音频 → Qwen3-TTS → GSV-TTS (13 条合成)
  └── 13 条音频 → Ditto/Wav2Lip/… → SyncNet 评估 (中英文对比)

HDTF (英文视频+音频)
  └── 13 帧画面 → Ditto/Wav2Lip/… → SyncNet 评估

LibriSpeech (英文 1000h ASR)
  └── 13 条 test-clean → Ditto/Wav2Lip/… → SyncNet 评估

CSS10、CML-TTS、KSS (旧版多语言 TTS 语料)
  └── 每语言 20 条 → 多语言 TTS 参考音频（旧版 baseline）
```

## TTS 编码器训练重叠说明

严格 unseen 候选优先使用发布时间晚于 Qwen3-TTS 报告的配对语音数据，或
明确与已知来源隔离的数据。Qwen3-TTS 没有公开完整训练清单，因此只能
报告来源级风险，不能声称绝对未见。Common Voice/MDC 数据的原始音频不
得重新托管或提交到仓库。
