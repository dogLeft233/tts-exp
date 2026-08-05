# TTS-TFG 实验项目

## 核心问题

**TTS 生成的语音是否比自然语音产生更好的唇形同步（TFG）？**

使用 AISHELL-1 中文语音 + faster_qwen3 0.6B ICL 模型生成 TTS，经 Ditto/多模型 TFG 生成口型视频，SyncNet 评分。

## 关键结论

1. **TTS > Natural 在中文唇形同步上成立**（Ditto: ΔSync-C=+1.25, p=0.0005）
2. **效应跨模型泛化**：Wav2Lip (+1.52) > IMTalker (+1.13) > 其余模型
3. **英文不成立**（两个独立数据集：LibriSpeech ΔC=+0.16 ns；HDTF 13 说话人 ΔC=−0.14 ns）
4. **单一声学特征无法解释**——可能是 Generator×Evaluator 共适应
5. **TTS 不损害帧级音素信息**——自然和 TTS 在 HuBERT 特征空间完全重叠

## 正在使用

| 类别 | 项目 | 说明 |
|------|------|------|
| TTS 引擎 | `faster_qwen3` (0.6B ICL) | 本地推理，24kHz 输出 |
| TFG 模型 | Ditto, Wav2Lip, MuseTalk, LatentSync, JoyVASA, V-Express, EchoMimic V2, IMTalker, StableAvatar | 9 个核心模型 |
| 评估 | SyncNet V2 | Sync-C (越高越好) + Sync-D (越低越好) |
| 参考帧预处理 | `38_prepare_frames.py` + `frame_quality.py` + `frame_detectors.py` | 第一帧策略（ADR-002）；mediapipe 检测器；原始分辨率输出 |
| 服务器 | Seetacloud RTX 4080 SUPER × N | 动态分配端口 |
| 流水线 | `00_datacheck → 02_tts → 03_ditto → 04_eval` | 可复现 |

## 已弃用

| 项目 | 原因 |
|------|------|
| EchoMimic V1 | 无 xformers 时 5.6h/样本 |
| AniPortrait | 256×256 fp16 OOM (11GB) |
| Hallo2 | CUDA 11.8 硬要求 (服务器 13.0) |
| `dashscope_vc` (云 API) | 响度归一化抹平 TTS 优势 |
| `trt_online.pkl` | 输出 0.2s 视频 (已损坏) |
| 旧 1080Ti 集群 (10.1.114.114) | 11GB VRAM 不足 |
| `aishell1_ldn` loudnorm 实验 | 已是 null result |

## 数据

| 数据集 | 路径 | 说明 |
|--------|------|------|
| HDTF 英文原始音频 | `data/hdtf_audio_en/audio/` | 337 个 .m4a 文件（871MB），美国政客演讲录音 |
| LibriSpeech TTS 音频 | `data/data/audio_en/` | 英文 LibriSpeech subset 的 TTS 生成输出 |
| 多语言 TTS 参考音频（旧版） | `data/multilingual_tts/` | 旧版 9 语言、每语言 20 条；CSS10/CML-TTS 存在已知来源重合风险 |
| 多语言 TTS 参考音频（MDC） | `data/mdc_tts/` | 当前 5 语言 × 20 条 = 100 条；英语/德语/意大利语/西班牙语/韩语已提取，另 4 种 MDC 条款仍返回 403 |
| TTS 资产目录 | `data/tts/catalog.json` | 本地 TTS 音频索引，支持按 provider/model/language/dataset 检索 |
| AISHELL-1 100 音素 corpus | `data/aishell1_100_zh/` | AISHELL-1 test split 100 条（16 说话人）+ 同文本 faster_qwen3 TTS；MFA 对齐 + HuBERT/XLS-R 分析 → `docs/experiments/09-aishell100-phoneme-extension.md`，产物在 `results/aishell100_phoneme/` |

## 技术栈

- Python 3.10, PyTorch 2.5.1+cu124, CUDA 12.4
- SyncNet V2 (评估)
- Ditto TRT (TensorRT 离线推理)
- HuBERT / XLSR (SSL 特征分析)
- MFA (强制对齐)

## 文档结构

```
CONTEXT.md              ← 你在这里
docs/
├── adr/                ← 架构决策记录
├── experiments/        ← 实验报告（按阶段编号）
├── deployment/         ← 模型部署记录
├── results/            ← 评分汇总
└── reference/          ← 流水线/服务器/部署参考
```
