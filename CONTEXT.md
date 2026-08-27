# TTS-TFG 实验项目

## 核心问题

**TTS 生成的语音是否比自然语音产生更好的唇形同步（TFG）？**

使用 AISHELL-1 中文语音 + faster_qwen3 0.6B ICL 模型生成 TTS，经 Ditto/多模型 TFG 生成口型视频，SyncNet 评分。

## 关键结论

1. **TTS > Natural 在中文唇形同步上成立**，n=100 稳健复现（Ditto: ΔSync-C=+1.02, d=1.24, p<0.0001, 92% 样本；详见 `15`）
2. **效应跨模型泛化**：Wav2Lip (+1.52) > IMTalker (+1.13) > 其余模型（n=13 旧数据）
3. **英文不成立**（LibriSpeech ΔC=+0.16 ns；HDTF 13 说话人 ΔC=−0.14 ns）
4. **差异在声学/韵律层而非音素可分离性**：早层（l0-2）最强（`12`/`13`）；TTS 更快、停顿少 63%、句内时长更均匀（`11`）；**无可分离性指标能预测逐样本 lip-sync 增益**（`15`，Wav2Sem 假设不被支持）
5. **音素结构非说话人混杂**（`14`）；早期"TTS 与自然在 HuBERT 空间完全重叠"结论已被 `13` 修正——逐音素差异在早层可检测（AUC 0.87-0.95）
6. **LRS3 MFA3/`english_mfa` 严格 replacement 未保留增益**：133 条记录的 Wav2Lip 四-cell 矩阵、532 个 strict mux 和 532 个 fresh SyncNet score 全部通过工程核验，但 natural-audio replacement 的 `benefit_C` 均值为 −1.201、`benefit_D` 均值为 −1.061，科学结论为 `NO_GO`；详见 `runs/lrs3_mfa_linear_replacement_mfa3_exploratory_20260825/04_statistics_exploratory_retry2/`
7. **Duration-compatible TTS visual teacher audit 当前工程 BLOCKED**：133 条 fit-only record 全部完成 real/natural/candidate landmark extraction 与独立 artifact review，但仅 115/133 达到固定 eligibility、21/23 effective groups 有 eligible record，低于预注册 minimum 122；scientific decision 为 `not_available`，不得进入 Stage02 或降低阈值；详见 `runs/lrs3_tts_visual_control_20260825/01_visual_teacher_audit_retry4/`

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
| AISHELL-1 100 音素 corpus | `data/aishell1_100_zh/` | AISHELL-1 test split 100 条（16 说话人）+ 同文本 faster_qwen3 TTS；MFA 对齐 + HuBERT/XLS-R 分析 → `docs/experiments/09`，B 系列深化（`11-15`：时长 JSD / 逐层曲线 / 逐音素 KLD / 解纠缠 / TFG 关联）产物在 `results/aishell100_phoneme/` |
| TED-LIUM 3 (30%) | `data/tedlium3_30pct/` | SpeechColab TED-LIUM 3 子集，32 个 HF parquet 分片，81,760 条（15.97GB，≈277h，均值 12.2s）；字段 `audio(bytes)/text/speaker_id/gender/file/id`；16kHz 内嵌音频，尚未被代码引用 |

## 技术栈

- Python 3.10, PyTorch 2.5.1+cu124, CUDA 12.4
- SyncNet V2 (评估)
- Ditto TRT (TensorRT 离线推理)
- HuBERT / XLSR (SSL 特征分析)
- MFA (强制对齐)

## 文档结构

文档由 Basic Memory 管理（`basic-memory/docs/`，经 MCP 索引）。仓库根不再维护独立 `docs/`。

```
CONTEXT.md              ← 你在这里（仓库根，agent 入口）
basic-memory/docs/
├── adr/                ← 架构决策记录
├── experiments/        ← 实验报告（按阶段编号；索引见 experiments/README.md）
├── deployment/         ← 模型部署记录
├── results/            ← 评分汇总
└── reference/          ← 流水线/服务器/部署参考
```

实验报告索引：`basic-memory/docs/experiments/README.md`。B 系列（11-15）为 2026-08 的音素/韵律/TFG 深化实验，均经对抗子 agent 审查修复。
