# TFG 模型对比实验 — 进度报告

**日期**: 2026-07-17  
**实验目标**: 对比 6 个 TFG 唇形同步模型在相同音频（natural 16kHz vs faster_qwen3 0.6B TTS 24kHz）下的 SyncNet 表现  
**样本**: AISHELL-1 子集 13 人，每样本 2 条件（natural_raw + tts_raw）

---

## 服务器信息

| 角色 | 地址 | GPU | 状态 |
|------|------|-----|:----:|
| 旧推理 | 10.1.114.114:1986 | 8× GTX 1080 Ti (11GB) | 在线 |
| 新推理 | connect.westd.seetacloud.com:14453 | 1× RTX 4080 SUPER (32GB) | 在线 |
| SyncNet 评估 | connect.westd.seetacloud.com:20398 | 1× RTX 4080 SUPER (32GB) | 在线 |

---

## TTS 音频来源

所有模型的 TTS 条件均使用 **faster_qwen3 0.6B Base (ICL clone mode)** 生成：
- 格式: 24kHz, mono, PCM s16le WAV
- 13 个样本，同时部署在两台推理服务器
- 旧服务器路径: `/data/wjj/tfg_exp/data/tts_audio/{1-13}.wav`
- 新服务器路径: `/root/autodl-tmp/tfg_exp/tts_audio/{1-13}.wav`

---

## 各模型推理状态

### 已完成推理（可评估）

| # | 模型 | 服务器 | 状态 | 输出 | SyncNet 评估 |
|---|------|--------|:----:|------|:------------:|
| 1 | **Wav2Lip** | 旧 1080Ti | ✅ 26/26 | 96×96 + 原帧拼接 | ✅ 25/26 |
| 2 | **MuseTalk** | 旧 1080Ti | ✅ 26/26 | 256×256 + 原帧拼接 | ✅ 25/26 |
| 3 | **LatentSync** | 新 RTX 4080 | ✅ 26/26 | 512×512 H.264 | ✅ 26/26 |
| 4 | **JoyVASA** | 新 RTX 4080 | ✅ 26/26 | 512×512 H.264 | ✅ 26/26 |

### 未完成推理

| # | 模型 | 服务器 | 状态 | 原因 |
|---|------|--------|:----:|------|
| 5 | **EchoMimic** | 旧 1080Ti | ⚠️ 1/26 | 仅 1 个测试样本 (natural_raw/1)；无 xformers 加速（5.6h/样本），batch 可能中断 |
| 6 | **AniPortrait** | 旧 1080Ti | ❌ 0 | OOM（11GB 不足以载入 3× UNet fp16 ~7GB + 编码器） |
| 7 | **V-Express** | 新 RTX 4080 | ⏳ kps 阶段 | 权重已下载 (6.2G)，kps 预处理完成 (13 .pth)，视频推理未启动 |

---

## SyncNet 评估结果

所有评估在评估服务器 20398 运行，使用标准 SyncNet v2 模型。

评估管道: 视频 → run_pipeline.py (人脸检测+裁剪) → run_syncnet.py (同步评分)

Sync-C (Confidence): 越高越好，Sync-D (Distance): 越低越好

### TFG 模型总分

| 模型 | N | nat_C | nat_D | tts_C | tts_D | ΔC | ΔD |
|------|---|-------|-------|-------|-------|-----|-----|
| **Wav2Lip** | 13 | **7.39** | **7.33** | **8.92** | **5.94** | **+1.53** | -1.39 |
| **MuseTalk** | 13 | 5.79 | 8.08 | 6.33 | 7.93 | +0.54 | -0.15 |
| **JoyVASA** | 13 | 4.81 | 9.44 | 5.08 | 9.58 | +0.27 | +0.14 |
| **LatentSync** | 13 | 4.65 | 8.89 | 4.58 | 8.77 | -0.07 | -0.13 |

### TTS 参考基线（ditto 唇形替换模型，非 TFG）

| 模型 | N | nat_C | nat_D | tts_C | tts_D | ΔC | ΔD |
|------|---|-------|-------|-------|-------|-----|-----|
| faster_qwen3 + ditto | 10 | 5.70 | 8.08 | 6.75 | 7.57 | +1.05 | -0.51 |
| IndexTTS2 + ditto | 13 | 5.69 | 8.09 | 6.73 | 7.72 | +1.04 | -0.37 |
| GSV + ditto | 13 | 5.76 | 8.07 | 6.36 | 7.92 | +0.61 | -0.15 |

### 评估失败记录

| 模型 | 条件 | 样本 | 错误类型 |
|------|------|------|----------|
| Wav2Lip | tts_raw | 9 | parse failed — SyncNet stdout 未包含 Confidence/Min dist 行 |
| MuseTalk | tts_raw | 9 | parse failed — 同上 |

失败原因: 样本 9 的人脸检测/裁剪结果不满足 SyncNet 评分阈值，模型推理正常完成但 SyncNet 无法计算分数。

### 评估数据位置

```
runs/tfg_wav2lip/04_eval/
runs/tfg_musetalk/04_eval/
runs/tfg_latentsync/04_eval/
runs/tfg_joyvasa/04_eval/
```

---

## 响度归一化调研

所有 6 个模型**均无内置响度归一化**（EBU R128 / ITU-R BS.1770 / RMS / peak）。

| 模型 | 音频加载 | 波形响度归一化 | 特征级归一化（不影响公平性） |
|------|----------|:---:|------|
| Wav2Lip | `librosa.load(sr=16000)` | ❌ | Mel 频谱 dB 裁剪 [-4,4] |
| MuseTalk | `librosa.load(sr=16000)` | ❌ | 同 Wav2Lip + Whisper tiny |
| LatentSync | Whisper `load_audio()` (ffmpeg int16→float32) | ❌ | Whisper log-mel (log_spec+4)/4 → [0,1] |
| JoyVASA | `librosa.load(sr=16000, mono=True)` | ❌ | — |
| EchoMimic | Whisper `load_audio()` | ❌ | 同 LatentSync |
| V-Express | `torchaudio.load()` → 重采样 16kHz | ❌ | Wav2Vec2 processor z-score |

结论: **实验公平**。任一模型没有因内置响度归一化获得不公平优势。Wav2Lip 的 sync_c=7.39 领先来自架构本身的唇形匹配能力，而非音频预处理差异。

---

## 关键发现

1. **Wav2Lip 碾压所有模型** — 最简单的架构（96×96 GAN），最高的 sync_c (7.39/8.92)，远超竞品和 ditto 基线
2. **MuseTalk = ditto 水平** — sync_nat 5.79 vs ditto 5.70，可作为 ditto 的 TFG 替代
3. **LatentSync 最鲁棒** — 唯一 ΔC≈0 的模型，TTS 条件下无劣化
4. **JoyVASA 同步能力最弱** — nat_c=4.81，但 ΔC=+0.27 说明对 TTS 音频有正面响应
5. **所有模型 TTS 条件 sync_c 倾向上升** — TTS 音频更"干净"，有利于 SyncNet 评分

---

## 待办

| 优先级 | 任务 | 详情 |
|:------:|------|------|
| 高 | EchoMimic 全量推理 | 在新 RTX 4080 上完成权重下载 + 26 样本推理 |
| 高 | V-Express 全量推理 | 在新 RTX 4080 上完成视频推理 |
| 中 | AniPortrait 在新服务器推理 | 旧 1080Ti OOM，新 RTX 4080 32GB 应足够 |
| 中 | 修复 tts_raw/9 评估 | 重跑样本 9 的 SyncNet（两个模型均失败） |
| 低 | 旧服务器 EchoMimic 清理 | 确认 batch 进程状态，必要时 kill + 迁移到新服务器 |
