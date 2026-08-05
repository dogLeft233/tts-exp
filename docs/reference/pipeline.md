# 实验流水线

## 标准流程

```
00_datacheck → 01_asr → 02_tts → 03_ditto → 04_eval → 05_report
```

但 TFG 跨模型对比跳过了 01_asr 和 03_ditto：直接用 TTS 音频驱动各 TFG 模型，再统一用 SyncNet 评估。

## 各步骤说明

### 00_datacheck — 数据体检
- 检查音频时长（≥5s 阈值）
- 验证配对完整性（face↔audio）

### 02_tts — TTS 生成
- Provider: `faster_qwen3` (本地 0.6B ICL)
- 输出: 24kHz mono WAV
- 速度: ~40s/样本（RTX 4080）

### TFG 推理 — 各模型独立
不同模型使用各自的推理接口：
- **Ditto**: `03_ditto.py` → TRT 离线引擎
- **Wav2Lip/MuseTalk/LatentSync**: 旧 1080Ti 服务器
- **JoyVASA/IMTalker/EchoMimic V2**: RTX 4080 独立容器

### 04_eval — SyncNet 评估
```bash
cd repos/tts-exp
python scripts/04_eval.py --run_id tfg_<model>
```
- 输入: `runs/tfg_<model>/03_ditto/{natural_raw,tts_raw}/*.mp4`
- 输出: `runs/tfg_<model>/04_eval/eval_meta.json`
- 每视频 ~20s（人脸检测 + SyncNet 评分）
- 默认人脸轨迹阈值为 `min_track=100`；若模型输出的场景切分使有效轨迹过短，可显式使用 `--min-track 25`，并在结果中记录该协议差异。

## 新增模型评估

```bash
# 1. 创建目录结构
mkdir -p runs/tfg_<model>/03_ditto/{natural_raw,tts_raw}

# 2. 放入 MP4（按 1.mp4 ~ 13.mp4 命名）

# 3. 运行评估
python scripts/04_eval.py --run_id tfg_<model>
```

## 配置文件

`scripts/config.yaml` 是单一真相源，包含所有路径和参数。
