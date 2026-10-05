# `rms_baseline_v1` 使用说明

这是可复用的固定 RMS 正对照配置，入口是现有音频 CLI；不需要加载 checkpoint，也不会调用 TFG 模型。配置把单声道 16 kHz float32 音频的全局 RMS 调到 `0.0376838172675746`，不做峰值限制或硬削波，并保持规范化后的样本数不变。

## 命令

在仓库根目录执行：

```bash
python -m scripts.audio_enhancement.cli \
  INPUT.wav OUTPUT.rms_16k_f32.wav \
  --config scripts/audio_enhancement/configs/rms_baseline_v1.json \
  --natural-reference-wav OUTPUT.natural_16k_f32.wav
```

`INPUT.wav` 是自然语音。CLI 会留下未增强的规范化自然参考，并写出 RMS 增强 WAV。两个输出都必须使用新文件名；CLI 默认拒绝覆盖已有文件。结果 JSON 记录输入、自然参考、增强音频的文件哈希、PCM 哈希、RMS 和样本时钟。

把 `OUTPUT.rms_16k_f32.wav` 用作生成器的音频条件。评分时仍用原始自然语音；不要把 RMS WAV 当作评分声轨。若同时通过 CLI 封装无声视频，保留原始 WAV 的 PCM：

```bash
python -m scripts.audio_enhancement.cli \
  INPUT.wav OUTPUT.rms_16k_f32.wav \
  --config scripts/audio_enhancement/configs/rms_baseline_v1.json \
  --generated-video VIDEO.silent.mp4 \
  --mux-original-wav INPUT.wav \
  --mux-output VIDEO.rms_baseline.mkv
```

当输入本来就是单声道 16 kHz 时，RMS 变换不改变样本数或采样时钟。其他 PCM/IEEE-float RIFF/WAVE 输入会先按固定音频边界转成 16 kHz 单声道；声道用算术平均混合，重采样使用配置中绑定的确定性实现。不要用压缩音频容器代替原始 WAV 作为评分来源。

## 参数和证据

- 方法 ID：`rms_fixed`；CLI 配置中的方法名：`rms`。
- 目标 RMS：`0.0376838172675746`；峰值限制：关闭。
- CLI 会把 `rms_baseline_v1.json` 的原始哈希写入结果。配置内的源文件字节数和 SHA-256 是来源绑定；使用前可与本地文件核对。
- 零参数 V3 CPU 工程核验在 40 条历史 fit 音频上逐字节匹配 NumPy 固定 RMS 基线。该核验只说明初始化边界与 RMS 控制等价，不把零参数头称为训练完成的候选。
- 完整共享开发集上，`rms_fixed − identity` 的 FULL `Cbest` 按说话人等权后，Wav2Lip 为 `+0.234`、MuseTalk 为 `+0.053`、Ditto 为 `+0.173`；每个前端 10/10 说话人完整。它是描述性开发集对照结果，不代表对所有输入都有效。
