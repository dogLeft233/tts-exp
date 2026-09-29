## Why

项目记忆 `Wav2Lip ROI retiming oracle 2026-09-06` 的最终 run `20260907_oracle_v4` 已通过独立验证：baseline/B/C_oracle/O_oracle=22/22/21/22，但 own-audio 的 C/D 均值为 −0.470/−0.361，95% CI 下界 −0.634/−0.508，未过 −0.10 门槛，终态 `ORACLE_OWN_AUDIO_UNRESOLVED`。

下一步检验一个具体假设：将最近帧取整改为相邻像素帧线性插值，能否改善 own-audio，并同时保留局部 timing 和 damage 控制。该干预也会改变画面锐度，因此只检验控制对插值方式的敏感性，不预设“量化就是原因”。

## What Changes

- 固定原 22 records / 22 source groups、N/W 音频、连续映射、掩码和全部控制门槛，只新增 V_LINEAR 视频臂。
- V_LINEAR × N/W：22 个新无损视频流、44 个新 mux 和 44 个新 SyncNet cell；复用并重新核验父 88 个历史 cell。
- 交付一个 CPU 小实验、独立 validator、中文结果及同一条 BM 实验记忆的状态更新。

## Capabilities

### New Capabilities

- `wav2lip-oracle-frame-interpolation`: 固定线性混帧的 oracle 对照与自身配对有效性诊断。

### Modified Capabilities

无。

## Impact

新增 `scripts/experiments/wav2lip_oracle_frame_interpolation/`、对应测试和新 run。复用已有媒体 I/O 与 SyncNet worker；无需 GPU、新依赖或模型下载。历史实验规则和结论保持原状；本 change 不包含 TTS/TFG 生成、bridge、audio head、训练或跨模型实验。

本轮交付为设计，尚未执行。下游预计由 GPT Luna Max 实现，按 `design.md` 的入口与 `tasks.md` 完成一次固定实验。
