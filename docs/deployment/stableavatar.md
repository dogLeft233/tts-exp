# StableAvatar 部署记录

## 状态

- **状态**: ✅ AISHELL-1 26 条批量推理完成；SyncNet 已评分；服务器已关闭
- **容器**: `connect.westb.seetacloud.com:28302`（已关闭）
- **官方代码**: [StableAvatar](https://github.com/neosun100/StableAvatar)
- **模型权重**: `FrancisRing/StableAvatar`

## 环境

- 代码目录：`/root/autodl-tmp/StableAvatar`
- Conda 环境：`/root/autodl-tmp/envs/stableavatar`
- PyTorch `2.6.0+cu124`、TorchVision `0.21.0+cu124`、TorchAudio `2.6.0+cu124`
- GPU：`NVIDIA vGPU-32GB`
- 权重下载约 23GB；数据盘为 50GB
- full-load 峰值显存约 27.4GB；CPU offload 可作为低显存备选

## 推理协议

- 输入：13 张 AISHELL-1 参考图 × 13 条 natural raw 音频 + 13 条 faster_qwen3 0.6B ICL TTS 音频
- 输出：512×512、25 FPS、MP4
- 批量脚本：`scripts/stableavatar_batch.py`
- 远端输出：`/root/autodl-tmp/StableAvatar/tfg_outputs/{natural_raw,tts_raw}/`
- 本地输出：`results/stableavatar/{natural_raw,tts_raw}/`
- 26/26 条推理成功；本地文件与远端 SHA-256 全部一致

## SyncNet 评分

| 条件 | 样本数 | Sync-C ↑ | Sync-D ↓ |
|------|:---:|:---:|:---:|
| `natural_raw` | 13 | 3.938 | 10.791 |
| `tts_raw` | 13 | 3.914 | 11.059 |
| Δ (TTS − Natural) | — | −0.024 | +0.268 |

- 26/26 个视频评分成功，`complete cases=13/13`。
- 因默认 `min_track=100` 对 StableAvatar 的场景切分过严，本次使用 `--min-track 25`；结果暂作为补充组，不直接混入默认参数排名。
- 逐样本结果：`results/stableavatar/04_eval/`

## 注意事项

- 远端 SyncNet 环境的 `ffmpeg` 位于 `/root/autodl-tmp/envs/syncnet/bin/ffmpeg`，运行评分前需加入 `PATH`。
- 权重和环境占用较多数据盘空间，部署前应检查 `/root/autodl-tmp` 剩余空间。
- `28302` 推理服务器和 `26502` SyncNet 服务器均已关机，端口探测返回 `Connection refused`。
