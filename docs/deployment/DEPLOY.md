# TFG 模型部署总览

## 服务器

| 角色 | 地址 | GPU | 状态 |
|------|------|-----|:---:|
| TFG 推理 A | 16461 | RTX 4080 SUPER 32GB | ⚠️ TCP refused 2026-07-30 |
| TFG 推理 B | 43417 | RTX 4080 SUPER 32GB | ⚠️ TCP refused 2026-07-30 |
| StableAvatar 推理 | 28302 | NVIDIA vGPU-32GB | ⚠️ TCP refused 2026-07-31 |
| SyncNet 评估 | 26502 | RTX 4080 SUPER 32GB | ⚠️ TCP refused 2026-07-31 |
| 实验数据 | 22035 | — | ⚠️ TCP refused 2026-07-30 |
| 旧 1080Ti | 10.1.114.114:1986 | 8×1080Ti 11GB | ❌ 下线 |
| 旧 RTX 4080 | 14453 | RTX 4080 SUPER 32GB | ❌ 下线 |

## 模型一览

| 模型 | 服务器 | 分辨率 | 速度 | VRAM | 状态 |
|------|--------|:---:|:---:|:---:|:---:|
| Wav2Lip | 旧 1080Ti | 96×96 | ~30s | 4.4G | ✅ |
| MuseTalk 1.5 | 旧 1080Ti | 256×256 | ~90s | 8.4G | ✅ |
| LatentSync | 旧 14453 | 512×512 | ~2m50s | 3.5G | ✅ |
| JoyVASA | 43417 | 512×512 | ~23s | 2.2G | ✅ |
| V-Express | 旧 14453 | 512×512 | ~6m20s | 9.0G | ✅ |
| IMTalker | 43417 | 512×512 | ~30s | 5.0G | 🆕 |
| EchoMimic V2 | 16461 | 512×512 | ~35s | ~8G | 🆕 |
| SoulX-FlashHead Pro | 43794 | 512×512 | 约 3.77s/chunk（预热后） | 8.106G | ✅ 已完成；服务器已关闭 |
| AVTR-1 | 48111 | 1280×720 | 约 3–5s/样本 | ~8G | ✅ 已完成；SyncNet 已评分；服务器已关闭 |
| AvatarForcing | 26795 | 512×512 | 约 42s/样本 | 约 18–20G | ✅ 已完成；SyncNet 已评分；服务器已关闭 |
| LeapTalk | 41601 | 512×512 | 约 40s/样本 | ~8–9G | ✅ 已完成；SyncNet 已评分；服务器已关闭 |
| StableAvatar | 28302（已关闭） | 512×512 | 10 steps；约 3min/5s（参考） | 峰值约 27.4G | ✅ 已完成；SyncNet 已评分；服务器已关闭 |
| AptAvatar | 待分配 | 720P I2V | 待实测 | 待实测；18.882B BF16 | ⏸ 磁盘不足 |
| LipForcing | 待分配 | V2V | H200 参考约 37G | 14B | ❌ 当前 32G 不适合 |
| EchoMimic V1 | — | — | — | — | ❌ |
| AniPortrait | — | — | — | — | ❌ |
| Hallo2 | — | — | — | — | ❌ |

模型一览中的 `✅` 表示历史部署或批量推理记录，不表示对应服务器当前在线；当前服务器可达性以“服务器”表为准。

## 部署模式

各模型部署在独立 Seetacloud 容器中：
- 输入: AISHELL-1 face image + natural/TTS audio
- 输出: 25fps MP4 (video+audio)
- 统一使用 faster_qwen3 0.6B ICL TTS 音频

## 常见问题

- **HF 下载失败**: `source /etc/network_turbo` 或 `export HF_ENDPOINT=https://hf-mirror.com`
- **torch.load 报错**: PyTorch ≥2.6 默认 `weights_only=True`，加 `weights_only=False`
- **OOM**: 降低分辨率或帧数
- **IMTalker HF 离线**: `export HF_HUB_OFFLINE=1`

## 2026 新候选部署判定

| 优先级 | 模型 | 适配现有能力的原因 | 主要阻塞 |
|:---:|------|--------------------|----------|
| 1 | LeapTalk | ✅ 已完成 26/26 条推理和 SyncNet 评分；基于 SoulX-FlashHead 1.3B；单步流式 I2V；~40s/样本；需 mediapipe 0.10.21 | 2026-07-30 已完成，服务器已关闭 |
| 2 | StableAvatar | Wan2.1 1.3B；官方提供 Docker 和 CPU offload；32GB 显存可覆盖 | ✅ 2026-07-31 完成 26/26；SyncNet 使用 `min_track=25` |
| 3 | AptAvatar | 2-step、中文 wav2vec2、长时 I2V | HF 仓库约 54.5GB，超过当前 50GB 数据盘；显存峰值未知 |
| V2V | LipForcing | 2026-07 已有官方代码和 14B 权重 | 需要约 37GB VRAM；1.3B student 尚未发布，不进入当前 I2V 组 |

2026-07-31 对 `28302` 和 `26502` 做 TCP 探测均返回 connection refused；StableAvatar 已完成 26 条推理和评分，服务器已关闭。
