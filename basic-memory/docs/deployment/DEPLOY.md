---
title: DEPLOY
type: note
permalink: tts-exp/docs/deployment/deploy
---


# TFG 模型部署总览

## 当前本地 TTS（2026-10-23）

| 模型 | 主机 | 验证 | 状态 |
|---|---|---|---|
| [Qwen3-TTS-12Hz-1.7B-Base](qwen3-tts-1p7b.md) | wjj@192.168.50.2，V100 16GB | 完整 self-ICL；3.92秒音频用时13.03秒；FP32/eager，PyTorch峰值reserved 8.48GiB | ✅ 独立环境部署通过，尚未进行评分对照 |

## 服务器

| 角色 | 地址 | GPU | 状态 |
|------|------|-----|:---:|
| 192个人目录 TFG/九评分器 | 10.108.16.192:22，wangjiajun | 6×A6000 + 2×3090；本次单A6000 | ✅ 2026-10-30 SSH及完整实验已验收 |
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
| [DyStream](dystream.md) | 192个人目录，单A6000 | 512×512，25fps，77帧 | 3.08秒成片用时32.573秒，含加载保存 | NVIDIA本人PID峰值8.080GiB | ✅ 官方组件I2V／真实CUDA前向／媒体解码通过；CPU人脸预处理，未评分 |
| [PC-Talk](pc-talk.md) | 192个人目录，单A6000 | 512×512，25fps，73帧 | 2.92秒成片用时9.289秒，含加载保存 | NVIDIA本人PID峰值2.268GiB | ✅ 官方I2V／CUDA前向／媒体解码通过，3.png须face-index1；未评分 |
| [TalkLikeYou](talklikeyou.md) | 192个人目录，单3090 24GB | 1034×1034，25fps | 19.8秒官方I2V样片用时48.57秒，含导入加载保存；渲染23.16FPS | NVIDIA本人PID采样峰值5.631GiB | ✅ 官方I2V及本机495帧读回通过；Sync-C未测 |
| [InfiniteTalk](infinitetalk.md) | 192个人目录，单3090 24GB | 896×448（480P桶） | 3.24秒样片用时86分49秒，含加载保存 | NVIDIA采样峰值16.000GiB | ❌ 官方示例已验证；用户因速度停止采用 |
| [MultiTalk](multitalk.md) | 192个人目录，单3090 24GB | 官方480P桶 | 运行29分34秒后用户中止，最后24/40步 | 中止前采样峰值14.342GiB，非完整峰值 | ❌ 未完成成片；用户因速度停止采用 |
| Wav2Lip | 旧 1080Ti | 96×96 | ~30s | 4.4G | ✅ |
| MuseTalk 1.5 | 旧 1080Ti | 256×256 | ~90s | 8.4G | ✅ |
| LatentSync | 旧 14453 | 512×512 | ~2m50s | 3.5G | ✅ |
| JoyVASA | 43417 | 512×512 | ~23s | 2.2G | ✅ |
| V-Express | 旧 14453 | 512×512 | ~6m20s | 9.0G | ✅ |
| IMTalker | 43417 | 512×512 | ~30s | 5.0G | 🆕 |
| EchoMimic V2 | 16461 | 512×512 | ~35s | ~8G | 🆕 |
| [EchoMimicV3-Flash](echomimic-v3-flash.md) | 192个人目录 | 512×512 | 约63.5s/变长输入，不含加载 | NVIDIA采样峰值14.86GiB | ✅ 80逻辑视频/九评分器/本地读回通过；RMS24受控配置 |
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

上表旧云端部署使用独立Seetacloud容器；2026-10-30的192部署使用个人目录和独立环境，见[[echomimic-v3-flash]]及[[77-echomimic-v3-chinese20-nine-scorers]]。下列是旧批次约定：
- 输入: AISHELL-1 face image + natural/TTS audio
- 输出: 25fps MP4 (video+audio)
- 统一使用 faster_qwen3 0.6B ICL TTS 音频

## 本地工作站 TTS 模型 (V100)

| 模型 | 位置 | 速度 (V100) | 状态 |
|------|------|:---:|:---:|
| Qwen3-TTS 0.6B (faster_qwen3) | tts-exp/models/ + ~/.venvs/qwen3 | ~5s/条 | ✅ 2026-08-11 |
| IndexTTS-2 | /home/wjj/tts-audio/index-tts (uv venv, fp32) | 4.7s 音频 ~13s | ✅ 2026-08-14 |
| GPT-SoVITS v2 | /home/wjj/tts-audio/GPT-SoVITS (conda GPTSoVits) | RTF~0.3 | ✅ 2026-08-14 |

详见 [[index-tts-2]] / [[gpt-sovits]] / [[qwen3-tts]]。V100 = Volta sm_70: 无 bf16/flash-attn2/sageattention; torch 用 cu124 构建; mkl<2024 (GPTSoVits env)。

## 常见问题

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

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| Add verified private192 EchoMimicV3-Flash deployment and completed Chinese20 evidence; preserve historical cloud rows | October 30, 2026 | user |
