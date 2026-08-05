# ACTalker

## 来源

- 论文: [Audio-visual Controlled Video Diffusion with Masked Selective State Spaces Modeling for Natural Talking Head Generation](https://arxiv.org/abs/2504.02542) (ICCV 2025)
- 代码: <https://github.com/harlanhong/ACTalker>
- 项目页: <https://harlanhong.github.io/publications/actalker/index.html>
- 机构: 香港科技大学 + Tencent + 清华大学

## 架构

端到端视频扩散框架，基于 Stable Video Diffusion (SVD-XT-1.1) 主干，引入并行 Mamba 架构 + 门控机制，将不同控制信号（音频、表情、姿态）分配到特定面部区域，实现细粒度无冲突生成。Mask-drop 策略增强区域独立性和控制稳定性。

支持三种推理模式：
- Mode 0: 仅音频驱动
- Mode 1: 仅表情驱动（VASA 控制）
- Mode 2: 音频 + 表情联合控制

## 硬件要求

| 资源 | 最低 | 推荐 |
|---|---|---|
| VRAM | 24GB (RTX 4090) | 40GB (A100) / 80GB (H100) |
| 系统内存 | 32GB | — |
| 硬盘 | 20GB+ (SVD ~9.5GB + 权重若干) | — |
| CUDA | 11.8 | 11.8（12.x 已知不兼容 Mamba SSM） |
| PyTorch | 2.0.1 | 2.0.1 |
| mamba-ssm | 1.2.0.post1 | — |

## 推理速度

| GPU | VRAM | 速度 (25 steps) |
|---|---|---|
| H100 | 80GB | ~6 min/样本 |
| A100 | 40GB | ~8 min/样本 |
| RTX 4090 | 24GB | ~12 min/样本 |
| RTX 3090 | 24GB | ~15 min/样本 |

## 部署状态

2026-07-26 在 Seetacloud 实例上完成部署：

| 项目 | 详情 |
|---|---|
| 服务器 | `connect.weste.seetacloud.com:30528` |
| GPU | RTX 4080 SUPER 32GB |
| 环境路径 | `/root/autodl-tmp/envs/actalker` |
| 代码路径 | `/root/autodl-tmp/repos/ACTalker` (commit `109de52`) |
| PyTorch | 2.0.1+cu118 (CUDA 可用) |
| CUDA | 11.8 |
| Mamba SSM | 1.2.0.post1 |
| mediapipe | 0.10.9 (修复了 0.10.35 无 `solutions` 模块的兼容性问题) |
| 数据盘 | 50GB，已用 ~25GB |
| 测试结果 | `test_environment.py`: 6/6 passed |

### 输入数据

来自 AISHELL-1 数据集，13 个中文样本（ID 1-13），与 TFG 实验使用相同数据。

| 数据类型 | 本地路径 | 远程路径 (weste:30528) | 数量 | 格式 | 大小 |
|---|---|---|---|---|---|
| 参考图片 | `data/data/image/{1..13}.png` | `repos/ACTalker/data/test_images/{1..13}.png` | 13 张 | PNG | 2.8MB |
| 自然音频 | `data/data/audio/{1..13}.wav` | `repos/ACTalker/data/natural_audio/{1..13}.wav` | 13 段 | 16kHz mono WAV | 2.5MB |
| TTS 合成音频 | `runs/<run_id>/02_tts/{1..13}.wav` | 待从 GPU 服务器复制 | — | 24kHz mono WAV | — |

本地绝对路径：`/mnt/e/Documents/tts-audio/tts-exp/data/data/`
远程绝对路径：`/root/autodl-tmp/repos/ACTalker/data/`

推理命令：`python Inference.py --config config/inference.yaml --ref data/test_images/1.png --audio data/natural_audio/1.wav --video <vasa_source>.mp4 --mode 0`

### 权重下载（按 README 要求）

README 仅列出 2 类必需权重：

| # | 权重 | 来源 | 大小 | 状态 | 详情 |
|---|------|------|------|------|------|
| 1 | SVD-XT-1.1 | HuggingFace `stabilityai/stable-video-diffusion-img2vid-xt-1-1` | 18GB (52 文件) | ✅ 已下载 | `pretrained_models/stable-video-diffusion-img2vid-xt-1-1` |
| 2 | Pretrained Checkpoints | SharePoint `hkustconnect-my.sharepoint.com` | 未知 | ❌ 不可访问 | 返回 403，跳转 OneDrive 登录页，无 HF 镜像 |

SharePoint 链接详情：
- URL: `https://hkustconnect-my.sharepoint.com/:f:/g/personal/fhongac_connect_ust_hk/ErbIGCYPpIRDjFcPgtYXoMABt5HWl5XS_He18Tii4wGQ5A`
- 实际路径: `Documents/Checkpoint/ICCV2025_ACTalker`
- 状态: 页面存在 (200 OK)，但下载需身份认证 (403 Forbidden)
- OneDrive API 尝试: 返回 `"User migrated"` 错误
- HF 搜索: 无 `harlanhong/ACTalker` 模型仓库
- GitHub 标注: "Pre-trained checkpoints will be released gradually"

增强模型 (`enhance_model/`: Whisper-Tiny, RIFE, BFR, ArcFace, 人脸对齐, 牙齿增强) README 未列为必需下载，属于可选后处理模块，各模型分散在公共源（HF/GitHub）。

### 阻塞因素

1. **ACTalker Checkpoints 未公开**: 代码已开源 (MIT)，预训练权重仅通过 SharePoint 分发且当前不可访问。HF 讨论中作者回复 "probably not"。**唯一硬阻塞**。
2. **实例已切为无卡模式**: `nvidia-smi` 返回 Permission denied，GPU 不可用。即使权重就绪也需切回 GPU 实例。
3. ~~CUDA 版本冲突~~: 已通过独立 CUDA 11.8 环境绕过。
4. ~~VRAM 不足~~: 实例实际配置 RTX 4080 SUPER 32GB，满足 24GB 最低要求。

## 与现有模型对比

| | ACTalker | EchoMimic V2 | IMTalker | Ditto |
|---|---|---|---|---|
| 主干 | SVD 扩散 | DiT 扩散 | Flow Matching | TRT 离线 |
| VRAM | 24GB+ | ~8GB | ~5GB | ~4GB |
| 速度 | 12min | 35s | 30s | 0s (预计算) |
| CUDA | 11.8 | 12.x | 11.8 | 12.x |
| 状态 | 🟡 环境就绪，缺权重 | ✅ 已部署 | ✅ 已部署 | ✅ 已部署 |
