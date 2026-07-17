# TFG 模型部署与推理配置总结（最终版）

## 服务器概览

### 旧服务器（1080 Ti × 8）
- **地址**: 10.1.114.114:1986, wjj / zTnKbLMb1MXhtJz2
- **GPU**: 8× GTX 1080 Ti (11GB, Pascal CC 6.1)
- **驱动**: 470.103.01 (CUDA 11.4 max)
- **系统**: Ubuntu 20.04.3, 157GB RAM
- **磁盘**: `/data/wjj/` (HDD)
- **部署模型**: Wav2Lip, MuseTalk 1.5

### 新服务器（RTX 4080 SUPER）
- **地址**: connect.westd.seetacloud.com:14453, root / UT2C6O0ZI8TF
- **GPU**: 1× RTX 4080 SUPER (32GB, Ada CC 8.9)
- **驱动**: 580.105.08 (CUDA 13.0)
- **系统**: Ubuntu 22.04, 503GB RAM
- **磁盘**: `/root/autodl-tmp/` (50GB XFS, 可写)
- **部署模型**: LatentSync, JoyVASA, V-Express
- **网络**: GitHub/HF 需 proxy (`source /etc/network_turbo`) 或 `hf-mirror.com`

### 评估服务器（20398）
- **地址**: connect.westd.seetacloud.com:20398, root / qUz0z/KoV/Ik
- **GPU**: 1× RTX 4080 SUPER (32GB)
- **用途**: SyncNet 评估 + 之前实验的主控服务器
- **环境**: `syncnet`, `ditto` conda envs

---

## 模型部署状态

### 1. Wav2Lip ✅ (旧服务器)

| 项目 | 说明 |
|------|------|
| 输出 | 26/26, 96×96 面部 |
| 每视频 | ~30s |
| 显存 | ~4.4 GB |
| 架构 | GAN + SyncNet 判别器 |
| 发布 | ACM MM 2020 |

**踩坑**: TorchScript → state_dict 转换; `temp/result.avi` CWD 相对路径

---

### 2. MuseTalk 1.5 ✅ (旧服务器)

| 项目 | 说明 |
|------|------|
| 版本 | **v1.5** (2025-03-28 发布，两阶段训练 + 时空采样) |
| 输出 | 26/26, 256×256 面部 |
| 每视频 | ~90s (fp32, 无 xformers) |
| 显存 | ~8.4 GB |
| 架构 | VAE latent + 多尺度 U-Net inpainting |
| 发布 | 2024-04 (代码) / arXiv 2410.10122 |

**踩坑**: sd-vae `.bin` 回退 unsafe pickle; vae_type=`"v15"`

---

### 3. LatentSync ✅ (新服务器)

| 项目 | 说明 |
|------|------|
| 输出 | 26/26, 512×512 |
| 每视频 | ~2m50s |
| 显存 | ~3.5 GB (峰值) |
| 架构 | 潜在扩散模型 (ByteDance) |
| 权重 | `latentsync_unet.pt` 4.8GB, Whisper tiny 73MB |
| 环境 | latentsync conda env (torch 2.6.0+cu124) |
| 发布 | 2024-12-12 |

**踩坑**: 旧服务器 11GB VRAM OOM → 迁移到 32GB RTX 4080

---

### 4. JoyVASA ✅ (新服务器)

| 项目 | 说明 |
|------|------|
| 输出 | 26/26, 512×512 |
| 每视频 | ~23s |
| 显存 | ~2.2 GB |
| 架构 | 扩散 Transformer (LivePortrait backbone) |
| 权重 | 2.6GB: motion_generator 414MB + HuBERT 361MB + LivePortrait 628MB |
| 环境 | joyvasa conda env (torch 2.4.1+cu124) |
| 发布 | 2024-11-14 (arXiv 2411.09209) |

---

### 5. V-Express ✅ (新服务器)

| 项目 | 说明 |
|------|------|
| 输出 | 26/26, 512×512 |
| 每视频 | ~6m20s |
| 显存 | ~9 GB |
| 架构 | 渐进 Dropout 扩散模型 (腾讯 AI Lab) + keypoints |
| 权重 | 6GB: denoising_unet 2.6G + reference_net 1.7G + motion_module 867M |
| 环境 | **latentsync** conda env (torch 2.6.0+cu124) — 复用避免依赖地狱 |
| 发布 | 2024-06-04 (arXiv 2406.02511) |

**关键修复**:
1. `torch.load(weights_only=False)` — PyTorch 2.6 默认 `weights_only=True`
2. `temp_output_path` path `replace()` bug — 文件名含 `4` 会在 `.mp4` 中也替换
3. env 问题: GraalPy 不兼容 → 重建 CPython → 依赖地狱 → 最终复用 latentsync env

---

### 6. EchoMimic ❌ 放弃

- 旧服务器: 权重下载完成但推理极度缓慢（无 xformers 时 5.6h/样本）
- 新服务器: 权重下载中删除（腾空间给 JoyVASA/V-Express）

### 7. AniPortrait ❌ 放弃

- 旧服务器: 256×256 fp16 仍 OOM（~9.8GB，模型过重）
- 权重 10GB+ 全部下载完成但无法推理

### 8. Hallo2 ❌ 不兼容

- ICLR 2025，4K 长时生成，质量最优
- 但: **CUDA 11.8 硬要求**（服务器 CUDA 13.0 不兼容）+ **仅英文**

---

## 交叉对比总表（最终）

| 模型 | 服务器 | 状态 | 输出 | 每视频 | 显存 | Sync-C (nat) | Sync-D (nat) |
|------|:------:|:----:|:----:|:------:|:----:|:---:|:---:|
| Wav2Lip | 旧 | ✅ 26/26 | 96×96 | ~30s | 4.4G | 7.393 | 7.327 |
| MuseTalk 1.5 | 旧 | ✅ 26/26 | 256×256 | ~90s | 8.4G | 5.789 | 8.081 |
| V-Express | 新 | ✅ 26/26 | 512×512 | ~6m20s | 9.0G | 5.975 | 8.818 |
| JoyVASA | 新 | ✅ 26/26 | 512×512 | ~23s | 2.2G | 4.815 | 9.443 |
| LatentSync | 新 | ✅ 26/26 | 512×512 | ~2m50s | 3.5G | 4.653 | 8.894 |
| EchoMimic | — | ❌ | — | — | — | — | — |
| AniPortrait | — | ❌ | — | — | — | — | — |

---

## SyncNet 评测结果

评测在 20398 服务器上完成（`04_eval.py`）：

| 排名 | 模型 | natural Sync-C↑ | tts Sync-C↑ | natural Sync-D↓ | tts Sync-D↓ |
|:---:|:---|:---:|:---:|:---:|:---:|
| 1 | **Wav2Lip** | **7.393** | **8.922** | 7.327 | **5.941** |
| 2 | **V-Express** | 5.975 | 6.506 | 8.818 | 8.236 |
| 3 | **MuseTalk 1.5** | 5.789 | 6.326 | **8.081** | 7.928 |
| 4 | JoyVASA | 4.815 | 5.084 | 9.443 | 9.581 |
| 5 | LatentSync | 4.653 | 4.581 | 8.894 | 8.768 |

所有模型 TTS 条件下 Sync-C 均高于 natural（TTS 更清晰利于唇形检测）。

---

## 可复用工作经验

### 网络受限环境下的权重下载策略
1. **hf-mirror.com** — 优先使用（无需 proxy），`wget --no-check-certificate`
2. **`source /etc/network_turbo`** — SeetaCloud proxy，稳定性差
3. **本地下载 + SCP** — 最可靠，适合大文件（>1GB）

### 环境隔离与复用
1. 每个模型首选独立 conda env，但**优先复用已有环境**（如 V-Express 复用 latentsync env）
2. 避免 `pip install` 级联升级 PyTorch（→ 依赖地狱）
3. `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple` 比 conda 安装更快

### 镜像站汇总
| 用途 | 地址 |
|------|------|
| HuggingFace | `https://hf-mirror.com` |
| PyPI | `https://pypi.tuna.tsinghua.edu.cn/simple` |
| Conda | 清华镜像（自动配置于 `~/.condarc`） |

### PyTorch 版本兼容速查
| CUDA | PyTorch 推荐 | 备注 |
|:---|:---|:---|
| 11.4 (470 driver) | 2.0.1+cu118 | 旧 GTX 1080 Ti |
| 12.4+ | 2.4.1+cu124 | RTX 4080 |
| 13.0 | 2.6.0+cu124 | 驱动 580，兼容 CUDA 12.4 二进制 |

### 常见代码 Bug 修复模板
1. **`torch.load` weights_only**: PyTorch ≥2.6 默认 `True` → 加 `weights_only=False`
2. **Python `str.replace()` 路径陷阱**: `"4.mp4".replace("4", "4-temp")` → `"4-temp.mp4-temp"` → 用 `pathlib` 代替
3. **conda GraalPy 陷阱**: `conda create` 可能解析到 GraalPy → 影响 native 扩展兼容性

---

## 本地结果文件

```
results/
├── latentsync/
│   ├── natural_raw/  (13 mp4)
│   └── tts_raw/      (13 mp4)
├── joyvasa/
│   ├── natural_raw/  (13 mp4)
│   └── tts_raw/      (13 mp4)
└── vexpress/
    ├── natural_raw/  (13 mp4)
    └── tts_raw/      (13 mp4)
```

Wav2Lip 和 MuseTalk 结果仍在旧服务器 `/data/wjj/tfg_exp/generation/`。
