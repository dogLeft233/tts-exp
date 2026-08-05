# SoulX-FlashHead 部署记录

## 状态

- **状态**: ✅ 官方 Pro smoke test 和 AISHELL-1 26 条批量推理完成
- **容器**: `connect.weste.seetacloud.com:43794`
- **目标模型**: `Model_Pro`
- **官方代码**: [SoulX-FlashHead](https://github.com/Soul-AILab/SoulX-FlashHead)
- **官方权重**: [SoulX-FlashHead-1_3B](https://huggingface.co/Soul-AILab/SoulX-FlashHead-1_3B)
- **官方环境**: Python 3.10、PyTorch 2.7.1 + CUDA 12.8、FlashAttention 2.8.0.post2
- **当前运行环境**: 复用已有 `echomimic_v2` 的 `torch 2.5.1+cu124` CUDA 基础，仅将 FlashHead Python 包覆盖安装到数据盘；未修改 EchoMimic 原环境

## 磁盘检查

最近一次远端检查结果：

| 路径 | 文件系统 | 总容量 | 可用 | 说明 |
|------|----------|-------:|-----:|------|
| `/` | overlay | 30G | 18G | 不用于模型和 pip/HF 缓存 |
| `/root/autodl-tmp` | XFS `/dev/md0` | 50G | 34G | 首选部署位置；已有 `echomimic_v2` 不删除 |
| `/autodl-pub` | NFS | 7.3T | 5.3T | 空间备用位置 |

根分区不能容纳完整 HF 仓库（约 20.36GB）并安全保留环境和缓存。部署时必须将模型、环境、HF cache、pip cache 和输出统一放到数据盘。

## 选择性下载

Pro 只需要以下模型文件：

| 文件 | 大小 | 用途 |
|------|-----:|------|
| `Model_Pro/diffusion_pytorch_model.safetensors` | 6,030,864,656 bytes | Pro checkpoint |
| `VAE_Wan/Wan2.1_VAE.pth` | 507,609,880 bytes | Pro 使用的 Wan VAE |
| `facebook/wav2vec2-base-960h/model.safetensors` | 377,607,901 bytes | 音频编码器 |

Pro 核心文件合计约 6.9GB。不要下载 `Model_Lite` 或 `VAE_LTX`，除非后续需要 Lite 对照；完整仓库同时包含两种模型和两个 VAE。

建议目录布局：

```text
/root/autodl-tmp/flashhead/
├── repo/
├── models/SoulX-FlashHead-1_3B/
├── models/wav2vec2-base-960h/
├── hf-cache/
├── pip-cache/
└── outputs/
```

当前已完成：

- 代码已固定在 commit `9bc03de`。
- `Model_Pro`、`VAE_Wan` 和 wav2vec 最小文件已下载并通过本地读取检查。
- FlashHead Python 覆盖层位于 `/root/autodl-tmp/flashhead/site-packages`。
- 代码 `compileall`、FFmpeg 检查和非 GPU 依赖导入通过。
- `xfuser` 无卡时会在导入阶段探测 `torch.cuda.get_device_name()`；因此无卡状态不能做完整 pipeline import，这属于预期限制。

官方 PyTorch 2.7.1 wheel 在该容器网络中多次大文件传输中断，因此没有继续反复下载；当前复用的 CUDA 12.4 PyTorch 基础需在开卡后做实际兼容性验证。

## 推理入口

官方 Pro 单 GPU 入口：

```bash
bash inference_script_single_gpu_pro.sh
```

开卡后的运行准备命令：

```bash
cd /root/autodl-tmp/flashhead/repo
export PATH=/root/miniconda3/envs/echomimic_v2/bin:$PATH
export PYTHONPATH=/root/autodl-tmp/flashhead/site-packages
export HF_HOME=/root/autodl-tmp/flashhead/hf-cache
export CUDA_VISIBLE_DEVICES=0
nvidia-smi
bash inference_script_single_gpu_pro.sh
```

开启 GPU 后先执行：

```bash
nvidia-smi
```

然后只运行一个 AISHELL-1 样本，记录显存峰值、耗时、输出分辨率、帧率和失败信息，再决定是否扩展到 13 个样本。

## 资源预期与风险

- 官方报告：Pro 在单张 RTX 4090 上约 10.8 FPS；实时 25+ FPS 需要两张 RTX 5090。
- 32GB 显存已完成单卡 Pro 推理验证，无 OOM。
- 官方默认使用英文 `facebook/wav2vec2-base-960h`；AISHELL-1 普通话效果必须单独验证，不能直接沿用中文模型的预期。

## Smoke Test

- **时间**: 2026-07-28
- **GPU**: NVIDIA GeForce RTX 4080 SUPER，32,760 MiB
- **输入**: 官方 `examples/girl.png` + `examples/podcast_sichuan_16k.wav`
- **输出**: `/root/autodl-tmp/flashhead/repo/sample_results/res_20260728-15:03:41-217.mp4`
- **结果**: `inference_exit=0`；512×512、25 FPS、65.92s；H.264 视频 + 16kHz 单声道 AAC
- **文件大小**: 4,145,676 bytes；`ffmpeg` 解码检查通过
- **速度**: 首个 chunk 约 185.8s（编译/预热），之后约 3.77s/chunk
- **注意**: 未安装 `flash_attn`，实际使用 PyTorch attention；功能已跑通，但速度不是官方优化上限
- **补齐依赖**: `audioread`、`numba==0.59.1`、`llvmlite==0.42.0`、`msgpack`、`soupsieve`

## AISHELL-1 Batch

- **输入**: 13 个 `natural_raw` + 13 个 `tts_raw`；样本 ID `1..13`，包含样本 9
- **TTS 来源**: `runs/r2_faster_qwen3_20260707T145233Z/02_tts/`，`faster_qwen3` 0.6B ICL
- **结果**: 26/26 成功，无失败；全部 MP4 已在本地完成解码检查
- **峰值显存**: 8.106 GiB
- **本地结果**: `results/soulx_flashhead/{natural_raw,tts_raw}/{1..13}.mp4`
- **本地 manifest**: `results/soulx_flashhead/batch_manifest.json`

## SyncNet 评分

- **结果目录**: `results/soulx_flashhead/04_eval/`
- **成功**: 25/26 个视频；12/13 个完整配对
- **配对均值**: Natural Sync-C `6.034`，TTS Sync-C `6.774`，ΔC `+0.739`；Sync-D Δ `−0.594`
- **失败项**: `tts_raw:9`；视频约 4.0 秒，检测到人脸但连续轨迹不足，未生成可评分 track。未用补静音方式伪造分数。
