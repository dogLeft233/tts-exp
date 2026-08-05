# 02 — TFG 跨模型对比

## 问题

TTS 对唇形同步的提升是否跨 TFG 模型泛化？

## 方法

固定 TTS 引擎（faster_qwen3 0.6B ICL），对比 9 个 TFG 模型在相同 13 个 AISHELL-1 样本上的 SyncNet 表现。

## 结果

| 排名 | 模型 | 论文年份 | Nat C | TTS C | ΔC | 分辨率 | 推理时间 |
|:---:|------|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | Wav2Lip | 2020 | 7.39 | **8.92** | **+1.53** | 96×96 | ~30s |
| 2 | **IMTalker** 🆕 | 2025 | 6.24 | 7.37 | +1.13 | 512×512 | ~30s |
| 3 | V-Express | 2024 | 5.98 | 6.51 | +0.53 | 512×512 | ~6m20s |
| 4 | Hallo2 🆕 | 2025 | 5.92 | 5.85 | −0.07 | 512×512 | ~8m |
| 5 | MuseTalk 1.5 | 2024 | 5.79 | 6.33 | +0.54 | 256×256 | ~90s |
| 6 | Ditto (基线) | 2025 | 5.67 | 6.82 | +1.15 | 256×256 | ~20s |
| 7 | JoyVASA | 2024 | 4.94 | 5.59 | +0.66 | 512×512 | ~23s |
| 8 | LatentSync | 2024 | 4.65 | 4.58 | −0.07 | 512×512 | ~2m50s |
| 9 | EchoMimic V2 🆕 | 2025 | 2.98 | 3.17 | +0.19 | 512×512 | ~35s |

> 🆕 = 2026-07-25 之后新增；Hallo2 使用无 `Kim_Vocal_2` 人声分离版本，详见 `docs/results/SYNC-C-SCORES.md`。

## 模型论文来源

年份优先记录正式会议发表年份；仅有 arXiv 版本的模型记录首次提交年份。若 arXiv 首发年份与正式发表年份不同，两者同时标出。

| 模型 | 对应论文 | 来源年份 |
|------|---------|---------:|
| Wav2Lip | [A Lip Sync Expert Is All You Need for Speech to Lip Generation In The Wild](https://arxiv.org/abs/2008.10010) | **ACM MM 2020** |
| IMTalker | [IMTalker: Efficient Audio-driven Talking Face Generation with Implicit Motion Transfer](https://arxiv.org/abs/2511.22167) | arXiv **2025** |
| V-Express | [V-Express: Conditional Dropout for Progressive Training of Portrait Video Generation](https://arxiv.org/abs/2406.02511) | arXiv **2024** |
| Hallo2 | [Hallo2: Long-Duration and High-Resolution Audio-Driven Portrait Image Animation](https://arxiv.org/abs/2410.07718) | **ICLR 2025**（arXiv 2024） |
| MuseTalk 1.5 | [MuseTalk: Real-Time High-Fidelity Video Dubbing via Spatio-Temporal Sampling](https://arxiv.org/abs/2410.10122) | 论文首发 arXiv **2024**；v3 **2025**；**1.5 模型 2025-03-28 发布** |
| Ditto | [Ditto: Motion-Space Diffusion for Controllable Realtime Talking Head Synthesis](https://arxiv.org/abs/2411.19509) | **ACM MM 2025**（arXiv 2024） |
| JoyVASA | [JoyVASA: Portrait and Animal Image Animation with Diffusion-Based Audio-Driven Facial Dynamics and Head Motion Generation](https://arxiv.org/abs/2411.09209) | arXiv **2024** |
| LatentSync | [LatentSync: Taming Audio-Conditioned Latent Diffusion Models for Lip Sync with SyncNet Supervision](https://arxiv.org/abs/2412.09262) | arXiv **2024** |
| EchoMimic V2 | [EchoMimicV2: Towards Striking, Simplified, and Semi-Body Human Animation](https://arxiv.org/abs/2411.10061) | **CVPR 2025**（arXiv 2024） |
| EchoMimic V1 | [EchoMimic: Lifelike Audio-Driven Portrait Animations through Editable Landmark Conditions](https://arxiv.org/abs/2407.08136) | **AAAI 2025**（arXiv 2024） |
| AniPortrait | [AniPortrait: Audio-Driven Synthesis of Photorealistic Portrait Animation](https://arxiv.org/abs/2403.17694) | arXiv **2024** |

> 本实验使用 MuseTalk 1.5 checkpoint（`models/musetalkV15/unet.pth`，推理参数 `--version v15`），不是 MuseTalk 1.0。根据[官方仓库](https://github.com/TMElyralab/MuseTalk)，截至 2026-07-28 未公布 2.0 或更新的模型 checkpoint；2025-09-26 的最新代码提交是下载脚本更新，不是新模型版本。

## 分析

1. **7/9 模型对 TTS 有正向响应**——效应不是 Ditto 特有的
2. **LatentSync 和 Hallo2 接近中性**（ΔC≈−0.07）——未观察到明显的 TTS 提升
3. **Wav2Lip 最强**——尽管分辨率最低（96×96），但唇形匹配最准
4. **IMTalker 新晋第二**——512×512 分辨率下仅次于 Wav2Lip
5. **EchoMimic V2 垫底**——使用了合成静态姿态，非真实驱动视频

## 未纳入正式评分的模型

| 模型 | 原因 |
|------|------|
| EchoMimic V1 | 无 xformers 时 5.6h/样本 |
| AniPortrait | 256×256 fp16 OOM (~10GB) |
| Hallo2 官方默认 pipeline | CUDA 11.8 硬依赖，后改用无分离版本完成补充评分 |

## 状态

🔄 **进行中** — 原始跨模型表已有 9 个模型评分；AVTR-1、AvatarForcing 和 LeapTalk 的新增评分见 `docs/results/SYNC-C-SCORES.md`，后续可继续加入新模型

2026 年候选模型的代码、权重和任务边界已完成初筛，详见 [`06-2026-tfg-model-survey.md`](06-2026-tfg-model-survey.md)。其中 `SoulX-FlashHead`、KlingAIResearch 的 `AvatarForcing`、`LongCat-Video-Avatar 1.5` 和 `SoulX-LiveAct` 属于 I2V 候选；`HighSync`、`Lip Forcing` 和 `SoulX-FlashTalk` 属于 V2V/长时流式分支，不应直接混入当前排名。

## 相关文件

- `docs/results/SYNC-C-SCORES.md` — 完整排名表
- `docs/deployment/` — 各模型部署记录
