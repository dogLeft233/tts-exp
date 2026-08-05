# 06 — 2026 TFG 模型候选调查

**调查日期**: 2026-07-30
**目的**: 筛选可加入 AISHELL-1 `natural_raw` vs `tts_raw` 评估的 2026 年 TFG 模型。
**状态**: ✅ 初筛完成；AVTR-1、SoulX-FlashHead、AvatarForcing、LeapTalk 和 StableAvatar 已批量完成；其余新增候选已完成资料核验，尚未部署。

## 判定标准

- **I2V TFG**：参考图 + 外部音频生成视频，可直接与现有静态肖像 TFG 分支比较。
- **V2V lip-sync**：输入已有视频 + 外部音频，只进入独立的唇形同步分支，不与 I2V 结果混排。
- **可运行**：官方代码和可下载 checkpoint 均已找到；仅有论文或 demo 不算可运行。
- “2026 模型”优先按论文/技术报告或公开模型首次出现年份记录；若论文是 2025、代码/权重在 2026 发布，则单独标注。

## 可直接纳入 I2V 评估的候选

| 模型 | 2026 来源 | 输入/输出 | 官方代码与权重 | 资源与风险 | 评估建议 |
|------|-----------|-----------|----------------|------------|----------|
| **SoulX-FlashHead** | [arXiv 2602.07449](https://arxiv.org/abs/2602.07449)；代码和权重 2026-02-12 发布 | 单张肖像图 + 音频 → 无限长度 talking head | [GitHub](https://github.com/Soul-AILab/SoulX-FlashHead)；[HF 1.3B](https://huggingface.co/Soul-AILab/SoulX-FlashHead-1_3B)；Lite/Pro 已发布，Pretrained 尚未发布 | 官方报告 Lite 在单 RTX 4090 达 96 FPS，Pro 约 10.8 FPS；仓库使用 `facebook/wav2vec2-base-960h`，中文效果需单独验证 | **A：最先尝试**；优先 Pro，若显存/速度不足再用 Lite |
| **AvatarForcing（KlingAIResearch）** | [arXiv 2603.14331](https://arxiv.org/abs/2603.14331)，2026-03 | 参考图 + 语音 + 文本 prompt → 一步流式视频；I2V | [GitHub](https://github.com/KlingAIResearch/AvatarForcing)；[HF](https://huggingface.co/lycui/AvatarForcing) | 1.3B；论文设置 25 FPS、832×480、约 34 ms/frame；本批次实际使用 512×512；当前仓库明确只支持 I2V；使用 `facebook/wav2vec2-base-960h` | **A：第二优先**；固定中性 prompt，单独记录分辨率和 prompt 影响 |
| **LongCat-Video-Avatar 1.5** | [arXiv 2605.26486](https://arxiv.org/abs/2605.26486)；2026-05-21 发布 | 支持单音频、音频+文本、音频+图像、视频续写；本实验只使用音频+图像 I2V | [GitHub](https://github.com/meituan-longcat/LongCat-Video)；[HF 1.5](https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.5) | Whisper-large-v3、多任务 DiT、8-step distillation；支持 480P/720P 和 v1.5 INT8；模型负载明显高于现有 4080 基线 | **B：高价值候选**；若可使用高显存服务器，优先加入 |
| **SoulX-LiveAct** | [arXiv 2603.11746](https://arxiv.org/abs/2603.11746)；代码和权重 2026-03-16 发布 | 参考图 + 音频 + 文本 → 长时/流式人像视频 | [GitHub](https://github.com/Soul-AILab/SoulX-LiveAct)；[HF](https://huggingface.co/Soul-AILab/LiveAct) | 18B（14B Wan2.1 + 4B audio module）；官方报告单 RTX 5090 offload 约 6 FPS，2×H100/H200 才达到 20 FPS；使用中文 `chinese-wav2vec2-base` | **C：方法上适合、硬件上暂不适合** |

### 直接候选的关键兼容性

- **中文音频**：LongCat 1.5 使用 Whisper-large-v3；LiveAct 使用中文 wav2vec2，二者更适合 AISHELL-1。FlashHead 和 Kling AvatarForcing 默认下载英文 `wav2vec2-base-960h`；FlashHead 已在当前部署记录中完成 AISHELL-1 批量推理并得到 `ΔC=+0.739`，但这个结果不能未经测试地外推到其他模型或条件。
- **文本条件**：LongCat、AvatarForcing 和 LiveAct 支持文本或 prompt；应使用固定的中性描述，不能把 prompt 差异当成模型差异。
- **任务模式**：LongCat 只以音频+图像 I2V 模式进入本调查；其视频续写模式必须另列为 V2V，不能与本表结果混排。
- **当前实验硬件**：FlashHead 和 Kling AvatarForcing 最可能适配现有 RTX 4080；LongCat 需要先做短片 smoke test；LiveAct 应暂列为高显存服务器候选。

## 不应混入同一 I2V 排名的模型

| 模型 | 类型 | 已核验信息 | 处理方式 |
|------|------|------------|----------|
| **HighSync** | V2V lip-sync | [arXiv 2605.16918](https://arxiv.org/abs/2605.16918)；[代码](https://github.com/saeed5959/high_sync)；[checkpoint](https://huggingface.co/saeed-5959/high_sync)；输入视频+音频，原生 512×512 | 可另建 V2V 组；不能与单参考图 I2V 结果直接比较 |
| **Lip Forcing** | V2V lip-sync | [arXiv 2606.11180](https://arxiv.org/abs/2606.11180)；[官方代码](https://github.com/cvlab-kaist/LipForcing) 和 [14B checkpoint](https://huggingface.co/JinhyukJang/lipforcing) 已于 2026-07-07 发布；1.3B student 尚未发布 | 论文声称 1.3B 为 31 FPS，但当前可用权重是 14B；独立 V2V 组，当前 32GB 不部署 |
| **SoulX-FlashTalk** | V2V/长时流式 avatar | 技术报告 [arXiv 2512.23379](https://arxiv.org/abs/2512.23379)；代码和权重 2026-01-08 发布；[代码](https://github.com/Soul-AILab/SoulX-FlashTalk)；[14B HF](https://huggingface.co/Soul-AILab/SoulX-FlashTalk-14B) | 技术报告输入包含 motion/source video context 和 reference frame；单卡需要超过 64GB，官方实时结果依赖 8×H800；独立记录，不算 2026 首发 I2V |
| **Avatar Forcing（TaekyungKi）** | 交互式 avatar | [arXiv 2601.00664](https://arxiv.org/abs/2601.00664)；[代码](https://github.com/TaekyungKi/AvatarForcing)；推理需要参考图、avatar 音频和 user video | 与 KlingAIResearch 的同名模型分开记录；进入交互式分支 |
| **SEDTalker** | 3D facial animation | [arXiv 2604.13335](https://arxiv.org/abs/2604.13335)；[代码](https://github.com/FarzanehJafari1987/SEDTalker) | 进入 3D/avatar 分支；不直接参与 2D 视频 Sync-C 排名 |
| **MOVA** | 联合音视频生成 | [arXiv 2602.08794](https://arxiv.org/abs/2602.08794)；[代码](https://github.com/OpenMOSS/MOVA) | 主要是文本/图像驱动的音视频联合生成，不是外部音频驱动 TFG |

## 论文-only 或尚未确认公开权重

初次检索阶段未找到可确认官方 checkpoint 的 2026 论文包括 **TurboTalk**（[2604.14580](https://arxiv.org/abs/2604.14580)）、**TempoSyncDiff**（[2603.06057](https://arxiv.org/abs/2603.06057)）、**UniTalking**（[2603.01418](https://arxiv.org/abs/2603.01418)）、**AsymTalker**（[2605.02948](https://arxiv.org/abs/2605.02948)）和 **GaussianEmoTalker**（[2607.00959](https://arxiv.org/abs/2607.00959)）。这些论文可作为文献综述候选，但不应在“可复现实验模型”清单中标为已部署。

## 初次检索推荐接入顺序（2026-07-28）

1. **SoulX-FlashHead Pro**：输入协议最接近现有静态肖像 TFG，模型规模较小；先用 1 个 AISHELL-1 样本验证中文音频和输出格式。
2. **KlingAIResearch AvatarForcing**：1.3B、I2V、官方 checkpoint 已公开；需要固定 prompt，并检查英文 wav2vec2 对普通话的影响。
3. **LongCat-Video-Avatar 1.5**：中文音频兼容性更有吸引力，但应先做 480P、短时长、INT8 smoke test。
4. **SoulX-LiveAct**：保留为高显存服务器实验，不在现有 RTX 4080 阶段阻塞主线。
5. **HighSync / Lip Forcing**：若扩展研究问题，再建立独立 V2V lip-sync 对照组。

## 统一评估要求

- 保持 13 个 AISHELL-1 样本、`natural_raw`/`tts_raw` 两个条件和现有 SyncNet V2 评分流程不变。
- I2V 模型统一使用同一参考肖像；有文本条件的模型使用固定中性 prompt，并在结果表中记录 prompt。
- LongCat 固定使用音频+图像 I2V，不使用视频续写入口。
- 将输出统一转为评估脚本接受的帧率和视频格式；记录原生分辨率，不把重采样后的结果当作原生质量。
- 每个模型记录显存、每样本耗时、失败样本、实际音频编码器和 checkpoint 变体。
- I2V、V2V 和 3D/avatar 结果分组报告；不要把任务差异解释成单一模型架构差异。

## 初次检索结论（截至 2026-07-28）

截至 2026-07-28，已确认 4 个支持 I2V 且具有官方代码和公开权重的候选。最现实的新增实验是 **SoulX-FlashHead** 和 **KlingAIResearch AvatarForcing**；最值得在高显存服务器上验证的是 **LongCat-Video-Avatar 1.5**。初次检索阶段没有把候选分数加入现有模型排名。

## 二次检索（2026-07-29）

本轮通过 Tavily 轮询检索、arXiv 页面、GitHub API 和 Hugging Face 模型卡复核了 2026 年新增条目。重点区分“论文/代码存在”和“有可直接下载的推理 checkpoint”。

### 新确认的可运行候选

| 模型 | 2026 来源 | 输入协议 | 代码/权重 | 当前判断 |
|------|-----------|----------|-----------|----------|
| **SkyReels-V3 A2V-19B** | [arXiv 2601.17323](https://arxiv.org/abs/2601.17323)；代码和权重 2026-01-29 发布 | 单张图 + 单条音频 → talking avatar，最长 200s | [GitHub](https://github.com/SkyworkAI/SkyReels-V3)；[HF 19B](https://huggingface.co/Skywork/SkyReels-V3-A2V-19B) | **高价值但重**；HF 仓库约 55.97GB，19B 参数，支持 `--low_vram`/FP8/offload；现有 50GB 数据盘不够，需更大磁盘和更长 smoke test |
| **AVTR-1** | 2026-05 发布的开放权重模型 | 单张肖像图 + 音频 → 25 FPS 流式 talking head；可选第二条 listening 音频 | [GitHub](https://github.com/avaturn-live/avtr-1)；[HF](https://huggingface.co/avaturn-live/avtr-1) | **最适合先做 smoke test**；权重仓库约 1.94GB，TensorRT 推理，官方 RTX 4060 Ti 已达实时；需注意 Renderer/Streamer 为 PolyForm Noncommercial，InsightFace 预训练权重也有非商用限制 |
| **Talker-T2AV** | [arXiv 2604.23586](https://arxiv.org/abs/2604.23586)；ACM MM 2026 | 文本 + 参考音频 + 身份图/视频 → 联合生成音频和视频；同一 checkpoint 也声明支持 A2V/V2A | [GitHub](https://github.com/zhenye234/Talker-T2AV)；[HF](https://huggingface.co/HKUSTAudio/Talker-T2AV) | **独立联合生成组**；HF 权重约 9.69GB、中文支持；默认会重新生成输出音频，不能把生成视频直接混入当前“固定外部音频”的 Natural/TTS 排名 |
| **Live Avatar** | [arXiv 2512.04677](https://arxiv.org/abs/2512.04677)；ECCV 2026 Oral | 参考图 + 音频/文本 → 无限时长流式 avatar | [GitHub](https://github.com/Alibaba-Quark/LiveAvatar)；[HF](https://huggingface.co/Quark-Vision/Live-Avatar) | **高显存候选**；14B，FP8 版本最低约 48GB、单卡模式约 80GB；不适合当前 RTX 4080 32GB，但应保留在高显存服务器清单 |

### 二次检索中排除或降级的条目

- **TempoSyncDiff**（[arXiv 2603.06057](https://arxiv.org/abs/2603.06057)）：官方 GitHub 已存在，但 README 明确不含训练 checkpoint；代码是参考训练/推理骨架，不能直接复现论文结果。
- **AsymTalker**（[arXiv 2605.02948](https://arxiv.org/abs/2605.02948)）：论文声称 600 秒一致生成和 66 FPS，但尚未找到可核验的官方代码或权重。
- **GaussianEmoTalker**（[arXiv 2607.00959](https://arxiv.org/abs/2607.00959)）：有项目网页仓库，但当前仓库只包含网页素材，没有推理代码或 checkpoint。
- **SyncCache**（[arXiv 2606.30849](https://arxiv.org/abs/2606.30849)）：是针对 HunyuanVideo-Avatar/Wan-S2V 的训练-free 加速方法，不是独立 TFG/THG 模型。
- **PersonaLive**（[CVPR 2026](https://github.com/GVCLab/PersonaLive)）：代码和权重已公开，但当前入口是参考图 + driving video，不是外部音频驱动，不能直接加入本项目 I2V 音频比较。
- **FantasyTalking2**（AAAI 2026，[代码](https://github.com/Fantasy-AMAP/fantasy-talking2)）：论文和研究代码已确认，但仓库 README 没有可核验的 FantasyTalking2 推理 checkpoint；暂不标记为可运行模型。

### 更新后的接入顺序

本列表以 AVTR-1 作为完成进度锚点，其余条目是二次检索时仍待决定的候选；SoulX-FlashHead、KlingAIResearch AvatarForcing 和 LeapTalk 已完成部署、批量推理及 SyncNet 评分，因此不再作为待接入项重复列出。AvatarForcing 的 26/26 批量记录、约 42s/样本、18–20GB 显存和评分记录见 `docs/deployment/avatarforcing.md`。Talker-T2AV 和 Live Avatar 仍保留为独立的联合生成/高显存候选。

1. ~~**AVTR-1**~~ ✅ 2026-07-29 完成：26/26 条推理成功，输出 1280×720 720P；SyncNet Nat C/TTS C 为 5.460/5.788（ΔC=+0.327），详见 `docs/deployment/avtr-1.md` 和 `results/avtr1/`。
2. **SkyReels-V3 A2V-19B**：作为高质量新一代 diffusion TFG；换用大数据盘后做 480P/540P `--low_vram` smoke test，再决定是否跑 13×2。⚠️ HF 仓库约 55.97GB，当前数据盘 50GB 不够。
3. **Talker-T2AV**：建立独立的联合音频-视频生成实验，记录其重新生成的音频，不与固定音频 TFG 排名混排。
4. **LongCat-Video-Avatar 1.5**：继续作为中文音频友好的重型 I2V 候选。
5. **Live Avatar**：保留到 48GB/80GB 以上服务器，不在当前 32GB 机器上投入调试时间。

## 三次检索（2026-07-30）

本轮针对“2026 年新模型 + 现有 RTX 4080 SUPER 32GB 可部署性”复核了官方代码、模型卡和依赖。判断将模型质量潜力与本项目的 50GB 数据盘、中文音频、单图 I2V 输入协议分开记录。

### 新增候选

| 模型 | 2026 来源 | 输入协议 | 代码/权重与资源 | 当前部署判断 |
|------|-----------|----------|-----------------|--------------|
| **LeapTalk** | [ViBT, arXiv 2511.23199](https://arxiv.org/abs/2511.23199)；代码和权重 2026-07-29 发布 | 单张参考图 + 音频 → chunk-by-chunk 流式 talking head；1 NFE | [GitHub](https://github.com/zhangrongxiang/LeapTalk)；[HF](https://huggingface.co/z-rx/leaptalk)；基于 SoulX-FlashHead 1.3B；新增 LoRA、audio projector 和 TAE 权重合计约 385MB；官方环境为 Python 3.12、PyTorch 2.7.1 | **A：已完成并评分**；2026-07-30 完成 26/26 条推理和 SyncNet 评分；Nat C/TTS C 为 6.222/7.221（ΔC=+0.999）；512×512、25 FPS；~40s/样本、~8–9GB VRAM；需 mediapipe 0.10.21；结果已下载至 `results/leaptalk/`（详见 `docs/deployment/leaptalk.md`） |
| **StableAvatar** | 原始研究模型非 2026 首发；[Docker 封装 v1.0.0 于 2026-01-04 发布](https://github.com/neosun100/StableAvatar) | 单张参考图 + 音频 → 无限长度 talking avatar；512×512/480×832 | [部署仓库](https://github.com/neosun100/StableAvatar)；Wan2.1 1.3B；官方文档约 30GB checkpoint 磁盘；本次 full load 峰值约 27.4GB | **✅ 已完成**：26/26 条推理和 SyncNet 评分；Nat/TTS C 为 3.938/3.914；评分使用 `min_track=25` |
| **AptAvatar** | [arXiv 2607.24013](https://arxiv.org/abs/2607.24013)；代码 2026-06-29 发布，checkpoint 2026-07-27 更新 | 单张参考图 + 中文音频 + prompt → 长时 I2V；2 NFE | [GitHub](https://github.com/TaoLiveAIGC/AptAvatar)；[HF](https://huggingface.co/TaoLiveAIGC/AptAvatar)；18.882B BF16 参数；HF 仓库约 54.5GB；依赖 PyTorch 2.8.0/cu128、xformers、optimum-quanto，使用 `chinese-wav2vec2-base` | **B：研究价值高，当前暂缓**；完整权重已超过 50GB 数据盘，32GB 显存峰值未知；需更大磁盘后再做量化/offload smoke test |

### 4080 部署判定

- **下一步顺序**：AptAvatar（扩容磁盘后）→ LongCat-Video-Avatar 1.5；StableAvatar 已完成。
- **环境隔离**：LeapTalk 要求 Python 3.12/PyTorch 2.7.1；AptAvatar 要求 PyTorch 2.8.0/cu128；不要修改现有 SoulX、SyncNet 或 TFG 环境。
- **中文音频风险**：AptAvatar 官方使用 `chinese-wav2vec2-base`；LeapTalk 和 StableAvatar 仍使用 `facebook/wav2vec2-base-960h` 或兼容路线，必须用 AISHELL-1 单样本验证后再批量。
- **磁盘约束**：当前 50GB 数据盘无法完整容纳 AptAvatar HF 仓库；StableAvatar 的约 30GB需求仍需扣除 Docker/环境/输出空间，部署前应先检查 `df -h /root/autodl-tmp`。
- **评估边界**：LeapTalk、StableAvatar、AptAvatar 均可进入单图 I2V 组；LipForcing 仍只能进入独立 V2V 组。

### 更新后的接入顺序

本列表针对当前 RTX 4080/50GB 数据盘的下一步；LeapTalk 已完成因此不再作为待测项；SkyReels、Talker-T2AV 和 Live Avatar 仍在调查范围内，但分别受磁盘、任务定义或显存约束，不是本轮的首选。

1. ~~**AVTR-1**~~ ✅ 已完成 26/26 条推理和 SyncNet 评分（Nat C/TTS C=5.460/5.788）。
2. ~~**LeapTalk**~~ ✅ 已完成 26/26 条推理和 SyncNet 评分（Nat C/TTS C=6.222/7.221）；512×512、25 FPS；~40s/样本，详见 `docs/deployment/leaptalk.md`。
3. ~~**StableAvatar**~~ ✅ 已完成 26/26 条推理和 SyncNet 评分；Nat/TTS C=3.938/3.914，详见 `docs/deployment/stableavatar.md`。
4. **AptAvatar**：先扩容到至少 70GB 数据盘，再测试 2-step、量化/offload 和显存峰值。
5. **SkyReels-V3 / LongCat-Video-Avatar 1.5**：保留给更大磁盘或更高显存服务器。
6. **LipForcing**：只在建立 V2V 对照组且获得 48GB 以上显存时部署。

## 当前结论（截至 2026-07-30）

AVTR-1、AvatarForcing、LeapTalk 和 StableAvatar 已完成 26/26 批量推理及 SyncNet 评分：AVTR-1 的 ΔC 为 +0.327，AvatarForcing 为 −0.110，LeapTalk 为 +0.999，StableAvatar 为 −0.024。StableAvatar 使用 `min_track=25`，因此先作为补充结果记录；AptAvatar 等候选仍待部署。
