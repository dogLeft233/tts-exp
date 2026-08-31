# TTS-TFG Audio Replacement Experiments

本仓库研究一个严格的问题：TTS 或音频增强是否能让冻结的 talking-face generator（TFG）生成更好的唇形同步视频，并且在最终把音轨换回原始自然音频后，增益仍然存在。

> 当前结论先说：**TTS 驱动的视频有时优于 natural 驱动的视频，但这不等价于 replacement-safe 增益。** 截至 2026-08-31，直接重合成、自然视频上的相似度以及单样本 replacement 训练都没有证明一个可泛化的 replacement-safe 音频头。

## 研究目标与唯一成功判据

对同一原始自然音频 `N`、同一人脸/源视频 `F` 和同一冻结 TFG `M`：

```text
C = audio_enhancement_head(N, optional TTS conditioning)
V_C = M(F, C)
Target   = mux(V_C, N)
Baseline = mux(M(F, N), N)
```

最终只接受 official file-level SyncNet 的同条件配对结果：`Target` 相对 `Baseline` 的 Sync-C 上升、Sync-D 降低，并且在 held-out clips/source groups 上通过预先固定的统计判据。以下现象单独都不算成功：

- `M(F, C) + C` 自洽分数较高；
- 固定真实视频上的 proxy 或 SyncNet loss 改善；
- candidate 音频与 TTS/natural 的特征距离变小；
- 只在训练记录、单个 speaker 或单一 TFG 上有效。

## 最近结果

### 1. WavLM→HiFi-GAN 的自编码式重合成不能保持 replacement

这里更准确地说是**冻结声学重合成链**，不是训练好的无损 autoencoder：natural waveform → frozen WavLM-Large layer 6 → prematched frozen HiFi-GAN。LRS3 n=50 的 150 个 cell 全部完成，mux 后 decoded PCM 与输入一致，但重合成音频并没有成为可替换的原始波形：

| 视频/音轨 | Sync-C ↑ | Sync-D ↓ |
|---|---:|---:|
| natural video + natural audio | 7.228 | 7.359 |
| direct video + direct audio | 7.128 | 7.420 |
| direct video + natural audio（strict replacement） | 6.997 | 7.590 |

相对 natural video + natural audio，strict replacement 的 Sync-C 变化为 **−0.231**，Sync-D benefit 为 **−0.232**；只有 6/7/4（C/D/joint）记录获胜。重合成音频的 WavLM feature cosine distance 平均为 0.126，也不能推出 waveform 可替换性。结论是：保留高层 SSL 表征或完成无损 mux，不足以解除 TFG 对生成视频时所用音频的声学/波形适配。

详见：`basic-memory/Experiments/LRS3 WavLM-HiFi-GAN direct resynthesis replacement.md`、`basic-memory/Experiments/AISHELL-1 n100 WavLM direct resynthesis decoder audit 2026-08-16.md`。

### 2. 自然视频上的 SyncNet 接近，不保证生成视频 replacement 保持

固定自然视频监督确实能在音频层面得到小幅改善：15 条记录的 official fixed-video audit 为 Sync-C **+0.050**、Sync-D **−0.095**，两项各 10/15 改善。但同一类 candidate 音频独立送入 Wav2Lip 后，换回 original natural audio 的增益几乎消失：Sync-C 只剩 **+0.015**，而 Sync-D 反而恶化 **+0.170**。speaker 分组的换轨 Sync-C 变化为 S0914 **+0.384**、S0915 **−0.245**、S0916 **−0.093**，迁移并不稳定。

因此，“两段音频在自然视频上 SyncNet 分数相近”或“candidate 在固定自然视频上表现好”，都不能证明 candidate 驱动的视频可以换回自然音频后仍保持增益。真正要测的是同一 TFG 的 `Target` 与 `Baseline` 的 official replacement 配对比较。

详见：`basic-memory/Experiments/Encoder-only SyncNet 微调与固定视频迁移性.md`、`basic-memory/Experiments/Direct WavLM 重合成 2×2 音轨 Identity Control.md`。

### 3. Wav2Lip + SyncNet 的单样本 replacement 增强有效，但泛化失败

P1 原型在被训练和选择的单条记录上得到 `F D=6.7253 < A D=6.79`，看起来 replacement 优于自然基线。随后按 source-group 完全分离的协议验证：20 条 train、8 条 val、12 条 unseen test，WavLM、HiFi-GAN、Wav2Lip 和 SyncNet 全部冻结。

- val 选择的 step 40 在 proxy paired gate 上为 4/8；
- official unseen test 只有 **3/12 favorable**，低于预注册的 7/12 门槛；
- `f_beats_a` 为 **0/12**，没有任何记录满足 F 的两次 D 都优于自然直接驱动；
- 判定：`GENERALIZATION_FAIL`；小幅 D 改善常伴随 C 同步变差。

同一实验的三臂消融还没有证明 TTS cross-attention 是贡献来源：

| 条件 | unseen test favorable |
|---|---:|
| TTS-kv | 2/10 = 0.2 |
| NAT-kv | 4/10 = 0.4 |
| SHUF-kv | 4/10 = 0.4 |

`rate_tts − rate_nat = −0.2`，且 TTS-kv 低于 SHUF-kv，判定为 `TTS_CONTRIBUTION_NOT_CONFIRMED`。这说明单样本的 improvement 是过拟合构造，当前 adapter 没有在 unseen records 上学到可迁移的 TTS 条件信息。

详见：`basic-memory/docs/experiments/30-replacement-audio-head-validation.md`。

## 当前边界与下一步

- **已确认的正结果**：在规整中文 read-speech 上，TTS 驱动 TFG 的 generation Sync-C 可以优于 natural；这不能自动升级为 replacement-safe 结果。
- **当前 replacement 结论**：direct resynthesis、fixed-video supervision 和 LRS3 replacement audio-head 验证均未证明可泛化的 `Target > Baseline`。
- **当前在研方向**：phone-aligned cross-attention。Week 1 诊断在 Exp 30 的 split 上得到 CKA **0.383**、val probe accuracy **0.941**，表明 TTS/natural WavLM 特征可分；同时 Exp 30 的帧级 attention 近似均匀，支持先修复 phone-level 对齐再做 Week 2 验证。该诊断是方向性证据，不是 replacement 成功结果。
- 当前协议草案：[`docs/specs/replacement-salvage-protocol.md`](docs/specs/replacement-salvage-protocol.md)。

## Basic Memory 文档系统

本项目使用 Basic Memory 作为跨会话知识图谱，而不是把实验进度散落在聊天记录或临时 Markdown 中，github链接：`https://github.com/basicmachines-co/basic-memory`：

- **项目记忆**：MCP 项目 `tts-exp`，本地源文件为 `basic-memory/`；记录本仓库的实验、任务、决策、部署和结果。
- **全局记忆**：MCP 项目 `main`，只用于跨项目的通用知识；项目结论优先写入 `tts-exp`，避免重复。
- **启动路由**：任何知识库读写前先读 `Startup Router`，按任务类型加载对应指令。
- **搜索顺序**：关于历史实验或决策，先在 `tts-exp` 用全文 `search_type="text"` 搜索；项目内无结果时再查 `main`。
- **实验位置**：正式记忆笔记在 `basic-memory/Experiments/`；编号报告/结果镜像在 `basic-memory/docs/experiments/`。负结果也要记录，不能用 progress 日志替代。
- **关系与复现**：笔记用 `[[Wiki Link]]` 连接实验、任务和决策；仓库中的 `runs/` 保存可审计的数字产物，git log 保存代码变更历史。

典型查阅顺序（MCP 伪代码）：

```text
read_note(identifier="memory://instructions/startup-router", project="tts-exp")
search_notes(query="replacement audio-head", search_type="text", project="tts-exp")
read_note(identifier="tts-exp/docs/experiments/30-replacement-audio-head-validation.md", project="tts-exp")
build_context(url="memory://tts-exp/experiments/...", project="tts-exp")
```

这里的 `project="tts-exp"` 是必要的路由参数；不要把项目笔记误写入 `main`。新增笔记前先搜索同名/近义标题，已有实体应更新而不是重复创建。
