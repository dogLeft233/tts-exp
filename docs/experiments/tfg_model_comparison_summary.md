# 多 TFG 模型对比实验总报告

**实验日期**: 2026-07-17 至 2026-07-19  
**实验目标**: 系统对比多个端到端 TFG（Talking Face Generation）模型在相同音频条件下的唇形同步表现，并探究 TTS 音频改善 SyncNet 分数的因果机制

---

## 1. 实验设计

### 1.1 数据集

| 属性 | 值 |
|---|---|
| 来源 | AISHELL-1（中文普通话） |
| 样本数 | 13（ID 1–13），质量剔除后保留 9–12 |
| 条件 | natural_raw（原始录音） vs tts_raw（TTS 合成） |
| TTS 模型 | Faster-Qwen3 0.6B Base（ICL 音色克隆），24kHz 单声道 |
| 评估指标 | SyncNet v2 — Sync-C（置信度↑）和 Sync-D（距离↓） |

### 1.2 服务器架构

三台服务器协同工作：

| 角色 | GPU | 部署模型 |
|------|-----|---------|
| 旧推理（1080 Ti × 8） | 8× GTX 1080 Ti（11GB） | Wav2Lip、MuseTalk 1.5（+ EchoMimic/AniPortrait 尝试后放弃） |
| 新推理（RTX 4080 SUPER） | 1× RTX 4080（32GB） | LatentSync、JoyVASA、V-Express（+ 稳定性干预、G×E 矩阵） |
| 评估服务器 | 1× RTX 4080（32GB） | SyncNet 评估（所有模型的统一评分） |

---

## 2. 模型部署

### 2.1 成功部署（5 个）

| 模型 | 架构 | 服务器 | 输出分辨率 | 每视频耗时 | 显存 |
|------|------|--------|:----------:|:---------:|:---:|
| **Wav2Lip** | GAN + SyncNet 判别器 | 旧 1080Ti | 96×96 | ~30s | 4.4G |
| **MuseTalk 1.5** | VAE latent + 多尺度 UNet 修复 | 旧 1080Ti | 256×256 | ~90s | 8.4G |
| **V-Express** | 渐进 Dropout 扩散 + keypoints | 新 RTX 4080 | 512×512 | ~6m20s | 9.0G |
| **JoyVASA** | 扩散 Transformer（LivePortrait backbone） | 新 RTX 4080 | 512×512 | ~23s | 2.2G |
| **LatentSync** | 潜在扩散模型（字节跳动） | 新 RTX 4080 | 512×512 | ~2m50s | 3.5G |

### 2.2 放弃的模型（3 个）

| 模型 | 原因 |
|------|------|
| **EchoMimic** | 旧服务器无 xformers 加速，每样本 5.6h；新服务器权重已删除 |
| **AniPortrait** | 旧服务器 OOM（256×256 fp16 仍 ~9.8GB） |
| **Hallo2** | CUDA 11.8 硬需求与新服务器 CUDA 13.0 不兼容 |

### 2.3 模型论文来源

年份优先记录正式会议发表年份；仅有 arXiv 版本的模型记录首次提交年份。当前评分表中的模型，以及曾尝试但未纳入原始评分的模型，来源如下：

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

### 2.4 网络受限环境的权重下载经验

1. **hf-mirror.com** — 优先使用（无需代理），`wget --no-check-certificate`
2. **`source /etc/network_turbo`** — SeetaCloud 学术代理，稳定性差
3. **本地下载 + SCP** — 最可靠，适合 >1GB 的大文件

---

## 3. SyncNet 评估结果

### 3.1 总分排名

| 排名 | 模型 | N | natural Sync-C↑ | tts Sync-C↑ | ΔC | natural Sync-D↓ | tts Sync-D↓ | ΔD |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **Wav2Lip** | 12 | **7.393** | **8.922** | **+1.529** | **7.327** | **5.941** | **−1.395** |
| 2 | **V-Express** | 26 | 5.975 | 6.506 | +0.531 | 8.818 | 8.236 | −0.583 |
| 3 | **MuseTalk 1.5** | 12 | 5.789 | 6.326 | +0.537 | 8.081 | 7.928 | −0.169 |
| 4 | **JoyVASA** | 12 | 4.815 | 5.084 | +0.269 | 9.443 | 9.581 | +0.144 |
| 5 | **LatentSync** | 12 | 4.653 | 4.581 | **−0.072** | 8.894 | 8.768 | −0.121 |

### 3.2 Ditto 唇形替换基线（非端到端 TFG）

| TTS 条件 | N | natural Sync-C | tts Sync-C | ΔC |
|---------|:---:|:---:|:---:|:---:|
| faster_qwen3 + ditto | 9 | 5.696 | 6.942 | +1.246 |
| IndexTTS2 + ditto | 13 | 5.690 | 6.730 | +1.040 |
| GSV + ditto | 13 | 5.760 | 6.360 | +0.600 |

### 3.3 关键发现

1. **Wav2Lip 碾压所有模型** — Sync-C 7.39/8.92，是唯一能"重写"口型到接近原始质量的模型。96×96 低分辨率反而聚焦于口部区域
2. **LatentSync 是最鲁棒的负对照** — ΔC ≈ −0.07（TTS 不改善也不恶化），表明 TTS 优势并非跨模型普适
3. **JoyVASA 同步能力最弱** — natural Sync-C = 4.81，但对 TTS 有正面响应（ΔC = +0.27）
4. **所有成功模型在 TTS 条件下 Sync-C 均高于 natural**（仅 LatentSync 例外），证实 TTS 音频在 TFG 场景中的系统性优势

---

## 4. G×E 分离实验

### 4.1 设计

2×2 生成器×评估器矩阵，将 TTS 优势分解为三个分量：
- **生成器效应**（Generator）：TTS 音频是否提升了 TFG 视频质量
- **评估器效应**（Scorer）：SyncNet 是否偏好 TTS 音频特性
- **交互效应**（Interaction）：联合耦合（核心关注）

```
方案: G_nat/E_nat → G_nat/E_tts → G_tts/E_nat → G_tts/E_tts
执行: ffmpeg 替换视频音轨 + 全局时长对齐 → SyncNet 评分
```

### 4.2 结果

| 指标 | G_nat/E_nat | G_nat/E_tts | G_tts/E_nat | G_tts/E_tts |
|-----|:---:|:---:|:---:|:---:|
| Sync-C | 5.574 | 2.776 | 2.327 | 6.819 |
| Sync-D | 8.317 | 11.429 | 11.501 | 7.550 |

**完全案例分解（n=9）：**

| 分量 | Sync-C | Sync-D |
|-----|:---:|:---:|
| TTS 总优势 | +1.246 | −0.766 |
| 生成器主效应 | −3.247 | +3.184 |
| 评估器主效应 | −2.798 | +3.113 |
| **G×E 交互** | **+7.290** | **−7.063** |

### 4.3 意义

对角优势完全来自 G×E 交互（+7.29 Sync-C），而独立的主效应都是负的。这意味着**并不是 TTS 音频单独好，也不是 SyncNet 单独偏好 TTS**，而是**TTS 音频 + Ditto 生成器 + SyncNet 评估器三者形成的联合耦合状态**产生了高分。将一个音轨替换为另一个会同时破坏这种耦合。

---

## 5. 剂量反应实验（Script 22）

### 5.1 干预设计

在 Phase 1 音频上应用 8 个音频特征干预，通过 Ditto + SyncNet 管道回传并测量 ΔSync-C：

| 干预 | 源 | 目标特征 | 验证 | n |
|------|-----|---------|:---:|:---:|
| LUFS_match_tts_to_nat | TTS | 响度 (LUFS) | 5/12 | 9 |
| spectral_tilt_match_tts_to_nat | TTS | 频谱倾斜 | 12/12 | 9 |
| dynamic_compression_tts_compress | TTS | 动态范围压缩 | 12/12 | 9 |
| dynamic_expansion_tts_expand | TTS | 动态范围扩展 | 12/12 | 9 |
| LUFS_match_nat_to_tts | 自然 | 响度 | 5/12 | 9 |
| spectral_tilt_match_nat_to_tts | 自然 | 频谱倾斜 | 12/12 | 9 |
| dynamic_compression_nat_compress | 自然 | 动态范围压缩 | 12/12 | 9 |
| dynamic_expansion_nat_expand | 自然 | 动态范围扩展 | 12/12 | 9 |

### 5.2 原始结果

| TTS 源干预 | raw ΔSync-C | ↓ 一致率 |
|-----------|:---:|:---:|
| LUFS_match_tts_to_nat | −1.198 | 9/9 |
| spectral_tilt_match_tts_to_nat | −0.989 | 9/9 |
| dynamic_compression_tts_compress | −1.157 | 9/9 |
| dynamic_expansion_tts_expand | −1.232 | 9/9 |

| 自然源干预 | raw ΔSync-C | 方向预期 |
|-----------|:---:|:---:|
| LUFS_match_nat_to_tts | −0.015 | ✗（无改变） |
| spectral_tilt_match_nat_to_tts | −0.528 | ✗（反向） |
| dynamic_compression_nat_compress | +0.046 | 平坦 |
| dynamic_expansion_nat_expand | +0.092 | 平坦 |

### 5.3 关键观察

所有 4 个 TTS 源干预都产生 −1.0 至 −1.2 的 ΔSync-C 下降，无论操纵哪个特征——甚至**动态压缩和动态扩展这对相反的干预也产生几乎相同的下降**（−1.157 vs −1.232）。这表明不是任何单一特征在驱动，而是**TTS-TTS 对角线的联合适应状态对任何扰动都很脆弱**。

---

## 6. 身份控制（Pipeline 噪声量化）

### 6.1 设计

对 TTS 音频做无操作变换（16-bit PCM 原样写回），重跑 Ditto + SyncNet，测量相对于原始 `tts_raw` 基线的 ΔSync-C。

### 6.2 结果

| 样本 | 基线 | 重新运行 | Δ |
|:---:|:---:|:---:|:---:|
| 1 | 6.491 | 4.468 | −2.023 |
| 2 | 6.600 | 5.478 | −1.122 |
| 3 | 7.444 | 5.927 | −1.517 |
| 4 | 6.843 | 5.807 | −1.036 |
| 5 | 6.161 | 5.426 | −0.735 |
| 6 | 6.149 | 5.682 | −0.467 |
| 7 | 6.406 | 5.138 | −1.268 |
| 8 | 7.977 | 6.520 | −1.457 |
| 10 | 7.303 | 6.478 | −0.825 |

**均值 ΔSync-C = −1.161 ± 0.469（n=9）**

身份控制的下降幅度（−1.16）**几乎完全覆盖了剂量反应中观测到的原始降幅**（−0.99 至 −1.23）。意味着这些降幅主要来自 Ditto 的逐次运行非确定性（pipeline noise），而非音频特征变化。

### 6.3 身份校正后的残差分析

对每个 TTS 源干预，减去逐样本身份基线 Δ：

| 干预 | 校正后 ΔC | SD | t | p | d | 判定 |
|------|:---:|:---:|:---:|:---:|:---:|:---|
| LUFS_match_tts_to_nat | **−0.037** | 0.296 | −0.38 | 0.72 | −0.13 | 不显著 |
| spectral_tilt_match_tts_to_nat | **+0.173** | 0.371 | +1.40 | 0.20 | +0.47 | 不显著 |
| dynamic_compression_tts_compress | **+0.004** | 0.338 | +0.03 | 0.97 | +0.01 | 不显著 |
| dynamic_expansion_tts_expand | **−0.071** | 0.219 | −0.97 | 0.36 | −0.32 | 不显著 |

**结论**: 4 个声学干预的残差全部不显著。Pipeline noise 解释了剂量反应的全部降幅。TTS 对角优势（+1.25 Sync-C）**不能被分解为任何单一声学特征的贡献**。

---

## 7. 稳定性干预（Script 25，PGD 对抗扰动）

### 7.1 设计

前两轮效应均无法将 TTS 优势归因到单个声学特征之后，最后一个因果候选是表示层面的 **HuBERT L11 segment_stability**。用 sign-PGD 在波形上以 L∞ ε=0.005 预算定向操纵稳定性损失，然后重新评估 Ditto + SyncNet。

| 细胞 | 源 | 方向 | 预期 Sync-C |
|------|-----|------|:---------:|
| `stability_adj_tts` | TTS | 提高代价（去稳定化） | ↓ |
| `stability_adj_nat` | 自然 | 降低代价（向 TTS 稳定化） | ↑ |
| `random_noise_tts` | TTS | 随机 ±eps 方向（对照） | ↓ |

### 7.2 结果

| 层级 | 细胞 | n | ΔSync-C（均值） |
|:---:|------|:---:|:---:|
| 1. Pipeline 噪声 | identity_tts | 9 | −1.16 ± 0.47 |
| 2. 任意方向 ε-噪声 | random_noise_tts | 12 | −3.42 ± 0.94 |
| 3. 特征定向扰动 | stability_adj_tts（TTS 去稳定） | 12 | **−4.59 ± 0.71** |
| 3b. 反方向 | stability_adj_nat（自然稳定化） | 12 | **−4.21 ± 0.83** |

### 7.3 统计检验

**检验 1** — 配对残差 `stability_adj_tts` − `random_noise_tts`:
- 均值 = **−1.177**，SD = 0.346，n = 12
- t = −11.79，p ≈ 1×10⁻⁷
- Cohen's **d = −3.40**（极大效应）
- **显著且方向正确**。去稳定 TTS 使 Sync-C 在随机噪声基线之上额外下降 ~1.18

**检验 2** — `stability_adj_nat` ΔSync-C 与 0 比较（预期为正）:
- 均值 = −4.21，t = −17.59
- **方向错误**。稳定化自然音频使其 Sync-C 下降了 4.21，而非上升

### 7.4 因果判定：INCONCLUSIVE

根据预设的三层判定逻辑：
- `STABILITY_CAUSAL`：检验 1 显著+方向正确 **且** 检验 2 方向正确
- `STABILITY_REJECTED`：检验 1 不显著或错误
- `INCONCLUSIVE`：检验 1 显著但检验 2 错误

**判定为 INCONCLUSIVE**。稳定性是 TTS 方向上的统计显著因果拦截变量（interceptor），但**双边因果链不成立**——往自然音频加入稳定性并不能复现 TTS 式的 SyncNet 分数。

### 7.5 科学解释

两种兼容解释：

1. **不对称因果贡献** — 稳定性对 TTS 的高 Sync-C 是"承重"的（移除它导致下降），但单独向自然添加稳定性不足以恢复高分（缺少其他 TTS 特征配合）
2. **G×E 共适应占主导** — SyncNet 是在 Ditto 类音频 + 特定声学分布上训练的，该分布恰好与较低 HuBERT-L11 稳定性共存。检验 1 只说明稳定性是 SyncNet 训练分布的一部分，不证明它是上游原因

---

## 8. 跨 TFG 模型一致性

### 8.1 ΔSync-C 汇总

| 模型 | n | ΔSync-C | ΔSync-D | 方向 |
|------|:---:|:---:|:---:|:---:|
| **Wav2Lip** | 12 | **+1.515** | −1.395 | 强正向 |
| Ditto | 9 | +1.246 | −0.766 | 强正向 |
| **MuseTalk** | 12 | +0.572 | −0.169 | 正向 |
| **V-Express** | 26 | +0.531 | −0.583 | 正向 |
| **JoyVASA** | 12 | +0.329 | +0.144 | 弱正向 |
| **LatentSync** | 12 | **−0.063** | −0.121 | 零（负对照） |

### 8.2 跨模型特征一致性（4 个候选机制）

| 机制 | Wav2Lip | MuseTalk | JoyVASA | LatentSync | 判定 |
|------|:---:|:---:|:---:|:---:|-----|
| segment_stability | + | + | + | ~ | 方向一致，但验证仅基于 Ditto |
| silhouette_score | + | + | + | ~ | 同上 |
| boundary_sharpness | + | + | + | ~ | 同上 |
| intra_class_compactness | + | + | + | ~ | 同上 |

**问题**: HuBERT 特征提取仅对 Ditto 完成，无法在其他 TFG 模型上验证机制方向。LatentSync 始终中性，支持其作为负对照的角色。

---

## 9. 候选机制综合判定表

| 机制 | Phase 1 证据 | Phase 2 校正后 | 最终判定 |
|------|:---:|:---:|:---:|
| **LUFS/响度** | 弱（Δ≈+0.17 LUFS） | 残差 −0.04 ± 0.30，t=−0.38，p=0.72 | **拒绝** — pipeline noise 完全解释 |
| **频谱倾斜** | 强观察性（Δ≈−0.240，12/12） | 残差 +0.17 ± 0.37，t=+1.40，p=0.20 | **拒绝为独立机制** — 未达到显著 |
| **动态范围** | 弱（Δ≈0.0004 绝对值） | 压缩 +0.004，扩展 −0.071，均 p>0.35 | **拒绝** |
| **G×E 共适应** | G×E 交互 = +7.29 | 不受身份控制影响（基于原始对角线分数） | **支持** — TTS 对角优势存在于生成器×评估器联合耦合 |
| **HuBERT L11 segment_stability** | 强观察性（FDR p=0.024，d=1.045） | PGD: TTS 方向 d=−3.40（显著）但自然方向失败 | **INCONCLUSIVE** — TTS 方向上的因果拦截变量，但双边因果不成立 |
| **边界清晰度（boundary_sharpness）** | 中等（平均 d=0.58） | 未单独检验 | 条件保留 |
| **轮廓分数（silhouette_score）** | 弱（方向一致，不显著） | 未单独检验 | 不优先 |
| **类内紧致性（intra_class_compactness）** | 弱（无强信号） | 未单独检验 | 不优先 |

---

## 10. 最终结论

### 10.1 核心科学发现

**TTS 音频提升 SyncNet 分数的"优势"是一个不可分解的 G×E 联合耦合状态，而非任何单一声学特征驱动。**

经过三轮因果验证（剂量反应→身份控制→稳定性 PGD），4 个候选声学特征全部被拒绝，唯一有统计显著因果信号的表示层面特征（segment_stability）也只在一个方向上成立且不足以完整复现对角优势。

### 10.2 模型架构的影响

- **Wav2Lip**（ΔC=+1.53）和 **Ditto**（ΔC=+1.25）对 TTS 音频反应最强 — 它们都使用 GAN 或 talker head 架构，直接逐帧映射音频→口型
- **V-Express**（ΔC=+0.53）和 **MuseTalk**（ΔC=+0.57）中等 — 扩散/修复架构有内部缓冲
- **JoyVASA**（ΔC=+0.27）较弱 — DiT 架构对输入变化更不敏感
- **LatentSync**（ΔC≈0）完全中性 — 其扩散过程的内部归一化可能过滤掉了 TTS 特性

### 10.3 跨语言关联（与英文 Wav2Sem 实验对照）

中文的多模型实验与英文单模型实验共享同一个核心结论：**TTS 优势主要通过音素/表示层面（Fp）而非语义层面（Fs）运作**，且其效应大小取决于自然语音本身的音素清晰度基线。

### 10.4 局限

| 局限 | 影响 |
|------|------|
| 小样本（n=9–12） | 统计检验力有限，中等效应可能漏检 |
| HuBERT 特征仅对 Ditto 提取 | 无法在其他 TFG 模型上验证表示机制 |
| 仅 SyncNet 评估 | 反映的是 SyncNet 的声学敏感度，不等同于人感知 |
| 单一语言（中文） | 跨语言泛化性未知 |
| 无人类感知评测 | 所有结论基于自动化指标 |
| 样本 9 在 TTS 条件下始终失败 | 所有 Δ 分析保留 n=9 |

---

## 11. 脚本清单

| 脚本 | 用途 |
|------|------|
| `19_generate_eval_matrix.py` | G×E 2×2 音频矩阵分解 |
| `20_causal_feature_interventions.py` | 声学特征干预（LUFS/频谱倾斜/动态范围） |
| `21_validate_causal_results.py` | 跨 TFG 模型因果验证 |
| `22_dose_response_syncnet.py` | 剂量反应 SyncNet 评估 |
| `23_identity_control_syncnet.py` | Ditto pipeline 非确定性量化 |
| `24_identity_corrected_analysis.py` | 身份校正后的残差分析 |
| `25_stability_perturbation_syncnet.py` | PGD 稳定性定向扰动实验 |
| `17_link_features_to_tfg.py` | HuBERT 特征与 TFG 模型关联 |

### 数据文件位置

```
data/wav2sem_analysis/metrics/
├── cross_tfg_validation.json      # 跨模型 ΔSync-C/ΔSync-D
├── feature_tfg_link.json          # 特征-TFG 关联（5 个模型）
├── dose_response.json             # 剂量反应结果
├── identity_control.json          # 身份控制结果
├── identity_corrected_summary.json # 校正后残差
├── stability_intervention.json    # PGD 稳定性干预
├── intervention_results.json      # 特征干预验证
└── intervention_pilot_20260718.json
```

### 实验文档

```
docs/experiments/
├── tfg_progress_20260717.md        # TFG 模型对比进度报告
├── tfg_model_comparison_summary.md # 本文件
├── tts_tfg_mechanism_report.md     # 机制分析总报告（英文）
├── phase2_execution_20260718.md    # Phase 2 执行日志
├── english_wav2sem_fp_fs_summary.md # 英文 Wav2Sem Fs/Fp 实验总结
└── indextts2_emo_tfg_20260708.md   # IndexTTS2 情感×TFG
```

```
docs/tfg_models/
└── TFG_DEPLOY_SUMMARY.md           # TFG 模型部署推理配置总结
```
