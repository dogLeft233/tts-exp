---
title: TTS原生优势的机制与训练效应
type: note
permalink: tts-exp/research/tts-tfg-inference-training-proposal-zh
status: proposal
project: tts-exp
date: '2026-10-19'
evidence_cutoff: '2026-10-19'
presentation: meeting
tags:
- proposal
- tts
- talking-face-generation
- mechanism
- training
---

# 从 TTS 原生评分优势到模型训练

> **主线：复现评分优势 → 定位参与因素 → 检验训练影响。**

TTS＝文本转语音；TFG＝音频驱动说话人脸生成。下文“音量”专指整段声音的整体幅度。

## 1. Abstract

| 环节 | 核心内容 |
|---|---|
| **Background** | 说话人脸生成需要让嘴部运动准确匹配语音；可靠的性能评价与训练数据选择，是提升生成效果的基础。 |
| **Existing method** | |
| **Problem** | 主流生成模型与同步评价器，是否存在尚未充分识别和控制的语音来源／音量依赖？这种潜在偏置可能出现在推理、评分或训练中，干扰模型能力比较与训练数据选择。 |
| **Solution** | 从自然／TTS比较出发，固定视频检查评分器、固定原音轨检查生成模型，区分输入变化影响哪个环节。再保持编码器权重不变、切换输入标准化，并交叉比较训练与推理音量，定位作用来源、检验训练阶段的影响。 |
| **Experiments** |  |
| **Contribution / implication** |  |

## 2. Method

### 2.1 三个问题与假设

| 问题 | 可检验假设 | 当前证据 |
|---|---|---|
| **Q1：TTS 驱动是否存在可重复观察的评分优势？** | 多个数据集、TFG与TTS设置中，TTS比自然语音获得更高分 | 已完成多种设置下的自然／TTS比较 |
| **Q2：什么因素让TTS获得更高分，为什么？** | 音量差参与优势；编码器对音量的处理影响后续输出 | 已统一两类声音的音量，比较多种编码器，并只改编码方向或长度生成视频 |
| **Q3：这些因素也会影响模型训练吗？** | 训练音量会改变模型面对测试声音时的表现 | Patrick三种音量的粗训练及评分已完成；更新次数不等，先报告趋势；TTS训练未完成 |

### 2.2 已有贡献

- **音量差确实参与TTS高分：** 统一音量后tts相比自然音频的优势会缩小。
- **音量是否改变编码，取决于模型怎样处理输入：** 已比较11个预训练编码器版本及Wav2Lip，并对8个版本单独关闭输入标准化。
- **训练与推理的音量偏好：** 训练时音量较低的数据集有利于模型获得较高sync C分数，推理时驱动音频音量较高有利于获得较高sync C分数。

## 3. Experiments

### 3.1 Q1｜原生评分优势已在多个设置复现

**表1｜自然／TTS 各自驱动、各自音轨评分；差值为 TTS−自然**

| 数据／设置 | 生成器 | 配对数 | ΔSync-C ↑ | ΔSync-D ↓ |
|---|---|---:|---:|---:|
| AISHELL-1，历史协议 | Ditto | 100 | +1.022 | −0.069 |
| 五数据集，历史协议 | Ditto | 250 | +0.980 | +0.117 |
| 与上一行相同的来源样本 | LeapTalk | 248 | +1.256 | −0.338 |
| LRS3，四种 TTS、官方评分 | Wav2Lip | 72／36来源组 | +0.539 | −0.166 |

五数据集＝MEAD／VFHQ／TalkVid／LRS3／GRID。英文四 TTS＝云Qwen、本地Qwen、IndexTTS-2、CosyVoice2；各18对、子集不同，汇总先来源组内平均。

> **结论：如表1，原生 Sync-C 优势可重复观察，但两个同步指标并非总是同时改善。**

**表1a｜历史多TFG对比中，TTS平均Sync-C提高的模型**

| TFG模型 | 自然语音 Sync-C | TTS Sync-C | 均值差 ↑ |
|---|---:|---:|---:|
| Wav2Lip | 7.393 | 8.922 | +1.529 |
| IMTalker | 6.235 | 7.369 | +1.134 |
| V-Express | 5.975 | 6.506 | +0.531 |
| MuseTalk 1.5 | 5.789 | 6.326 | +0.537 |
| Ditto | 5.670 | 6.819 | +1.149 |
| JoyVASA | 4.936 | 5.591 | +0.655 |
| EchoMimic V2 | 2.980 | 3.165 | +0.185 |
| SoulX-FlashHead Pro | 6.034 | 6.774 | +0.740 |
| LeapTalk | 6.222 | 7.221 | +0.999 |
| AVTR-1 | 5.460 | 5.788 | +0.328 |

AISHELL-113条输入

### 3.2 Q2｜音量差参与TTS高分，影响取决于编码器的输入处理

**表2｜把自然语音与TTS调到相同音量，分差会怎样？**

| 语音样本与操作 | 原TTS分差 | 统一音量后分差 | 分差缩小；99%置信区间 |
|---|---:|---:|---|
| 15人／71对：每对自然／TTS调到相同音量 | +0.438 | +0.266 | +0.172；[+0.041,+0.315] |
| 另外20人／40对：统一到事先选定的音量 | +0.530 | +0.259 | +0.272；[+0.137,+0.424] |

**音量目标：** 15人组每对调到自然／TTS原始RMS的几何平均值（各对可不同）；20人组全部调到 **RMS≈0.03768**。RMS衡量声音的整体幅度。

> **音量差是TTS高分的一部分原因。** 如表2，两组说话人统一音量后，TTS分差都缩小。

**变化不只发生在评分器：** 在20人组上，只调整生成视频时使用的自然声音，仍用原音轨评分，ΔC＝**+0.250**

**表3｜编码变化能否影响生成？在Wav2Lip中分开改变方向和长度**

```mermaid
flowchart LR
    A["声音变大<br/>内容、时刻不变"] --> M["Wav2Lip输入频谱改变"] --> Z["音频编码方向改变"] --> V["生成视频"] --> S["仍用原自然音轨评分"]
```

编码方向＝各维特征的相对组成；编码长度＝特征向量的整体大小。两项可以分开替换，检验哪项改变了视频评分。

| 只改变哪一项 | ΔSync-C | 99%置信区间 |
|---|---:|---|
| 方向改为声音放大后的方向，长度不变 | +0.161 | [+0.094,+0.232] |
| 长度改为声音放大后的长度，方向不变 | +0.020 | [−0.006,+0.046] |

> **Wav2Lip中，单独改变方向就能提高评分。** 40人实验中，方向效应比长度效应高0.142

**表4｜同一句话只改变音量，不同编码器会怎样？**

| 编码器 | 输入怎样处理 | 放大声音后的编码方向 |
|---|---|---|
| Wav2Lip、WavLM base-plus、Whisper base／small | 保留原有处理 | 明显改变 |
| HuBERT英／中／large、wav2vec2／XLS-R、data2vec、WavLM large（7个版本） | 保留输入标准化 | 变化很小 |
| 上述7个版本，权重不变 | 只关闭输入标准化 | 变化明显增大 |
| W2vBERT2，权重不变 | 关闭输入频谱标准化 | 仍几乎不变；后续逐帧标准化继续抵消音量变化 |

编码器＝把声音转成模型使用的数字特征；输入标准化＝按声音自身的统计量调整尺度，可消除大部分整体音量差。

### 3.3 Q3｜Patrick已完成三种音量粗训练；目前报告观察结果

**表6｜Patrick第10轮粗训练：仍用原自然音轨评分**

| 训练音量 | 推理−6dB C | 推理0dB C | 推理+6dB C |
|---|---:|---:|---:|
| −6dB | 8.106 | 8.213 | 8.158 |
| 0dB | 7.797 | 8.068 | 8.086 |
| +6dB | 7.336 | 7.770 | 7.994 |

SyncTalk／AVE编码器、Patrick一位人物、一次随机初始化；各60000次训练尝试，固定第10轮粗训练模型，在六个10秒测试片段上取均值；
 
**结论：**

- **较小训练音量对应更高Sync-C：** 本次Patrick粗训练的三个模型，在相同驱动／评分音量下，均为训练−6＞0＞+6；完整27种组合中，这一排序都保留。
- **较大推理音量呈提高Sync-C的趋势：** 固定原自然音轨评分，驱动音量从−6升至+6dB，三个模型的C分别增加 **+0.053／+0.289／+0.658**；中间各档并非严格单调上升。
- **评分音量也会改分：** 固定九条视频，仅将评分音量−6→+6dB，C变化范围为−0.131～+0.039；视频本身未变。

### 3.4 后续实验｜先核验音量归一化，再做TTS训练

| 阶段 | 怎么比较 | 回答什么 |
|---|---|---|
| **1：核验归一化在训练与推理中的利弊** | 先完成当前Patrick归一化试验；比较不归一化／仅训练时归一化／仅推理时归一化／两阶段均归一化，采用同一套归一化规则 | 能否减少不同音量下的表现波动？正常音量下的评分、口部重建和训练稳定性是否付出代价？ |
| **2：开展TTS训练实验** | 确定音量处理规则后，比较自然原视频／自然视频经相同对齐转码／真实视频换TTS音轨并对齐；统一在留出的自然音视频上测试 | 相比自然数据及相同处理流程，TTS训练数据是否带来额外收益？ |

**归一化看收益也看代价：** 同时报正常音量表现、跨音量／跨录制波动、口部重建误差与训练稳定性；不只比较分数是否更稳定。[E11]

**TTS先对齐再训练：** 检查实际发音时刻、重复／跳帧及有效训练时长；保留自然视频经过相同处理的对照，区分TTS与对齐转码的作用。[E12]

两阶段均固定配对初始参数与训练预算，核对实际更新次数，并在更多人物、重复训练及不同录制上验证；主要评分使用原自然音轨。

**结果表头：** 训练数据／归一化设置｜原音量表现｜跨音量波动｜跨录制表现｜口部重建误差｜训练稳定性。

### 3.5 关键图表规格

| 图 | 横轴 | 纵轴／颜色 | 结果状态 |
|---|---|---|---|
| 图1：原生优势 | 数据集／TFG／TTS分面 | 配对ΔC与ΔD，各批次单列 | 已有；来源分组区间有据才画 |
| 图2：统一音量是否缩小TTS优势 | 调整前／调整后 | TTS−自然分差及置信区间 | 已完成；对应表2 |
| 图3：音量如何影响编码与视频 | 编码器／只改方向或长度 | 编码方向变化／ΔC，分面展示 | 已完成；对应表3、5 |
| 图4：声音放大多少，视频变化多少 | 增益dB | ΔC、D0、口部开度分面 | 另20人已完成 |
| 图5：Patrick粗训练的音量比较 | 推理dB | 训练dB，颜色为C；评分音量分面 | 27种组合已完成；训练次数限制随图标注 |

统计以说话人／身份为单位；种子为身份内重复，不将帧或片段数当独立样本。95%、99%及同时区间按原实验分别标注，不跨队列计算优势解释百分比。

## 参考文献

1. Prajwal et al. **Wav2Lip**. ACM MM 2020. [论文](https://arxiv.org/abs/2008.10010)。
2. Peng et al. **SyncTalk**. CVPR 2024. [论文](https://arxiv.org/abs/2311.17590) · [官方代码](https://github.com/ZiqiaoPeng/SyncTalk)。
3. Salehi et al. **Comparative Analysis of Audio Feature Extraction for Real-Time Talking Portrait Synthesis**. BDCC 2025；自然／合成及音频前端已有比较。 [正式版](https://www.mdpi.com/2504-2289/9/3/59) · [预印本](https://arxiv.org/abs/2411.13209)。
4. Smeu et al. **Circumventing Shortcuts in Audio-visual Deepfake Detection Datasets with Unsupervised Learning**. CVPR 2025；前置静默为真假分类捷径，不直接证明本项目的生成偏置。[论文](https://arxiv.org/abs/2412.00175)。

<details>
<summary>证据索引与已完成实验核对</summary>

| 编号 | 已核对来源 |
|---|---|
| E1 | [AISHELL100报告15](../docs/experiments/15-tfg-link.md)、[五数据集](<../Experiments/跨数据集 TFG 测评（5×50 multiset）.md>)、[跨TFG评分汇总](../docs/results/SYNC-C-SCORES.md) |
| E2 | [英文四TTS、来源分组复算与排除记录](论文实验数据整理与写作主线.md) |
| E3 | [15人：分别调整驱动与评分声音的音量](<../Experiments/原生全局电平与谱组合生成评价四格 2026-09-27.md>)；[报告46第7节](../docs/experiments/46-amplitude-gain-independent-validation.md) |
| E4 | [另外20位女性：统一音量后的结果](<../Experiments/女性说话人上的 TTS 来源方向独立复核.md>) |
| E5 | [报告25：幅度、编码方向与谱形机制](../docs/experiments/25-tts-tfg-mechanism.md) |
| E6 | [报告46：另20人放大／缩小声音，160个视频](../docs/experiments/46-amplitude-gain-independent-validation.md) |
| E7 | [只加入多个人共有的编码变化，再在另外10人上测试](../Experiments/增益共同方向与残差的留出说话人生成验证.md) |
| E8 | [报告52：11个预训练编码器与Wav2Lip的比较](../docs/experiments/52-encoder-direction-commonality.md)、[报告53：8个编码器关闭输入标准化](../docs/experiments/53-encoder-input-normalization-ablation.md) |
| E9 | [前期小规模实验：May三种训练音量](../docs/experiments/50-synctalk-may-gain-training-pilot.md)，保留记录，不作本文训练主表 |
| E10 | [前期小规模实验：May训练／驱动／评分音量](../docs/experiments/51-synctalk-may-eval-gain.md)，保留记录，不作本文训练主表 |
| E11 | [Patrick已有27种组合的评分](<../docs/results/SyncTalk Patrick coarse10 seed42 27格描述性预览.md>)、[Patrick统一音量后重新训练](<../Experiments/Patrick RMS归一化同初始化独立复跑.md>)、[四位人物的完整训练方案与失败记录](<../Experiments/SyncTalk 四新身份官方完整训练电平实验 Spec.md>) |
| E12 | [NAS十人重对齐](<../Experiments/NAS 短窗口 10 说话人扩大样本试验 2026-09-24.md>)、[Ubuntu历史核查](../codex/tts-exp/Codex%20checkpoint%20-%202026-10-16T10-52-00Z%20-%20Training%20amplitude%20design%20and%20Ubuntu%20history%20verified.md) |

**模型／TTS覆盖已核对：** 早期 Wav2Lip、IMTalker、V-Express、MuseTalk、Ditto、JoyVASA、EchoMimicV2、Hallo2、LatentSync；后续 FlashHead、LeapTalk、AVTR-1、AvatarForcing、StableAvatar；另有 Hallo3、Seedance 补充协议。各自版本、失败分母及近零／负向结果见E1，不拼成统一排名。中文早期另有 IndexTTS-2、GPT-SoVITS 的 Ditto 对比；英文四TTS见E2。F5／ChatTTS的特征或部署记录不冒充下游验证。

**已有结果，不重复安排：** 统一自然／TTS音量、在另外20位女性上重复、80句自然音频的驱动／评分音量比较、只改编码方向或长度、20人放大／缩小声音、移植共同编码变化、多编码器比较及关闭标准化、Patrick粗训练27种组合评分；May前期实验保留历史记录。

**其他已完成路线：** [处理链预测审计43](../docs/experiments/43-processing-chain-predictive-audit.md)、[原生因素筛查39](../docs/experiments/39-native-input-factor-mechanism-screen.md)、[MFA停顿修复33](../docs/experiments/33-mfa-linear-content-pause-prosody-mapping.md)、[后续复核36](../docs/experiments/36-mfa-linear-heldout-speech-residual.md)不承担当前主线；其失败不能推出相应因素普遍无效。

[局部时序／谱形全因子](<../Experiments/等电平等总长的局部时钟声源谱包络全因子 2026-09-28.md>)已完成，时序有正响应，但原生波形→WORLD重合成的等效检验未通过；只能在该重合成域解释。LibriSpeech按已有资格决定排除；公共数据不自动视为无预训练重叠。

核对截至2026-10-19的BM报告与修订；本次为文献和证据整理，未新增训练、生成或评分。

## Observations

- [scope] Q1复现原生评分现象；Q2解释音量参与及编码变化；Q3主文采用Patrick粗训练描述结果，训练次数限制保留，TTS训练仍待验证。
- [decision] 已完成实验进入证据；在途、未完成确认与条件性扩展分列，不重复安排。

## Relations

- synthesizes [[研究交接总览]]
- follows [[从 TTS 高分到可控驱动：唇形生成的输入机制与迁移边界]]
- supported_by [[25-tts-tfg-mechanism]]
- supported_by [[46-amplitude-gain-independent-validation]]
- supported_by [[SyncTalk Patrick coarse10 seed42 27格描述性预览]]
- relates_to [[50-synctalk-may-gain-training-pilot]]

## Changelog

| 日期 | 修订 |
|---|---|
| 2026-10-19 | 按三问新建投屏版；核对已完成、在途和未执行证据，沿用Abstract／Method／Experiments模板。 |
| 2026-10-19 | 保留用户删除的章节；将实验简称改成实际操作与结论，突出多编码器的音量处理结果，并与Wav2Lip生成实验分开。 |
| 2026-10-19 | 训练主表改用Patrick第10轮粗训练，联动修改摘要、Q3、结论及图规格；May转为历史来源，保留Patrick训练次数不等的限制。 |

</details>