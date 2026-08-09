# 17 — TTS 特征迁移与 natural 节奏对齐深度调研

## 目的

调研如何把 TTS 音频中的有用声学特征赋予 natural 音频，同时保持 natural 的内容、phone timing 和停顿节奏。重点回答：

1. 哪些因素可以迁移：音色/谱包络、音素清晰度、F0、能量、韵律和时长；
2. 如何显式控制 natural 与 TTS 的节奏关系；
3. 现有 voice conversion、prosody transfer 和 TTS 系统哪些机制可以借鉴；
4. 哪些论文或仓库的 disentanglement/节奏保持表述不能直接作为已验证结论；
5. 对当前 `phone-local-warp` No-Go 结果应采取什么下一步。

本报告来自多角度检索、23 个来源提取、65 条候选 claim 和 25 条 adversarial verification。最终确认 6 条，反驳或降级 19 条；结论只保留被证据支持的窄命题。

## 执行摘要

最可靠的路线不是把 TTS waveform 逐 phone 重采样后拼回 natural waveform，而是把内容、时序和声学属性显式分解：

```text
natural:
    content + phone boundaries + phone duration + pause mask
TTS/reference:
    selected timbre/spectral attributes
optional:
    F0 + energy + expressive prosody

output:
    natural content
  + explicitly selected duration/pause
  + selected TTS acoustic attributes
```

核心原则：

- `duration`、`pause` 和 boundary 必须是显式变量，不能由 speaker embedding 或声码器隐式决定；
- TTS 的音色、谱包络、F0、energy 和 duration 必须用独立开关做 factorial ablation；
- 逐帧 MCD/F0-RMSE 不能单独评价节奏，必须加入 duration、pause、boundary 和 speaking-rate 指标；
- voice conversion 的“保留源韵律”或“目标音色迁移”通常是系统目标，不等于已证明的属性独立性；
- 如果推理时没有 TTS/reference 输入，模型只能学习 generic TTS-like prior，不能实例级迁移某一条 TTS 的音色、情绪或韵律。

对当前项目的结论是：

```text
No-Go：普通 phone-local waveform warp 不适合作为训练 target。
Go：继续验证 explicit-duration-controlled acoustic transfer。
```

## 1. 为什么当前 phone-local-warp 失败并不只是 coverage 问题

当前实现做的是：

```text
TTS phone waveform
    -> ordinary waveform resampling
    -> natural phone sample length
    -> overwrite natural region
```

普通 waveform resampling 会同时改变：

- pitch；
- formant 和 spectral envelope；
- phase；
- transient；
- phone 内部的非线性时间结构；
- 跨 phone coarticulation 和连续 prosody。

8 ms crossfade 只能减小局部边界 click，不能恢复被切断的 transition 和句内韵律。V2 中 high-coverage 子集仍退化，因此不能把失败归因于 MFA coverage 不足：

```text
raw TTS Sync-C delta       = +0.4175
phone-local-warp delta     = -0.4877
high-coverage local-warp   = -0.5546
```

这否定的是“逐段重采样到 natural timing 能保留 raw TTS TFG 收益”的具体 target 假设，不是否定 raw TTS→TFG 结果，也不证明所有形式的 acoustic transfer 都不可行。

## 2. 文献和开源系统的证据分级

### 2.1 高可信：显式 duration 是节奏控制接口

ACL 2025 的 R-VC 将 rhythm 纳入 zero-shot voice conversion，并通过 Mask Generative Transformer 做 in-context duration modeling，使内容 unit 的持续时长适配目标 speaking style。这支持以下窄命题：

```text
content units + explicit duration model + speaker/style condition
```

比“让 decoder 自己从 speaker embedding 中产生正确节奏”更适合作为节奏控制接口。

但 R-VC 的证据边界必须保留：

- rhythm 主要指 unit/phoneme duration 和 speaking rate，不等同于完整 expressive prosody；
- 90.2% 是三档节奏类别准确率，不是逐 phoneme duration error；
- 论文承认细粒度 duration prediction 可能出现局部过长或过短；
- 它不是 arbitrary natural waveform 与 TTS waveform 的逐帧同步系统。

因此，R-VC 可作为 duration-conditioned generator 的架构参考，不能直接作为当前 waveform enhancer 已经解决节奏对齐的证据。

### 2.2 中等可信：prosody 可作为独立条件，但不应声称完全解耦

PACE 等工作将内容、目标音色和 prosody prompt 分开作为条件，并报告了 prosody 控制与目标说话人相似度的结果。这说明 prosody-conditioned generation 是合理候选。

但现有证据不足以证明纯粹、无串扰的 prosody representation：

- 部分结果来自规模有限的 arXiv 预印本；
- 缺少充分的 reference-swap factorial test、置信区间和显著性检验；
- 韵律、音色和说话人条件仍可能互相泄漏；
- “isolate/refine prosody”通常是设计目标，不能直接写成已验证的统计独立性。

早期多因素 voice conversion 工作虽然显式编码 content、timbre、rhythm、pitch，并用 adversarial mask-and-predict 降低表示相关性，但 MAP/adversarial objective 只减少可预测性，不能保证独立。其 rhythm-only conversion 仍存在 content leakage。

### 2.3 仅部分支持：高层 voice-conversion 系统

OpenVoice、Seed-VC、Chatterbox 和 Jets 都有可借鉴组件，但不能直接解决当前任务：

| 系统 | 可借鉴部分 | 不能直接推出的结论 |
|---|---|---|
| OpenVoice | tone-color embedding、多语言/风格控制 | tone-color 接口不是参考 waveform 的逐帧 duration 轨迹 |
| Seed-VC | zero-shot speaker conversion、整段 `length-adjust` | 整句变速不等于 phone-level natural rhythm alignment |
| Chatterbox | source token 的固定 25 Hz 时间网格、speaker/reference 条件 | 25 Hz token 只提供离散时间索引，不证明准确编码 pause、duration 或完整 rhythm；target 条件也不只有 speaker identity |
| Amphion Jets | text↔acoustic frame 的 monotonic alignment、token duration | TTS 内部文本对齐不是 natural waveform↔TTS waveform 对齐 |
| FACodec/Prosody2Vec/FC-TTS | 因素化表示和多条件生成的研究方向 | README/论文中的 disentanglement 目标不等于 reference-swap 后单属性独立变化 |

尤其需要避免以下过强表述：

```text
source token length determines source rhythm
    ≠
output preserves source phone timing and pause exactly
```

```text
speaker embedding + source content tokens
    ≠
clean timbre/prosody disentanglement
```

## 3. 推荐的目标架构

### 3.1 有 TTS/reference 输入：条件式 acoustic transfer

如果推理阶段允许输入 TTS/reference audio，推荐：

```text
natural.wav
    -> content encoder
    -> natural phone/unit sequence C_n
    -> natural duration D_n and pause mask P_n

TTS/reference.wav
    -> timbre/spectral encoder
    -> selected style representation S_tts

[C_n, D_n, P_n, S_tts]
    -> duration-controlled acoustic generator
    -> mel/codec representation
    -> vocoder
    -> output.wav
```

严格 natural timing 的版本应满足：

```text
D_output = D_natural
P_output = P_natural
```

并把其他属性拆成开关：

```text
F0_output     = natural 或 TTS
Energy_output = natural 或 TTS
Timbre_output = natural 或 TTS
```

这比 waveform splice 更容易解释，也更容易对 TFG 做因果对照。

### 3.2 只有 natural waveform：generic enhancer

如果推理时只有 natural waveform，没有 TTS/reference 输入，则不能动态迁移某一条 TTS 的特定属性。模型最多学习：

```text
natural audio -> generic TTS-like acoustic prior
```

因此必须区分：

- **conditional transfer model**：输入 `natural + TTS/reference`，实现实例级属性迁移；
- **generic enhancer**：只输入 `natural`，学习跨样本的声学先验。

此前的 waveform-only enhancer 方案属于第二类，不能用第一类系统的 speaker/style transfer 结果作为直接证明。

## 4. 推荐的 duration 参数化

对于同一文本的第 `i` 个 phone：

```text
natural duration: d_i^n
TTS duration:     d_i^t
```

严格保留 natural timing：

```text
d_i^out = d_i^n
```

为了测试 TTS timing/prosody 的贡献，使用受控插值：

```text
d_i^out = d_i^n * exp(alpha * (log d_i^t - log d_i^n))
```

其中：

```text
alpha = 0      完全 natural timing
alpha = 1      完全 TTS phone timing
0 < alpha < 1  部分 TTS timing
```

对 `d_i^out` 增加：

- phone duration 的合理上下限；
- 整句总时长约束；
- pause duration 和 pause count 约束；
- unmatched/low-confidence phone mask；
- inserted/deleted token 的显式处理。

这个插值实验可以直接测试：

```text
alpha -> Sync-C
alpha -> phone duration error
alpha -> pause error
alpha -> F0/energy change
```

如果 Sync-C 只有在 alpha 增大时恢复，说明 raw TTS 的收益可能依赖 timing/prosody，而不是纯 timbre 或 phonetic clarity。

## 5. 当前项目应优先实现的实验矩阵

### 5.1 第一阶段：可解释声学 controls

先不要扩大 residual waveform enhancer。构造以下条件：

| 条件 | duration/pause | F0 | energy | timbre/spectrum |
|---|---|---|---|---|
| Natural identity | natural | natural | natural | natural |
| Raw TTS oracle | TTS | TTS | TTS | TTS |
| Timbre-only | natural | natural | natural | TTS |
| Timbre + clarity | natural | natural | natural | TTS/SSL |
| Timbre + F0 | natural | TTS | natural | TTS |
| Timbre + energy | natural | natural | TTS | TTS |
| Partial timing | interpolated | selected | selected | TTS |
| Loudness control | natural | natural | gain-matched | selected |

最关键的是 `Timbre-only`：

- 若 `Timbre-only` 保留 raw TTS 的 TFG 增益，说明可以继续 strict-natural-timing acoustic transfer；
- 若只有 `Partial timing` 或 TTS duration 恢复增益，则 strict natural timing 不是合理主目标；
- 若所有声学 controls 都失败，而 raw TTS 成功，则应重点怀疑 coarticulation、global prosody 或 TFG evaluator interaction。

### 5.2 第二阶段：pitch-preserving local warp

将普通 waveform resampling 替换为 PSOLA、WSOLA 或 phase vocoder，目的不是直接作为最终模型，而是隔离以下假设：

```text
local-warp 退化是否主要来自 pitch/formant/phase distortion？
```

即使 pitch-preserving 版本改善，也仍需检查逐 phone splice 对 coarticulation 和连续 prosody 的破坏。

### 5.3 第三阶段：duration-controlled latent generator

只有前两阶段支持 acoustic transfer 后，才训练：

```text
natural content tokens
+ forced natural duration
+ TTS timbre/style condition
    -> mel/codec decoder
```

decoder 的输出长度由 `D_natural` 或 natural token sequence 显式控制，而不是由 TTS reference 隐式决定。

建议初始损失：

```text
L = 1.00 * MR-STFT
  + 0.30 * log-mel
  + 0.10 * frozen SSL feature
  + 0.05 * energy envelope
  + 0.01 * residual/identity regularization
```

同 speaker target 才使用较高 waveform L1；不同 speaker 的 TTS target 应降低 waveform L1，避免复制 TTS 的音色、录音条件和 gain。

## 6. 必须报告的指标

### 6.1 时间和节奏

- phone duration MAE/MAPE；
- phone duration Pearson/Spearman correlation；
- speaking-rate ratio；
- pause boundary precision/recall/F1；
- pause duration error；
- total duration error；
- phone boundary error，至少报告 ±20 ms 与 ±50 ms 容差；
- exact sample count。

### 6.2 韵律

- voiced-region F0 RMSE；
- semitone correlation；
- F0 contour DTW distance；
- energy-envelope correlation；
- voiced/unvoiced agreement。

### 6.3 声学和内容

- 对齐后的 MCD；
- log-mel/STFT distance；
- HuBERT/XLS-R middle-layer distance；
- CER/PER/WER；
- speaker embedding cosine；
- loudness、谱倾斜、clipping、naturalness。

未经变长对齐的逐帧 MCD 和 F0-RMSE 不能作为唯一节奏指标，因为 duration change 本身会显著影响它们。

### 6.4 TFG

- Sync-C、Sync-D；
- AV offset；
- audio event 与 mouth landmark lag；
- video failure rate；
- paired utterance delta；
- bootstrap CI 或 paired permutation test；
- provider、speaker、TFG 分层结果。

不能只报告 Sync-C；必须检查模型是否通过 LUFS、谱倾斜或其他 SyncNet shortcut 获得表面收益。

## 7. 对未经证实 claim 的统一降级

以下表述在当前证据下不应直接使用：

1. “参考音频交换只会改变对应属性。”
2. “content/rhythm/timbre/pitch 表示已经完全独立。”
3. “固定 25 Hz token 就准确承载了源语音节奏。”
4. “整句 `length-adjust` 就能实现 phone-level timing 对齐。”
5. “Jets 的内部 alignment module 是 natural/TTS waveform transfer baseline。”
6. “OpenVoice 的 rhythm/pauses/intonation 控制就是外部 waveform 同步接口。”
7. “source token length 决定了输出一定保留 source 的 pause 和真实 phone duration。”
8. “speaker embedding 只提供音色，而不会携带 F0 或 prosody。”

更准确的写法是：

```text
这些系统提供了可借鉴的 duration、content、speaker/style 或 prosody
conditioning primitive；属性独立性和严格 natural-time preservation
仍需在目标数据和 TFG 上通过 factorial swap 与 duration-aware metrics 验证。
```

## 8. 最终决策

当前最有证据支持的工程路线是：

```text
natural content
+ explicit natural duration/pause
+ separately selected TTS acoustic/style factors
    -> duration-controlled acoustic generator
```

而不是：

```text
TTS waveform
    -> phone-wise ordinary resample
    -> natural waveform splice
```

下一步顺序：

1. pitch-preserving local warp；
2. duration/F0/energy/timbre 的可解释 factorial controls；
3. 第二个 TFG 复验；
4. 只有 controls 支持后才训练 duration-controlled latent generator；
5. 若只有 TTS duration/prosody 才恢复 TFG 收益，放弃严格 natural timing preservation 作为主 target；
6. 若 timbre-only 或 spectral-only 能恢复趋势，再继续 waveform/latent enhancer 路线。

## 参考来源

### 主要论文

- [R-VC: Rhythm Controllable and Efficient Zero-Shot Voice Conversion](https://aclanthology.org/2025.acl-long.790/)
- [R-VC arXiv HTML](https://arxiv.org/html/2506.01014v1)
- [Prosody-Adaptable Audio Codecs for Zero-Shot Voice Conversion](https://arxiv.org/abs/2505.15402)
- [Adversarially Learning Disentangled Speech Representations](https://arxiv.org/abs/2102.00184)
- [Disentangling Prosody Representations with Unsupervised Speech Reconstruction](https://arxiv.org/html/2212.06972)
- [On Prosody Modeling for ASR+TTS based Voice Conversion](https://ar5iv.labs.arxiv.org/html/2107.09477)
- [A Holistic Cascade System for Expressive Speech-to-Speech Translation](https://ar5iv.labs.arxiv.org/html/2301.10606)

### 工程实现和仓库

- [OpenVoice](https://github.com/myshell-ai/OpenVoice)
- [Seed-VC](https://github.com/Plachtaa/seed-vc)
- [Amphion](https://github.com/open-mmlab/Amphion)
- [Chatterbox Voice Conversion](https://github.com/LAION-AI/chatterbox-voice-conversion)

## 与本项目实验的关系

本报告补充但不覆盖 [16 — phone-local-warp 目标验证](16-phone-local-warp-validation.md)。实验 16 的 No-Go 结论仍然有效；本报告将下一阶段从“改进 waveform splice”收敛为“显式 duration-controlled acoustic transfer + 可解释 controls”。
