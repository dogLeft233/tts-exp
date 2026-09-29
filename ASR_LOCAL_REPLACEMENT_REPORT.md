# ASR 错误定位与局部 TTS 替换实验

## 1. 研究问题

项目中曾观察到：TTS 音频有时比 natural audio 更容易驱动口型生成模型产生高 SyncNet 分数。

本实验进一步问：

> 如果 ASR 在某些时间段识别错误，能否只把这些局部时间段的 natural audio 替换成对应的 TTS audio，从而改善生成视频的唇形同步？

这个问题被拆成两个实验：

1. **相关性实验**：ASR 错误区间是否对应局部唇形同步较差的区间？
2. **局部替换实验**：用对齐的 TTS waveform 替换 ASR 错误区间后，视频是否真的变好？

---

## 2. 先给结论

1. Natural audio 上，ASR 错误区间和局部同步较差区间存在弱正相关，但重合率很低。
2. TTS audio 上没有得到稳定的同类相关性。
3. 用 TTS 替换 ASR 错误区间后，严格 replacement 评分没有改善，Sync-C 和 Sync-D 反而都出现轻度下降。
4. 因此，**ASR 错误位置不是可靠的 TTS 局部替换目标**。
5. 这个结果只否定当前的 ASR-targeted 局部 waveform 方案，不等于所有局部替换方法或所有生成器都不可行。

---

## 3. 实验一：ASR 错误与局部同步的相关性

### 3.1 数据

- 数据集：LRS3
- 初始 cohort：24 条样本、24 个 source groups
- 条件：natural audio 和 TTS audio 分别分析
- 数据范围：fit-only，不访问 validation/test sealed split

最终由于部分记录缺少完整有效单元，统计时使用：

- Natural：20 条记录、20 个 source groups
- TTS：19 条记录、19 个 source groups

### 3.2 ASR 计算流程

使用固定版本的 Wav2Vec2-CTC：

```text
音频
  ↓
逐帧 argmax
  ↓
CTC collapse
  ↓
greedy ASR 转写
  ↓
与参考转写比较
  ↓
得到 substitution / deletion 等 ASR error 区间
```

ASR 阶段没有使用：

- beam search
- 语言模型
- 词典修正
- 人工修正
- reference-conditioned decoding

另外，对参考转写执行 CTC forced alignment，用来确定参考词的时间范围。forced alignment 只提供时间定位，不修改 greedy ASR 的识别结果。

### 3.3 局部同步的计算流程

SyncNet 对每段视听组合给出一个**同步距离矩阵**：行对应视频时间轴上的时刻（每秒 25 行），列对应 31 个候选音视频相对偏移（−15 到 +15 帧），矩阵中的数值表示"在该时刻、该偏移下口型与语音有多不匹配"，越小越同步。对每个 audio/video arm：

1. 读取完整的 SyncNet V2 distance matrix，形状为 `[T, 31]`。
2. 按官方 SyncNet 的全局评分口径求最佳偏移：先把矩阵按列在时间上取平均（记作 `mdist`），取均值最小的列 `j_star`，得到官方的最小距离 `Sync-D = mdist[j_star]`、置信度 `Sync-C = median(mdist) - Sync-D`，以及最佳偏移 `AV offset = 15 - j_star`。
3. 固定这条最佳偏移列，逐时刻套用与官方相同的量，得到一维的局部同步置信度曲线 `local_c[t] = median(mdist) - distance[t, j_star]`，再施加一次 9 点中值滤波去除单帧尖刺。这一步就是本报告所说的 **vendor-equivalent reduction**：reduction 指把二维距离矩阵压缩成一维逐时刻曲线，vendor-equivalent 指所用的量与官方 SyncNet 完全一致——重新计算出的 Sync-C、Sync-D 和 offset 与官方输出相同（经过 parity 校验），它不是项目自创的指标。使用时还会把每个时刻映射回音频时间轴，并丢弃偏移填充行、中值滤波边缘行以及落在所选人脸轨迹或音频范围之外的行，避免把 SyncNet 的 padding 误当成真实的局部低同步。
4. 使用该 arm 自己的分布定义低同步区间：

```text
low_sync_threshold = mean(local_c) - population_std(local_c)
low_sync_mask = local_c < low_sync_threshold
```

5. 将 `ASR_error_mask` 与 `low_sync_mask` 对齐，计算相关性和重合指标。

统计单位是 whole sample/source group，并使用 10,000 次 bootstrap；没有把相邻视频帧当成相互独立的样本。

### 3.4 相关性结果

| 条件 | 有效记录 / source groups | median Spearman | 95% bootstrap CI | low-sync precision | error recall | micro IoU |
|---|---:|---:|---:|---:|---:|---:|
| Natural | 20 / 20 | 0.1448 | [0.0555, 0.1799] | 13.71% | 23.21% | 0.0943 |
| TTS | 19 / 19 | 0.0899 | [-0.0283, 0.1595] | 5.51% | 14.91% | 0.0419 |

### 3.5 如何解释

Natural arm 的 Spearman 相关置信区间高于 0，因此达到了预先设定的弱支持门槛。但它的实际重合率不高：

- Natural 的低同步位置中，只有 `13.71%` 同时是 ASR 错误位置；
- Natural 的 ASR 错误位置中，只有 `23.21%` 同时属于低同步位置；
- TTS arm 的两个置信区间都跨过 0，不能认为存在稳定相关性。

所以，“ASR 识别错误”和“口型同步较差”不是同一个现象。Natural 上的弱相关不足以支持直接用 ASR 错误作为替换目标。

---

## 4. 实验二：ASR-targeted 局部 TTS 替换

### 4.1 候选区间筛选

替换目标不是所有 ASR 错误区间，而是经过多重门禁筛选的区间。候选必须满足：

- 属于 substitution 或 deletion 类型；
- 对应连续的参考词；
- natural 和 TTS 两侧都有有限且可对齐的 forced-alignment span；
- TTS donor 区间不能含 insertion 或混合错误；
- 替换后仍为单声道、16 kHz、PCM16；
- TTS 片段可以被精确映射到 natural 目标区间长度。

候选数量的变化如下：

```text
71 个 natural ASR error blocks
        ↓ 排除无法映射参考时间的区间
52 个 reference-mappable blocks
        ↓ 排除 TTS donor 区间含插入/混合错误的区间
21 个 TTS-clean target blocks
        ↓ 按 source group 组织成对推断
13 个完整 source groups
```

这个收缩是预先定义的质量控制，不是根据 SyncNet 得分挑选样本。

### 4.2 局部 waveform 替换

对每个保留下来的区间：

1. 从 TTS 音频中取出与参考内容对应的 waveform 片段；
2. 使用 `scipy.signal.resample` 将 TTS 片段重采样到 natural 目标区间的精确长度；
3. 在替换边界使用固定 raised-cosine ramp；
4. 不做全局归一化；
5. 不改变整句时间轴；
6. 不做全局 time-warp；
7. 将替换片段放回 natural waveform，形成 `asr_targeted` 音频。

### 4.3 视频生成和严格评分

每个样本生成四类条件：

- `natural`：natural audio 驱动的视频基线；
- `asr_targeted`：ASR 错误区域替换为 TTS 后驱动的视频；
- `target_control_0`；
- `target_control_1`。

视频生成使用冻结的 Wav2Lip。关键评估协议如下：

```text
candidate / natural audio
          ↓
      冻结 Wav2Lip
          ↓
       生成视频
          ↓
所有视频都换回同一份 untouched natural PCM
          ↓
       SyncNet 评分
```

这样测量的是：

> TTS 局部片段是否改变了视频的口型运动，使视频在换回 natural audio 后仍比 natural baseline 更同步。

这比直接计算 candidate audio 与 candidate video 的匹配更严格。后者只作为 candidate self-consistency proxy，不作为主要科学结论。

### 4.4 Gain 的定义

项目中 Sync-C 越高越好，Sync-D 越低越好。因此对 target 相对于 natural baseline 定义：

```text
Sync-C gain = Sync-C(target) - Sync-C(natural)
Sync-D gain = Sync-D(natural) - Sync-D(target)
```

两个 gain 都大于 0，才表示 target 在两个指标上同时改善。

统计以 source group 为单位做 10,000 次 bootstrap，并要求至少 12 个完整 source groups 才进入 paired inference。最终有 13 个 source groups 完成严格推断。

---

## 5. 局部替换结果

### 5.1 严格 replacement endpoint

严格评分要求视频全部换回 untouched natural PCM。结果为：

| 指标 | 中位 gain | 95% bootstrap CI | 解释 |
|---|---:|---:|---|
| Strict Sync-C gain | -0.0341 | [-0.2092, -0.0054] | 没有改善，整体轻度下降 |
| Strict Sync-D gain | -0.0501 | [-0.2184, -0.0016] | 没有改善，整体轻度下降 |

逐组情况：

- Sync-C 改善：3/13；
- Sync-D 改善：3/13；
- 两个指标同时改善：2/13；
- 两个 control contrast 的置信区间均跨 0。

严格科学结论为：

```text
NO_PROTOTYPE_SUPPORT
```

也就是说，ASR-targeted 局部 TTS 替换没有证明能改善 strict natural-audio replacement 分数。

### 5.2 Candidate-audio proxy

另外计算 candidate audio 与 candidate video 直接配对的 proxy：

| 指标 | 中位 gain | 95% bootstrap CI |
|---|---:|---:|
| Proxy Sync-C gain | -0.0210 | [-0.0885, 0.0631] |
| Proxy Sync-D gain | -0.0072 | [-0.0463, 0.0440] |

proxy 的区间跨过 0，也没有可靠正收益。更重要的是，即使 proxy 为正，也不能替代 strict replacement endpoint，因为 candidate audio 可能和 candidate video 共同适应，而不一定能和 natural audio 对齐。

---

## 6. 最终结论

本实验没有支持以下假设：

> ASR 错误位置就是 TTS 最值得替换的位置。

目前更合理的解释是：

- ASR 错误主要描述内容识别问题；
- 局部唇形同步还受到发音时序、声学表示、视频生成器和 SyncNet 评分窗口影响；
- 两者只有有限重合，无法用 ASR 错误 mask 作为可靠的视觉同步 mask；
- 局部 waveform 重采样和边界拼接还可能引入额外声学变化；
- candidate 自洽不等于换回 natural audio 后仍然有效。

因此，本实验的科学结论是：

> 在当前 LRS3 fit-only cohort、冻结 Wav2Lip 和严格 untouched-natural-audio 评分协议下，ASR-targeted 局部 TTS waveform 替换没有获得支持，且总体表现为轻度负向。

这不等于证明所有局部替换方法都不可行，也不等于证明其他生成器一定失败；它只否定了当前这条 ASR-targeted waveform 路线作为可靠 enhancement target 的依据。

---

## 7. 可复现实验产物

- ASR 与局部同步相关性：`runs/lrs3_asr_sync_error_correlation_20260831/`
- ASR-targeted 局部替换原型：`runs/lrs3_asr_targeted_local_replacement_prototype_20260901/`
- 相关实验代码：`scripts/experiments/asr_sync_error_correlation/`

## 8. 一句话汇报版

 我们先用 greedy CTC ASR 定位识别错误，再将其与 SyncNet 的局部低同步区间对齐；natural 音频上只有弱相关，TTS 上不稳定。随后把可严格对齐的错误区间替换成 TTS waveform，重新驱动冻结 Wav2Lip，并将所有视频换回 untouched natural audio 进行严格评分。最终 Sync-C 和 Sync-D gain 均为负，因此 ASR 错误区域不能作为可靠的 TTS 局部替换目标。

---

## 9. Natural-to-TTS Bridge：从音频侧匹配到静态图 replacement 验证

最新发现是：**bridge 音频在真实自然视频上获得接近 natural 的 SyncNet 分数，不保证它驱动生成的视频换回 natural 音频后仍能保持分数。** 这不是“SyncNet 已证明两条音频完全对齐，生成却失败”：SyncNet 测量音视频匹配，不直接测音频之间的时间等价性；自然视频上的平均近似保分也不是逐样本、逐音素对齐证明。

### 9.1 为什么要做 bridge 实验

上面的 ASR-targeted 实验是在 natural waveform 中寻找局部错误区间，再把这些区间替换成 TTS waveform。bridge 实验测试的是另一条路线：不根据 ASR 错误选局部区间，而是对整句音频构造一个处在 natural 和 TTS 之间的受控音频。

这个实验要回答的问题是：

> 如果保留 natural 音频的时间长度和相位结构，只把它的声学频谱表示向 TTS 移动，能否让 Wav2Lip 产生不同的口型运动，同时仍然能和原始 natural 音频同步？

这里有一个容易误解的地方。项目真正关心的不是 candidate audio 和它自己生成的视频能否在对角线上取得高分，而是下面这个 replacement 关系：

~~~text
natural audio N + face video → Wav2Lip → video V_N → 换回 N → SyncNet 基线
bridge audio B  + face video → Wav2Lip → video V_B → 换回 N → SyncNet 主评分
~~~

第二行的音频仍然换回 untouched natural audio。这样测量的是：bridge 改变的口型运动，是否真的和原始 natural 音频匹配。如果只计算 V_B 和 B 的对角线分数，bridge audio 和它驱动的视频可能共同适应，不能证明它对 natural-audio replacement 有帮助。

为方便阅读，后文使用以下记号：

- **N（natural）**：视频原本对应的真实人声；
- **M（MFA-linear TTS）**：前一阶段已经生成并冻结的、与 natural 记录对应的 TTS 音频。MFA 是 forced alignment 工具，用已知文本给音频中的词/音素估计时间边界；这里的 linear 表示此前已按这些边界把 TTS 音频映射到 natural 的时间网格。本次实验把它作为声学目标，不重新生成 TTS，也不重新运行 MFA 或 DTW；
- **B（BRIDGE_075）**：本实验根据 N 和 M 构造的 bridge 音频；
- **V_X/A_Y**：用音频 X 驱动 Wav2Lip 生成视频，再把音频 Y 放回视频后交给 SyncNet 评分。

### 9.2 数据和实验边界

实验使用 LRS3 的 fit-only 数据。fit-only 表示只使用开发阶段已有的记录，不访问最终 validation/test 或其他 sealed 数据。

样本不是按 SyncNet 分数挑出来的，而是由固定规则提前决定：

~~~text
父实验：133 条记录、23 个 source groups
        ↓ 每个 source group 取按原顺序出现的第二条记录
确认 cohort：22 条记录、22 个 source groups
~~~

其中剩余的一个 source group 没有第二条记录，因此按规则不进入确认 cohort。这个 22-record cohort 与前一轮 discovery 使用的 23 条记录完全不重叠。所有选择在读取确认分数之前完成。

每条记录绑定三类已有资产：原始 natural 音频、对应的 face video、MFA-linear TTS 音频。两条音频都必须是相同长度的 16 kHz、单声道、PCM16。实验固定了输入文件、模型权重和工具版本，并且不执行以下操作：

- 新的 TTS 生成、MFA 或 DTW 对齐；
- 训练或微调模型；
- 搜索其他 bridge 强度；
- 根据分数筛选记录、重试评分或修改父实验产物；
- 访问 validation/test sealed media。

### 9.3 三个音频臂

每条记录固定生成三个音频版本：

| 音频臂 | 如何构造 | 作用 |
|---|---|---|
| N | 原始 natural PCM，逐样本保持不变 | natural baseline |
| N_REPEAT | 解码后的 PCM 字节与 N 完全相同，但在独立工作目录中重新运行 Wav2Lip | 检查重复渲染和评分是否稳定 |
| BRIDGE_075 | 在 STFT 的 log-magnitude 上使用 75% 的 MFA-linear TTS 方向，同时保留 natural phase | 实验候选 |

### 9.4 BRIDGE_alpha 的音频计算

bridge 不是把两条 waveform 按比例直接相加，也不是把整句 TTS 做一次时间拉伸。它在短时频谱上进行受控插值。STFT 可以理解为把一条长 waveform 切成许多重叠的短窗口，然后记录每个时间窗口中不同频率的能量。

先把两条等长音频分别做 STFT。固定参数为：

~~~text
采样率：16,000 Hz
n_fft：1024
window length：1024
hop length：256
window：periodic Hann
center：True
边界 padding：reflect
数值类型：CPU float64
幅度下限：epsilon = 1e-7
~~~

记 natural 的 STFT 为 S_N，MFA-linear TTS 的 STFT 为 S_M。固定 bridge 强度 alpha = 0.75，先在 log-magnitude 上计算：

~~~text
L_B(f,t) = (1 - alpha) × log(max(|S_N(f,t)|, epsilon))
           + alpha × log(max(|S_M(f,t)|, epsilon))

           = （1 - ） × natural log-magnitude
           + alpha × MFA-linear TTS log-magnitude
~~~

再使用 natural 的相位：

~~~text
P_N(f,t) = S_N(f,t) / max(|S_N(f,t)|, epsilon)
S_B(f,t) = exp(L_B(f,t)) × P_N(f,t)
~~~

最后对 S_B 做 inverse STFT，并强制输出长度等于 natural 的原始长度 L。因此 bridge 的目标是保持 natural 的时间网格和相位骨架，只改变短时能量/频谱结构，使其朝 MFA-linear TTS 方向移动。

重建后只允许以下固定处理：

1. 做一次全局 RMS 匹配，使 bridge 的整体能量与 natural 相同；
2. 如果峰值达到或超过 0.999，做一次全局衰减到 0.999；
3. 做一次 PCM16 canonicalization。

不允许滤波、降噪、compressor、limiter、重采样、时间调整或迭代修复。这样可以把结果归因到预先登记的 bridge 构造，而不是事后音频修补。

### 9.5 如何判断 bridge 是否真的改变了 Wav2Lip 输入

在读取任何确认阶段的 SyncNet 分数之前，先使用官方 Wav2Lip 的 mel-spectrogram 函数计算三份表示：

- mel(N)：natural 的 Wav2Lip 输入；
- mel(M)：MFA-linear TTS 的目标表示；
- mel(B)：bridge 的实际 Wav2Lip 输入。

这里的 mel 是 Wav2Lip 使用的时间-频率特征，不是最终的 SyncNet 分数。定义：

~~~text
d = mel(M) - mel(N)       # natural 到 TTS 的目标方向
u = mel(B) - mel(N)       # bridge 相对 natural 的实际移动

progress = dot(u, d) / max(dot(d, d), 1e-12)
~~~

progress = 0 表示没有离开 natural，progress = 1 表示在这个 mel 向量意义上走到了 MFA-linear TTS 的方向终点。它不要求等于 waveform 插值参数 0.75，因为 STFT 重建和 Wav2Lip mel 提取都不是简单的线性映射。

本实验预先登记的 movement 门槛是：至少 20/22 条记录的 progress >= 0.15，并且均值的 95% 置信区间下界严格大于 0.15。下表把同一批 22 条记录上的三个 bridge 强度并列，其中只有 BRIDGE_075 是本实验预先登记的候选；B025 / B050 的数值来自之后对同一批音频、同一公式的官方 mel 复核（`runs/bridge_mel_progress_official_20260913/`），仅作对照：

| Bridge | mean progress | median | bootstrap 95% CI | 达到 0.15 的记录 |
|---|---:|---:|---|---:|
| B025（alpha = 0.25） | 0.1313 | 0.1185 | [0.1002, 0.1634] | 8/22 |
| B050（alpha = 0.50） | 0.3116 | 0.3007 | [0.2620, 0.3608] | 20/22 |
| B075（alpha = 0.75，确认候选） | 0.5118 | 0.5119 | [0.4539, 0.5685] | 22/22 |

可以看到 progress 随强度单调上升：强度越高，bridge 在 mel 空间里越靠近 MFA-linear TTS。BRIDGE_075 达到预先登记的两项门槛，判定为：

~~~text
movement = PASS
~~~

在静态图子实验实际使用的前 11 条记录上，同一指标为 B025 mean 0.125 / median 0.112 / 95% CI [0.083, 0.172]，B050 mean 0.307 / median 0.296 / 95% CI [0.236, 0.380]，B075 mean 0.509 / median 0.503 / 95% CI [0.419, 0.596]。

这说明 bridge 确实把 Wav2Lip 看到的音频特征稳定地从 natural 向 TTS 方向移动了。这个结论只说明“输入变化成功发生”，还没有说明视频因此变好。

### 9.11 固定真实自然视频，只改变评分音频

这一阶段不运行 Wav2Lip。固定 22 条真实自然视频，将评分音频从 N 改成不同强度的 bridge，观察 bridge 是否仍匹配原始真实口型。所有音频共享同一视频轨迹、视觉特征和有效评分窗口。

bridge 沿用 9.4 的构造，只改变 alpha：在 natural 与 MFA-linear TTS 的 log 幅度谱之间插值，保留 natural 相位，重建为等长音频，再做固定 RMS 匹配和 PCM16 量化。alpha=0.25 表示频谱插值强度为 25%，不是“25% 时间对齐”，也不是替换 25% 的片段。

| 评分音频 | 相对 N 的平均 ΔSync-C | 95% bootstrap CI | 解释 |
|---|---:|---|---|
| Bridge 0.25 | +0.001 | [-0.077, +0.074] | 平均基本不掉分，但未证明严格等效 |
| Bridge 0.50 | -0.347 | [-0.519, -0.188] | 已有下降，不能称作与 N 等效 |
| Bridge 0.75 | -0.783 | [-1.052, -0.545] | 进一步下降 |
| Bridge 1.00 | -1.184 | [-1.506, -0.900] | 进一步下降；仍保留 natural 相位，不等于原始 MFA-linear 音频 |

总体最佳 offset 大体保留，但匹配分数随强度增加而下降。**整体节奏或 offset 接近，不等于局部发音和声学表示完全匹配。** 单凭这个实验不能区分局部音素时序变化与频谱/发音表征变化。

控制也揭示了分数的解释限制：将 N 延迟 200 ms 后，最佳 offset 全部移动 5 帧，但重新搜索 offset 的平均 ΔC 仍为 +0.057。也就是说，允许搜索偏移的 Sync-C 可以掩盖可由全局平移补偿的延迟，不能单凭高分断言零偏移对齐。

### 9.12 第二步：优质静态图生成，同时比较原生与 replacement

使用本地三张 512×512、近正脸且嘴部无遮挡的静态人脸图，固定生成区域、评分裁剪和 25 fps。每张图分别用 N、Bridge 0.25、Bridge 0.50 驱动冻结的 Wav2Lip GAN。复用上一阶段的 bridge 波形并核验 hash，不重新调整音频。

最终使用原队列前 11 条音频，与三张图全部交叉，每个强度有 33 个配对，但独立音频来源组只有 11 个。原计划为 22 条，运行中按用户要求收束到 11 条，此前已披露部分结果；因此这是探索性分析，不是预先固定样本量的确认性试验。

| 组合 | 视频如何生成 | SyncNet 使用的音频 | 回答的问题 |
|---|---|---|---|
| 自然基线 V_N/A_N | 静态图 + N | N | 自然音频驱动的基准 |
| Bridge 原生 V_B/A_B | 同一静态图 + B | B，不更换音轨 | bridge 音视频组合本身是否优于自然基线？ |
| Bridge replacement V_B/A_N | 与原生相同的 bridge 视频 | 换回未经修改的 N | bridge 生成的口型是否更适合自然音频？ |

两轮新实验均在所有候选共享的真实有效时间窗口上，先平均 SyncNet 距离曲线，再计算 `C = 曲线中位数 − 最小值`、`D = 最小值`，搜索范围为 ±15 帧。该共同支持口径与历史含边界 padding 的全局 Sync-C 不直接比较绝对数值。

统计时先将同一音频在三张图上的配对差值平均，再对 11 个音频来源组进行 10,000 次 bootstrap。下表区间没有进行多重比较或运行中停止校正。

| 组合 | 平均 Sync-C | 相对自然基线 ΔC（95% CI） | C 改善配对数 | 平均 Sync-D |
|---|---:|---|---:|---:|
| 自然基线 V_N/A_N | 5.595 | — | — | 7.170 |
| Bridge 0.25 原生 | 5.518 | -0.078 [-0.145, -0.026] | 8/33 | 7.266 |
| Bridge 0.50 原生 | 5.416 | -0.179 [-0.290, -0.071] | 7/33 | 7.255 |
| Bridge 0.25 replacement | 5.488 | -0.107 [-0.167, -0.042] | 6/33 | 7.279 |
| Bridge 0.50 replacement | 5.249 | -0.347 [-0.446, -0.244] | 1/33 | 7.532 |
| Bridge 0.75 原生 | — | -0.323 [-0.527, -0.135] | 7/33 | — |
| Bridge 0.75 replacement | — | -0.661 [-0.871, -0.483] | 0/33 | — |
| MFA-linear 原生 | — | -0.169 [-0.436, +0.057] | 13/33 | — |
| MFA-linear replacement | — | -0.967 [-1.299, -0.695] | 0/33 | — |

Sync-C 越高越好，Sync-D 越低越好。表中所有条件的平均 ΔC 都为负；除 MFA-linear 原生（区间跨过 0）外，其余未校正 ΔC 区间上界均低于零。B025 / B050 的 replacement C/D 同时改善比例分别为 5/33 和 0/33，而 Bridge 0.75 和 MFA-linear 的 replacement 没有任何配对改善（均为 0/33）。这些结果支持本探索性样本集上的下降，而非增强。

Bridge 0.75 与 MFA-linear 两行来自同口径后续运行（`runs/static_image_b075_mfa_20260913/`）：同样的 3 张固定图片与前 11 条音频，四个候选共用同一有效窗口，因此与前三行的绝对分数口径略有差异。该运行只报告 Δ 类指标与 C 正向计数，所以这两行的绝对 Sync-C / Sync-D 留空。

### 9.13 这些结果说明什么，不说明什么

1. **自然视频上的音频侧近似保分，不足以保证生成后的 replacement 保分。** 最清楚的例子是 0.25：在 22 条真实自然视频上平均 ΔC=+0.001，但在其中前 11 条音频的静态图生成实验中，replacement 平均 ΔC=-0.107。两阶段样本数不同，不能将均值差当作严格的逐条因果分解，也不能声称每条音频都先通过了等效检验。
2. **下降不只发生在换回自然音轨之后。** 0.25/0.50 原生组合相对自然基线也分别下降 0.078/0.179。换回 N 后进一步变低，表明 bridge 视频平均更匹配其自身驱动音轨，但自身匹配仍未超过自然基线。原生比较同时改变视频和评分音频，不能据此单独判断视觉质量下降。
3. **输入向 TTS 移动、与真实口型匹配、驱动生成、换回自然音频，是不同环节。** 前一环节通过不能替代后一环节实测。直观上，生成器可能对被改变的频谱作出不同口型响应；这是与结果一致的机制猜测，不是已被单独验证的原因。
4. **不需要动态自然口型作为输入，也能观察到本次负结果。** 静态图没有原视频随时间变化的口型轨迹，且生成区域外已核验始终等于静态图。因此本次下降不能归因于输入动态口型泄漏。但静态与历史动态实验还存在图像身份、裁剪及评分口径等差异，不能反推历史微小正增益必然由泄漏造成。
5. **不能据此宣布 SyncNet 普遍失灵或 bridge 普遍无效。** 把 natural 整体延迟 200 ms 后，最佳 offset 全部移动 5 帧，正确反映了时序被破坏；这支持当前测量链路可感知相应变化。结果限于这些已使用的 LRS3 样本、三张静态图、当前构造与冻结生成器，不能授权 audio head 训练，也不是跨身份或跨模型泛化结论。

### 9.14 核验与可信度边界

纳入产物的输入、模型、视频和矩阵 hash、音视频配对绑定、无损转码逐帧一致性及生成区域外静态性已核验。独立重算全部 66 个强度配对的原生/replacement 差值，最大误差为 0；首条音频 × 三张图共 15 个评分 cell 的独立官方前向与保存矩阵逐元素一致。不是全部评分 cell 都重新做过模型前向。

首条音频 × 三张图的重复生成矩阵误差为 0，200 ms 延迟均识别出 5 帧变化；这些控制没有覆盖全部 11 条。相关 28 项测试及 Ruff 通过。复核器曾因张量内存布局不同出现约 0.000011444 的距离差异，匹配原评分布局后误差归零，没有修改原分数或放宽阈值。现有核验未发现能解释负结果的实现错误，但不等于证明所有潜在问题均已排除。

数值依据为自然视频强度扫描与低强度静态图实验的配对统计；本节不附生成视频地址。

## 10. Bridge 实验一句话汇报版

我们发现，低强度 bridge 在真实自然视频上平均基本保持 SyncNet 分数，并不保证它驱动静态人脸生成后、换回自然音频仍能保持分数：0.25 强度的 replacement 平均 ΔSync-C=-0.107，0.50 为 -0.347，而且不换音轨的原生组合也低于自然基线。因此，音频侧匹配只能作为前置检查，不能替代生成后的严格 replacement 验证；当前 11 条音频 × 三张静态图的探索性结果未支持增强。