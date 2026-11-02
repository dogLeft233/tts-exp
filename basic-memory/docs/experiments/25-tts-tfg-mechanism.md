---
title: 25-tts-tfg-mechanism
type: experiment
permalink: tts-exp/docs/experiments/25-tts-tfg-mechanism
concluded: '2026-10-02T17:39:48.9609677+00:00'
repository: dogLeft233/tts-exp
project: tts-exp
verification: 11776 statistical endpoints and 54 geometry endpoints independently
  reconstructed; full evidence restored; root figure QA pass
scope: AISHELL-1 Chinese; cloud Qwen TTS; Wav2Lip; separate native, fixed-A, local-G
  and motion estimands
status: concluded
---

# 报告25：TTS-TFG 的增益驱动与局部谱形相容性

**状态：CONCLUDED**

**研究范围：** AISHELL-1 中文、cloud Qwen TTS、Wav2Lip 视频生成与 SyncNet 评分链。下述结论针对本研究的语料、模型与冻结评估设计。

## 结论摘要

幅度增益改变 log-mel 输入，并沿音频编码方向带来固定评分音下的正向 SyncNet 响应；固定 RMS 的 native 几何干预还对应自然语音的可见 aperture 与 aperture 速度增加。男性、女性的原生 TTS−自然 gap 在配对 q00→q11 处理中都缩小，方向与增益路径一致。

μ/ρ phone 坐标揭示另一层机制：token 内相对频带比值轨迹与 recipient 的发声、时间结构共同约束音视频匹配。Natural donor→TTS recipient 的 trajectory 相对 SELF_GRID 使固定评分端 C 下降，而独立视觉族测得 aperture 增加；TTS donor→自然 recipient 的 joint−SELF_GRID 也下降。谱形移植对 SyncNet 的作用取决于 recipient 的局部声学与时间结构相容性；TTS 供体移入自然 recipient 时，joint−SELF_GRID 也下降。正向增强来源由独立增益路径定位。

男性 native gap 配对估计、女性 native gap 配对估计与固定 A 生成响应分别报告。男性 full-A 补充、primary face、raw、guard15 的 native time-excess 从 q00 RAW/RAW 的 +0.235（99% CI [+0.054,+0.399]）到 q11 FIXED/FIXED 的 +0.031（[−0.118,+0.179]），配对 q00−q11 缩小 +0.204（[+0.119,+0.295]）。这项配对同时改变生成条件和评分输入。男性另一个固定 A 端点是 primary 旧共用音频截止、raw、guard20 的 `G_visual_fixed_minus_raw_audio_raw`：observed +0.192、time-excess +0.185（[+0.108,+0.266]）。女性 full-A、guard20 的配对 TTS−自然 gap 变化为 +0.245（[+0.118,+0.388]）。它们使用不同 cohort、支持与评分设计，不合并为一个 native 中介比例。

## 1. 增益、log-mel 与编码方向

MEL 分区协议对幅度做 STFT/mel 投影、20-log 变换和对称裁剪。对未裁剪坐标，增益 `g` 对应 `1.6 log10(g)` 的归一化 mel 位移；被裁剪至边界的坐标不保留完整增益信息。独立 MEL 四格在 raw 几何、guard20、40 名等权 speaker 的固定自然音评分下得到：

| MEL 对比 | Sync-C 均值 | speaker bootstrap 99% CI |
|---|---:|---:|
| q10−q00（仅内部坐标） | +0.171 | [+0.101, +0.245] |
| q01−q00（仅边界坐标） | −0.009 | [−0.037, +0.019] |
| q11−q00（联合） | +0.190 | [+0.117, +0.274] |
| q10−q01 | +0.180 | [+0.109, +0.253] |

这里 q10/q01 是 MEL 裁剪边界分区，与 RMS 或 μ/ρ 分解中同名 cell 的定义不同。RMS 潜变量实验在同一 raw C、guard20 端点上，方向对比为 **+0.161 [+0.094,+0.232]**，半径对比为 **+0.020 [−0.006,+0.046]**，方向−半径为 **+0.142 [+0.077,+0.212]**。在冻结 SyncNet 端点，方向变化的 99% CI 为正，半径 CI 跨零。

固定 RMS 的 native 几何补充包含 80 条语句、40 名 speaker。自然语音经 FIXED 后，aperture 增加 **+0.002899 [+0.001844,+0.003940]**，绝对 aperture 速度增加 **+0.022588 [+0.014555,+0.030767]/秒**；TTS 经 FIXED 后，aperture 为 **−0.001254 [−0.002108,−0.000396]**，速度为 **−0.007101 [−0.013010,−0.001405]/秒**。配对 TTS−自然 gap 变化分别为 **−0.004153 [−0.005565,−0.002759] aperture** 和 **−0.029689 [−0.041260,−0.018409] aperture 速度/秒**，speaker-bootstrap 99% CI 均低于零。原生 gap 为 **+0.010824 [+0.008752,+0.012922] aperture** 和 **+0.074732 [+0.063392,+0.086525] aperture 速度/秒**。因此，固定 RMS 干预下两类语音的可见动作差缩小；native gap 与 fixed-A 响应仍是各自的估计量。

## 2. μ/ρ 谱形与局部 phone-window 响应

对每个精确匹配的 phone token，重建谱坐标为 `p(t,k)=softmax(μ(k)+ρ(t,k))`。重建 logits 满足恒等式 `log(p(t,k)/p(t,l))=(μ(k)−μ(l))+(ρ(t,k)−ρ(t,l))`：μ 控制 token 平均的相对频带 log 比，ρ 控制这些比值在 token 内如何随时间变化。q10 替换 μ 并保留 recipient 的 ρ；q01 替换 ρ 并保留 recipient μ；q11 同时替换两者。SELF_GRID 是预先规定的双步自身网格循环。所有生成 cell 保留 recipient 的 F0、AP/清浊音、时间网格、逐帧 SP 功率，并满足固定全局 RMS 合同。该分解定位相对频带比值坐标；它不单独指定某个 formant、antiformant 或平滑度标量。

方向标签按**供体→受体**定义：Natural→TTS 表示自然语音供体参数移植至 TTS 受体，TTS→Natural 表示 TTS 供体参数移植至自然语音受体。统计族覆盖 80 条语句、40 名 speaker、两条方向、A_RAW 与 A_RT 两个固定评分音域、raw/unit 两种几何，以及 FULL、VALID、guard20、guard15 四种支持。共计划 11,776 个端点，其中 10,304 个可估计、1,472 个保留为 `INCOMPLETE_SUPPORT`。共同 20,000 次 speaker bootstrap 的 max-|t| 临界值为 4.955307；不完整端点保留在计划分母。

### 全局 fixed-anchor 端点

主要 Sync-C 指标为固定 anchor 的 `C_fixed_anchor = B_median − D_fixed_anchor`；`C_best = B_median − D_min` 是描述性指标。主族还保留 B_median、D_min、D0、search_gain 与最佳 lag。以下为预先指定的 A_RAW、raw 几何、guard15、C_fixed_anchor 切片；数值为等 speaker 均值及同时 99% CI。

| 供体→受体方向 | 对比 | Sync-C 均值 [同时 99% CI] |
|---|---|---:|
| Natural→TTS | mean（q10−q00） | −0.149 [−0.280,−0.019] |
| Natural→TTS | trajectory（q01−q00） | −0.201 [−0.346,−0.056] |
| Natural→TTS | joint（q11−q00） | −0.283 [−0.520,−0.046] |
| Natural→TTS | trajectory−SELF_GRID | −0.192 [−0.336,−0.047] |
| Natural→TTS | joint−SELF_GRID | −0.275 [−0.512,−0.037] |
| TTS→Natural | joint−SELF_GRID | −0.197 [−0.381,−0.013] |

Natural→TTS 的 trajectory−SELF_GRID time-excess 在 A_RAW/A_RT × raw/unit 四种组合下均为负；joint−SELF_GRID time-excess 在两个方向也均为负。mean−trajectory 与交互 I 的直接同时区间包含零。完整结果保留九个预定对比、六项指标、全部方向/音域/几何/支持和 observed/time-excess；上表为固定切片。

### 局部定位

局部统计是固定 anchor 距离缩减 G 的窗口贡献，按匹配 phone 覆盖权重汇总，并以全部固定支持窗口数为分母。每个评估 query 对应五帧视频使用的 16 列 Wav2Lip mel-block 并集，约为 350 ms 的 mel 坐标邻域；phone 权重按 mel 列中心是否落入 recipient 对齐区间计算。窗口可以跨相邻 phone 并包含协同发音，故这些权重描述 query 的 mel 输入覆盖，而不是独立音素干预或完整网络的波形感受野。

FULL 支持下有 35 个不同 phone 对比行，其同时 99% CI 上界低于零：q10 的 `a˥˩` 4 行；q01 的 `n` 12 行；q11 的 `n` 10 行、`ŋ` 7 行、早段 `tɕ` 2 行，均为 Natural→TTS。FULL 与 VALID 在本 cohort 的逐端点数值完全重复。phone-local guard15 的同时 CI 均包含零，因此位置结论限于 FULL 支持。q01 `/n/` whole-phone、raw G 的 FULL observed 为 **−0.018 [−0.034,−0.001]**，time-excess 为 **−0.014 [−0.027,−0.002]**；beginning 与 middle 阶段也为负。`ŋ` 与早段 `tɕ` 出现在 joint cell 的 FULL 响应中。全部 16 个冻结标签和全部 phase 均纳入；没有依据 EVAL 结果筛选标签。

1,472 个不完整端点都来自短 T 受体 context `S0099_BAC009S0099W0123` 的 Natural→TTS 方向：432 个 global、1,024 个 phone-phase、16 个 overall-window。它们保留在原计划中，没有删语句或以 guard15 替代。

## 3. 与既有 WORLD 和视觉结果的关系

较早的 WORLD factorial 以 `E=q01−q00`、`G=q10−q00`、`I=q11−q10−q01+q00` 表示；SP 与 SP_EM 是处理条件名，cell 定义独立于 μ/ρ。该报告使用逐点 speaker bootstrap，μ/ρ 族使用同时 max-|t| 区间。旧 WORLD SP/SP_EM 的 E、G 为负、I 为正，显示组合响应有非加性。

下表是较早 WORLD factorial 的 **within-WORLD A_RT** 对比：raw 几何、`C_fixed_anchor`、guard15，逐点 speaker-bootstrap 99% CI。来源为 `world_consumer_statistics.json` SHA `b11f5530820ac82d654d77240bbb05b93ef2b6ab554a8e2c05e1b34fb9313b88` 的 `/bootstrap/summaries/7688–7690`、`/8136–8138`、`/16200–16202`、`/16648–16650`。

| 供体→受体方向 | WORLD 条件 | E | G | I |
|---|---|---:|---:|---:|
| Natural→TTS | SP | −0.392 [−0.505,−0.267] | −0.337 [−0.473,−0.203] | +0.536 [+0.424,+0.663] |
| TTS→Natural | SP | −0.201 [−0.291,−0.118] | −0.311 [−0.402,−0.219] | +0.512 [+0.442,+0.591] |
| Natural→TTS | SP_EM | −0.395 [−0.507,−0.271] | −0.325 [−0.449,−0.203] | +0.518 [+0.403,+0.647] |
| TTS→Natural | SP_EM | −0.195 [−0.289,−0.109] | −0.316 [−0.408,−0.227] | +0.511 [+0.441,+0.590] |

更早的 16,640 端点 local-response 同时族有 14,560 个可估计、2,080 个 `INCOMPLETE_SUPPORT`，237 个区间完全低于零、0 个完全高于零。FULL 与 VALID 的 76 个 phone-local、26 个 overall-window 负向端点在本 cohort 完全重复；guard20 有 12 个 overall-window 负向端点且没有 phone-local 负向端点；guard15 有 21 个 overall-window 负向端点且没有 phone-local 负向端点。旧族 n/ŋ 子集在 FULL/VALID 有 71 个负向、0 个正向、441 个跨零端点（512 个计划 phone-local 端点）；guard15 全部跨零；guard20 有 256 个 incomplete、其余 256 个跨零。旧视觉十臂结果中，八个处理相对 RT 的主 phone-separation 区间均跨零。

新视觉 geometry 族有 54 个预定端点：两方向 × aperture、绝对 aperture 速度、phone-class separation 三指标 × 九个对比，40 名 speaker 共用 bootstrap，family max-|t| 临界值 3.734945。方向按供体→受体定义：Natural→TTS 是自然语音供体移植到 TTS 受体；TTS→Natural 是 TTS 供体移植到自然语音受体。mean aperture 是以双眼距离归一化的口部开度（无量纲）；绝对 aperture 速度是相邻有效帧之间开度绝对差除以 q00 参考时间间隔，单位为归一化开度/秒，缺失帧不跨越连接；phone-class separation 是每条语句内有支持的 token 开口元音平均开度减双唇音平均开度，不是 AUC。总体几何端点使用 40/40 完整 speaker 对；phone-class 指标按预冻完整配对掩码，Natural→TTS 有 17/40 对、TTS→Natural 有 16/40 对。phone 对齐使用冻结的 −85 ms 偏移。唯一同时 99% CI 不跨零的是 **Natural donor→TTS recipient 的 mean-aperture trajectory-beyond-grid（q01−SELF_GRID）**：+0.001519 eye-distance-normalized aperture（精确值 +0.0015189082417374758；CI [+0.00008576689016922038,+0.002952049593305731]）。其余 53 项同时 CI 均跨零。该动作端点与 Sync-C 固定评分音端点分别估计；aperture 增加和 C 下降在该干预下同时出现。

## 4. 结论范围

增益路径识别出正向生成响应：gain 改变 log-mel 和实际编码方向，在固定评分音下提高 SyncNet 端点，并在固定 RMS 干预下伴随可见动作变化。μ/ρ 路径定位出 recipient 相容性约束：ρ 操作的是 token 内相对频带比值的时间轨迹，Natural→TTS 的局部 trajectory/joint 处理与 TTS→Natural 的 donor 谱形移植均显示 recipient 结构影响匹配结果。视觉 geometry 族同时测得一个 aperture 运动端点。三类读数分别描述音频评分、native gap 与可见运动，按各自的评分输入和分母报告。

结论限于冻结的语料、模型与评分设计。FULL/VALID 在本 cohort 的重复数值、各支持之间的区间差异及短 T 的 1,472 个 `INCOMPLETE_SUPPORT` 端点均保留；phone-local 定位仅在 FULL 支持下有同时区间证据。AP 是预定帧上的非周期性参数，Sync-C 是 SyncNet 模型端点；局部 mel-column 覆盖是约 350 ms 坐标邻域，不能直接读作某个独立 phone 的因果效应。

## 5. 证据与复核入口

- 机器可读来源索引：`.codex/scratch/source_registry_report25_attempt08.json`（raw SHA `dd62486123cf1f485ec65c9c62cd2c6e1a53b3614b0e3d1e54de25b610e19eb6`）。它记录全局端点、计划/估计/不完整分母、native 与固定-A estimand、来源、绘图及复核收据；digest audit08 对 150 项文件/载荷哈希及字节数核对全部通过，0 项失败，收据 SHA `ecf50ae67cabd00933cf03973ff3f70e1bb5b40da2aaa9ba67a765e38670a28e`。
- μ/ρ 消费者的 clean restore 共 646 个文件（645 个清单成员加输出 manifest）；完整 family 有 11,776 行。family endpoints 文件 SHA `c47cb2e0fb5b6eb014a54ae9b13fc86774cf8ce983753495e8a020e5a819de1c`，restore receipt SHA `d918f44bb1e124f31aef4601327740d8b8ac09a68f92fa2ed1e84c9bd6422871`。ROOT exact-join 比较 33,868 个投影单元、0 mismatch。
- 局部重建覆盖 160 contexts、80 clips、40 speakers，重算 665,600 个 clip-level 比较及 8,320 个局部端点；最大绝对残差 `4.440892098500626e-16`，official mel starts 与五帧 window-union CSR 逐项相等。
- μ/ρ 最终图表包含 8 个资产：9 页 global forest、32 页 local heatmap、CSV 表格及 verifier/QA 收据。render manifest raw SHA `70d101c2435d0693ba70ae5a6fc357bf57c9c2f0c67ff3bc3e57b822678cb0fc`；所有输出与 PDF QA 的逐文件 SHA 见来源索引。
- 54 项视觉 geometry 的独立重建 receipt 有 160 contexts、800 tracks、630 token rows；primary、独立重建、54 行 render、PDF raster QA 和 ROOT 六图 visual QA 均通过。跨运行时复算使用 NumPy 2.3.5 与 1.26.4：speaker 值、bootstrap draw/count、分母与 mask 一致；178 项比较中派生 SE/max-|t|/CI 最大绝对残差 `8.881784197001252e-16`，原始容差 `atol=2e-12, rtol=2e-10` 内，54 项 CI 零分类相同。
- 完整视觉原始输入档案经独立恢复与逐成员核验：receipt `.codex/scratch/shape_visual_family_archive_attempt10/full_visual_root_qualification_attempt03.json` raw SHA `44428b1853cdc89185d5116ff9927826b98a1e3d85ba72aff9e19feffa78c418`；覆盖 1,898 个归档成员、1,896 个源文件、493,892,958 字节和 29 个输入封印，源/归档清单一致且所有恢复成员均重新计算哈希。转移清单 SHA `719b3c61dea682929068b55b20555935f6b71fe99533236103b713c71831f3f5`。
- 54 项可视化：`visual_geometry_endpoints_all54.csv`、`visual_geometry_family_all54.pdf` 及三张指标 PNG；render manifest raw SHA `4ceff74663a19d22c8dad4abf4f7f646f41284c767f0089b9f3af33985761c76`、payload SHA `983765b6f26f76391fa103e7503ba372787551db41c1f8a69cad1ddae4d9bd54`。独立 render verifier receipt SHA `6bf57fa712059edb942f072a3c4e1dd4dbb49debbba339dd08437484f08e985c`，PDF raster QA manifest raw SHA `737d6f056386a865113ce7631faeea8b09b51ffed5b93afa8ef83e1470f54867`，ROOT visual QA receipt SHA `343fc4b5a70fd94c0b48493faa40b9af81f8519e6cd7bdf35d470945eacc64cb`。