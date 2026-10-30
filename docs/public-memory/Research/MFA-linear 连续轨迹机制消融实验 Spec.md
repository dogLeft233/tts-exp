---
title: MFA-linear 连续轨迹机制消融实验 Spec
type: spec
permalink: tts-exp/research/mfa-linear-连续轨迹机制消融实验-spec
status: planned
date: '2026-09-16'
tags:
- mfa-linear
- mechanism
- spec
- trajectory-ablation
---

# MFA-linear 连续轨迹机制消融实验 Spec

状态：🔄 已设计，未实现、未运行。面向下游 GPT Luna Max。
任务：实现并运行本 spec，得到可判读的小规模机制实验；不训练模型、不调参、不扩展数据集。
本文件是科学协议的单一真相源。历史脚本仅提供函数与资产，历史脚本的条件数、缓存分数、分支策略不覆盖本协议。

## 1. 问题、假设与实验边界

问题：历史 MFA-linear 的原生 SyncNet 增益，主要与重合成处理、TTS 来源特征，还是音素内部随时间变化的特征有关？

三个假设可以同时成立：
- H1 重合成贡献：自然音频经过同一 WavLM→HiFi-GAN，也可能变得更容易驱动 Wav2Lip。
- H2 来源贡献：同样在自然时间网格、同样经过声码器，TTS 来源轨迹仍比自然来源轨迹有额外作用。
- H3 动态贡献：保留音素内均值、时长与音素顺序，逐步减弱音素内变化，会逐步削弱 TTS 的优势；自然来源的对称处理用于检查是否只是任何语音都怕这种操作。

本轮只研究历史阳性条件中的“音素内特征变化幅度”。不独立识别帧顺序、共发音、纯音色或纯韵律，不证明整个特征空间线性。H2 仍包含历史 TTS 处理链与插值操作，不能称为纯 TTS 因果效应。
保持特征均值不等于保持输出的音素身份、音质、响度或真实发音边界。声码器是非线性的，需结合下述检查解释。
15 条来自同一 S0765 speaker，历史结果已看过；所有分析均为探索性、该条件内的配对诊断，不称为独立验证或跨说话人泛化。

## 2. 固定资产与选样：使用历史 15 条，不重新选样

仓库根：`[redacted-local-path]`。以下路径相对仓库根。

A. cohort、历史 TTS 条件特征、源音频身份：
`runs/knn_vc_poc_valid15_mfa_linear_bridge_20260813/summary.json`

B. 自然/TTS MFA token：
`runs/two_stage_hubert_aishell1_20260810/data_boundary/aishell1_400_raw_mfa_faster_qwen3_heldout.json`

C. natural_raw、raw_tts 评估音频、逐样本人脸视频、Wav2Lip/SyncNet checkpoint hash：
`runs/rhythm_timing/20260813_syncnet_valid15_mfa_bridge/manifest.json`

严格按 A 的 `selection.ordered_paired_keys`，15 条全部使用，逐条按 paired_key 关联 A/B/C；sample_id 仅作显示名。A 中 speaker_group 必须是 S0765。不要从完整数据池重新选“最好”或“最干净”的 15 条。
本 spec 编写时 A/B/C、15 对源音频、15 个 reference face video 均存在。未来路径移动允许按相同 SHA 找回资产，不允许换样本、换 TTS 或换人脸。

实现 prepare 时：
1. 校验 A/B/C 文件与每条已记录的输入 SHA；B 的 hash 应与 A.manifest.alignment_sha256 一致。
2. B.records 按 (paired_key, condition) 读取 natural/tts tokens，核对源 SHA 和 A 的 token hash。旧绝对路径前缀可映射到当前 repo，内容 hash 必须一致。
3. C 的 natural_raw/raw_tts 行分别确定 N_RAW/T_RAW 音频；同一 paired_key 的 face 必须一致。
4. A.items[].conditions 中 `paired_tts_mfa_linear` 行的 feature_path 指向 .pt，取 **conditioning**，不是 **output**。
5. 将这些定位结果、输入 SHA、模型 SHA、样本顺序、代码版本写入一次普通 `inputs.json`。不需要另建数据库、注册表或多级签名系统。

若缺资产或身份不一致：列出具体缺项，状态 incomplete；不得用另一套实验静默替代。

## 3. 八个音频条件

记 E 为冻结 WavLM-Large L6，D 为冻结 prematched HiFi-GAN。

| arm | 输入与生成方式 | 意义 |
|---|---|---|
| N_RAW | C 中已有 natural_raw 波形 | 自然基线 |
| T_RAW | C 中已有 raw_tts 波形，保留自身长度 | 历史 raw-TTS 参照 |
| N_100 | D(Z_N) | 自然直接重合成 |
| N_050 | D(S_0.5(Z_N)) | 自然音素内变化保留 50% |
| N_000 | D(S_0(Z_N)) | 自然音素内变化归零 |
| T_100 | D(Z_T) | 历史 MFA-linear 的完整条件轨迹 |
| T_050 | D(S_0.5(Z_T)) | TTS 音素内变化保留 50% |
| T_000 | D(S_0(Z_T)) | TTS 音素内变化归零 |

Z_N = E(N_RAW)，形状 [F,1024]。
Z_T = A 的历史 .pt 中 conditioning，已经映射到自然时间网格，形状必须与 Z_N 一样。

这里刻意以历史 conditioning 为 T_100 的真相源，避免现有代码的重采样修正改变待解释的处理条件。不要重新提取 T_RAW 特征替代它，也不要对 Z_T 再做一次 MFA 映射。报告明确写“历史处理链的条件特征”；该实验不验证当前最新版 MFA-linear 生成器。

A 的 feature 文件 SHA 必须与 conditioning_sha256 字段一致；该字段是整个文件的 SHA，不能误当 tensor bytes 的 SHA。
所有六个重合成条件都重新经过同一 D 生成；不把历史 T_100 音频与新生成其他条件混用。

模型使用：
- `scripts/wavlm_knn_vc_adapter.py:WavLMKNNVCAdapter`
- revision `c616845c4e309e24d5927f15adbdf277a3d65358`
- 16 kHz、L6、1024 维、320 samples hop；eval + inference_mode。
- WavLM/HiFi-GAN hash 取 A.model 并校验；Wav2Lip/SyncNet hash 取 C 并校验。
- 无训练、无 kNN 检索、无新 TTS API 请求。

## 4. 消融的精确定义

### 4.1 哪些帧参与

调用现有 `knn_vc_retrieval.frame_owners(F, natural_tokens)`，沿用帧中心 (i+0.5)×320/16000，不自创另一套时间偏移。
调用 `matched_span_map(natural_tokens, tts_tokens)` 得到音素匹配关系。

某自然 token occurrence 只有同时满足以下条件才可消融：
- owner.is_silence 为 false；
- 该自然 span_index 在 matched_span_map 中；
- 该 occurrence 在自然网格上至少拥有 2 帧。

同一组 occurrence/frame indices 同时用于 Z_N 和 Z_T。两次出现的相同音素是两组，不能按 label 跨 occurrence 汇总。
静音、spn、未匹配 fallback、仅 0/1 帧的音素保持各自原值。历史 fallback 不丢样本、不换 natural 特征、不参与消融。
frame_owners 的 coverage 错误要报出；本轮不重新 MFA、不临时延长 token。

### 4.2 运算公式：先对齐，再在自然网格内求均值

对来源 X∈{N,T}、可消融 occurrence p、帧集合 I_p：

```python
mu = Z_X[I_p].mean(dim=0, keepdim=True)  # [1,1024]
out[I_p] = mu + keep * (Z_X[I_p] - mu)
# keep = 1.0 / 0.5 / 0.0
```

out 初始为 Z_X.clone()，未选中帧保持不变。keep=1 直接复用完整 Z_X。
均值在每个来源自己的 Z_X 内求；不能混合 N/T 均值、不能用全语料音素中心、不能对 waveform/mel 做这个操作。
不加 crossfade、平滑、随机扰动、幅度归一化；它们都会增加实验因素。

可核验性质：
- 每个可消融 occurrence 的均值不变；
- keep=0 时该 occurrence 所有向量相等；
- keep=0.5 时，对均值的残差幅度为原来的 0.5，平方能量为 0.25；
- occurrence 内相邻帧差的平方能量同样变为 keep² 倍；
- occurrence 边界处的跳变不保证保持，必须单独报告。

这测的是动态幅度，不是打乱时间顺序。均值化可能让声码器输入偏离常见分布；不能只看 T_000 降分就证明 TTS 特有机制。

## 5. 波形、长度与最低限度质量检查

- N_RAW/T_RAW 直接使用 C 中的 16 kHz mono 音频，核验采样率与 hash。编码用解码 float32；不再重采样历史基线。
- 六个重合成条件均用同一 D；调用 `run_knn_vc_poc.exact_natural_length`，仅末尾裁剪/补零到 N_RAW 的样本数。记录 raw decoder length 和 adjustment。
- 保存 FLOAT WAV 作分析母版；生成和评分统一使用同一份 PCM16 WAV，六条件都按同一量化规则转换。N_RAW/T_RAW 若已是 PCM16 则直接用原文件。
- 不做 loudnorm、peak normalization、VAD trim、变速或强制 phone 边界修复。若 FLOAT 越界，记录样本数与最大值；不可隐式 clip 后当合格。该 cell 标记失败并报告。
- 每条每臂记录 N、duration、peak、RMS、非有限值数、越界数、末尾补零长度。
- 每条记录可消融 occurrence 数、可消融帧比例、matched coverage、fallback 帧数；没有可消融帧时保留该样本并标记无干预，不用其他样本补位。
- 每条每来源记录原始/消融后的音素内差分能量与跨音素边界差分能量，分别统计，不把边界跳变当作音素内部变化。

固定听检样本为前 3 条（IDs 1/2/3），所有八臂生成后听同一句的差异，记录是否明显破音、吞字、静音、边界点击。不因听感或分数删样本。若执行环境无法听检，写“未完成”，不伪造主观结果。
本轮不引入新 ASR、MOS 模型或声学训练。质量检查不充分时，结论保留“声码器适配/可懂度混杂”。

## 6. 视频生成

每条都用 C 中同一 reference face **视频**，不改成静态首帧或新图片。它是历史 Ditto 生成的参考，不是真实自然视频。
冻结 Wav2Lip checkpoint，复用 `eval_mfa_linear_wav2lip_syncnet.py` 中的命令参数：
`--face_det_batch_size 4 --wav2lip_batch_size 4 --nosmooth`。
沿用原视频帧率并验证输出 25 fps；同一 paired_key 的所有臂使用同一 face 和生成参数。
每个 sample/arm 独立 cwd/temp，这是已有脚本真实出现过的污染点。
八臂全部 fresh render，15×8=120 个视频；历史分数仅供背景阅读，不能填入新矩阵。
T_RAW 保留自己的音频长度。其他七臂统一自然长度。不要靠 `-shortest`、音频拉伸或视频位移制造等长。

先做第 1 条全八臂 smoke，检查音频/视频/评分连通性；通过后继续同一协议全部 15 条。smoke 不能用来选择强度或删臂。

## 7. 评分：原生 + 固定音轨 + 最小 2×2

记 V_a 为 arm a 的生成视频，A_b 为 arm b 的评分音频，S(a,b) 为 SyncNet 分数。

每条计算以下 **15 个唯一评分 cell**：
1. 八个原生 cell：S(a,a)，a 遍历八臂。
2. 六个共同自然音轨 cell：S(a,N_RAW)，a 遍历六个重合成臂。
3. 一个补充 cell：S(N_RAW,T_100)。

其中 S(N_RAW,N_RAW) 已在第 1 项，不重复计算。总计 15×15=225 cell。
T_RAW 仅参与原生描述性比较，因时长不同，不作 natural replacement，也不进入后面的配对机制主对比。

核心 2×2 为：
| video / score audio | N_RAW | T_100 |
|---|---|---|
| N_RAW | S(N_RAW,N_RAW) | S(N_RAW,T_100) |
| T_100 | S(T_100,N_RAW) | S(T_100,T_100) |

直接给同一 video frames 输入不同评分音频即可，不必实际封装 105 个 replacement 文件。这样避免把 AAC/PCM 或 mux 差异混入矩阵。

### 7.1 固定 crop 与共同时间支持

使用冻结 SyncNet V2，保留完整 lag 曲线，lag 从 -15 到 +15 帧。复用：
- `scripts/experiments/static_image_bridge/score_worker.py` 的 audio_embedding、visual_embedding 及逐 lag 距离计算；
- `scripts/experiments/static_image_bridge/images.py` 的检测选择与 score_box 函数；
- 该 worker 的网络/frontend 可以复用，但不要搬来其整套历史实验 runner。

评分 crop：在每条 C.reference face 视频的第 0 帧上检测一次，按现有 select_detection 规则得到 score_box；保存到 inputs/score_box_<id>.json。同一条所有 video/audio cell 都用这个 box，按现有 crop_zero_padded→224×224 处理，每条可以有不同 box。
准备时检查参考视频首/中/末三帧 crop 确实覆盖脸和嘴；明显不覆盖则报告该协议准备失败，不通过试不同 box 的 SyncNet 分数挑 crop。
这使用固定 crop + 共同支持，与旧官方 pipeline 自动 tracking 的汇总分数不是完全相同协议，因此必须先检查本轮阳性条件是否仍存在，不能直接引用旧 +0.964 当本轮结果。

按以下公式汇总，避免各 lag 用不同数量的帧：
- visual[t] 为从视频第 t 帧开始的连续 5 帧 embedding。
- audio[t] 为 MFCC 从 4t 开始的连续 20 列 embedding。
- M[t,k] = L2(visual[t], audio[t+k])，k=-15…15；k 正号明确表示取更晚音频，不要与官方 stdout offset 符号混写。
- 对同一条除 T_RAW 原生外的 14 个 cell，F=min(所有这些 cell 的 visual_count、audio_count)，共同有效 t 为 range(15,F-15)。要求至少 50 个 t。
- T_RAW 原生单独以同样规则取自身支持；其分数与其他臂的差值是描述性参照，不能作等时长机制证据。
- d(k)=mean_t M[t,k]；D=min_k d(k)；C=median_k d(k)-D；k*=argmin，平局按 -15…15 顺序取第一个。
- 同时保存 median_k d(k)、k* 和 d(0)。不能把 C 上升一律写成匹配距离改善，C 的参考中位数也可能变了。
- 自然音轨列额外保存 d(k_N)，k_N 来自本条 S(N_RAW,N_RAW) 的 k*，不对每个候选重新选这个坐标。这是固定 lag 的描述性检查，不重排视频或音频。

每个样本、视频臂只提取一次 visual embedding；每个音频臂只提取一次 audio embedding，重复 cell 复用。无需并行 GPU 调度。保存每个 cell 的 d(k)，不要求存巨大的逐帧哈希清单。

## 8. 预先固定的对比和统计

所有减法先逐条计算，再对 15 个 utterance 求均值。C 越高越好；D 对比统一转换为“基线 D − 候选 D”，正数代表改善。

定义两种分数视角：
- q_own(a)=C(a,a)：原生视角，音频与视频同时随 arm 改变。
- q_N(a)=C(a,N_RAW)：固定自然音轨视角，评分音频固定。

### 8.1 先报告阳性条件

G_own=q_own(T_100)-q_own(N_RAW)。
均值≤0：报告“本轮协议没有复现原生正向效应”，完成既定矩阵后停止解释“为何保留增益”，不换样本、crop、强度或模型重试。
均值>0但普通95% bootstrap CI跨0：报告“阳性参照不确定”，后续仅作机制线索。
普通95% CI下界>0：报告“本历史样本/当前协议中阳性参照成立”。
同时报告对应 D 改善。若 C/D 冲突，分别表述。

### 8.2 三个核心对比

分别用 q=q_own、q=q_N 计算：
- R = q(N_100)-q(N_RAW)：自然编码/重合成相对 raw 的贡献。
- E = q(T_100)-q(N_100)：完整轨迹的来源处理链差异。
- I = [q(T_100)-q(T_000)] - [q(N_100)-q(N_000)]：TTS 相比自然是否更依赖所消融的音素内变化。

注意：R+E=G 在同一视角下是代数分解，不代表两个独立因果机制。R 包含编码、重合成和统一量化，不能单独归因于 HiFi-GAN。
I>0 仍可能受两类输入对声码器消融的不同敏感度影响，只是来源×动态操作的交互线索。

**q_N 下 R/E/I 为三个机制主对比。** bootstrap 的抽样单位为 utterance，每次有放回抽 15 个 index，各条件/两种指标使用同一组 index，seed=20260916，10000 次。
报告均值、中位数、正向条数、普通95% percentile CI；三个主对比另给 Bonferroni 98.333% CI（分位数0.0083333、0.9916667）。D 为辅助指标，不另设显著性结论。原生视角、剂量与2×2只给描述性95% CI，明确未校正。
同一 speaker 不能做 speaker bootstrap，也不能把帧当独立样本。没有必要新增 p 值/复杂混合模型。

### 8.3 50% 中间强度与输出检查

分别画 N/T 的 keep=0、0.5、1 曲线，画原生 C/D 与自然音轨 C/D。
报告逐条和总体的 q(T_100)-q(T_050)、q(T_050)-q(T_000)，自然来源同样报告。
只有 0% 极端条件崩坏、50% 几乎不变时，不称为清晰剂量响应。
q_N(T_000)-q_N(N_000) 可作为保留均值后的来源差异线索，但均值仍带有音色/声学信息，不叫“纯音素语义”。

2×2 报告：
- 固定 A_N 的视频差：C(T_100,N_RAW)-C(N_RAW,N_RAW)。
- 固定 A_T 的视频差：C(T_100,T_100)-C(N_RAW,T_100)。
- 固定 V_N 的音频差：C(N_RAW,T_100)-C(N_RAW,N_RAW)。
- 固定 V_T 的音频差：C(T_100,T_100)-C(T_100,N_RAW)。

以上不是对 SyncNet 偏差和真实运动改善的完美因果分解，不相加成“音频贡献百分比”。

## 9. 结果判读表

| 观察 | 允许的解释 |
|---|---|
| N_100 比 N_RAW 好，T_100 比 N_100 的差异很小/不确定 | 重合成处理可能解释部分结果；差异不显著不等于两者等效 |
| T_100 比 N_100 好，固定自然音轨下也成立 | 该历史 TTS 来源处理链提供额外、能影响视频输出的有用信息 |
| T 动态减弱时下降，N 同样下降，I 不确定 | 音素内变化可能普遍重要，未定位 TTS 特有来源 |
| T 动态逐步减弱时下降更多，I 主CI>0，50%也受损，质量检查无明显崩坏 | 支持“TTS 的额外作用部分依赖这些音素内变化”的机制线索 |
| 原生上升但自然音轨列不升，2×2对音轨敏感 | 只能说明条件内AV一致性改善；自然兼容性未建立，可能是声学域或局部时序匹配 |
| N/T均值化都破音、边界差分大幅增大或可懂度可疑 | 不能把降分归结为有用动态；本干预存在声码器/音质混杂 |
| 50%好于100% | 更符合适度平滑有利，不能写成“动态越多越好” |
| T_100未复现原生优势 | 本轮未获得可解释的阳性参照；不能宣布所有历史MFA-linear增益不存在 |

所有结论必须区分“数据观察”和“机制猜测”。本轮没有真人嘴型真值或系统主观评估，不宣称真实唇运动改善。

## 10. 最小实现与交付

建议只新增：
- `scripts/experiments/mfa_linear_trajectory_ablation.py`：prepare/audio/render/score/analyze 五个 CLI stage；可把独立评分 worker 放同名前缀文件，避免环境冲突。
- `tests/experiments/test_mfa_linear_trajectory_ablation.py`：少量科学协议测试。
- `runs/mfa_linear_trajectory_ablation_20260916/`：本轮产物。

预计命令接口（下游需要实现，不是现在已存在的命令）：
```bash
python scripts/experiments/mfa_linear_trajectory_ablation.py --stage prepare --output-dir runs/mfa_linear_trajectory_ablation_20260916
python scripts/experiments/mfa_linear_trajectory_ablation.py --stage audio --output-dir runs/mfa_linear_trajectory_ablation_20260916
python scripts/experiments/mfa_linear_trajectory_ablation.py --stage render --output-dir runs/mfa_linear_trajectory_ablation_20260916
python scripts/experiments/mfa_linear_trajectory_ablation.py --stage score --output-dir runs/mfa_linear_trajectory_ablation_20260916
python scripts/experiments/mfa_linear_trajectory_ablation.py --stage analyze --output-dir runs/mfa_linear_trajectory_ablation_20260916
```

提供 --smoke：仅处理既定第1条，不另选样本；完整执行后可复用本轮相同输入/参数已完成的 smoke cell。
环境路径允许 CLI 指定，默认借用现有脚本中的 Python 路径；不要自动重装模型环境。

必须交付：
- inputs.json：15条身份、路径、输入与模型SHA、评分box、协议常量。
- audio_qc.csv：120行；包含两条raw条件，六条resynthesis条件的干预/长度记录。
- features/：至少保存每条 Z_N、Z_T、occurrence masks，消融特征可由固定公式重建。
- audio/、videos/：全部八臂可查看；音频母版和PCM16明确区分。
- scores.csv：225行，含 sample_id、video_arm、audio_arm、C、D、curve_median、k_star、d_zero、共同支持数；自然列另含固定k_N距离。
- curves.npz 或每条 curves.json：完整31点曲线，可复算C/D。
- analysis.json + report.md + 一张 keep剂量曲线图。
- 报告首段写阳性参照状态、R/E/I、主要混杂和下一步；分数展示3位小数，原始数据保留完整精度。

最低测试：
1. 已知小矩阵验证 keep=1恒等、keep=0.5残差减半、keep=0均值常量及均值保持。
2. 两个同label不同occurrence互不混合；静音/未匹配/单帧保持原值；N/T共用同一mask。
3. 测试先对齐后pool的顺序；禁止 .pt 的 output 错作 conditioning，可用哨兵张量验证载入键。
4. 验证每条15个唯一cell和总225个；T_RAW不进入自然音轨列。
5. 人工距离曲线验证C/D、lag符号、共同支持与固定k_N；bootstrap按utterance共同抽样。

完成条件：完整样本与cell数、所有结果有限、必要输入身份校验通过、可从scores/curves重算主对比、报告解释与判读表一致。若有未修复的cell失败，报告incomplete及失败列表，不删掉后宣称15条完成。
除了与这次计算直接相关的函数和测试，不重构旧实验框架，不加新的任务编排、训练平台或审批门禁。实验完成后的正式记忆更新遵循 Startup Router；本spec本身不是实验结果。

## 11. 下游执行顺序

1. 阅读本spec，核验第2节资产；写prepare和输入清单。
2. 实现纯特征消融函数并通过最低测试。
3. 生成第1条八臂音频/视频与15个评分cell，检查shape、crop、支持和分数复算。
4. 保持参数不变跑全15条；技术失败修复后仅补失败cell，不根据科学结果改协议。
5. 计算既定对比，完成报告。阳性参照弱或结果为负，也正常交付。
6. 返回report路径、主要结果和限制。任何扩样、训练、换脸或新一轮参数试验作为后续建议，不并入本轮。

## Observations

- [status] planned
- [question] 历史MFA-linear原生增益是否依赖TTS来源及音素内连续变化，还是可由重合成处理解释？
- [design] 固定历史S0765 n=15，八臂、120视频、225评分cell；冻结模型、无训练。
- [boundary] 当前为探索性机制spec；单speaker、历史已见样本、音素均值化存在声码器适配混杂。

## Relations

- relates_to [[MFA-linear 冻结 VC 重合成与下游验证]]
- relates_to [[MFA-linear 机制小样本诊断 2026-09-13]]
- relates_to [[MFA-linear 音素时序渐进修复 2026-09-13]]

## Changelog

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 根据用户要求设计供GPT Luna Max实现的冻结模型轨迹消融实验；仅写spec，未运行 | September 16, 2026 | user request / agent design |
