---
title: VSR 视觉内容辨识与 TTS 配对验证实施部署 Spec
type: note
permalink: tts-exp/research/vsr-视觉内容辨识与-tts-配对验证实施部署-spec
status: implemented
question: 冻结中文 VSR 能否在历史配对视频上辨认目标内容，以及 TTS 是否改善该辨识？
tags:
- vsr
- tts
- implementation-architect
- spec
---

# VSR 视觉内容辨识与 TTS 配对验证实施部署 Spec

## 1. Objective

为下游 agent 实现一个小型、可复算的中文 VSR 实验：在同一目标文本、同一人像的历史自然/TTS 驱动 AVTR-1 视频上，检验 TTS 视频是否包含更容易被冻结视觉识别模型辨认的目标内容。先验证测量工具，再解释组间差异。

本轮交付是实现与部署说明；实现、全量运行和独立复算已完成，结果见对应 Experiments 笔记。采用 implementation-architect 的 11 节结构。只需要一个运行脚本、一个纯指标模块和一个对应测试文件；不训练、不生成新 TFG、不新建通用框架。

**必须纠正上一轮 pilot**：此前将 `results/avtr1/*/1.mp4` 与 `data/aishell1_100_zh/transcripts/1.txt` 配对，混用了两套样本编号。AVTR-1 的源音频是 `data/data/audio/`；sample 1 的历史目标文本是“组织了第一批七个地区城市开展三网融合试点”，并非“甚至出现交易几乎停滞的情况”。`/tmp/avsr_avtr1_pilot_20260918.json` 和 `/tmp/avsr_avtr1_intermediate_20260918.json` 中的旧 CER、目标 CTC loss 及其 T/N 差异全部标为 `INVALID_REFERENCE_BINDING`，不得用来选择指标、样本或支持结论。工程推理成功和输出维度仍有效。旧结果保留，不覆盖；本轮必须重新绑定文本、重新计算。

科学问题分开：
- H_measure：模型在本数据上更支持正确文本，并且这种支持至少部分来自嘴部的时间变化。
- H_tts：在 H_measure 通过后，TTS 视频的目标内容辨识优势高于自然视频。
- 本实验不直接检验嘴型与音频的毫秒级同步，不证明人类感知改善，不证明 TTS 增益机制已被因果定位，也不授权训练 replace 增强头。

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 初始实验与下游部署规范；纠正历史 pilot 的跨数据集文本绑定 | September 18, 2026 | user requested |
| 自审补齐 native 嘴部张量、随机性与多脸检查，保证对照可复算 | September 18, 2026 | agent self-review within requested scope |
| 完成实现与全量复算；130 视图、独立 CTC/统计验证通过，最终判读 NO_CLEAR_PAIRED_GAIN | September 18, 2026 | agent implementation |

## 2. Repository Model

当前已验证链路：

`results/avtr1/{natural_raw,tts_raw}/{1..13}.mp4`
→ MediaPipe 四点人脸定位
→ 官方 `VideoProcess` 灰度嘴部裁剪 96×96
→ `VideoTransform` 中心裁剪 88×88、标准化
→ `E2E.encode` 输出 [T,256]
→ `model.ctc.log_softmax` 输出 [1,T,3363]
→ 独立计算文本 CTC 分数。

官方 `AVSR.infer` 另外接 attention decoder + CTC + 中文 LM 的 beam search；它只作次要 CER，不进入主指标。无 LM 的 CTC head 仍包含训练数据的统计偏好，因此必须有内容与运动对照。

部署实情：
- 本地 `third_party/AVSR` origin 为 `https://github.com/AV-LLM/AVSR.git`，它是作者仓库的 fork，commit `84c148a33c14189eda9dcabcc39fed96f8da30d9`。
- 作者来源：[Visual Speech Recognition for Multiple Languages](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages)。本地 CMLR 权重属于其多语言 VSR 系列，不应写成最新 Auto-AVSR 英文权重。
- 环境 `[redacted-local-path]`：Python 3.8.20、torch 2.1.2+cu121、torchvision 0.16.2+cu121、torchaudio 2.1.2+cu121、mediapipe 0.10.9，pip check 已通过。
- 仅中文 CMLR 视觉权重和中文 LM 已落地，其他 VSR 权重不能假设存在。
- 当前可用磁盘约 10 GB；预计本实验产物 <1 GB，不复制原视频，不进行空间清理。

输入来源：
- 生成器记录 `basic-memory/docs/deployment/avtr-1.md` 的 AISHELL-1 Batch 节绑定音频、参考图和 TTS run。
- `runs/r2_assets/README.md` 的 Natural Audio 表提供全部 13 个 ID、AISHELL WID、文本。
- `runs/r2_assets/transcript.json` 只覆盖 1..10，需与 README 交叉核对；不能将其当作完整 13 条。
- `runs/r2_assets/transcripts/all_transcripts.txt` 当前为空，不能作来源。
- `data/wav2sem_analysis/manifest/alignment.json` 为历史交叉检查，缺 sample 9；其中时间是均匀分段，不能冒充 MFA 时间真值。
- 原 `runs/r2_faster_qwen3_20260707T145233Z/` 顶层目录当前不存在；历史部署记录给出的路径不等于当前资产存在。
- 13 条自然语音全部属于说话人 S0764。任何 CI 仅描述本说话人的这些话语，不作跨说话人泛化推断。

## 3. Code Anchors

| path | symbol / section | current role | required change |
|---|---|---|---|
| `third_party/AVSR/infer.py` | `main` | 已成功的 Hydra 单视频入口 | 仅复用作部署 smoke；不修改 |
| `third_party/AVSR/pipelines/pipeline.py` | `InferencePipeline.__init__` | 加载配置、模型、检测器 | 新 runner 初始化一次；记录配置和权重哈希 |
| 同上 | `process_landmarks` | 检测或读取 pickle | 复用检测；在官方插值前记录原始缺失率，不传空字符串当路径 |
| `third_party/AVSR/pipelines/data/data_module.py` | `AVSRDataLoader.load_data`、`load_video` | video 分支仅使用帧张量 | 通过新 runner 生成无音轨临时输入并复用预处理，防止模型接收到声音 |
| `third_party/AVSR/pipelines/detectors/mediapipe/video_process.py` | `VideoProcess.__call__`、`interpolate_landmarks` | 四点稳定、补缺失、灰度嘴部裁剪 | 保持原行为；先保存原始 landmarks 有效掩码的副本 |
| `third_party/AVSR/espnet/nets/pytorch_backend/e2e_asr_transformer.py` | `E2E.encode` | 输出最后一层 encoder；可提取 resnet | 主实验只使用最后一层，不扫层挑显著结果 |
| `third_party/AVSR/espnet/nets/pytorch_backend/ctc.py` | `CTC.log_softmax` | 汉字类别后验 | 复用；不调用其默认 zero_infinity=True 的训练 loss |
| `third_party/AVSR/pipelines/model.py` | `AVSR.infer` | beam 解码 | 仅原始 N/T 视频运行一次，附录报告 CER |
| `scripts/experiments/vsr_tts_pilot.py`（新增） | `audit_inputs, load_vsr, extract_views, analyze_run, validate_run, main` | 无 | 编排四个阶段，保存完整证据 |
| `scripts/experiments/vsr_tts_metrics.py`（新增） | `normalize_text, build_decoys, ctc_nll, content_margin, make_views, paired_summary` | 无 | 纯计算，可小张量测试 |
| `tests/experiments/test_vsr_tts_pilot.py`（新增） | 对应语义测试 | 无 | 见第 8 节 |

所有新增 Python 必须兼容 3.8。不要复制 `zip(..., strict=True)`、`X | None` 等新版本特性。第三方源码不修改；模型预处理以所固定版本为准。

## 4. Reference Pattern

- `scripts/experiments/tts_time_instance.py::write_json`：仿照临时文件 + os.replace，JSON 使用 allow_nan=False；不要 import 整个旧实验来复用十行 I/O。
- `scripts/experiments/tts_time_instance_recompute.py::bootstrap`：仿照先形成每个独立单位的配对差、再固定种子重采样的思路；本实验的单位为话语，n=13 且单说话人。不要复制旧实验的最小组数、多重比较 alpha 或 Python 新版本语法。
- AVSR 官方 `InferencePipeline` 和 `AVSR.infer` 已验证可运行，复用此装载链路，避免复制 encoder 架构。
- 没有足够相似的现成“VSR 内容校准 + TTS 比较”脚本；新增薄封装即可，不接入旧 F0/SyncNet gate 框架。

## 5. Invariants

1. 样本全集固定为 AVTR-1 IDs 1..13；这是已经观察过的探索性 cohort，绝不称为独立验证集。失败记录保留，禁止按 CER、CTC 或 SyncNet 删除样本。
2. 每条绑定 video 路径及 SHA、目标文本及来源 SHA、WID、source speaker、face 路径及 SHA、condition。两个视频按同一 WID 配对，文件名数字本身不是证据。
3. 目标是“预期说出的内容”；原 TTS 是否逐字无误尚未逐条确认。视频文字指标包含任何上游漏字影响，不能单独归因于生成器。
4. 视觉模型实际输入不含音轨，text 仅用于 encoder 之后计算分数，不参与裁剪、时长设置、特征提取或候选选择。
5. 全句主分析保留每个视频的全部帧；禁止截取 N/T 共同最短时段后仍使用完整句子真值。
6. 对照全部在相同已裁剪 tensor 上制作，禁止重新检测造成对照裁剪漂移。
7. CTC blank=0；目标不含 blank、unk、eos。遇 OOV、空文本、不可达路径或非有限值，明确失败；不静默转 unk、不用 zero_infinity 把失败变成零损失。
8. CTC 是汉字输出，不是音素后验；blank 不等于静音；置信度高或 blank 少本身不等于嘴型更好。
9. Frozen encoder 不训练，参数 eval、no_grad；主指标不用 beam/外部 LM；全部统计从保存的 float32 logp 复算。
10. 只写新 run；输入变化时拒绝 resume，重试用新目录。科学中性/反向结果也必须产出报告。
11. 任何实验结论都限定“该冻结 VSR 对这些视频的内容辨识”；VSR 未检出提升不能证明 SyncNet 提升全是评价器偏差。

## 6. Implementation Plan

### 6.1 部署检查与命令

默认复用已装环境，不更新第三方仓库、不升级依赖。每个 subprocess 用参数列表执行，避免 shell 拼接路径。

在仓库根检查：

```bash
[redacted-local-path] -m pip check
git -C third_party/AVSR rev-parse HEAD
df -h .
```

模型锁四个 SHA-256：
- `benchmarks/CMLR/models/CMLR_V_WER8.0/model.pth` = `b6e8238522d709ea9234dc680ce03b77b202ff3ce2179bc34eb364988bae4737`
- 同目录 `model.json` = `fca2428d9ac6d9e70bf4348bdd90787bb7d8f698ec5b3e4fac2436185487f334`
- `benchmarks/CMLR/language_models/lm_zh/model.pth` = `85470cc3e98c1875a007113780ea2966ada73394f6d86e0658adbf02f0dfca4c`
- 同目录 `model.json` = `bcfb190a447936524fbc95543ae4d8151fbffb812aaf241c2fdaabb81ddefae7`

相对路径均基于 `third_party/AVSR/`。保持主模型和 LM 现有路径。将官方 ini 复制进 run 并把其中四个路径变成绝对路径；模型参数不变。这样新脚本从仓库根运行也可用。import 前仅把 AVSR 根添加到 sys.path；不要永久 chdir。

需要核对原始官方入口时，在 `third_party/AVSR/` 内执行：
```bash
[redacted-local-path] infer.py config_filename=configs/CMLR_V_WER8.0.ini data_filename=[redacted-local-path] detector=mediapipe gpu_idx=0
```

如果环境/权重缺失，报告具体缺项；按作者 README 恢复 Python 3.8 环境和固定版本，权重仍取官方 CMLR model-zoo 链接并校验上述哈希。不要切换到英文权重继续评分。正常情况下不需要下载。

下游新增 CLI，所有命令在仓库根：
```bash
[redacted-local-path] scripts/experiments/vsr_tts_pilot.py --run-dir runs/vsr_tts_content_v1 --stage audit
[redacted-local-path] scripts/experiments/vsr_tts_pilot.py --run-dir runs/vsr_tts_content_v1 --stage extract --smoke
[redacted-local-path] scripts/experiments/vsr_tts_pilot.py --run-dir runs/vsr_tts_content_v1 --stage extract --resume
[redacted-local-path] scripts/experiments/vsr_tts_pilot.py --run-dir runs/vsr_tts_content_v1 --stage analyze
[redacted-local-path] scripts/experiments/vsr_tts_pilot.py --run-dir runs/vsr_tts_content_v1 --stage validate
```

支持 `--device cpu|cuda:0`，默认 cuda:0；无 CUDA 则明确提示使用 CPU，不切换模型。smoke 只计算 sample 1 双臂并写 smoke 状态；它不改变 13 对分母，不能生成正式结论。续跑同一已冻结输入只补缺失 cell，不重新选样本。现有科学任务的分支限制不自动迁移到本新 pilot；仍遵守用户当前 GPU 使用约定。

### 6.2 audit：来源、文本与冻结 manifest

以 `runs/r2_assets/README.md` 的 13 行为主，解析编号、WID、文本；与 transcript.json 的 1..10 逐字核对（见下方规范化）。发现冲突就阻塞对应 pair，禁止用 VSR 输出解决真值冲突。README 的 sample 5 原文“从体偏紧”即使疑似错字也保留并标记，不擅自改写。

源清单：
- N 视频 `results/avtr1/natural_raw/{id}.mp4`
- T 视频 `results/avtr1/tts_raw/{id}.mp4`
- N 音频 `data/data/audio/{id}.wav`
- face `data/data/image/{id}.png`

核验生成器部署记录所述来源，记录它与 README 的哈希及具体条目；所有 26 个 MP4 用 ffprobe 检查 25 fps、可解码、frame count、start time、duration；不要求 N/T 时长相同。N 音频时长与历史表允许因表格四舍五入相差 ≤0.03s，否则标记需调查。

原始 generation manifest 如可找到，额外绑定；找不到则 `provenance_level=historical_documented`，明确当前 SHA 只能防止本轮文件变化，不能追溯证明历史文件身份。这允许开展历史探索性 pilot，不能声称 cryptographically verified provenance。若文本/来源记录互相矛盾，`BLOCKED_REFERENCE_BINDING`，禁止靠同名文件猜测。

文本规范化：Unicode NFKC，去所有空白以及 Unicode category 以 P 开头的标点；不繁简转换、不自行将数字改汉字。保留 raw_text、normalized_text、token_ids。映射使用模型 JSON 的实际 char_list，确认 3363 类、blank=0、eos 在末尾；OOV 拒绝该 pair，记录字符。不删除难字以改善分数。

冻结 `manifest.json`：schema_version=1、protocol_id、candidate_count=13、所有 records（成功和失败）、模型/代码/配置/source 文档哈希、固定 seed=20260918、阈值、变换与 decoy。每条含 id/WID/speaker/face/N/T/target/eligibility/reason。不要把旧 /tmp 的 reference 搬过来。

### 6.3 正确文本与错误文本：在看 VSR 结果前构建

每个 target 从其余 12 条文本中选择 5 个不同的完整句子作 decoy，按 `(abs(L_j-L_i), numeric_sample_id)` 排序取前 5。排除规范化后与 target 完全相同的句子和 OOV 句子；不足 5 标记不可评分，不补造文本。N/T 使用同一组 decoy。保留这 6 个文本的长度与身份。

这使主指标询问“视频是否更支持目标内容，而不是仅对所有句子给出更高分”。不同文本长度的残余偏好由静态对照检查，仍需报告其局限。

### 6.4 extract：视频输入与五种视图

每个原视频以 ffmpeg `-map 0:v:0 -c:v copy -an` 生成 run 内临时无音轨 MP4，ffprobe 验证无音轨；解码原版与去音轨版视频帧 hash 序列相同，PTS 相对起点相同。处理完成删除本 runner 创建的临时 MP4，保留核验 JSON。单个样本替换音轨/去音轨的视觉特征一致性另见测试。

复用官方检测与预处理；在插值前复制 raw validity mask。每侧 detection_fraction ≥0.95，且最长连续缺失 ≤5 帧；否则整个 pair 从配对统计排除但记录失败。保存首、中、末嘴部小图用于检查，预览不得用于按口型质量选择样本。非单人脸/明显错脸明确失败，不尝试调 detector 阈值挑高分。

得到官方标准化 `x[1,T,88,88]`，T 必须与完整 25fps 视频帧数相同。定义：

| view | 构造 | 用途 |
|---|---|---|
| native | x 原样 | 原始内容分数与主比较 |
| frozen | 将 x 的第 floor(T/2) 帧复制 T 次 | 同人脸、同长度的静态对照 |
| reversed | x 时间维反转 | 保留帧集合、破坏原时间顺序的诊断 |
| matched | 将 x 重采样到 M 帧 | N/T 总长度相同的敏感性检查 |
| matched_frozen | matched 的中间帧复制 M 次 | 同长度敏感性检查的静态参照 |

每个 pair 的 `M=floor((T_N+T_T)/2+0.5)`。matched 用沿时间轴的线性插值、align_corners=True；每个像素独立，端点保留，空间不插值。这是视频速度改变对照，不叫自然时间结构不变，也不能保证完全去除了语速影响。不裁掉句子尾部；reversed 仍针对原始文本计算，禁止反转文本配合它。

五个视图各自重新前向编码；不能反转 logp 假装反转视频实验。保存每个视频的官方标准化 native 输入 `x[1,T,88,88]` float32 压缩 npz 一份，以便独立检查 frozen/reversed/matched 的构造；不为五个视图重复保存图像张量。保存每个视图 `encoder[T,256]` 和 `logp[T,3363]` float32 压缩 npz，原始 validity mask 和 frame 时间轴单独保存。matched 时间轴只标 normalized index，不能解释成 MFA 边界时间。

启动提取时固定 Python random、NumPy、torch seed=20260918，设置 cudnn.benchmark=False、cudnn.deterministic=True；smoke 实测同设备重复性，不声称跨设备 bitwise 一致。原始检测时统计每帧人脸数量；多脸帧记录并令对应视频不通过单人脸 QC，不能依赖官方 detector 的有缺陷最大框逻辑。计数可以用同一 MediaPipe detector 的额外无状态 CPU 检查，保持官方预处理和原始模型权重不变。

正常预算：13×2×5=130 个 encoder 前向，另有 26 次 beam 解码可直接复用 native encoder；同一模型加载一次。主流程无需 resnet、全部层 hooks、额外模型或新 TFG。保存真实耗时，不承诺固定 GPU 时长。

### 6.5 指标的唯一公式

每个 logp 和候选文本 y：
`ell(V,y) = CTC_NLL(logp, y) / len(y)`，越低越支持 y。
使用 CPU float64 的 `torch.nn.functional.ctc_loss`，输入 [T,1,C]，`reduction="sum", blank=0, zero_infinity=False`，然后手工除一次字符数。输入来自保存的 float32 logp，不重复 softmax 或按类别重归一化。

先检查 `T >= len(y) + adjacent_repeated_character_count(y)`。目标或任一 decoy 不可达则整个 cell 无效；不得丢掉这个 decoy再平均。非有限 loss 属失败，不置零。PyTorch 的 mean reduction 本身除目标长度，避免二次除长度；详见 [PyTorch 2.1 CTCLoss](https://docs.pytorch.org/docs/2.1/generated/torch.nn.CTCLoss.html)。

定义正确内容优势：
`M(V) = mean_j ell(V,decoy_j) - ell(V,target)`，越大越好。
定义依赖运动的内容优势：
`Q(V) = M(native) - M(frozen)`。
主要配对效果：
`G_i = Q(T_i) - Q(N_i)`。
配套原始内容效果：
`B_i = M(T_i_native) - M(N_i_native)`。
长度敏感性效果：
`Gmatched_i = [M(T_matched)-M(T_matched_frozen)] - [M(N_matched)-M(N_matched_frozen)]`。
时间顺序诊断：
`R(V) = M(native)-M(reversed)`。

对原始视频另外报告：target 在六候选中的严格 top1（平局计失败）、rank、target ell、greedy CTC CER、官方 beam CER、平均 blank 概率、argmax blank 占比、平均 entropy、时长、原始检测率。CER = 字符 Levenshtein / 真值字符数，可超过 100%。greedy 先合并相邻重复再去 blank；特殊输出符号单独记录并按单 token 错误计数，不能靠删除非 blank 特殊 token 降低错误率。beam 的 eos 是终止符按官方处理。

不把 encoder 范数、blank 占比、entropy 作为“好口型”的替代主指标；不使用本版本 CTC.forced_align 直接计算音素 onset error。CTC 的发射峰和声学音素边界并不等价，本轮不做时间边界结论。

### 6.6 测量校准、配对统计与解释

先检查工程完整性。paired-complete = 两臂全部五视图及六文本分数都有效；所有统计使用同一 paired-complete 集合，打印 13 → eligible → extracted → paired-complete 的数量和每条失败原因。至少 10 对才能做以下判读；这是本次小 pilot 的实用门槛，不是功效分析，不等于大样本确认。

测量校准 `CALIBRATED_ON_COHORT` 必须同时满足：
- natural 和 TTS 两侧各自：native 的六候选严格 top1 比例 ≥0.60；
- 两侧各自 Q>0 的话语比例 ≥0.75，且各自 mean(Q)>0；
- 技术一致性测试通过（去音轨、确定性、CTC 数值校验）。

此门槛仅是提前固定的工程/科学适用性筛选。不要因为差一点通过而改阈值、改候选文本或换层。失败为 `INCONCLUSIVE_VSR_VALIDITY`，仍输出所有原始指标和差异，但不能据此判断 TTS 是否改善口型。两侧分别报告通过情况，单侧通过记为测量适用性不对称。

对 G、B、Gmatched、R 以及辅助 CER 差异，输出每条值、均值、中位数、正比例。主 G 的 95% percentile CI 用 np.random.default_rng(20260918)，20,000 次有放回抽取话语 pair，每次均值；N/T 不拆开重采样。Gmatched 同法报告。辅助指标的 CI 是描述性的，不作多终点挑选显著性。无独立说话人重复，因此这些区间仅条件于 S0764 和该批固定视频。

提前固定判读：
- 所有校准通过 + G 的 CI 下界 >0 + mean(B)>0 + mean(Gmatched)>0：`EXPLORATORY_VISUAL_CONTENT_SUPPORT`。更强的表述是“该 VSR 对 TTS 视频的目标内容及动态证据更敏感”，不是嘴型真实性已被证明。
- G 正向但 B≤0：`CONTROL_DRIVEN_DIFFERENCE`，可能仅因 TTS 静态基线更差；不得宣称绝对可读性更好。
- native 正向但 matched 不为正：`DURATION_SENSITIVE`；同长度重采样本身改变速度，不能反推“时长是唯一原因”。
- G CI 跨零：`NO_CLEAR_PAIRED_GAIN`，不等于零效应。
- G CI 上界 <0：`EXPLORATORY_REVERSE_EFFECT`，限定模型与样本。
- 其他不满足支持规则的组合统一 `INCONCLUSIVE_PATTERN`，列出具体未满足项。

判读 precedence：工程失败/样本不足 → 测量校准失败 → CONTROL_DRIVEN_DIFFERENCE（G均值>0但B≤0）→ DURATION_SENSITIVE（G均值>0但Gmatched≤0）→ G区间支持/跨零/反向 → INCONCLUSIVE_PATTERN。代码以这些顺序实现，避免同一结果被多个状态同时描述。

SyncNet 的已有 `results/avtr1/04_eval/eval_meta.json` 只作描述性背景，标注历史来源未与当前视频逐文件哈希绑定；不重新生成、评分或依据它筛选视频。VSR 阳性可以增加独立视觉证据；VSR 阴性/无效不能把 SyncNet 效应全部归结为评价器偏好。

### 6.7 产物与自审

run 目录只需：
- `manifest.json`：锁定输入、配置、来源证据、decoy、排除原因。
- `environment.json`：版本、设备、代码/权重 SHA。
- `features/{id}/{condition}/{view}.npz` 与原始 QC JSON、三帧预览。
- `records.jsonl`：每个 id/condition/view 的状态、所有六候选 loss、M 和辅助指标；错误也占一条。
- `analysis.json`、`report.md`：完整分母、校准和配对结果、解释边界、旧 pilot 撤销说明。
- `validation.json`：独立重载 logp，用小型 log-space CTC 动态规划复算全部 loss、统计和判读。
- `smoke.json`：只记录实现 smoke，不能伪装全量完成。

`validate` 不调用 producer 的 ctc_nll/content_margin/paired_summary 来证明自己正确。用不同实现复算；输入允许相同冻结 manifest。无需第二套框架或多个 review agent。结果报告中的 numerical tolerance 明确写出：logp 概率行和误差 <1e-5；主/独立 CPU float64 CTC NLL 绝对差 ≤1e-5；统计差 ≤1e-8。同设备重复 encoder/logp 最大绝对差 ≤1e-5；失败先诊断，不放宽阈值。

更新 BM 时先遵守 Startup Router，写 Experiments 结果笔记，包含 planned spec 的链接；HANDOFF 只添加当前结论和产物路径，不写流水日志。明确校准 PASS 和工程 PASS 与 TTS 科学支持不是同一个状态。

## 7. Expected Change Surface

### Must change

- 新增 `scripts/experiments/vsr_tts_pilot.py`
- 新增 `scripts/experiments/vsr_tts_metrics.py`
- 新增 `tests/experiments/test_vsr_tts_pilot.py`
- 新建 `runs/vsr_tts_content_v1/`（下游实际执行时）
- 实验执行后添加 BM Experiments 结果笔记并更新 HANDOFF 的 VSR 当前状态。

### May change

- 如 runner 需要的小型环境锁文件，放新 run 内。
- 若真实输入与本 spec 存在矛盾，修订本 spec 的来源说明；先记录矛盾，不静默改变 cohort 或目标。

### Should not change

- 所有第三方模型源码与权重、原视频、原文本和源图、历史 runs、旧 /tmp pilot。
- F0、MFA-linear、Wav2Lip/SyncNet pipeline、全局 config.yaml、其他未提交工作。
- 不删除压缩包、不重装现有可用环境、不下载第二个 VSR、不生成新视频、不训练 head、不 commit/push。

## 8. Validation Plan

| command / target | behavior | expected result |
|---|---|---|
| 新两脚本 `python -m py_compile ...`（autoavsr 环境） | Python 3.8 兼容 | 无语法错误 |
| `[redacted-local-path] -m pytest tests/experiments/test_vsr_tts_pilot.py -q` | 下列真正语义测试 | 全通过；pytest 缺失时仅安装兼容 Py3.8 的 pytest==8.3.5 |
| audit | 13 WID/text 配对，错误 AISHELL100 文本应被拒绝 | 正确来源能绑定；缺失/冲突明确记录 |
| extract --smoke | sample 1 双臂五视图、音轨隔离和缓存 | 10 个有效视图或明确失败，非整轮 PASS |
| extract --resume + analyze | 全 26 视频、固定规则和分母 | 完整完成或诚实的无效/不足结论 |
| validate | 从保存 logp 独立复算 | 数值与判读一致，完整性检查 PASS |

必须的单元测试：
1. 明确制造两个数据集同名 ID 不同 WID/text，不能被合并；1..10 来源冲突必须失败，11..13 能从 README 绑定，空文件不能作来源。
2. CTC tiny vocab=3、T≤4，枚举全部路径并用“先合并再去 blank”计算 target 概率，和主 CTC 及独立 DP 比较；重复字符要 blank 分隔；不可达不是零分。
3. 正确/错误文本使用已知小概率张量验证 M、Q、G 的符号；删除静态对照、两次除长度、调换 N/T 都应被检测。
4. frozen 每帧完全相同；reversed 保留帧集合；matched 两端一致、时间轴长度 M 一致、N/T 不截尾；变换不修改原 tensor。
5. OOV/空文本/NaN/Inf/目标可达但 decoy不可达/缺失一臂 都保留明确失败，不能变成配对数据。
6. greedy 重复字符 + blank、CER 可>1、特殊符号不被当正确字符删除。
7. 缓存数据或 input SHA 改变必须拒绝 resume；smoke 的 1 对不得通过 n≥10。
8. synthetic 配对例验证 CI 只重采样 pairs，判读各分支及 precedence。
9. GPU smoke 中，同一视频原音轨、无音轨、替换成零音轨且视频流copy 三版，经提取的输入 tensor/encoder/logp 在上述容差内一致。只生成 sample1 的临时 copy，验证后删除。
10. GPU smoke 重复同一 native 前向，eval/no_grad 和一致性满足要求。beam CER 变化不得改变主指标。
11. validate 的独立 DP 端点为最后字符与末尾 blank 的 logsumexp，不能漏终态或允许重复字符非法跳转。

不要运行整个历史实验测试套件代替这些检查，也不要要求负面科学结果改成 PASS。自审发现实现错误修复后新建 run，旧 run 标为无效并保留；不得在相同数据上调整科学指标追求显著。

## 9. Risks and Edge Cases

- **真值串错是已发现的具体错误**：必须撤销旧 CER/CTC 比较，不能只加一个脚注继续引用数字。
- CMLR benchmark 的公开 CER 不能直接迁移到合成头像；差解码可能由文本错误、裁剪、视觉域偏移或模型能力造成，本轮校准用于区分是否具备最低可测性。
- 本地仓库是 fork；`VSR.py` 含旧接口/缺失 utilities，不用它作新入口。用已经跑通的 infer.py 和 Pipeline。
- MediaPipe 官方多脸选择代码不适合直接保证选中了目标人；当前单人头像应审计，异常多脸视频失败，不盲信最大框。
- 官方插值会修改 landmarks 列表，保存 QC 时必须复制原始 mask；否则全部检测率会被错误写为 100%。
- 稳定脸部裁剪使用嘴中心，可能影响运动特征；固定官方处理并在解释中说明，不能因结果改裁剪。
- CTC 稀疏峰值不是实际字符/音素持续区间；本轮不声称 onset error 改善。
- 全 blank 较高可让部分 summary 看似自信；正确文本对比和静态对照比 max probability 更必要，但也不能消除全部模型偏好。
- paired mean duration 相同不等于韵律相同；matched 是敏感性分析，不是无偏因果分解。
- 13 个样本仅一位音频说话人，且部分 face 图重复；报告具体 face SHA 复用，不能按 26 视频当 n=26 或声称跨身份效果。
- 旧文章可能说 VSR 曾阻塞：本中文模型的部署不能解除历史英文 LRS3 VSR teacher gate。
- 不通过下游比较结果挑更好的模型层、语句子集、decoy 或 beam 参数。

## 10. Assumptions / Unknowns

- VERIFIED: 本地模型已能提取 T×256 encoder 和 T×3363 汉字 CTC 输出，CUDA/MediaPipe 可用。
- VERIFIED: AVTR-1 的历史来源是 data/data/audio 和 R2 Faster-Qwen3；不是 AISHELL100 同名 ID。
- VERIFIED: r2_assets/README 提供完整 13 文本/WID；transcript.json 只提供前 10；全部音频 speaker=S0764。
- VERIFIED: 现有 26 视频跑通过工程 smoke；这不验证新指标、不证明内容质量。
- VERIFIED: 作者 upstream 与本地 fork 区别已在来源说明中标注。
- LIKELY: 历史部署记录正确描述现存 AVTR-1 视频和 TTS 目标文本；缺少原 generation hash 时必须保留 historical_documented 等级。
- UNKNOWN: 源图和原音频在下游启动时是否完整可访问；audit 实测，禁止虚构路径。
- UNKNOWN: CMLR 是否通过本 cohort 的内容/运动校准；不能承诺一定有科学结论。
- UNKNOWN: TTS 是否逐句准确表达目标、视频是否完整包含音频内容；报告来源边界，失败不归因于单一模块。
- UNKNOWN: 新增两文件/测试名在下游实施时是否已有其他 agent 工作；存在时先读取，不覆盖。
- UNKNOWN: 单模型阳性是否符合人类判断或跨说话人可复现；留给后续独立数据，不在本 pilot 扩张范围。

## 11. Handoff Contract

按第 3 节 anchors 实现两个薄脚本和一个对应测试文件，遵守第 5 节全部 invariants，仿照第 4 节原子写盘及配对统计，不复制旧实验复杂框架。先完成真值来源纠正与环境检查，再 smoke、全量、分析、独立复算。不得只交付“模型成功加载”。

完成标准：源码/测试、冻结 manifest、26 视频处理状态、5 视图证据、内容/运动校准、配对指标、独立复算、自审修复、BM 结果与 HANDOFF 指针齐全。若测量无效或增益不成立，仍是正确交付，结论如实写出。

如证据与本 spec 冲突，停止依赖该假设的阶段并报告具体矛盾；可以继续不依赖它的实现与单元测试。不得通过改真值、减分母、调阈值、换模型或进入 head training 来绕过问题。
