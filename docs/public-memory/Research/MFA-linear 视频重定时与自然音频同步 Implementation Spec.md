---
title: MFA-linear 视频重定时与自然音频同步 Implementation Spec
type: research_topic
permalink: tts-exp/research/mfa-linear-视频重定时与自然音频同步-implementation-spec
status: active
implementation_status: implemented
scientific_status: NOT_RUN
protocol: mfa_linear_video_retiming_v1
question: 受 Sync-C 引导和时间修改约束的视频重定时能否使 MFA-linear 驱动视频更好匹配原自然音频？
tags:
- mfa-linear
- video-retiming
- syncnet
- implementation-spec
---

# MFA-linear 视频重定时与自然音频同步 Implementation Spec

## 1. Objective

在冻结的 Wav2Lip 下，用已有 MFA-linear 音频 M 驱动同一静态肖像生成 V_M；固定配对自然音频 N，通过有界、单调的视频时间重采样得到 V_R，使 V_R/N 尽量同步。Sync-C 负责引导候选选择，时间位移、局部速度及画质约束负责限制修改幅度。最终输出带原始 N 音轨的可播放视频、逐帧时间映射、搜索轨迹和真实模型评分凭据。

本轮是逐视频有监督搜索的可行性实验；不训练音频增强器，不声称得到可泛化的增强器。若搜索成功，可把 V_R 作为后续视觉教师候选，必须另行验证蒸馏和泛化。设计首选 Wav2Lip；用户若指定 Ditto，应先补齐其独立生成 adapter，不能直接替换模型名称沿用本合同。

实现与单样本 Wav2Lip engineering smoke 已完成；M/N 与 R/N 通过官方 SyncNet cell 和独立 checker。Formal cohort/scientific evaluation is still NOT_RUN.

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 根据用户请求设计；核实静态生成、插值、评分、当前资产及历史裁剪缺陷 | September 23, 2026 | user（设计任务）；agent（具体参数提案） |
| 实现与 Wav2Lip 单样本工程 smoke 完成；科学评估仍未运行 | September 23, 2026 | user（实施任务）；agent（实现与验证） |
| 按用户要求扩展 Stage B 候选评分并完成官方复评及 fresh checker；最佳候选仍未过 offset 门，科学评估未运行 | September 23, 2026 | user（预算扩展）；agent（实施与验证） |

### Observations

- [status] active
- [progress] implementation_status=implemented；scientific_status=NOT_RUN。Wav2Lip smoke 结果见 [[MFA-linear 视频重定时 Sync-C 约束工程 smoke 结果 2026-09-23]]。
- [result] sample1/portrait3 engineering smoke PASS；扩展预算后累计256个候选评分（7 Stage A、249 Stage B），最佳非identity候选固定裁剪 C/D/D0/offset=6.177/7.257/13.068/-2，因未达到 |offset|≤1 而回退 identity；官方 ΔSync-C=0.000，正式科学评估未运行。
- [budget_expansion] 增加Stage B候选上限后仍因原rounds_per_step=2提前结束于197次；rounds_per_step=4后用满256评分预算并在Stage B预算门停止，结果见 [[MFA-linear 视频重定时 Sync-C 约束工程 smoke 结果 2026-09-23]]。
- [question] MFA-linear 驱动嘴部动作能否通过小幅视频重定时，更好地匹配完全不变的自然音频？
- [decision] 先做 3 个已核验的 clean-MFA 样本，肖像 3 搜索，肖像 6/9 原样迁移时间映射；没有训练阶段。
- [constraint] generation_audio=M、optimization_audio=N、final_mux_audio=N，三个角色必须显式保存；M/M 只作诊断。
- [constraint] 搜索指标和官方完整评分链分开命名；同一 SyncNet 权重的重新评分不是独立模型验证。

## 2. Repository Model

已核实的音频链：

`scripts/pilot_generate_mfa_linear.py::generate`
→ natural/TTS 的 WavLM-Large L6 特征
→ `knn_vc_retrieval.mfa_linear_target` 按 MFA 音素映射到 natural 时钟
→ 冻结 prematched HiFi-GAN
→ `exact_natural_length`
→ MFA-linear WAV。

因此本实验的 M 是已声码还原的音频，不是 raw TTS，也不是 A/B/C 增强器输出。视频重采样是其后增加的独立阶段，不再重做音频 MFA-linear。

已核实的生成/评分链：

`static_image_bridge/render_worker.py::main`
→ PNG 紧脸框 → Wav2Lip 官方 mel 前端 → Wav2Lip 模型 → FFV1 25 fps。

`static_image_bridge/score_worker.py::read_video/audio_embedding`
→ 固定 score box 的 BGR 224×224 五帧窗口 + 16 kHz PCM 的 MFCC
→ SyncNet V2 两塔 → 距离矩阵 → 固定支持上的 C/D/offset/D0。

`phone_gain_static_tfg_mfa/official_score.py::strict_mux`
→ 视频流 copy + 原自然 PCM
→ `third_party/syncnet_python/run_pipeline.py`
→ S3FD 跟踪/裁剪
→ `run_syncnet.py` / `SyncNetInstance.evaluate`
→ 官方 C/D/offset 和 activesd.pckl。

新增流程：

`audit → render → calibrate → search → seal → transfer → official → check → report`。

生成器每个音频/肖像只调用一次；搜索阶段反复调用冻结 SyncNet 的视觉塔，不反复生成 Wav2Lip。N 的音频 embedding 可以按完整指纹缓存。

资产核验结果：

- LRS3 旧 confirmation 清单还在，22/22 natural WAV 的 SHA 一致，但其 22 条 MFA-linear WAV 目前均不可用；不能以该清单假称数据齐全。
- clean 小样本源：`runs/aishell1_mfa_linear_n25_resample_poly_20260814/mfa3_clean_mfa_linear_gate_fix2_20260814/summary.json`，ID 固定为 1、101、201；3/3 M 文件存在且 SHA 与 summary 一致。
- 配对来源：同根 `mfa3_clean_gate_20260814/manifest.json` 的 natural_source、transcripts；以 summary.results 的 sample_id/paired_key/speaker_id 连接，禁止数字 ID 跨数据集裸连接。
- 旧 n25 原版 25 对 WAV 仍在，但不能把原版音频混入 clean 小样本。扩展队列属于后续协议修订。
- 肖像与修正框：`runs/phone_gain_static_tfg_mfa_repair_20260923_sync_c_v3/00_protocol/registry.json::portraits`；只读取 portraits，不读取其 A/B/C 音频或结论。

## 3. Code Anchors

新增包记为 P=`scripts/experiments/mfa_linear_video_retiming/`；下表新符号为需要实现的接口。

| path | symbol | current role | required change |
|:--|:--|:--|:--|
| scripts/pilot_generate_mfa_linear.py | generate, exact_natural_length | 生成有来源的 MFA-linear 音频 | 只作为来源契约；首轮直接复用 clean WAV，不改实现 |
| scripts/experiments/static_image_bridge/render_worker.py | main, chunk_mels, encode_ffv1_stream | 正确的静态 Wav2Lip 前向及无损视频 | 由新 adapter 以 subprocess 原样调用；保留 PNG/PCM/紧框合同 |
| scripts/experiments/phone_gain_static_tfg_mfa/assets.py | _portrait_meta | 从真实人脸检测生成 generation/score boxes，拒绝整图 fallback | 参考其检查；新 audit 复核已存 PNG、框和 detector provenance |
| scripts/experiments/static_image_bridge/images.py | generation_box, score_box | 明确两种裁剪的几何定义 | 只读复用，不将两种框混用 |
| scripts/experiments/wav2lip_oracle_frame_interpolation/media.py | interpolate_frames | float64 相邻原始像素混合、half-up uint8 | 复用像素算法；本实验自行构造 q/j/k/w，不调用旧 build_linear_indices |
| scripts/experiments/static_image_bridge/score_worker.py | crop_zero_padded, read_video, audio_embedding | 固定裁剪和官方 MFCC 前端 | 复用预处理；新 scorer 显式检查 PCM16、模型 keyset、dtype 和形状 |
| scripts/experiments/phone_gain_static_tfg_mfa/batch_score_worker.py | _visual_embedding, _write_video_scores | 一次加载模型、音频缓存、真实视觉前向 | 仿照其批处理；新增持久搜索 worker，不将全部候选编码写盘 |
| scripts/experiments/phone_gain_static_tfg_mfa/official_score.py | strict_mux, run_logged, parse_score, evaluate | 官方完整复评与 PCM 审计 | 复用 strict_mux；仿照 evaluate 的两条命令，写新 cell adapter；不调用绑定六音频臂的 load_cells |
| third_party/syncnet_python/SyncNetInstance.py | evaluate, calc_pdist | 官方窗口和 offset 语义 | 只读；作为 parity 及完整官方验收基准 |
| P/config.py, P/assets.py | FrozenProtocol, freeze_inputs, audit_portraits | 新增 | 参数冻结、资产/schema/哈希、typed 输入角色 |
| P/generation.py | render_baselines, canonicalize_tail | 新增 | 生成 N/M 两臂、规范尾部帧数、保存原始和规范化证据 |
| P/retime.py | build_knots, validate_map, render_map | 新增 | 时间映射参数化、硬约束、uint8 插值 |
| P/scorer.py, P/search_worker.py | FrozenSyncNetScorer, score_frames, search_record | 新增 | 正确评分、有限预算搜索和候选轨迹 |
| P/official.py | score_official_cell, recompute_official_curve | 新增 | 从 sealed manifest 调用官方链；保存所有 crop 与矩阵出处 |
| P/run.py, P/check.py, P/report.py | main, check_run, write_report | 新增 | 阶段状态、恢复、独立验证、条件化结论 |
| scripts/configs/mfa_linear_video_retiming_v1.yaml | 唯一参数源 | 新增 | 固定资产、模型、预算、变换和验收阈值 |
| tests/experiments/mfa_linear_video_retiming/ | 见第8节 | 新增 | 媒体、调用、评分、搜索和错误路径验证 |

## 4. Reference Pattern

最接近的是 `wav2lip_oracle_frame_interpolation/media.py::interpolate_frames` 和其 runner/validator：从原始像素构造无损重定时视频、解码复查、重新神经网络前向、独立重算矩阵。不要复制旧 forward_map、旧22条队列、旧 masks 或 own-audio 门；本实验目标音轨是 N，改善 N 匹配可能牺牲 M 匹配。

正确模型调用仿照 `static_image_bridge/render_worker.py::main`，官方复评仿照 `phone_gain_static_tfg_mfa/official_score.py::evaluate`。后者的固定 ARMS/SELECTED_CELLS 不适用于本实验，不能直接运行其完整 main 期待自动理解 R。

现有代码没有完整的“Sync-C 引导的受约束逐视频搜索”runner；新增小包，以 adapter 连接现有稳定边界，不改旧实验。

历史依据：旧 nearest oracle 的 own C 为 −0.470，线性混帧后为 −0.202，仍未过其 own gate。这是插值实现可参考的证据，不是本任务成功的先验保证。

## 5. Invariants

1. N、M 的原始 WAV、PCM、采样数、音量、时间轴均不变；优化过程中只改视频。最终音轨必须与输入 N 解码 PCM 逐样本相同。
2. 用 M 调用 Wav2Lip 生成 V_M；V_N 用 N；主要搜索 cell 恒为 V_R/N，不能优化 M/M 或 R/M 替代它。
3. static PNG 三张：3/6/9；搜索肖像固定3。不得按 Sync-C 挑图/样本；不用自然视频帧当参考。
4. Wav2Lip 与 SyncNet 权重冻结、eval、无梯度；生成和评分使用独立 Python 环境，核验实际导入模块 __file__。
5. 每个候选从原始 V_M 像素重采样；不得在上一个候选上再插值，不得插值 embedding 冒充新视频。
6. 输出25 fps、帧数和分辨率固定；q 单调、有界。所有音频和帧坐标从 t=0 定义，保持相同 PTS。
7. 同一 record 的评分支持 W 在搜索前由长度决定并冻结；不能随候选换行、删难片段或降低 min support。
8. 搜索冻结 crop；最终官方链动态 crop 单独记录。不同前端的 C 不直接混为一个数。
9. 每个候选的 Sync-C、D、D0、B、offset、参数、可行性和拒绝原因均保存；数据缺失不写零分。
10. V_R 在官方复评前封存；官方结果不得用于重选候选、改阈值、扩大搜索或挑样本。
11. 人物3的 q 原样迁移到6/9；不重新搜索。三个肖像不是三个独立语音样本。
12. 所有输出在新 run；内容哈希覆盖未提交源码。历史失败结论、旧run和用户工作不覆盖。
13. 搜索成功只证明当前视频与该评分器上的优化可行性。官方完整链与搜索链使用相同权重，不构成独立模型证据。

## 6. Implementation Plan

### 6.1 P0：冻结输入、环境和预算

新 CLI：
`python -m scripts.experiments.mfa_linear_video_retiming.run --config scripts/configs/mfa_linear_video_retiming_v1.yaml --run-id <id> --stage <stage|all>`。
`--smoke` 仅 sample1/portrait3，run 标记 engineering_only；formal 固定3样本×3肖像，不能将 smoke 结果作为调参集反复改协议。

资产导入时复制轻量清单，不复制父模型/音频。检查 summary 的 M 文件 SHA、自然 source 路径与旧 eval manifest 中同 paired_key 的自然 SHA；解码 N/M 均为16k/mono/PCM16且样本数完全相同。冻结 transcript、paired_key、speaker_id、M来源summary/tokens哈希。任何 mismatch 阻断该 cohort，不在另一个目录按同名 WAV 补齐。

固定模型路径/权重：

- Wav2Lip Python：`[redacted-local-path]`。
- Wav2Lip root：`third_party/Wav2Lip`；checkpoint `checkpoints/wav2lip_gan.pth`；SHA256 `ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8`。
- SyncNet Python：`[redacted-local-path]`。
- SyncNet root：`third_party/syncnet_python`；checkpoint `data/syncnet_v2.model`；SHA256 `961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442`。
- ffmpeg/ffprobe：`[redacted-local-path]}`。记录当前版本和哈希；本轮 freeze 后不得换，不要求与失效的旧 executable 哈希硬匹配。
- 记录 torch/numpy/opencv/scipy/python_speech_features、CUDA/device、源码文件哈希和 git commit；不是只验证文件存在。

首轮单GPU串行；生成 batch4、SyncNet batch20；seed=20260923。GPU任务前检查外部 compute PID、空闲显存≥5 GiB、RAM≥8 GiB、磁盘≥4 GiB且足够预计输出的1.2倍。发现其他任务占用就 RESOURCE_WAIT，不杀进程。搜索上限每样本256次真实候选评分或20分钟，先到者停止；总 active GPU预算3小时。资源等待不计active时间。保留 best-so-far 和精确恢复状态。

### 6.2 P1：正确生成与媒体规范

优先使用现存 portrait registry 里的 PNG/检测/几何凭据并重新验证哈希。portrait3 应得到：

- RGB SHA `bd5659ec3560bea57c34aa98d97ec9f60916a85a9414e70a3cf7a538db3f3903`；
- generation box xyxy=[138,90,357,387]；
- score box xyxy=[33,18,462,447]，429像素正方形。

6/9 的框从同 registry 冻结，不能拿3的框套用。生成框不是512×512整图；拒绝 full-frame fallback。score crop 与 generation crop 不相等是正常的。

生成命令模板（每个 sample、portrait、N/M各一次，REPO绝对路径）：

```bash
[redacted-local-path] \
  <REPO>/scripts/experiments/static_image_bridge/render_worker.py \
  --image <PNG> --image-rgb-sha256 <RGB_SHA> \
  --audio <N_OR_M_WAV> --box <x1> <y1> <x2> <y2> \
  --checkpoint <REPO>/third_party/Wav2Lip/checkpoints/wav2lip_gan.pth \
  --ffmpeg [redacted-local-path] \
  --outfile <RUN>/generated/<portrait>/<sample>/<N_OR_M>.mkv \
  --result <RECEIPT_JSON> --batch-size 4 --seed 20260923 --device cuda
```

cwd=REPO，subprocess argv列表，无 shell 拼接。receipt 必须满足 input_mode=one_png_only、source_frame_indices全部0、audio SHA匹配相应角色、帧数>0、FFV1/25fps、框外像素静态且相同。追加真实 checkpoint/source/version bindings；不能仅在receipt中抄预期hash。

明确 Wav2Lip 尾部策略：mel chunk 可能使原生视频比音频短几帧。令目标 F=ceil(len(N)/640)。若原生帧数 Fraw<F 且差≤5，统一在 N/M 尾部重复末帧补到F；若 Fraw>F 且差≤5，统一右裁到F；差>5报 MEDIA_LENGTH_MISMATCH。记录原始帧数、裁补数、raw与canonical哈希。不得调整音频或用 -shortest。最后一个视频帧覆盖到 ceil 时刻，音频结束可早于视频结尾不足40ms，这是明确的时基余数。

令 Fvalid=min(F,Fraw_N,Fraw_M)，所有搜索支持只使用未补帧的有效前缀，尾部补帧及其5帧感受野排除。原视频与候选都使用相同规范化流程。首尾保护区及所有补帧区不许重定时。

### 6.3 P2：实现正确的 SyncNet adapter，并先校准

新 search_worker 由 SYNCNET_PYTHON 启动；在导入模型前只把冻结 SYNCNET_ROOT 插入 sys.path，检查 `SyncNetModel.__file__` 和 `SyncNetInstance.__file__`。禁止误用 Wav2Lip 训练用 SyncNet_color、其 mel前端、其他 eval目录同名模型或默认未加载权重的实例。

模型使用 `SyncNetInstance(device="cuda")` 的官方 `__S__`。加载前验证 checkpoint 的 keyset/shape 与模型完全一致；随后载入、eval、requires_grad=False，用 inference_mode。保存参数状态哈希/实际模型路径，加载失败不得 strict=False继续。

评分契约：

- 视频在完整uint8 BGR帧上先插值，再按固定score box零填充/resize到224；值域0..255、float32，不除255、不转RGB、不额外ImageNet归一化。
- 视频窗口张量 [B,3,5,224,224]，起始帧i对应音频MFCC列4i..4i+19。
- 音频由 scipy.io.wavfile 读入原PCM16，直接复用 python_speech_features.mfcc 的官方默认参数；13×20，不是Wav2Lip的80×16 mel。N音频特征仅算一次，M只用于锁定后的诊断。
- 两塔原始1024维向量，不额外L2 normalize。
- d(i,k)=pairwise_distance(v_i,a_(i+k),p=2,eps=1e-6)，k=-15..15。边界缺失以NaN保存，但冻结W上必须全有限。
- 仿照官方 evaluate 保守窗口数 n=min(Fvalid,floor(L/640))-5；音频至少计算 n 个合法窗口。
- W=整数i满足15≤i<n-15；若少于25行，记录 INSUFFICIENT_SUPPORT，不缩小vshift或改短音频门。
- curve[k]=mean_W d(i,k)，D=min(curve)，B=median(curve)，C=B-D，D0=curve[15]，offset=15-argmin(curve)；argmin平分取最左。float32距离/时间平均，随后转float64汇总，C展示3位、选择用未舍入数值。
- 除全局值外，每连续25个W行构成局部窗，尾块不足25丢弃；保存局部D0、C、offset。不能只看全局峰掩盖局部错位。

校准顺序：

1. 同源 q=id 的像素严格一致；搜索内存评分与保存FFV1后 fresh评分：矩阵max_abs≤1e-4、C/D差≤1e-4、offset完全相同。
2. 直接复用旧固定裁剪worker给同一V_M/N评分，并在相同W重算；差≤1e-4。这是前端复现检查，不要求与S3FD官方整链C相等。
3. 对一个由官方pipeline产生的crop AVI调用官方 evaluate；新adapter使用完全同样的JPEG提取、窗口数、zero padding、全行均值，复现日志C/D到0.001和offset精确一致。此 adapter parity 模式只用于验证，不改变搜索固定W定义。
4. 对 N/N 施加已知±3帧的音频延迟诊断（仅校准临时拷贝）：在共同内区检查最优lag按已知方向移动，误差≤1帧，并核验符号。Sync-C无需下降，因为其搜索了offset。若基线峰在边界或无稳定峰，明确校准不足。
5. 同一原始输入重复评分数值一致；出现不稳定先排查eval/device/preprocess，不放宽门。

成功后锁定 scorer_fingerprint。任何校准失败暂停依赖评分的搜索；保留可复现诊断。

### 6.4 P3：有界时间映射与像素插值

定义 output frame j 从 source coordinate q_j=j+δ_j 取样；δ>0表示读取后面的动作，即画面动作提前。只改原始 V_M 时间顺序的速度，不能逆序。

固定参数：

- 位移上限 |δ_j|≤3帧=120ms。
- 相邻速度 0.5≤q_(j+1)-q_j≤1.5。
- 首5帧、最后5个未补帧以及全部尾部补帧：δ=0。
- δ由最多16个等距内部knot线性插值得到，knot最小间隔12帧；整数knot位置唯一排序。短片无法容纳内部knot时只保留identity，标明不可优化。
- knot δ为0.25帧网格；所有q严格在[0,F-1]，端点保持不变；相邻斜率跳变≤0.5。按帧审计而非只审计knot。
- 只用相邻原始帧线性像素插值：j0=floor(q)、j1=ceil(q)、w=q-j0；float64混合，floor(x+0.5)转uint8。整帧插值；静态肖像框外像素因源帧相同应严格不变。
- R=mean((δ/3)^2)+mean((diff δ/0.5)^2)，用于同分候选的最小改动选择；保存最大位移、R、最大速度变化、混帧比例。

这里“抽帧”对应q增长快于1，“插帧”对应q增长慢于1且取中间帧；输出帧率和总帧数不变。首轮无需RIFE/FILM等额外网络。线性混帧可能产生重影，质量与同步收益分别验收。

### 6.5 P4：两阶段、固定预算、无梯度搜索

identity的固定裁剪基线记 C_b、D_b、D0_b、offset_b。候选需先满足全部媒体硬约束，再满足 C≥C_b−0.050、D≤D_b+0.050、D0≤D0_b+0.050，容差是本轮设计值，不是已验证的人类可感阈值。

为防止“只提高错位背景B”获胜，最终接受还需D0至少改善0.020。纯背景增益单列 SCORE_ONLY，不作为可用教师。

搜索细则：

1. 第0候选identity。预设6个整体平滑位移种子 g∈{-3,-2,-1,1,2,3}，δ=g×首尾10帧线性ramp形成的平台；ramp在保护区外起落，投影到knot网格后重新全量审计，非法种子跳过不修剪作弊。identity及6种子计入256预算。
2. 阶段A最多80次合法且唯一的真实候选评分，按 (D0, abs(offset), -C, R, candidate_id) 排序寻找时间对齐起点；上面的C/D/D0非劣限制仍生效。保存过程中所有达到 |offset|≤1 的候选。
3. 阶段B余下最多176次：有 |offset|≤1 候选时取其中C最大者开始；没有则从阶段A最佳开始，但最终不能用未同步输出充当成功。
4. 确定性coordinate search：步长依次[1.0,0.5,0.25]帧；每个步长最多2轮；按时间从早到晚逐knot尝试±step，另尝试相邻两个knot同向±step（允许平滑移动整段）。每次从当前incumbent产生候选；本轮候选都评完后再更新。参数相同则缓存命中，不计真实前向预算。
5. 阶段B优先级：满足|offset|≤1优先，然后C最大；以1e-6为数值同分容差，依次选D0低、R小、candidate_id小。不跨seed反复重启，不按官方评分回头搜索。
6. 最终在全部候选中筛选：C≥C_b+0.100、D0≤D0_b−0.020、D≤D_b+0.050、|offset|≤1；局部25行窗的median D0不得恶化，局部|offset|的90分位不得大于基线。
7. 在满足门的候选中先取Cmax，再从 C≥Cmax−0.020 的集合选R最小者。这把“尽量同步”和“少改视频”落实为确定规则。不存在符合者则返回identity，status=NO_ACCEPTABLE_WARP；保留best_attempt供分析，不作为成功产物。
8. 预算/时间耗尽可产生 BUDGET_LIMITED 且保留已达门的候选；报告停止原因及实际搜索次数，不能宣称全局最优。

日志每行含 map_sha、父baseline指纹、候选参数、阶段、W_sha、C/D/B/D0/offset/局部统计、R、feasibility、reject_reason、elapsed、incumbent。保存原始帧hash和音频embedding；失败候选只保存参数/指标，不写成百上千MP4。

### 6.6 P5：封存、对照、跨肖像迁移

portrait3每个sample最终导出：

- V_M：原M驱动规范化视频，主要基线。
- V_R：选定重定时视频，失败时为identity。
- V_GLOBAL：预设平滑整体位移种子中按同一最终规则选择的对照；无可接受种子则identity。
- V_NEAREST：对V_R同一q取最近原始帧（half-up），仅诊断插值贡献。
- V_MIRROR：δ取反；若不满足硬约束则标记unavailable，不偷偷换成另一随机扰动。
- V_N：自然音频驱动参考，提供实际方法对比，不是必需超越的理论真值。

封存 map、模型、媒体与优化轨迹SHA后，把同一q迁移到 portrait6/9 的 V_M；因为输入M和mel chunk相同，F必须相同，若不相同先报告工程失败，不能缩放q迁就。6/9仅评 M/N、R/N、N/N；不重新搜索、不按图选择。

固定裁剪M/M和R/M在封存后作为诊断；M本身是生成音频，与N可能声学不同，不以R/M非劣作为最终硬门。raw TTS、A/B/C不进本轮cell矩阵。

### 6.7 P6：官方完整评分与播放

主官方复评：portrait3的 N/M/R/GLOBAL/NEAREST/MIRROR 各配N（镜像非法则缺失有原因）；portrait6/9的 N/M/R 各配N；另外portrait3的M/M、R/M。总数上限每样本14 cells、3样本42 cells。无需所有音频全交叉。

对每cell，调用strict_mux（视频流copy、pcm_s16le），显式-map 0:v:0 -map 1:a:0，不使用-shortest、不重定时音频。补做解码帧hash和PTS逐一相等检查；strict_mux目前只验证视频metadata，不能把metadata相同当像素证明。

官方命令与参数：

```bash
[redacted-local-path] <SYNCNET_ROOT>/run_pipeline.py \
  --videofile <CELL.mkv> --reference <UNIQUE_CELL_KEY> \
  --data_dir <CELL_WORKDIR> --min_track 25 --overwrite
[redacted-local-path] <SYNCNET_ROOT>/run_syncnet.py \
  --videofile <CELL.mkv> --reference <UNIQUE_CELL_KEY> \
  --data_dir <CELL_WORKDIR> \
  --initial_model <SYNCNET_ROOT>/data/syncnet_v2.model \
  --vshift 15 --batch_size 20
```

cwd=SYNCNET_ROOT。官方代码内部使用PATH中的ffmpeg，新adapter必须把已冻结ffmpeg所在目录放到子进程PATH首位并记录实际解析路径；不能只给外层mux正确ffmpeg，内层却调用另一个。

要求returncode=0、恰好一个人脸track/crop、恰好一组有限C/D/offset、activesd存在且形状符合31列。零track、多个track、多段日志、NaN、±15边界峰均有独立状态，不能按最高C挑track。parse_score允许offset缺失，新adapter需进一步强制offset存在。

保存官方crop的帧数、源时间覆盖及距离矩阵；每个R/N与M/N必须使用相同长度/时段的track才可配对官方日志C。不同则标 SUPPORT_MISMATCH，并另外在双方共同原始时间区间复算 common-support 诊断；不可把不同窗口的日志C强行作主差值。即使轨迹不同也记录检测框变化，避免误称纯嘴型收益。

官方 D0/B/完整curve从activesd矩阵按官方全行float32平均计算，复现日志C/D的0.001打印容差。官方评分采用其自身边界zero-padding；与固定W的优化C分别存 `search_sync_c` 与 `official_sync_c`，不得共用无前缀字段。

主播放文件为无损MKV+N；如额外导出AAC MP4仅供预览，另记录音频有损，不用它评分或做PCM一致性证明。

### 6.8 P7：报告与结论

每条sample报告搜索前后 C/D/D0/offset、官方同项、最大/平均位移、预算、可行性、相对N驱动基线及对照差异。表中Sync-C三位，JSON保留原精度。输出带时间映射曲线和官方距离曲线的静态图；盲看包固定包含所有3条，不挑成功条。

3样本不做“稳定泛化”显著性宣称。预定义描述性判读：

- ENGINEERING_BLOCKED：输入/模型/媒体/评分校准无法通过。
- SEARCH_NO_ACCEPTABLE_WARP：没有sample得到受约束可接受q。
- SEARCH_GAIN_ONLY：固定裁剪搜索有增益，但官方复评未支持。
- ORACLE_RETIMING_PROMISING：3条全部完成官方M/N和R/N配对；至少2条满足官方ΔC≥+0.100、D0改善≥0.020、|offset|≤1，剩余条官方ΔC不低于−0.050，3条mean ΔC≥+0.100。身份fallback按Δ=0计入，不剔除。
- PORTRAIT_TRANSFER_DESCRIPTIVE：另报6/9上六个配对中的成功比例及逐条数值，不增加独立n。只有来源3条，不算9条。

影像质量独立状态：未有人类盲看时 HUMAN_NOT_ASSESSED；出现明显口部双影、跳动则QUALITY_CONCERN。搜索/官方同步指标达门不自动覆盖画质状态。盲评最小形式为全3条 baseline/R 随机左右顺序，N音轨相同，分别询问同步偏好和重影/跳动；随机key在答复前封存。不代替用户填写。

使用相同SyncNet进行搜索和复评具有评价器过拟合风险，即使 ORACLE_RETIMING_PROMISING 也只支持启动后续验证。后续音频增强器训练前，需要新语句/说话人、独立同步模型或盲评，及冻结TFG路径的可训练性设计；本轮不自动开训。

## 7. Expected Change Surface

### Must change

新增P包的 `__init__.py/config.py/assets.py/generation.py/retime.py/scorer.py/search_worker.py/official.py/run.py/check.py/report.py`；新增YAML；新增第8节的定向测试；实施后更新本spec状态并在Experiments写最终结果笔记。不得在Research写逐步进度日志。

### May change

只有发现复用helper的确切兼容缺陷且定向复现时，才最小修改 `static_image_bridge/render_worker.py` 或 `phone_gain_static_tfg_mfa/official_score.py`，同步回归调用者。优先在新adapter补充检查，不能为了新协议改写旧实验的默认门槛或统计。

### Should not change

`third_party/Wav2Lip`、`third_party/syncnet_python`、权重、MFA/声码器、增强器训练包、旧run、旧结果、根config和全流水线。不得重置/清理当前脏工作树。旧LRS3缺失音频不静默由不同版本替换。

## 8. Validation Plan

实现阶段已运行本包专项 pytest、compileall 与真实 Wav2Lip/SyncNet 单样本 smoke；formal cohort 未运行。下表定义完整验证目标，实际 smoke 范围与限制记录在结果笔记中；合成 fixture 不冒充真实神经模型结果。

| command / target | behavior | expected result |
|:--|:--|:--|
| `python -m compileall -q scripts/experiments/mfa_linear_video_retiming` | CLI/schema导入 | 无语法错误 |
| `python -m pytest tests/experiments/mfa_linear_video_retiming/test_assets.py` | ID连接、N/M role、SHA、缺文件、错版本、错采样率、禁止旧n25混入 | 正例通过；坏输入在GPU工作前拒绝 |
| test_generation.py | subprocess argv、XYXY紧框、禁止fullframe、static PNG、checkpoint路径、PCM保持 | stub捕获正确调用；shape/hash错误失败 |
| test_retime.py | q方向/单调/位移/速度/保护区、identity、half-up、原帧取样、nearest/mirror | 手算像素样例相等；逆序/越界/多次插值被拒绝 |
| test_scorer.py | BGR值域、[B,3,5,224,224]、MFCC[13,20]、eps、offset符号、边界W、local windows | 独立距离公式和标注延迟与实现一致 |
| test_search.py | identity fallback、阶段预算、确定性次序、不得官方调参、背景刷分被D0门拒绝、低R选择 | 可构造目标的mock scorer得到预期q；单独标synthetic |
| test_media.py | F与尾部补帧、无-shortest、full PCM、stream copy像素/PTS | 支持集不含补帧，音频无裁切，变更被发现 |
| test_official.py | 0/多track、stdout缺offset、超时、activesd错误、31列、不同支持 | 逐类明确失败，无静默默认值 |
| test_resume.py | 同run重入、部分文件、改音频/框/模型/W/code后缓存失效、预算继续累计 | 没有重复前向或读入旧分数 |
| test_check.py | 篡改q/像素/PCM/curve/summary/角色/权重凭据 | 独立checker拒绝，而非只重新调用生产摘要 |
| `python -m pytest tests/experiments/wav2lip_oracle_frame_interpolation/test_interpolation.py tests/experiments/static_image_natural_to_tts_bridge/test_geometry.py` | 直接复用边界回归 | 现有测试通过 |
| 新CLI `--run-id <smoke> --smoke --stage all` | 真Wav2Lip+真SyncNet，sample1/portrait3完整链 | 正确PCM/像素与模型调用；工程通过允许科学无增益 |
| 新CLI `--run-id <formal> --stage all` | 固定3×3、封存后官方复评 | 所有预定cells或明确失败分类，禁止自动扩展到旧25 |
| `python -m scripts.experiments.mfa_linear_video_retiming.check --run-dir <RUN>` | 独立媒体/PCM/参数/embedding/矩阵/统计重算 | PASS仅代表工程证据一致，不代表同步阳性 |

独立checker不得调用生产 `validate_map`、`score_metrics`、`select_candidate` 作为唯一验证；直接由保存knot重建q，检查每帧约束，从原始帧重建导出像素，由保存embeddings用numpy公式重算距离/curve，用搜索日志重算最终选择。M与R最终全部fresh前向至少各一次，用另一进程重载权重，检查内存与导出后分数一致。

错误注入必须覆盖：把N/M对调、wav2lip_gan与SyncNet权重互换、RGB/归一化误用、XYXY与Y1Y2X1X2混用、从旧embedding插值评分、首尾padding进入W、静默删失败样本、用官方评分重选R。

## 9. Risks and Edge Cases

- 官方完整链和固定crop优化可能方向不一致；这是真实负结果，不能换成更好看的评分口径。
- 全局高C不保证offset≈0或每个局部都同步；必须报告D0/offset/局部窗口。嘴型类别错误靠重定时可能无法修复。
- 120ms上限和速度0.5..1.5可能不够修复大错位；首轮阴性仅限制该变换族，不自行扩大范围直到阳性。
- 所有候选反复看同一自然音频和SyncNet，存在选择偏差。fresh评分只验证可复现，不能把搜索增益的bootstrap CI称为外部验证。
- 线性像素混合会模糊和双影；nearest对照能描述插值贡献，不能单独证明视觉自然度。
- Wav2Lip尾部帧数与完整自然音频未必相同；明确裁补、固定有效支持，严禁-shortest偷偷删音频。
- PNG3/6/9的框不可混用；历史整图输入错误曾把官方C压得极低，所以“代码能跑”不够。
- 文件名相同不等于同样本；1/101/201必须同时绑定paired_key/speaker/transcript/PCM。
- GPU可用性随时变化；20分钟/3小时预算是上限，部分运行状态必须可恢复，不越权终止其他实验。
- 基线峰在±15边界、官方track缺失或片段不等长时，不把数字C强行解释为同步改善。
- 原M音频可能保留声码器误差；本实验问的是给定M视频是否能校时，不以它解释全部TTS优势。

## 10. Assumptions / Unknowns

- VERIFIED: 两个Python环境入口与Wav2Lip/SyncNet权重文件存在；源码有实际模型加载与前向路径。此轮未运行权重加载，执行时仍须验证内容SHA与模型keyset。
- VERIFIED: clean修正版M的1/101/201三文件与summary SHA一致；其配对自然来源在manifest中明确，旧n25自然文件25/25哈希一致。执行时仍须做三条PCM采样格式/长度联合核验。
- VERIFIED: 肖像PNG3/6/9存在；修正registry具备检测凭据、紧生成框、独立score box。
- VERIFIED: 旧LRS322条M音频路径不可用，不能直接恢复旧静态实验当作本轮输入。
- VERIFIED: 官方完整链、固定crop worker的预处理和支持规则不同，不能假设绝对C一致。
- LIKELY: 低维单调时间映射可修复部分生成动作提前/滞后，适合先做有限预算搜索。
- LIKELY: 相同M与同一Wav2Lip前端在三肖像上产生相同帧数；需工程断言。
- UNKNOWN: 当前GPU空闲/健康、三条官方检测是否稳定、120ms预算是否充分、插值质量是否可接受。
- UNKNOWN: 搜索收益能否在官方链及6/9肖像上保留；即使保留，是否能被未来音频增强器蒸馏仍未确定。

## 11. Handoff Contract

实现者沿第3节代码锚点新增独立包，保留第5节全部不变量；生成调用仿照static render worker，插值仿照原始像素half-up算法，官方评分仿照official_score的双命令。将改动限制在第7节，不做无关重构。

必须先完成资产/模型绑定、正确紧框、identity与评分校准，再进入固定预算搜索；随后封存q、迁移肖像、完整官方复评、独立checker和报告。不得用代理embedding、假权重、错误音频cell或只有stdout成功替代真实端到端证据。科学无增益可以是有效交付，未运行/阻塞不能冒充阴性。

若仓库现状与本spec冲突，保存具体路径、hash或形状证据，停止依赖该事实的分支并报告；继续不依赖它的检查。不要自动换数据、模型、裁剪、评分前端、支持或阈值。扩展25条、新语言、Ditto、RIFE或增强器训练需要单独协议修订。

### Relations

- inspired_by [[Wav2Lip oracle frame interpolation 2026-09-07]]
- relates_to [[Wav2Lip ROI retiming oracle 2026-09-06]]
- relates_to [[LRS3 音素增强静态肖像 TFG 与 MFA 条件实验 2026-09-22]]
- relates_to [[自然时钟受约束音素增强与机制对照 v2 Implementation Spec]]
- relates_to [[MFA-linear 视频重定时 Sync-C 约束工程 smoke 结果 2026-09-23]]
