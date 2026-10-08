---
title: Wav2Lip 等时钟加噪的生成端与评价端四格实验 Spec
type: spec
permalink: tts-exp/research/wav2-lip-等时钟加噪的生成端与评价端四格实验-spec
status: implemented
protocol: wav2lip_noise_cross_v1
tags:
- tts
- wav2lip
- syncnet
- causal-intervention
- implementation-spec
engineering_status: COMPLETE
scientific_status: INCONCLUSIVE_NARROW_NEGATIVE
implementation_run: runs/wav2lip_noise_cross_v1
---

## 1. Objective

实现并运行一个范围有限的四格实验，回答：在固定时钟的加噪干预下，SyncNet 分数变化有多少发生在评价音频变化时，有多少伴随生成视频变化？比较自然音频 N 与 TTS 音频 T 的响应，但不把加噪响应当成 TTS 原生增益的完整因果解释。

协议名：`wav2lip_noise_cross_v1`。主模型为本地冻结 Wav2Lip GAN，评价器为冻结 SyncNet V2。使用 24 个 LRS3 来源、单一 20 dB 有色噪声干预、同一来源的固定静态肖像。零新 TTS、零训练、零云调用；无需 MFA、VSR、视觉事件检测、自然时间轴迁移或 LeapTalk 环境。

本 spec 授权下游完成实现、CPU 检查、独立 smoke、正式运行、自审修正和 BM 结果记录；本次文档编写没有执行推理。

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 首版：核对历史缺口和本地数据后，设计独立 Wav2Lip 四格实验 | September 20, 2026 | user request |
| 自审：明确 RAW 缺失时四格仅作独立描述，主 E/G 使用同一完整五格集合，避免支持与分母不一致 | September 20, 2026 | user request |

问题分成两项主要检验：
- E_T：视频固定，仅给评价音频加噪，TTS 分支的 Sync-C 改变量。
- G_T：评价音频固定，仅改为加噪驱动生成视频，TTS 分支的 Sync-C 改变量。

N 分支提供对照；原生 T−N、交互作用和 N/T 响应差异为预先指定的次要描述。任何一项不显著都不是后续阶段停止条件。技术无效与科学证据不足分开。

## 2. Repository Model

既有流程与本次数据流：

`video_manifest_250.json` 的 ID → 历史 LeapTalk MP4 的嵌入音轨解码 → RAW / A0 / NOISE20 PCM → 同一真实视频首帧肖像 → `static_image_bridge/render_worker.py::main` → FFV1 无声音视频 → `SyncNetEngine.extract_visual/extract_audio/distance_matrix` → 固定共同支持上的五格评分 → 来源等权配对统计 → 独立复算。

注意：历史 LeapTalk 文件只充当已有音频载体，旧生成画面和旧分数不参与新四格。新模型结果不得宣称补完了旧 LeapTalk B 阶段。

已核实历史：
- [[LRS3 TTS 原生优势的分数分解与曲线诊断]] 已完成 12 来源曲线。
- [[Ditto TTS 局部时间辨识与共同语速控制结果 2026-09-20]] 已完成局部辨识与共同语速控制；无需本轮重复。
- [[TTS 原生增益来源的生成端与评估端交叉诊断]] 已完成固定视频 A 阶段，重新生成 B 阶段未完成。
- [[TTS 视觉时间校准与自然时间轴迁移实验结果 v2 2026-09-20]] 的视觉可测性门槛与本实验无关，不作为输入门。

输入固定为 ID 151–174，来自 `data/dataset_samples/video_manifest_250.json` 第 ID−1 条。24 个真实视频以及两臂历史音频载体均已核实存在；真实视频均 224×224、25 fps。本次未核实所有音轨解码质量，须 preflight 复核。

每条输入：
- 来源分组：`Path(video_local_path).parent.name`，不可用均为 lrs3 的 speaker_key 代替；这是来源视频组，不保证不同人物身份。
- N 载体：`runs/leaptalk_multiset/natural_raw/<id>.mp4`。
- T 载体：`runs/leaptalk_multiset/tts_raw/<id>.mp4`。
- 肖像：manifest 中真实视频第一张解码帧，PNG，无按评分或嘴部姿态挑帧。
- 全长音频，不截成相同秒数、不变速、不对齐两臂。N/T 各自独立时钟，只有同一臂 RAW/A0/NOISE20 共享时钟。

RAW 指“归档载体解码后的基线”，不声称与最初 TTS WAV 逐样点相同。报告必须写明可能经过容器编码，以及与旧 crop/ORIGINAL.wav 的区别。

## 3. Code Anchors

| path / symbol | 当前职责 | 本次动作 |
|:--|:--|:--|
| `scripts/experiments/static_image_bridge/render_worker.py::main, chunk_mels, encode_ffv1` | 固定 PNG、固定 box、官方 Wav2Lip mel 前端、确定性推理、FFV1 | 原样以 subprocess 调用；不另写推理器 |
| `scripts/experiments/tts_visual_timing.py::_render_wav2lip, _portrait_rgb_hash` | 渲染命令和 worker receipt 验证 | 仿照调用与图像 RGB hash；不导入其视觉校准/迁移 gate |
| `scripts/experiments/tts_native_gain_attribution/audio.py::read_pcm16_wav, write_pcm16_wav, pcm16_to_float, quantize_pcm16` | 严格 PCM16 I/O | 复用 |
| 同文件 `frame_rms_and_activity, low_energy_noise_power, noise_realization, active_rms` | 活动掩码、低能量谱和确定性噪声 | 复用纯函数；不调用会附带 DENOISE 等条件的 construct_source/audio_stage |
| `scripts/experiments/tts_native_gain_attribution/syncnet.py::SyncNetEngine` | 官方 MFCC/视觉前向和 [T,31] 距离矩阵 | 新 worker 导入复用；不调用绑定旧队列的 stage |
| `scripts/experiments/tts_native_gain_attribution/analysis.py::curve_metrics, four_cell` | 先时间均值再 lag 聚合，四格分解 | 复用公式/纯函数；不复用旧 12 来源 bootstrap 和多重比较设置 |
| `scripts/experiments/tts_native_gain_attribution/common.py::gpu_lease` | 共享 GPU 锁和资源检查 | 在主进程 GPU 阶段使用；子 worker 不重复取得同一锁 |
| `scripts/experiments/tts_visual_timing_syncnet_worker.py::score` | 独立环境评分、保存 embedding/matrix | 参考进程边界；不要照搬每格重复特征前向、50 行支持门或单格汇总作为最终共同支持 |
| `scripts/experiments/wav2lip_noise_cross.py`（新增） | 尚无 | 编排 audit/audio/render/score/analyze/validate |
| `scripts/experiments/wav2lip_noise_cross_metrics.py`（新增） | 尚无 | 干预构造、支持、统计与报告所需纯函数 |
| `scripts/experiments/wav2lip_noise_cross_worker.py`（新增） | 尚无 | 一个样本/臂的 SyncNet 特征复用与矩阵保存 |
| `scripts/experiments/check_wav2lip_noise_cross.py`（新增） | 尚无 | 从 PCM、embedding、matrix 独立验证 |

路径以当前仓库为准，禁止误用 `pre_repair_snapshot/` 的同名模块。

## 4. Reference Pattern

最接近的模式是 `tts_native_gain_attribution/analysis.py::four_cell` 与 `generation.py::crossed_score_stage`：同一时钟内，生成条件和评分条件交叉，固定支持，再分解变化。

本轮应模仿四格定义、特征复用和原始产物复算。不要复制 LeapTalk adapter、144+4 旧固定分母、两个生成 seed、去噪支线、盲评包和父实验完成状态依赖。

采用 `tts_visual_timing.py::_render_wav2lip` 的独立环境调用模式。Wav2Lip eval 推理主 seed 固定 42；不把不同 seed 当独立样本。实际重复渲染只用于技术重复性。

## 5. Invariants

1. 固定 24 来源/24 对；缺文件、音频无效、输出失败均留台账，不找替代样本，不根据 Sync-C 或原生 TTS 增益筛选。
2. N 与 T 之间从不直接换音轨。四格严格在同一 ID、同一 N/T 臂内形成。
3. 每臂 RAW、A0、NOISE20 采样率、样点数相同；原发音不时移、不重新采样、不重复归一化。解码到统一 PCM 只在最初做一次。
4. 同一 ID 六个主要视频使用相同肖像/box/模型/推理配置；只允许驱动 PCM 不同。生成不得输入真实视频帧序列。
5. 同一视频的不同评分音频共享完全相同的视觉 embedding。固定音频的两条视频共享同一音频 embedding。
6. 主评分严格使用同一臂全部五格共同的有效时间窗口；不能每格自行裁边、择 lag 范围或换脸框。
7. 缓存以输入 PCM、肖像像素、checkpoint、代码、协议和条件 hash 绑定；仅“输出文件存在”不表示有效。显式重复控制不得命中主缓存。
8. 主结论只来自正式 cohort；smoke 隔离目录，science=NOT_TESTED。
9. 不把 Sync-C 改善命名为嘴型更准确；不把不显著命名为无效；不把四格分解称为原生 TTS 增益的中介比例。
10. G_T/E_T 未显著、RAW 原生优势未显著均不阻断其他技术有效 cell；指标技术验证失败才阻止相应推断。
11. 旧代码接口、旧 runs、旧 spec、旧实验完成状态不可覆盖。工程 PASS 和科学判定分别保存。

## 6. Implementation Plan

### 6.1 固定配置与 preflight

新增 `scripts/configs/wav2lip_noise_cross_v1.yaml`，保存本节所有常量；runner 把解析后的配置和代码 hash 冻结到 protocol.json。正式 run 默认 `runs/wav2lip_noise_cross_v1/`；smoke 使用 `runs/wav2lip_noise_cross_v1_smoke/`。

固定 ID=151..174，控制 ID=(151,152)，source=(N,T)，seed=42，noise SNR=20.0 dB，fps=25，sr=16000，batch_size=4，vshift=15，bootstrap_draws=20000，bootstrap_seed=20260920，primary_family=(E_T,G_T)。

环境：
- 编排：当前 Python 3，须满足 numpy/scipy/PyYAML 等已有 CPU 依赖。
- 生成：`[redacted-local-path]`。
- 评分：`[redacted-local-path]`，PYTHONPATH 包含仓库根和 `third_party/syncnet_python`。
- ffmpeg/ffprobe：`[redacted-local-path]}`。
- Wav2Lip checkpoint：`third_party/Wav2Lip/checkpoints/wav2lip_gan.pth`，SHA256 `ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8`。
- SyncNet checkpoint：`third_party/syncnet_python/data/syncnet_v2.model`，SHA256 `961e8696f888fce4f3f3a6c3d5b343100b238e79b2659bff2c605442`。
- 数据 manifest SHA256：`a0ac3e6b6a678fe12963746c238327a90ed0075a0e79049e215a833685805ac0`。

逐条验证 manifest 的 dataset=lrs3、ID 与路径绑定、source_group 唯一；保存载体/原视频 SHA256、音频流信息、原始起始 PTS、解码命令。不读取历史评分决定是否入选。音频使用明确 `-map 0:a:0 -ac 1 -ar 16000 -c:a pcm_s16le` 解码到 WAV，不使用 `-shortest`，不从裁剪后的评估视频提取。新视频从解码 PCM 的样点 0 开始，因此记录容器原时钟但不额外套用历史 AV offset。

首帧输出应为 224×224；generation box 固定 [0,0,224,224]。记录文件 SHA 和解码 RGB SHA，显示联系图供报告审阅，但不设人工审批门。尺寸或解码不符则报输入问题，不悄悄改变裁剪协议。

24 对为有资源上限的机制试验，不承诺检出细小效应。每个 ID 仅一个 source group，不能将帧、五格或重复控制算入样本量。

### 6.2 构造三种音频，保留未缩放对照

对于某 ID，先解码 x_N、x_T 为 float64（int16 / 32768）。
- RAW_S = 原始解码 PCM，必须逐样点相等。
- 只用 x_N 的低能量帧估计谱 P，复用 low_energy_noise_power。P 是噪声合成模板，不称为分离出的真实环境噪声。
- 对 S=N,T，复用 noise_realization(length, P, sample_id=id, source=S)。记录该函数现有确定性 PCG64 seed；每臂只生成一个实现。
- m_S = frame_rms_and_activity(x_S) 的活动样点掩码，不用加噪后的音频更新掩码。
- beta_S = active_rms(x_S,m_S) / (10 × active_rms(z_S,m_S))，u_S=x_S+beta_S*z_S。
- 对一个 ID 的两臂四个数组一起算 g = min(1, 0.98 / max_abs(x_N,x_T,u_N,u_T))。
- A0_S=quantize(g*x_S)，NOISE20_S=quantize(g*u_S)。

不再进行额外 RMS/loudness 归一化，不调用去噪，不加入额外剂量。噪声增加了多种声学统计，结论只能描述该复合加噪操作。

操纵检查：
- 各臂三种 WAV 样点数精确相同，mono/16k/PCM16；无非有限数值、无 int16 饱和样点。
- A0 与预期缩放误差≤1 LSB，RAW 完全相等。
- 在原 m_S 上，用实际落盘 `NOISE20−A0` 与 A0 计算 SNR，应在 [19.8,20.2] dB。
- 原音频至少 1 秒 active samples；谱、signal/noise RMS 为正且有限。失败写 AUDIO_INVALID，不悄悄换白噪声/掩码/样本。
- 落盘掩码、P、noise seed、beta、g、所有 PCM hash，便于独立重建。

RAW 主要用于判断本地归档音频的原生优势；A0 是四格的正确基线。分别报告 RAW 与 A0 的 T−N，以及缩放带来的差异，不能把 A0 叫完全未处理的 raw。

### 6.3 生成、评分与固定分母

每 ID、每 S 生成 V_RAW、V_A0、V_NOISE20，全部调用现有 render_worker；主视频共 24×2×3=144 个逻辑 cell。g=1 且 PCM 完全相同可让 RAW/A0 显式别名复用，但仍保留两个逻辑 cell 和 reused_from。

渲染 receipt 必须绑定音频 hash、RGB hash、box、seed、checkpoint/worker hash、帧数、fps、输出像素与文件 hash。检查同臂三条件 mel 分块数量与视频帧数相同。保留 FFV1 视频，不转有损编码后再评分。

每 ID/S 只评分以下 5 格，共 240 主要评分 cell：

| key | 视频 | 评价音频 |
|:--|:--|:--|
| RAW | V_RAW | RAW |
| q00 | V_A0 | A0 |
| q01 | V_A0 | NOISE20 |
| q10 | V_NOISE20 | A0 |
| q11 | V_NOISE20 | NOISE20 |

每臂 worker 一次加载 SyncNet，视觉前向最多 3 次、音频前向最多 3 次，组合生成五个 [T,31] 矩阵。静态肖像本来就是 224×224 face crop；对所有生成帧使用同一全幅 224×224 ROI，不运行条件相关的人脸检测。使用现有引擎的 MJPEG 解码契约，完整记录；不混用另一套像素解码缓存。

同一 ID/S 令 L 为五格矩阵行数的最小值，主共同支持 W=range(15,L−15)。至少 25 行才能进入正式分数统计；不足写 INSUFFICIENT_SUPPORT，不填 0。N/T 长度无需相同，也不能把两臂窗口当作逐音素对应。
若某臂成功而另一臂失败，仍保留该臂结果；N/T 配对推断只用两臂五格均完整的来源对。两项主检验 E_T/G_T 使用同一个完整 T 五格集合及对应 W。RAW 失败但四格有效时，可另用四格共同支持给出 DESCRIPTIVE_ONLY 的 E/G/I，不混入主检验或原生比较；报告该诊断的单独分母。禁止用分母或支持不同的数相减。

指标必须先对 W 的每个 lag 列做 float32 时间均值，得到 d[j]：
- D=min_j d[j]；B=median_j d[j]；C=B−D。
- offset=15−argmin_j d[j]（帧）；d0=d[15]。
- 保存 31 点曲线、argmin、边界最优标记、支持行、embedding 和矩阵。
主指标称 `C_interior`（官方模型/公式、统一内部窗口），不冒充全片官方 stdout 的同一个数。不用“先每行取最小再平均”。

### 6.4 必须完成的技术控制

smoke 用 ID151、152，执行全部主逻辑和控制，与正式 run 完全隔离。smoke 后不依据效果方向改参数。

正式控制也固定 ID151、152，各 N/T：
- 4 个独立 V_A0 重渲染，禁止复制/缓存冒充重复。对 A0 评分，共 4 评分 cell。比较解码像素、PTS、embedding 和原四格共同支持的 C/D。
- 4 个 +200ms 评价音频延迟控制：在 A0 前加 3200 个零并截掉尾部 3200 样点，总长不变，视频仍 V_A0。只用于控制，共 4 评分 cell。
- 总计 148 个生成逻辑 cell、248 个评分逻辑 cell。若 RAW/A0 别名或特征复用，报告实际前向数与逻辑数，不伪造新生成。

重复控制在同等支持上要求 |ΔC|、|ΔD|≤0.01 且 offset 差≤1 帧；另报告矩阵最大误差和像素是否完全相等。超过则标 GENERATION_UNSTABLE 并修复确定性，不能把阈值改大。

延迟控制不要求 C 下降。在保守行区间 range(20,L_control−20) 上，比较延迟矩阵列 j+5 与未延迟矩阵列 j（j=0..25），最大距离误差≤1e−3；该映射表达“评价音频晚 5 帧”。如未延迟 argmin 及其移动后均不触边，另要求官方 offset 移动−5±1 帧；边界或并列最优时这个辅助判据 NOT_ASSESSABLE，不判模型阴性。合成 embedding 的位移测试必须先确认符号。控制区间不足标技术不可判，不通过控制链。

至少一个 smoke 格将保存的 embedding 交给官方 SyncNetInstance 的 calc_pdist 参考路径，检查完整矩阵误差≤1e−4，以及同 W 的 C/D/offset；不重新跑条件相关的人脸追踪。重复评分前向误差另记录，不能只自比同一缓存文件。

若阈值失败，先排查时钟、MFCC padding、解码和实现差异；若确属协议假设不成立，应保留失败证据并报告，不自动放宽阈值继续确认性推断。

### 6.5 四格统计与解释规则

每 ID/S 定义：
- E_S=q01−q00（评价音频改变的效果）；
- G_S=q10−q00（固定评价音频下，生成视频改变的效果）；
- I_S=q11−q10−q01+q00；
- Total_S=q11−q00=E_S+G_S+I_S。
统一“加噪后减加噪前”，负值表示分数下降。不要在表中混用 harm 的正号。

主要族只有 E_T、G_T，报告均值、中位数、正/负/零个数、分母及来源配对 bootstrap：
- 按 source_group 等权；若实现后发现重复来源，先组内平均再采样，并公开实际来源数。
- PCG64(seed=20260920)，20,000 次，两个主量共享采样索引。
- 95% CI 供描述，97.5% 双侧百分位 CI 用于两主项 Bonferroni 控制，分位点 [0.0125,0.9875]，quantile method=linear。
- 97.5% CI 全负=NEGATIVE_RESPONSE，全正=POSITIVE_RESPONSE，跨0=INCONCLUSIVE。另报告区间是否完全落入 [−0.200,+0.200]：WITHIN_PREDECLARED_SCORE_RANGE。这只是本研究的分数尺度，不是人类可感知阈值。
- 24 中可用完整 T 五格不足20来源时，主推断标 LOW_COVERAGE，仍报告估计和 CI，但不输出确认性方向标签。不以达到20来源为统计功效充分的证明。
- 不做依据 p 值/CI 的追加样本或提前停止；本轮固定24，对小效应可能仍不确定。

次要量全部标 SECONDARY/DESCRIPTIVE，给95%来源 bootstrap CI，不做批量显著性叙事：
- E_N、G_N、I_N、I_T、Total_N/T。
- E_T−E_N、G_T−G_N（必须同一完整来源对；两个都负不等于差异成立）。
- RAW 的 C_T−C_N，A0 的 q00_T−q00_N；各自正向比例。
- A0−RAW 的每臂变化、原生优势差 (A0_T−A0_N)−(RAW_T−RAW_N)。
- 对四格的 B、D、d0 变化和 offset 变化，分解 C 变化的数值组成；不得称其为独立嘴型证据。
- 逐来源散点及四格图；曲线仅作诊断，不增加 H/rank/VSR 等新主要指标。

解释模板：
- E_T 明确为负，G_T 区间很宽：只确认评价路径对该噪声敏感，生成路径尚不确定。
- E_T 明确为负，G_T 整个主区间落入±0.200：在本剂量/模型/数据下，生成路径的分数变化被限制在预设范围；不是所有生成效应不存在。
- G_T 明确为负：同一评分音频下，加噪驱动导致视频变化并降低分数，支持本操作存在经生成视频传递的效果；不能证明真实嘴型变差。
- 两条路径都有响应：报告同时存在及 I，不计算“评价器贡献百分比”。
- RAW T−N 没有明确优势：本实验仍能回答 Wav2Lip 的噪声响应，但不能用来解释已经在 Ditto/LeapTalk 观察到的原生优势。
- RAW/A0 优势不同：缩放是条件改变，分开解释；禁止借 A0 阳性宣称 RAW 复现。
- 即使 T 比 N 更抗噪/更敏感，也不证明自然音频原有噪声就是 TTS 优势的原因。噪声合成是一个局部机制探针，不是 N→T 因果转换。

### 6.6 CLI、产物和资源

统一 CLI（必须实现）：

```bash
python scripts/experiments/wav2lip_noise_cross.py --config scripts/configs/wav2lip_noise_cross_v1.yaml --run-dir runs/wav2lip_noise_cross_v1_smoke --smoke --stage all
python scripts/experiments/wav2lip_noise_cross.py --config scripts/configs/wav2lip_noise_cross_v1.yaml --run-dir runs/wav2lip_noise_cross_v1 --stage all --resume
python scripts/experiments/check_wav2lip_noise_cross.py --run-dir runs/wav2lip_noise_cross_v1
```

stage 支持 audit/audio/render/score/analyze/validate/all。all 应运行独立 checker 并输出报告；遇到可隔离样本失败继续其余样本，最后如实 PARTIAL。输入协议身份不一致、模型 hash 不符、整体控制失败不能当作普通缺样本忽略。

产物保持简单：
- protocol.json、inputs.json（固定所有 cell key 和输入/环境 hash）；
- audio/、videos/、features/、scores/（WAV、FFV1、embedding、matrix 与简洁 metadata）；
- controls.json、per_source.csv、summary.json、validation.json、report.md、figures/；
- 每 cell 有 status/reason，失败不得只出现在 stdout。

GPU 串行共享锁；无 GPU则 RESOURCE_WAIT，可完成 CPU阶段。当前磁盘仅约7.3GiB余量，preflight重新测量：由 smoke 的每秒FFV1大小和正式总时长估算剩余产物，再乘1.5，另外保留1GiB及单cell临时空间；报告预算。不要写解码帧JPEG目录；不要因资源不足删除旧runs/权重。长音频ID156原视频约78秒，现有renderer会累计帧到RAM，应估算其raw帧缓冲与编码副本内存；如不够，先保存明确资源阻塞，不静默截断音频或更换样本。

科学状态只受技术有效性、覆盖和预定统计影响；未达到预想阳性不触发停跑。保持日志和短进度更新。

## 7. Expected Change Surface

### Must change

- 新增 `scripts/experiments/wav2lip_noise_cross.py`。
- 新增 `scripts/experiments/wav2lip_noise_cross_metrics.py`。
- 新增 `scripts/experiments/wav2lip_noise_cross_worker.py`。
- 新增 `scripts/experiments/check_wav2lip_noise_cross.py`。
- 新增 `scripts/configs/wav2lip_noise_cross_v1.yaml`。
- 新增 `tests/experiments/test_wav2lip_noise_cross.py`、`tests/experiments/test_check_wav2lip_noise_cross.py`。
- 完成后在 BM Experiments 写一篇结果，Research 核心枢纽增加链接；遵循 Startup Router，先搜后写，不复制进度日志。

### May change

- 仅若实测证明 renderer 的累计帧内存导致阻塞，可对 `static_image_bridge/render_worker.py::encode_ffv1/main` 做兼容的流式编码修复；CLI、输出像素、mel分块不得变，增加针对性回归测试并记录新hash。
- 共享纯函数若有确定 bug，先用失败fixture证明，再最小修复和回归；不得为了本实验修改旧协议常量。

### Should not change

- 旧 LeapTalk B 的 spec/status；tts_visual_timing 的门槛；旧结果；模型权重；主流水线 config 默认值。
- 不添加训练头、VSR、MFA、Rubber Band、新数据下载、另一个模型、去噪条件或多剂量。
- 不拆成通用插件系统、任务调度框架、十几个包或大量计划文档。

## 8. Validation Plan

实施顺序：CPU行为测试 → 两ID完整smoke → 独立checker → 固定24正式运行 → 独立checker → 结果与科学边界自审。

```bash
python -m py_compile scripts/experiments/wav2lip_noise_cross.py scripts/experiments/wav2lip_noise_cross_metrics.py scripts/experiments/wav2lip_noise_cross_worker.py scripts/experiments/check_wav2lip_noise_cross.py
python -m pytest -q tests/experiments/test_wav2lip_noise_cross.py tests/experiments/test_check_wav2lip_noise_cross.py
git diff --check
```

若 repo 的 ruff 已可用，针对新增文件运行；不要为了lint全仓重排代码。

必须有以下有意义的 CPU 用例：
1. 给定两臂不同长度PCM，生成三条件；RAW逐样点相等，共同g、20dB SNR、无裁剪且时钟不变。
2. g<1 的高峰值fixture，验证缩放计入A0而RAW不变；杜绝拿RAW作为加噪四格的q00。
3. 已知四格 q00=5,q01=4,q10=4.5,q11=3，得到 E=−1,G=−0.5,I=−0.5,Total=−2。
4. 人工矩阵使“先每行min”与“先时间均值”不同，确保实现采用后者。
5. 五格长度不同与边缘padding，确保同W、无补零窗口被当有效；少于25行不是0分。
6. 合成embedding延迟5帧，验证列移动和official_offset符号；不强制C下降。
7. 固定cohort与source bootstrap：重复帧/控制不增加样本量；原生差为负或CI跨0仍完成生成计划。
8. 篡改condition、PCM/model/spec hash、把q01错绑到另一来源、缺q10、重复控制指向主文件：checker必须失败。
9. 成功路径小fixture含全五格与两个主量，checker独立重算得到一致数值；仅台账数量不能算验收。
10. resume：有效缓存复用，改变任一依赖不复用；smoke不混入正式；错误重试不重复统计cell。
11. RAW失败但四格有效、N失败但T有效、样本不足20：分别测试分母/状态；不混用不同分母相减。

独立checker可以共享I/O，但不得调用runner的summary、主分析函数或four_cell来“验证”它自己。对所有主cell从保存的embedding以独立float32 pairwise-distance实现重建矩阵（含eps=1e−6的逐维加项），逐元素误差≤1e−4；独立重算W、C/B/D/d0、四格、固定seed bootstrap和状态。PCM操纵检查也从落盘WAV和掩码复算。对所需产物逐个核对绑定，检查同视频重复评估时视觉embedding身份一致。

正式工程 COMPLETE 条件：144主要生成逻辑cell+4真实重复、240主要评分+8控制均完成，控制PASS，checker PASS。如存在失败则PARTIAL并公开固定分母；不能因为科学LOW_COVERAGE而把失败cell藏掉。工程COMPLETE与科学INCONCLUSIVE可以同时成立。

报告数值使用原始精度计算，Sync-C展示3位小数；禁止展示舍入后再统计。

## 9. Risks and Edge Cases

- 首要解释风险：本实验定位“指定噪声干预的路径”，不是定位TTS全部原生优势。标题与结论必须保留这个范围。
- N/T并非随机处理、语速/内容可能不同；跨臂RAW比较为配对描述。四格内部同一PCM的加噪操作才是明确受控干预。
- 同一源码视频组可能出现相同人物，source group bootstrap不自动支持跨人物总体结论。
- 使用自然低能量谱不等于已识别自然录音噪声；低能量帧可能含语音。
- 同一生成视频重新裁脸会伪造评价端效应，因此固定全幅ROI并重用视觉embedding。
- 噪声与音量归一化混淆由RAW/A0双基线显式测量；不要把g不同当独立处理。
- 全局延迟能被SyncNet搜索补偿；控制验收看曲线位移，不看C必须下降。
- 噪声改变mel，但Wav2Lip可能响应很小；这是结果，不是自动认定worker没读音频。用receipt/PCM/mel身份和输出像素差分核验，不要求输出必须变化。
- 原native优势阴性不得触发整实验gate；这轮无视觉事件门，避免重复5/12可测链路。
- 24来源不能保证小效应结论；宽区间如实报告，不能靠放宽CI、挑子集或追加样本得到阳性。
- 同一对N/T的时长不同，不做逐帧差分、不把等窗口数叫相同发音内容。

## 10. Assumptions / Unknowns

- VERIFIED：本地checkpoint与上述SHA一致；两个venv Python路径存在。
- VERIFIED：ID151–174的24个真实视频和48个历史载体存在；来源父目录互异，真实视频为224×224/25fps。
- VERIFIED：renderer接受静态PNG、固定box、WAV并输出FFV1；SyncNetEngine提供两路特征和距离矩阵；已有音频纯函数能构造20dB有色噪声。
- VERIFIED：本轮没有实际模型前向；环境路径存在不等于CUDA/依赖已通过smoke。
- LIKELY：现有两个环境可直接运行；必须用smoke验证，不升级整套依赖。
- UNKNOWN：全部载体音轨解码/内容质量、TTS在此Wav2Lip静态脸协议上的RAW优势、正式有效支持比例和最终CI宽度。
- UNKNOWN：现有磁盘是否够保存全量FFV1、长句渲染RAM峰值；由smoke实测和时长估算解决。
- UNKNOWN：该干预响应能否推广到其他TFG或真实视觉准确性；本轮不作该推论。

## 11. Handoff Contract

下游必须遵循列出的代码锚点，保留全部不变量，复用现有Wav2Lip renderer、SyncNet引擎和音频纯函数；把改动限制在列明文件，不做无关重构。

按 audit→audio→render→score→analyze→validate 完成真实成功路径。smoke通过后自动跑固定24来源；不因native或中间统计未阳性询问是否继续。资源/依赖失败保留已完成cell并明确原因，解决可修复问题后续跑；若仓库事实与模型/数据/统计协议实质冲突，停止受影响分支并报告证据，不自行替换模型/样本/干预。

最终交付实现、可重跑命令、完整固定分母、真实媒体与特征、独立validator结果、简明报告与BM结果链接。报告必须回答：E_T、G_T分别是多少、区间多宽、多少来源方向一致；N对照如何；RAW优势是否复现；哪些结论仍不能得到。

BM完成记录时先读Startup Router并搜索既有节点。结果笔记链接本spec、历史交叉诊断和核心枢纽；不要把工程完成写成TTS机制已证明。

- [status] spec_ready
- [question] 固定时钟加噪引起的同步分数变化，经评价音频与生成视频两条路径分别如何表现？
- [decision] 使用24个固定LRS3来源和本地Wav2Lip，单剂量四格，RAW/A0分开；native结果不阻断生成
- implements_next [[TTS 原生增益来源的生成端与评估端交叉诊断]]
- relates_to [[TTS 增强 TFG 的原因：证据、排除项与因果分叉]]
- informed_by [[LRS3 TTS 原生优势的分数分解与曲线诊断]]
- informed_by [[Ditto TTS 局部时间辨识与共同语速控制结果 2026-09-20]]


## Implementation Record

- [status] implemented_and_validated
- [result] 正式运行 `runs/wav2lip_noise_cross_v1` 完成 240 个主评分单元与 8 个控制评分单元；独立 checker 为 PASS。
- [result] 主要 E_T/G_T 均值接近 0 且 97.5% CI 跨 0；该实验只给出“20 dB 有色噪声响应”这一窄机制的阴性/不确定结果。
- [report] [[Wav2Lip 等时钟加噪的生成端与评价端四格实验结果 2026-09-20]]