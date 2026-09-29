## 1. Objective

设计版本：2026-09-22；protocol_id：`tts_evidence_temporal_patch_v1`；状态：SPEC_ONLY，未实施、未运行。

分两阶段回答：

1. A：复用现有 108 条 Wav2Lip 视频，判断官方 Sync-C 的 TTS 增益，是否同时伴随局部时间辨识、独立视觉时间响应和人工感知同步改善。
2. B：在小规模配对样本中，只干预 Wav2Lip 的真实音频瓶颈表示，检验对齐后的短时动态残差是否影响固定 recipient 音频下的生成结果，以及其时间组织是否比同能量错时扰动更有用。

目标是补足证据并定位有限机制；不要求得到正结果，不训练增强头，不新增 TTS，不做 Ditto/多层扫描，不把分数变化直接写成嘴型准确性变化。A 是已观察 cohort 的证据补充，B 是探索性定位；均不是全新来源的确认实验。

## 2. Repository Model

现有父 run（下称 P）：`runs/phoneme_tfg_association_external_visual_cpu_audit_v1_20260922`。

`P/00_protocol/blocks.json` / `generation_plan.json`
→ `phoneme_tfg_association.generation.render_wav2lip`
→ `wav2lip_face_roi_replacement.generation_worker.mel_chunks/load_model/render_arm`
→ `Wav2Lip.forward → audio_encoder → face_decoder_blocks`
→ `P/03_video/wav2lip/<sample_id>/<arm>.mkv` 与 receipt。

评价有两条不同路径：

- 官方：`official_syncnet_eval.evaluate → run_pipeline.py → run_syncnet.py`，已有 `P/07_official_syncnet/full/summary.json`，108/108，失败 0，min_track=25。这里的 Confidence 是 full-track endpoint。
- 自定义特征：`P/04_syncnet/wav2lip/<sample_id>/<arm>/` 下已有 visual/audio 特征、distance/static_distance 与 receipt；`scoring.summarize_curve` 支持 C/B/D 分解。其裁剪、支持域和官方路径不一定相同，不能改名为官方值。

已核实 cohort 为 36 个不同 source_group，每组 natural + 两个 TTS，四个 TTS 各覆盖 18 组；分析单元是 source_group，不是 108 个独立视频，也不是 72 个独立对照。

历史 `tts_time_instance.run_a` 发现 SyncNet 局部时间排名信号，但不是独立感知评价；`tts_visual_timing_worker.extract` 与 `tts_visual_timing_metrics.aperture_events` 提供独立于 SyncNet 的嘴部运动测量，并可能因低运动或缺失而不可测。

Wav2Lip 的 `audio_encoder` 将每个 `[1,80,16]` mel 窗口压缩为 `[512,1,1]`。当前静态脸路径逐窗口独立解码，没有跨输出帧的递归状态。因此 B 所称动态是“窗口表示随时间变化”，不是循环网络记忆；完整 donor 瓶颈交换只证明接线，不能成为时间机制结论。

## 3. Code Anchors

下表中“复用”均为只读调用/适配，不授权修改历史实验。

| path | symbol | 当前职责 | 所需变化/用法 |
|:--|:--|:--|:--|
| `scripts/experiments/phoneme_tfg_association/protocol.py` | `file_sha256`, `canonical_hash`, `validate_receipt` | 输入身份、receipt 检查 | 新协议复用 hash；在新模块加强到文件/PCM/解码帧/模型/配置/代码绑定 |
| 同上 | `gpu_processes`, `assert_gpu_clear` | 检查计算进程 | 仅参考；目前查询失败会返回空列表，新 guard 必须 fail closed，不能直接视为空闲 |
| `scripts/experiments/phoneme_tfg_association/generation.py` | `render_wav2lip`, `_mux_pcm` | 外部单帧、变长渲染、无损复用 PCM | 复用既有基线入口与 PCM mux；不改其 API 或 P |
| `scripts/experiments/wav2lip_face_roi_replacement/generation_worker.py` | `mel_chunks`, `load_model`, `render_arm`, `encode_video` | mel、模型、96×96 ROI 推理、FFV1 | 新 worker 复用；薄模型包装器完成 hook，不复制 decoder |
| `third_party/Wav2Lip/models/wav2lip.py` | `Wav2Lip.audio_encoder`, `Wav2Lip.forward` | 512 维音频瓶颈与解码 | 运行时 forward hook，模型文件与权重只读 |
| `scripts/experiments/phoneme_tfg_association/official_syncnet_eval.py` | `discover_receipts`, `parse_score`, `evaluate` | 官方重评分 | A 只读导入已算结果；B 在新 output_root 运行官方命令；新适配器要求恰好一个合法 track，不能静默取第一个 Confidence |
| `scripts/experiments/phoneme_tfg_association/scoring.py` | `summarize_curve`, `distance_from_embeddings`, `equal_count_rows` | 曲线与支持域 | 复用纯函数；保留 `custom_fixed_roi_*` 命名和来源 |
| 同上；`scripts/experiments/wav2lip_roi_peak_recheck/worker.py` | `score_native`; `SyncNetScorer.score` | 固定ROI特征/矩阵提取 | 新评价适配器沿用前者已核验的调用边界；worker实际实现与返回schema在实施前进一步核对，不与官方tracking混用 |
| `scripts/experiments/tts_time_instance.py` | `parse_textgrid`, `classify_event`, `_calibrate_k`, `run_a` | 对齐、局部 rank、lag 校准 | 参考公式/控制；不复制硬编码 ID、历史数据路径或原地调参方式；新实现采用下述分离支持的 lag 校准 |
| `scripts/experiments/tts_visual_timing_metrics.py` | `audio_time_map`, `map_time`, `continuous_support_blocks`, `aperture_events`, `event_distance` | 严格音素映射、嘴部事件 | 使用有 word tier 的 token adapter；保留原有效性/低运动门槛，不放宽以凑样本 |
| `scripts/experiments/tts_visual_timing_worker.py` | `extract` | 原始 478 landmarks、PTS、validity | 作为隔离环境 subprocess 复用；不插值修补检测失败 |
| `scripts/experiments/wav2lip_probe_runtime.py` | `gpu_lock` | 项目 GPU 锁 | 使用相同锁路径，但新模块采用非阻塞锁/有限重试，并额外核验设备状态 |
| `scripts/experiments/tts_evidence_temporal_patch/protocol.py`（新） | `audit_parent`, `freeze_protocol`, `select_patch_cohort`, `gpu_lease` | 无 | 唯一输入、选择、资源与缓存契约 |
| `…/evaluation.py`（新） | `import_official`, `score_temporal_rank`, `extract_visual`, `export_blind_pack`, `import_ratings`, `score_patch_media` | 无 | A 证据与 B 共用评价适配器 |
| `…/intervention.py`（新） | `build_frame_map`, `decompose_dynamic`, `build_patch`, `validate_patch` | 无 | NumPy 纯变换与支持域，不引入 torch |
| `…/worker.py`（新） | `capture_bottleneck`, `PatchedForward`, `render_patches` | 无 | 独占 GPU 中捕获/移植；每个 batch 明确帧索引；finally 删除 hook |
| `…/analysis.py`（新） | `summarize_evidence`, `summarize_patch`, `cluster_bootstrap` | 无 | 组级统计、证据状态与报告 |
| `…/run.py`, `…/check.py`（新） | `main`, `validate_run` | 无 | 分阶段 CLI、独立验收，不隐式启动 GPU |

## 4. Reference Pattern

- 采用 `phoneme_tfg_association.generation.render_wav2lip` 的外部冻结单帧、真实 mel 长度、receipt 和文件校验；不复制早期 LRS3 原视频输入，不复制其他 worker 固定 93 帧的假设。
- 采用 `tts_visual_timing_worker.extract` 的“隔离环境只提特征，纯 metrics 做科学判断”边界，以及未测量/低运动的显式状态。
- 采用 `tts_time_instance.run_a` 的逐 query 时间错配明细和 source-group 聚合；不把旧 12 条实验的数值移入本 cohort，不使用同一支持选择最佳 lag 后再评价该支持。
- 采用 `official_syncnet_eval.evaluate` 的每 cell 日志、唯一 reference 和顺序执行；A 不重新调用该函数覆盖历史分数。
- 没有找到足够接近的 activation-patching 模式。新增局部 hook 包装器，禁止建立通用解释性框架或重写 Wav2Lip。

## 5. Invariants

1. P、所有音频、对齐、参考脸、权重和历史产物只读；所有新数据写 `runs/<new_run_id>/`。缓存身份不符即拒绝复用，不覆盖旧文件。
2. LRS3 原视频/其抽帧不得用于生成、donor、口型教师或本轮参考真值；只用 P 已冻结的外部非 LRS3 单帧。LRS3 音频和 TextGrid 合法。
3. A 的 108 视频零次 TFG 重生成，零次 TTS；缺失 cell 如实报告，不更换模型、不补抽高分样本。官方 full-track 与自定义 fixed-ROI 指标永不混列。
4. B 每个 recipient 的参考帧、ROI、mel、帧数、原始 PCM、时钟、seed 和模型固定；只替换指定瓶颈输出。N←T 始终用 N 原音频评价，T←N 始终用 T 原音频评价。
5. 不为每个处理单独寻找最佳 lag、DTW 或时间轴以提升主指标；所有处理共享 recipient baseline 的冻结校准和共同支持。
6. MFA occurrence 必须按词序与词内 phone 匹配，不用“最近同音素”；不跨不连续映射缺口插值。不把英语 phone 标签强转成中文标签。
7. 源组等权；重复 natural、同源两种 TTS、帧、事件及评分者不作为额外独立 n。缺失分母和原因逐层保存；NaN 不填 0。
8. GPU 使用前取得锁、成功核验设备/进程状态，再启动一个 worker。未知状态、有其他进程、锁占用均拒绝启动；不杀其他任务，不静默 CPU fallback。
9. 工程状态（COMPLETE/INCOMPLETE/BLOCKED）与科学状态（POSITIVE/NEGATIVE/INCONCLUSIVE/UNMEASURABLE）独立。没有人工标签时感知证据为 PENDING_HUMAN，不生成虚构评分。

## 6. Implementation Plan

### 6.1 冻结输入与运行协议

1. 从本目录 `input-bindings.json` 核验 P 的 5 个顶层文件，再遍历 108 receipts：视频 SHA、对应官方 record/日志、audio 文件和解码 PCM SHA、参考帧与 crop SHA、checkpoint SHA、source_group、arm、fps、PTS、tail/frame count。每个生成 cell 必须一对一连接，拒绝重复/多 track/缺字段。用解码帧 hash 辨别容器相同与像素相同。
2. `prepare` 只读 CPU：产出 `00_protocol/{parent_inventory,protocol,selection,cache_inventory}.json`。协议包含代码文件 hash、Python/依赖版本、权重、配置、支持域定义、seed=20260922、bootstrap_draws=20000、预算。启动科学阶段后不可改；改变即新 run/revision。
3. 新 CLI：`python -m scripts.experiments.tts_evidence_temporal_patch.run --run-id <id> --stage <stage> --config scripts/configs/tts_evidence_temporal_patch_v1.yaml`。stage 固定为 `prepare`, `evaluate-existing`, `export-blind`, `import-ratings`, `patch-preflight`, `patch-smoke`, `patch-render`, `patch-evaluate`, `analyze`, `check`。默认不提供自动跑完整 GPU 链路的 `all`。

### 6.2 A：复用 108 视频的评价证据

**官方事实。** 导入 108 个 full-track Confidence/Min dist/offset，不重算。每组计算两项 `ΔC_gm=C_T−C_N`，总体先组内平均再跨 36 组平均；四模型各 18 组只报告配对增益，不声称共同 natural 下的完整模型排名。输出 Sync-C 展示值统一三位小数，计算保留完整精度。

**SyncNet 时间辨识（仍属于评分器证据）。** 复用通过 receipt 身份检查的 `04_syncnet` 特征与距离矩阵。缺缓存时可重新提取特征但不得重生成视频；GPU 走同一 guard。按每个 arm 的真实连续支持划分前 1/3 校准与后 2/3 评价，二者间排除 0.8 秒 guard（覆盖 ±15 帧 lag 和窗口宽度），并剔除边界 padding/tail。校准枚举 k∈[-15,15]，最小平均距离，平局依次最小 |k|、最小 k；评价不得再优化 k。

主时间诊断 `R=mean[1(d(i,k+δ)>d(i,k))+0.5·1(tie)]`，δ∈{-5,-3,-2,+2,+3,+5} 帧。只有全部 δ 都可用的 query 才进入共同支持。每 arm 至少 10 个校准 query、25 个评价 query，否则 UNMEASURABLE。报告 raw 和单位范数版、跨/同 phone 事件分层、C=B−D、背景项 B、最小距离 D、trough width。不得将这些重算值冠以 `official_fulltrack`。raw 为预定主诊断，其他为探索性。

**独立视觉测量。** 对 108 个视频调用 `extract`，复用输入 hash/landmarker hash/PTS/算法均匹配的既有缓存。用 `aperture_events` 保存 aperture、事件、validity、LOW_MOTION；不使用 SyncNet 指导事件选择。这里测的是可见运动，不是准确性真值，不报告“运动更多=同步更好”。

用确定性选出的 4 个技术校准源组（每种 TTS 一个，选择规则见 6.3），对其 N/T 视频做已知 ±3 帧整体索引平移和局部 ±2 帧三角形时间变形的派生控制；不运行 TFG。冻结共同内区，事件位置由已知索引映射给出，不从评分器拟合。原事件至少 3 个、每个控制匹配事件 recall≥0.8、事件位置 MAE≤1 帧才称该视频时间量尺通过。未通过保留失败，不放宽阈值。这是量尺校准，不是生成器对音频扰动的响应。

**人工感知主证据。** 导出 72 个 N/T 原生 AV 配对，复用视频；每对由至少 3 名独立评分者评价。无模型名、分数和处理名，左右/播放顺序按冻结 seed 随机且平衡。问“哪段声音与嘴部动作更同步”：left/right/tie/unjudgeable；音质偏好与视觉质量另问，不并入同步主评分。原生条件时长不同，不截成同长假装同时间轴；相同显示尺度和播放设置。明确听觉上可能识别 TTS，所谓盲仅隐藏实验标签，不能完全消除音质偏好。

每名评分者另含 8 个 ±200 ms 错配注意力控制及 4 个重复题；控制≥7/8 正确且重复题≥3/4 一致才纳入。控制不能参与主统计；不足 3 名合格评分者的 pair 标为缺失。同步得分 T 胜=1、平=0.5、N 胜=0；先评分者均值，再组内两 TTS 均值，最后跨源组，主 estimand 为 `H−0.5`。至少 30/36 源组具有完整两对才作总体推断，否则仅描述缺失和部分结果。导出网页/CSV 不等于已经完成盲评；本任务不自动联系或雇用评分者。

**A 统计与判断。** 对 H−0.5 与 ΔR 两个预定问题分别给 97.5% bootstrap CI（两项 Bonferroni），同时给普通 95% CI。source-group bootstrap 20000 次，固定 PCG64 seed。ΔC 与 ΔH/ΔR 的 Pearson/Spearman 先组内聚合后计算，CI 按源组重采样；不把它们解释为因果中介。四模型与事件分层不另加“显著”标签。

评分者作为本次固定评分面板；源组 bootstrap 的推断不自动推广至所有人群。source_group 是源视频簇，不保证不同簇就是不同说话人；有可信 speaker_id 时追加 speaker-cluster 敏感性分析，无身份时明确这一独立性限制。任何多分支诊断均不能补做“挑最显著结果”的主检验。

报告分为：官方增益复现；时间响应一致/不一致/未定；感知一致/相反/未定/待标注；视觉量尺通过率。只有 H 的校正区间下界>0 才说本 cohort 感知证据正向；若只有 ΔC、ΔR 正向，结论止于评分器时间辨识增强。感知正向也不证明自然音频替换有效。

### 6.3 B：样本、对齐和固定表示干预

**队列。** 先从 72 个可用 N/T pair 构造二分配额选择，source_group 全局不得重复。稳定排序键为 SHA256(seed|group|tts_arm)，解有多个取字典序最小解。选择 4 个技术校准组（每模型 1）与 12 个科学组（每模型 3），两集合不交叠。技术组只按输入可用性选择；科学组再要求双向严格映射覆盖≥80%、每方向至少一个≥25 帧连续有效 core、合计 core≥50 帧。不读 C、silhouette、盲评或任何增益来挑选。不足即 INSUFFICIENT_SUPPORT；不得改配额或按结果补样。

**帧时钟。** 通过 worker 的实际 mel start 保存每帧 `start=floor(i·80/25)`、mel sample support 与 nominal center `(start+8)/80`；最后重复 tail chunk 单独标记并不进入干预支持。禁止简单假设瓶颈时间为 i/25 或使用 SyncNet 的 0.1075 秒中心替代 mel 中心。MFA word-tier adapter 输出 `audio_time_map` 所需 token 结构；取 recipient nominal center 经严格 piecewise map 到 donor，再用 donor 实际 centers 线性插值。插值左右帧及其 mel 支持均须落在同一连续匹配 block；斜率在 [0.5,2] 外、间隙、静音和边界支持均不移植。不得 clamp 或外推。

**唯一层。** 只取 `model.audio_encoder` 输出，缓存 H∈R^(F×512)，float32、eval、no_grad；保持脸分支原样。不扫描层、不训练 probe、不擦除所谓“natural/TTS方向”。

**动态定义（操作性定义，不叫纯时间因素）。** 每个共同 core block 上，对 recipient H_r 与映射后 H_d 分别做 5 帧中心 boxcar 平滑 L，窗口不得越过 block；先在扩展支持上平滑，再取去掉两端 2 帧后的 core。令 u=H_r−L_r、v=H_d−L_d，并各自减 core 时间均值；两者都作为 512 维时间序列。把 v 用单一标量调整到 u 的 Frobenius norm，不逐通道拟合。u/v norm≤1e−8 的 block 为不可干预。记录原始/调整后的范数、余弦、相邻差分与帧支持。保留 H_r 本身为基底，而非用 L_r 重建，确保基线绝对一致。

固定 λ=0.5，无结果驱动调参。同一 recipient 的四条件：

| 条件 | core 表示 | 意义 |
|:--|:--|:--|
| BASE | H_r | 同 runtime 基线 |
| COHERENT | H_r + λ(v−u) | donor 对齐动态替换 |
| SCRAMBLED | H_r + λ·π(v−u) | 将完整扰动向量按帧循环移动 floor(core_len/2)，保持均值/Frobenius 范数，破坏其时间位置 |
| ERASE | H_r − λu | 削弱 recipient 自身动态的诊断 |

π 在 core 内执行、不跨 gap；每 block 保存 permutation。core 外三条件严格等于 H_r；不另外 taper（否则能量控制改变）。接缝可能产生不自然变化，边界±5 帧不得进入局部主分析，并在质量审计中独立报告。SCRAMBLED 控制的是“同能量扰动放在错误时间”，不是证明除此之外所有语义保持不变。

双向 N←T、T←N 均执行。完整 donor 替换仅作张量级接线验证，不作为科学条件，因为静态脸模型的全瓶颈替换会近似复制 donor 解码结果。

### 6.4 B 工程校准、预算与资源

技术校准 4 组×2 recipient×4 条件（原生重放、恒等 hook、瓶颈 core 索引 +3/-3 帧）=最多 32 个完整视频。移位控制只从实际有效邻帧取 H，不循环/填充；共同内区上检查生成 ROI 是否等于按相同映射索引的原生 ROI，这是确定性传播检查，不是同步质量改善。像素接线应通过；landmark 时间量尺仍可因 LOW_MOTION 不通过，两者分开。

同 runtime 重复/恒等 hook 的 pre-quantization 输出 max_abs≤1e−6；解码帧应一致。原 P 的 CPU 输出和本次 GPU 输出做单独 parity 报告，不要求跨硬件视频容器 SHA 相同。若存在跨设备像素差异但内部重放正常，科学阶段统一使用本次 BASE，并记录与 P 分数差异；不得把旧 CPU BASE 与新 GPU patch 混算。

科学阶段最多 12组×2方向×4条件=96 个视频；加技术阶段上限 128。满足逐 cell 精确缓存身份的 BASE 可复用，实际渲染数可更少。不得把单帧 ROI 重放次数伪称独立样本。视频可全长但只解码一次输入脸、缓存 H；按 batch=4 顺序运行。

`gpu_lease` 先获取 `/tmp/tts-exp-wav2lip-gpu0.lock` 的非阻塞锁，再查询 nvidia-smi：目标 UUID、全部可见 PID（含 graphics/context）、显存、利用率；查询失败/无权限/进程不可识别为 RESOURCE_UNKNOWN，任何非本任务进程为 RESOURCE_BUSY。若无法安全界定目标卡则保守拒绝。父进程传 UUID 给 CUDA_VISIBLE_DEVICES；worker 初始化前二次核验，stage/cell 间检查新外来任务，发现冲突停止后续启动，保留已完成产物。锁只能协调本项目，不宣称能阻止外部任务抢占。每次检查落 receipt；不无限占锁等待、不杀进程。

smoke 输出每 cell 耗时/峰值显存/剩余预计时间。配置 `max_gpu_minutes=120`、`max_new_render_cells=128`；预计或累计超过预算则停止并报告，不能自动扩容。A/B 的特征提取和官方评分也计入 GPU 时间。

### 6.5 B 评价与科学判据

所有科学条件与 BASE 绑定同一 recipient PCM、同一轨道规则；官方 full-track Confidence 保留为次要 endpoint。主局部 SyncNet 指标沿用 A 的固定 lag rank，但 lag 只在 recipient BASE 的校准支持拟合，四条件共用，局部评价仅 core 去边界后完整窗口。若校准区不足则不可测，不用 full-clip 最佳 lag 顶替。

独立视觉指标：记录 COHERENT/SCRAMBLED/ERASE 相对 BASE 的 aperture 与事件变化，以及映射到 recipient 时钟的 donor aperture 轨迹距离变化（共同 valid core，使用原始 eye-normalized aperture，不单独对每条件重新 z-score）。donor 接近程度只能称 donor-like movement，不叫真实嘴型更准确；无事件时可以有像素响应，不能填补事件指标。

人工主对比只针对 N recipient：每组 COHERENT vs BASE、COHERENT vs SCRAMBLED，各 3 名合格评分者，12×2=24 对（至少72次主判断）。声音在对内逐 PCM 相同；配对身份完全隐藏；质量问题单列。复用 A 的盲评格式/QC。T recipient 的反向结果、ERASE 为诊断，不另行筛选有利 endpoint。

质量 gate：另设“明显脸部/嘴部破坏使同步无法公平判断”布尔项；任一处理在≥3/12科学组被至少2名合格评分者标记，则标 QUALITY_CONFOUNDED，不作机制正向结论。损坏组仍留在意向分析/缺失表，不能删除后重算为成功。没有标签时该 gate 为未审，不默认通过。

两个预定科学对比为 N←T 的：

1. `R_COHERENT−R_BASE`（能否恢复/增强 recipient 时间辨识）；
2. `R_COHERENT−R_SCRAMBLED`（是否超出同能量错时扰动）。

二者按源组 paired sign-flip exact test（12 组最多4096枚举，条件可交换为检验假设）并 Holm 校正；同时给组级 bootstrap 95% CI，不把帧当 n。少于 10 完整科学组则仅描述。人工两对比用同样两项 family 的组级检验/CI，分开列 family，不与 SyncNet 合并成一个显著性数值。四模型每种仅3组，不检验模型排名。

“时间组织参与本层生成响应”的有限支持要求：工程/范数/支持 gate 通过，COHERENT 相对 BASE 与 SCRAMBLED 的两个主对比均正向且校正 p<0.05，有独立视觉变化且无明显系统性质量损坏；若无人工正向结果，措辞限于固定音频的算法评价响应。升级到“感知同步改善”必须两个对应人工对比也正向并通过校正。T←N 对称损伤只加强证据，不设为必然条件；阴性不证明所有动态机制被排除。不得计算“解释了原生 TTS 增益的百分之几”：干预时钟/支持/音频与 native estimand 不同。

### 6.6 产物和可恢复性

新 run 目录固定为 `00_protocol/`, `01_existing_evidence/`, `02_visual/`, `03_blind/`, `04_patch_plan/`, `05_latents/`, `06_patch_video/`, `07_patch_eval/`, `08_analysis/`, `report.md`, `validation.json`。

每条 endpoint 行至少含 `protocol_id/revision`, `source_group`, `sample_id`, `tts_arm`, `recipient_arm`, `condition`, `metric_family`, `endpoint`, `value`, `support_hash`, `calibration_hash`, `video_sha256`, `audio_pcm_sha256`, `model_sha256`, `status/reason`。patch receipt 另含 donor/recipient latent hashes、映射 hash、逐帧 core/permutation、λ、module path、前后 norm、代码/environment、设备与 hook 已移除证明。

缓存 key = canonical hash(所有输入哈希、参数、代码/环境、checkpoint、支持域和层名)；必须校验产物 hash 后复用。写 temporary→原子 rename；失败不留 success receipt；重跑只补未完成 cell。report 强制列已复用/新增数量、每一级分母、历史/当前 endpoint 区别、人工缺口与可支持的结论层级。

## 7. Expected Change Surface

### Must change

- 新增 `scripts/experiments/tts_evidence_temporal_patch/{__init__,protocol,evaluation,intervention,worker,analysis,run,check}.py`。
- 新增 `scripts/configs/tts_evidence_temporal_patch_v1.yaml`。
- 新增 `tests/experiments/tts_evidence_temporal_patch/{test_protocol,test_evaluation,test_intervention,test_worker,test_analysis,test_check}.py`。
- 本 change 的 `spec.md`, `input-bindings.json` 为交接源；实施结果通过 BM 实验笔记跟进。

### May change

- 新协议目录内小型本地盲评 HTML 模板；仅离线导出，不建服务、不上传媒体。
- 新 run 下报告/验证/特征/评分/媒体。
- 如果运行时证明旧 helper 不支持当前输入，在新 adapter 实现最小兼容；必须改旧 helper 才能正确时先报告扩面及所需回归测试。

### Should not change

- 所有旧 run、历史结论、配置与科学门槛；`third_party/Wav2Lip`、`third_party/syncnet_python`、TTS provider、00–05 主流水线、模型权重。
- 不整顿脏 worktree，不处理无关 deletion/untracked，不升级依赖或替换评分模型。

## 8. Validation Plan

1. 静态：`python -m compileall -q scripts/experiments/tts_evidence_temporal_patch`；CLI `--help` 在无 torch/CUDA 的 CPU 环境可运行。
2. `pytest -q tests/experiments/tts_evidence_temporal_patch/test_protocol.py`：108→36组/72对唯一映射；输入 hash/重复 cell/多 track 拒绝；选择不读结果字段；16组不重复且配额满足；资源查询失败、外来 PID、锁冲突均不启动 worker；resume 参数变化拒绝缓存。
3. `…/test_intervention.py`：英语重复 phone/缺 word tier/间隙/短音素/变长 tail；实际 mel center 映射与解析线性信号吻合；不能跨 gap；core 外 bitwise 不变；λ=0 恒等；COHERENT/SCRAMBLED delta 均值与范数相同（float32容差1e−5）；零 norm 显式不可干预；shift 符号用解析序列验证。
4. `…/test_worker.py`：fake encoder 确认 batch 与绝对帧索引、尾 batch、hook 异常清理、donor 不触发递归 hook；真实模型 opt-in smoke 恒等误差≤1e−6，移位 ROI 与预期源帧一致。默认测试不能分配 GPU。
5. `…/test_evaluation.py`：A 不调用 TFG/官方重跑；原有 C 不改写；自定义指标命名不同；lag 校准与评价窗口及最大shift支持不重叠；固定 recipient PCM 未变化；landmark 原 invalid/LOW_MOTION 传播；缺人工标签返回 PENDING_HUMAN；盲评 key 不出现在受试者包。
6. `…/test_analysis.py`：手工 fixture 验证重复 natural 不重复加权，聚类 bootstrap 确定性，12组4096枚举正确，Holm正确；缺失/低组数不宣布阴性或完成；C=B−D 与 ΔC=ΔB−ΔD 数值恒等只针对同一 scorer/support。
7. `…/test_check.py`：篡改 recipient PCM/hash/support/层名/λ使验证失败；无盲评仍可工程部分完成，但最终证据不得写感知确认；预算不会因 resume 重置累计计数。
8. 回归：`pytest -q tests/experiments/phoneme_tfg_association tests/experiments/tts_time_instance/test_protocol.py tests/experiments/test_tts_visual_timing.py tests/experiments/test_check_tts_visual_timing.py`。随后 `pytest -q tests/experiments/tts_evidence_temporal_patch`。更广测试仅记录与改动有关失败，不修复无关环境问题。
9. 数据集验收：prepare 实际核验108视频/36组；A 逐 cell 导入官方值逐项一致；新增 TFG=0；B 明确技术/科学组、所有patch共同音频与支持、≤128新渲染cell；独立 CPU 从冻结逐组表复算主统计，与报告差≤1e−8。
10. Human-in-loop：导出盲评包后可继续做 CPU/GPU技术阶段；若评分未返回，analyze 输出 interim 且 check 报 PENDING_HUMAN，不能把任务标为科学结论完结。

## 9. Risks and Edge Cases

- 官方 full-track C 与 fixed-ROI C 曾有显著差异；旧 JSON 的 endpoint 名称也可能不准确。以 scorer命令/receipt/实际支持为准，新协议不修写历史数字。
- 当前生成器训练与 SyncNet 存在方法依赖风险；换一条 SyncNet 曲线不算独立评价。VSR 历史校准不足，本版不把未经校准 VSR/AV-HuBERT 加入主终点。
- 全瓶颈移植可能只是 donor 重播；残差也含音素、音色和上下文，不能叫单一纯时间因子。尺度匹配降低部分范数混淆，不排除 off-manifold 损伤。
- 时间打乱可能比 coherent 更不自然；必须同时优于 BASE 才能支持增益，单独赢过损坏控制不够。
- landmark 无法测量不等于视频没有响应；原生时长不同和自然换音频未对齐也不能作为自然替换失败证据。
- 高频残差/接缝的定义只涵盖本协议一个频段和层；不支持全模型或所有时间因素结论。4组校准/12组科学仍是小样本。
- 人工原生比较可能混入音质偏好；固定 recipient 的 B 比较更有识别力，但仍需盲法与质量审计。
- 显存空闲不代表无人使用，旧 helper 查询失败会误判空闲；新 guard 必须保守，不能只看 `torch.cuda.is_available()`。
- 现有仓库存在大量用户未提交/新增文件；本次不得覆盖或把缺失文件自动“恢复”。

## 10. Assumptions / Unknowns

- VERIFIED: 父官方 summary 为 complete，108/108、0失败；36组不完全平衡设计；输入绑定 hash 已记录。
- VERIFIED: 外部冻结单帧生成路径、瓶颈 `[B,512,1,1]`、当前 forward 不跨输出帧传递隐藏状态；可用 hook 不改权重。
- VERIFIED: 有 landmarks、严格 phone映射、局部rank与官方评分的可复用实现；有缓存目录不等于每个缓存都符合本协议。
- LIKELY: 108视频足以完成A主要证据导出；A新增成本以landmarks/盲评为主。
- UNKNOWN: 16个独立源组能否满足模型配额和双向连续支持；prepare 必须验证，未通过禁止降低阈值凑数。
- UNKNOWN: 当前 FaceLandmarker asset/runtime 与实际低运动可测率；按协议输出依赖缺失/不可测，不能自动换模型。
- UNKNOWN: 合格人工评分者及标签何时可获得；这是最终感知结论的外部依赖，不影响先交付导出工具和自动评价。
- UNKNOWN: CPU/GPU基线漂移、GPU空闲状态和预算内总耗时；运行前实测。本文未启动 GPU，也未声称现在空闲。
- UNKNOWN: donor短时动态是否为有益信息；科学失败/无显著变化是合格结果，不触发自动层扫描或放大样本。

## 11. Handoff Contract

实施 agent 必须沿 Code Anchors 建立单一新协议模块，遵守所有 Invariants，仿照既有外部单帧/receipt/纯metrics模式，把补丁限制在 Expected Change Surface；不做无关重构、不修改历史结果。先完成 CPU 输入审计/单测，再经独占 GPU guard 做4组技术校准，最后最多12组双向科学干预。A复用108视频；B先固定recipient音频与时钟再干预表示；人工缺失明确保留。若仓库证据与本计划矛盾（尤其层输出、映射支持、缓存身份或GPU安全），停止受影响阶段并报告，不能静默换实验设计。
