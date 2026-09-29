# 音素可分度与 TTS→TFG 增益：低成本配对关联实验 Implementation Spec

状态：✅ IMPLEMENTED / concluded；2026-09-22。协议：`phoneme_tfg_association_v1`。
本 spec 已实现并完成 corrected 主实验。首个试跑因误把 LRS3 源视频帧送入渲染器而作废；正式 run 使用外部非 LRS3 视频首帧，完成生成、评分、静态控制与关联分析。正式结果为 `INCONCLUSIVE`，不与作废 run 合并。
输入快照见同目录 `input-bindings.json`。科学范围：LRS3 English，四种现有 TTS，指定 TFG 与 SyncNet V2；不继承中文效应量或因果结论。

## 1. Objective

用有限的视频推理回答两个不同问题：

1. 在预先抽取的英文样本上，各 TTS 相对 natural 的原生 Sync-C 增益有多大？
2. 在同一句话内，扣除 TTS 组合的平均差异后，可分度更高的 TTS 是否有更大的 Sync-C 增益？这是唯一主关联假设。

推荐主方案：36 个不同 `source_group`，每组一条 utterance，每条生成 natural 与随机分配的两种 TTS，共 **108 条 Wav2Lip 视频**。四种 TTS 的六个无序组合各分配 6 条，每种 TTS 出现 18 次，得到 72 个 TTS−natural 配对、36 个独立来源块。相对同一 TFG 跑 100×5=500 条，生成条数减少 78.4%；这不是实测墙钟加速比。

可在运行前选择扩展档：预先固定其中 12 个来源、每个 TTS 组合 2 条，用 Ditto 再生成相同三臂，共 **36 条附加视频**。主方案加扩展为 144 条，跨生成器结果单独报告。12 组只做方向/效应区间核验，不足以单独证明跨 TFG 泛化。默认 `enable_ditto_replication: false`，不根据 Wav2Lip p 值决定是否扩展。

音频、MFA、HuBERT/XLS-R embedding 全部优先复用；按条重建可分度只需 CPU。主方案使用完整 utterance，保留原生语速、音色和时长。短片段、MFA 时间拉伸、四格换音轨不属于 v1，避免改变问题或把错时钟评分解释成视觉改善。

这是一项有预注册分析的关联实验。随机分配的是要渲染的 TTS 组合，可分度本身没有被随机干预；即使主关联成立，也不能写成“提高音素可分度导致 TFG 改善”。

## 2. Repository Model

数据流：`cohort_n100.json` + transfer manifest → 缓存和输入哈希核验 → 逐条音素指标 → 来源组抽样/组合分配/协议冻结 → 变长 Wav2Lip 生成 → SyncNet embeddings 与距离矩阵 → 来源块差分关联 → 独立复算与报告。

已核实的输入事实：

| 资产 | 当前情况 | 本实验用途 |
|---|---|---|
| `runs/lrs3_local_tts_n100_20260921/00_inputs/cohort_n100.json` | 100 条，42 个 source groups，speaker_id 缺失 | 官方文本、原视频、来源组 |
| `runs/lrs3_english_phoneme_transfer_n100_20260921/00_inputs/manifest.json` | natural、qwen_cloud、qwen_local、index_tts2、cosyvoice2 | canonical PCM、音频 SHA、各自 TextGrid |
| 同 run `01_mfa/alignment_manifest.json` | 497/500；CosyVoice2 缺 3 条 | 失败分母、对齐支持 |
| 同 run `02_embeddings/{hubert,xlsr}/` | pooled_vectors、token_records、sample_metrics、feature_meta | 不重新抽取 SSL、不训练新 probe |
| 同 run `03_metrics/metrics.json` | pooled 与配对汇总；没有最终逐条全部可分度表 | 描述性复核；不能把总体 Fisher 填到每条记录 |
| 中文 `results/aishell100_phoneme/tfg_link/` | 历史报告 15：104 个去重关联均未过 FDR | 设计背景，单独语种证据 |
| `openspec/changes/explain-lrs3-tts-syncnet-gain/input-bindings.json` | 历史 LRS3 50 条，与当前 n100 **同条 ID 交集为 0**、来源组交集 42 | 不能 join 视频分数；也不是来源组独立的外部验证集 |

缓存快速体检：97 条有五臂 HuBERT-L6 样本记录；五臂均 duration≥3s、valid_token_count≥20 的记录有 91 条，覆盖 41 个来源组。这只是预审计，不是最终资格：phone 标签支持、PCM/MFA 边界与脸部检查仍待实现。

当前调用边界中有重要特例：`wav2lip_probe_gpu.main` 把参考帧数量写死为 93；`wav2lip_probe_runtime.syncnet_matrix/score_metrics` 使用 88 行及固定 U_ROWS；`wav2lip_probe_score.main` 把 audio_arm 标为 N。不能直接用这些顶层封装处理本次变长五臂语音。

## 3. Code Anchors

以下新文件全部位于 `scripts/experiments/phoneme_tfg_association/`。表中“复用”是调用/参考，不表示修改旧实验。

| path / symbol | 当前职责 | 必须实现或复用的变化 |
|---|---|---|
| 新 `protocol.py::audit_inputs, select_blocks, freeze_protocol, validate_receipt` | 无 | 哈希审计、资格、确定性 36 来源/六组合分配、预算、resume 契约 |
| 新 `features.py::load_cached_tokens, compute_sample_features` | 无 | 读取 transfer 缓存；输出按 sample×arm 的指标和支持数；保留 layer 内 vector_index 语义 |
| `scripts/experiments/lrs3_english_phoneme_transfer.py::_load_store, _sample_metric_rows, _tokens_for_level` | 绑定 token 向量、逐条聚合、过滤静音 | 复用文件 schema/算法；可直接调用 `_sample_metric_rows`，不调用 `metrics()` 重做全部 KLD/FDR |
| `scripts/16_feature_separability.py::_silhouette_cosine, fisher_ratio, inter_class_separation` | 已有核心指标 | 复用原数值定义；冻结 sklearn 版本/实现路径，不能悄悄更换 singleton 处理 |
| 新 `generation.py::prepare_reference, render_wav2lip, render_ditto` | 无 | 变长、分 cell receipt、同句同脸/seed；Ditto 是可选适配器 |
| `scripts/38_prepare_frames.py::first_acceptable_frame, _sample_candidates, _make_detector` | 参考帧挑选与检测 | 复用 first-frame 策略；新适配器保留原 sample_id，不能调用整段 prepare_frames 后接受重新编号/补样 |
| `scripts/experiments/wav2lip_face_roi_replacement/generation_worker.py::load_model, mel_chunks, render_arm, encode_video` | 单模型加载、mel、渲染、FFV1 | 复用底层；参考静态帧/box 的数量按 mel_chunks 实际长度创建 |
| `scripts/03_ditto.py::run_ditto` | seeded subprocess adapter | 可选复用函数；冻结实际离线 cfg/环境/权重，不从函数参数名推断 TRT 模式 |
| 新 `scoring.py::score_native, score_static_control, summarize_curves` | 无 | 动态 T，缓存 A/V embeddings，完整/内部/等窗口评分，静态视觉对照 |
| `scripts/experiments/wav2lip_roi_peak_recheck/worker.py::SyncNetScorer.score, _forward, pairwise_distance` | 可变长 SyncNet、PCM 核验、[T,31] 矩阵 | 复用底层，确认冻结的 SyncNet checkpoint hash；标准 224×224 crop 后评分 |
| `scripts/experiments/lrs3_tts_gain_mechanism/analysis.py::_curve_metrics, equal_count_rows` | 明确支持行的 C/D/offset/背景计算 | 复用纯函数，传本实验支持行；不调用硬编码 LeapTalk/整数 sample_id 的 pair_curve_metrics |
| 新 `analysis.py::build_block_differences, fit_primary, summarize_gains, sensitivity_analysis` | 无 | 组合固定效应、HC3、wild bootstrap、group-aware 配对汇总 |
| `scripts/experiments/phone_separability_mechanism/metrics.py::paired_group_bootstrap, benjamini_hochberg` | 来源组 bootstrap、BH | 用于均值和次要检验族；不可拿均值 sign-flip 代替回归斜率检验 |
| 新 `check.py::validate_run` | 无 | 从原缓存/PCM/embeddings 独立复算；不调用 fit_primary 或 summarize_curves 复核它们自身 |
| 新 `run.py::main, run_stage` | 无 | 阶段 CLI、资源锁、失败状态、报告和 resume；父实验只读 |

## 4. Reference Pattern

参考 `lrs3_english_phoneme_transfer.py` 的分阶段、hash-verified reuse 和 explicit missingness；参考 `wav2lip_probe_runtime.run_gpu_plan/_validate_generation_result` 的 plan→worker→receipt 核验；参考 `lrs3_tts_gain_mechanism.analysis` 的 FULL/INTERIOR/EQUAL_COUNT 与 C 分解。

必须不同之处：本实验有四个独立 TTS 名称、变长视频、source_group 级设计、单次主检验。不能复制旧脚本的 93/88 帧、natural 固定音轨、整数 ID、硬编码语种/TFG、旧支持门槛或全 token 独立检验。

未找到足够相似的“平衡不完全区组＋同句 TTS 差分＋组合固定效应”现成统计实现；在新 `analysis.py` 中做小型纯 NumPy/SciPy 实现，并用合成反例验证。

## 5. Invariants

1. sample_id、source_group、transcript_sha256、audio container/PCM SHA、TextGrid SHA、SSL revision、layer、缓存文件 SHA 全链路可追踪；不通过路径相似或相同 source_group 冒充同条音频。
2. 所有 TTS 使用自己的英文 MFA；沿用当前 exact English label/NFC 规则，无中文 tone stripping，无均匀对齐 fallback。不同层的 vector_index 只索引该层矩阵。
3. 同一句三个视频用相同的外部非 LRS3 冻结参考帧/脸框、seed、TFG checkpoint 和渲染配置；声音只来自冻结的 canonical WAV。cohort LRS3 视频仅作音频/文本 provenance，永远不进入 renderer。禁止重合成 TTS、loudnorm、WSOLA、按评分选时段或挑 seed。
4. natural 视频每个 source×TFG×seed 只生成一次，两个 TTS 配对共用；统计保留这种依赖。source_group 不是已核验 speaker_id。
5. 组合分配、主指标、支持策略、样本资格、运行档、预算和分析方法在首次新 SyncNet 结果前冻结。已有音频指标被看过，因此这是新视频关联的前瞻性注册，不是完全未见数据的发现。
6. “主指标不可用”时不能把 Fisher、其他层或其他 TFG 升为主指标；只报告不可判定。按 arm 保存所有失败，包括三条 CosyVoice2 缺失；不生成伪边界/分数/0 值。
7. 完整 PCM 必须保留在科学评分 media 中，不用 AAC 重编码音频，不用 `-shortest` 截音频。评分只使用真实音频/视频共同支持；尾部不足显式记账，不合成末尾重复帧来补高分。
8. 旧 run、旧 provider、旧 checkpoint、旧测试语义不变；新配置经 `utils.load_config` 深合并，但本实验参数只在独立 YAML 段中生效。
9. 研究报告保留 full precision JSON；Sync-C 展示 3 位小数。p 值用 `(exceedances+1)/(B+1)`，不输出 p=0。
10. 科学状态与工程状态独立：checker PASS 不等于支持假设；INCONCLUSIVE 不等于没有关联。

## 6. Implementation Plan

### 6.1 固定配置、审计和逐条指标：不生成视频

新增 `scripts/configs/phoneme_tfg_association_v1.yaml`，至少包含：

```yaml
phoneme_tfg_association:
  protocol_id: phoneme_tfg_association_v1
  seed: 20260921
  source_groups: 36
  utterances_per_group: 1
  tts_per_utterance: 2
  pair_replicates: 6
  tts_order: [qwen_cloud, qwen_local, index_tts2, cosyvoice2]
  primary_tfg: wav2lip
  primary_feature: hubert_layer6_phoneme_silhouette
  primary_endpoint: native_interior_equal_count_sync_c
  enable_ditto_replication: false
  max_unique_wav2lip_videos: 108
  max_unique_ditto_videos: 36
  max_render_attempts_per_cell: 2
  fps: 25
  sample_rate: 16000
  visual_manifest: data/dataset_samples/video_manifest_250.json
  visual_dataset_allowlist: [mead, vfhq, talkvid, grid]
  detector: mediapipe
  face_detector_model: <hashed local MediaPipe face detector asset>
  vshift: 15
  minimum_interior_windows: 25
  bootstrap_draws: 10000
  wild_bootstrap_draws: 9999
  sensitivity_family: secondary_associations_v1
```

`audit_inputs` 核验 input-bindings 全部文件，然后按父 manifest 验证实际 WAV/PCM 和已存在 TextGrid。视觉输入必须来自显式哈希的外部多数据集 manifest（当前 allowlist 为 MEAD/VFHQ/TalkVid/GRID）；LRS3 记录、路径或与 cohort source video 相同的文件一律拒绝。记录每个样本、每臂：duration、speech/valid token 数、token frame support、label histogram、MFA 缺失、cohort source 视频 hash（仅音频样本 provenance）以及独立 visual source hash。没有声明 hash 的 TextGrid，在本协议第一次锁定时计算保存，不能声称旧 run 已核验过该 hash。

同阶段输出 `reuse_inventory.json`：先检查显式配置的历史视频/评分 manifests，按 exact sample_id+audio PCM SHA 查候选，再核验 reference、TFG checkpoint、seed、预处理、完整音频长度与代码合同。全部兼容才复用；视频兼容但评分不兼容时只重评分。历史50条的零交集不能外推为全仓库零可复用视频。部分旧133条replacement路径当前本地不存在，记 `SOURCE_NOT_LOCAL`，不要自动下载或退回错ID资产。108是主方案总cell上限，新增cell数等于该计划减去严格匹配的已有cell；抽样不得偏向缓存命中的样本。

`compute_sample_features` 主指标是 **每个 utterance×condition 的 HuBERT L6 cosine silhouette**。用 speech=true、valid=true、有限且非零的 pooled vectors，保留当前 phone labels。不是 corpus-level silhouette/Fisher；输出原始 S 与 `ΔS=S_T−S_N`。缓存全五臂逐条结果便于审计，但不会渲染所有五臂。

音频侧预资格：五臂都拥有可核验独立 MFA；各臂 ≥3 秒、≥20 有效 speech tokens、≥5 个有效标签、≥2 个出现至少两次的标签，silhouette 有限；phone span 在自身音频时长内，允许既有前端最多 20ms 数值尾差并显式记录。音素支持变化本身可能影响 S，保留重复标签数/valid-token 比例供诊断。不要按 S 大小、ΔS 方向或预想 TTS 排名筛样本。

低成本内容检查：文本一致、duration ratio、MFA speech coverage、词数/秒、波形峰值/RMS/静音占比。它们不证明 TTS 忠实读出了文本。预注册敏感性子集限制所有被选 TTS 的 duration ratio∈[0.5,1.5]；不据此删除主分析中的长短差异。ASR/人工转写校验若有 hash-matched 历史资产可作旁证；缺失则 `content_fidelity=unverified`，不写“纯音素效应”。v1 不强制启动新云 ASR。

次指标预先限定为 XLSR L10 phoneme silhouette、HuBERT L6 viseme silhouette、HuBERT L6 log(Fisher)，后者仅在两臂 Fisher>0 时使用；任何其他层曲线只是描述。以上可从现有 pooled vectors 在 CPU 重算，不重跑 MFA/SSL/logistic probe。

### 6.2 来源抽样、组合分配、预算冻结

1. 为每个音频合格记录从外部非 LRS3 visual manifest 准备一张冻结参考帧；cohort 的 LRS3 源视频只能作为音频/文本 provenance，严禁作为 renderer 输入，也不能加载其帧序列。视频 visual source 只读取第 0 帧，图像 source 直接读取单帧；同句三臂共用该帧和 box。对已是224×224的外部图保留全图作为固定 crop 与 box `[0,224,0,224]`；若不是该形状，使用冻结检测器的脸框裁切再 resize 至224×224，保存 visual source/首帧 hash、frame policy、检测器/模型 hash、坐标和插值规则。失败不重编号，保留原始 sample_id。
2. 从有候选的每个来源组选择一条：对候选按 `sha256(seed|utterance|source_group|sample_id)` 排序，取首个。随后来源按 `sha256(seed|group|source_group)` 排序，取前 36；所有 tie 以原字符串打破。至少 36 个合格来源才能锁正式协议，否则 `INSUFFICIENT_ELIGIBLE_GROUPS`，不降低门槛。
3. 用显式 PCG64(seed+1) 对已选 36 来源做一次 permutation；按固定 TTS 顺序构造六种 pair，每种连续分配 6 个来源。组合内 a/b 按 tts_order 固定，不能按 S 排序。此时任一 TTS 有 18 来源支持。
4. 每种 pair 按独立 hash 顺序选前 2 个来源，合成 12 个 diagnostics/optional-Ditto 子集；在任何新评分前固定。
5. 输出 `eligibility.jsonl`、`features.jsonl`、`blocks.json`、`generation_plan.json`、`protocol.json`。protocol 保存来源宇宙、排除/选择理由、所有 hashes、依赖版本、dirty-tree code hashes、主分析与精确计数。
6. 预算为不同视频 cell 数：主实验 108，附加档 +36，静态对照 0 个 TFG 视频。禁止每次重试重新计一个“可用样本”。每 cell 最多两次尝试，attempt log 计实际 GPU 秒与失败，第二次仍失败则保留缺失，不替换样本/组合。
7. 先运行冻结主 cohort 的前 2 个来源共 6 个主 cell，作为技术 smoke；成功 receipt 原样计入 108，不额外生成。只验 shape/PCM/timing/可复算与运行时，不能看 β/显著性后改抽样。

模型只加载一次，按 cell 顺序处理。从 smoke 的音频秒数与渲染秒数分别估计 median/max RTF，报告预计 GPU 秒、CPU 评分秒、磁盘峰值和不可预测范围；108/500 只是条数预算，不能承诺固定几分钟或固定倍数加速。

### 6.3 完整音频的变长生成与缓存

`render_wav2lip` 在 Wav2Lip 环境调用已核验底层函数：每臂计算自己的 mel_chunks，构造等长的**同一外部冻结参考帧/box**列表，再调用 render_arm。实现不得调用 `worker.load_frames(cohort_source_video)`；receipt 必须证明 `visual_source` 非 LRS3、`first_frame_only`，且 `fixed_reference_frame_repeated=true`。Wav2Lip 的 source-face 缓存可复用，但不得缓存或复用 cohort 源视频运动。

按冻结 seed 设局部 worker 的 torch/NumPy 状态，使用现有 batch=4、float32、固定 TF32/deterministic 配置，记录依赖和权重 SHA。内部科学产物沿用 FFV1+PCM MKV；MP4 只作为可选浏览副本，不进入科学评分。输出 `03_video/{tfg}/{sample_id}/{arm}.mkv`。

每个成功 cell 都写原子 receipt，绑定 `protocol_hash, sample_id, source_group, arm, tfg, seed, source_audio/pcm_sha, reference/crop_sha, checkpoint_sha, code/config_sha, mel_chunk_count, generated_frames, scoreable_frames, tail_excluded_ms, output_sha, elapsed_seconds`。resume 只复用完整匹配 receipt 的 cell；不复用仅仅同名的 MP4。主 cohort 各臂长度可以不同，不能统一填充到最长。

`render_ditto` 仅在已冻结扩展档时运行同一 12 来源三臂，使用 `run_ditto`/seeded adapter 与实际离线 cfg。独立 preflight 检查权重/环境/参考图规则。缺少部署条件就写 `DITTO_UNAVAILABLE` 并交付完整主实验，不改成另一个 TFG，也不以此阻断主方案。任何额外部署/扩大预算另立后续工作。

### 6.4 主评分、距离分解与静态视觉负对照

所有模型统一使用固定 SyncNet V2 权重和同一 crop 规则。Wav2Lip 为 224 源脸坐标下的固定 ROI；Ditto 对同一来源三臂使用共同规则的固定 crop，记录 box，不能逐帧按评分优化 crop。预处理不通过修改嘴型或时间轴增加支持。

`score_native` 使用各视频自身的原始驱动音频，只调用一次 audio/visual forward，保存 `[T,1024]` embeddings 与 `[T,31]` 距离矩阵。明确 T 来自真实媒体共同支持和现有 5-frame/MFCC 窗口契约，不取常数 88。FFmpeg 解码 PCM 必须与冻结 canonical WAV 完全一致；前后端 hash/浮点策略相同才能复用 audio embedding。

主支持 `INTERIOR_EQUAL_COUNT`：每臂内部行是 `15..T-16`（零基 inclusive）；三臂共同 `K=min(T_N−30,T_a−30,T_b−30)`，K≥25。各臂从内部行中按 `equal_count_rows` 均匀取 K 行，固定索引保存在 receipt。这样在同一句三个条件中去掉零填充并使用等量时间窗口；只是等数目，不声称相同 phone/时间对齐。

先对选定行求 31-lag 平均曲线，再 `D=min(curve)`、`B=median(curve)`、`C=B−D`、`offset=15−argmin(curve)`。保留 FULL 与全部 INTERIOR 两个敏感性支持；不混用三种结果。输出每个 natural→TTS 的 `ΔC=C_T−C_N`、`gain_D=D_N−D_T`、`ΔB=B_T−B_N`，逐条核验 `ΔC=gain_D+ΔB`。原始全支持分数可用于历史定义参照，主结果必须带支持名称。

静态对照无需新 TFG 视频：每来源用同一冻结 reference crop 的五帧静态输入取得一个 visual embedding，重复到各 arm 的 T，搭配该 arm 已缓存的真实 audio embeddings。用与主评分完全相同的 K 行和 lag 规则得到 `C_static`。缓存键包含 reference crop、色彩/resize、SyncNet 权重和代码；禁止用生成视频平均 embedding 冒充真实静态视觉输入。

静态对照回答“在没有嘴部运动的输入上，是否也会观察到类似的音频相关评分差异”。对 `dC_generated−dC_static` 拟合同样的 x/pair 回归，直接给斜率差与 CI。不能通过“generated 显著、static 不显著”断言二者不同；也不能把差值称作已识别的纯视觉因果收益。C 分解或静态对照都不能单独证明评价器偏好解释了全部效果。

独立视觉证据：自动产出 12 来源的盲评包（同句三臂随机匿名、同一浏览规格）和空 rating schema，供至少 3 名评者评价音画同步/口型可读性。无人工数据时照常交付，`perceptual_status=NOT_MEASURED`；不把未经英语校准的中文 VSR 当作独立真值。

### 6.5 主分析、统计功效与判定

令 `S_ij` 为句 i / TTS j 的逐句 silhouette，`Y_ij=C_ij−C_iN`，`X_ij=S_ij−S_iN`。每句分配两种 TTS a、b。唯一主回归用 36 行来源差分：

```text
x_i = (X_ib − X_ia) / 0.1 = (S_ib − S_ia) / 0.1
y_i =  Y_ib − Y_ia        =  C_ib − C_ia
y_i = pair_intercept[pair_i] + β*x_i + error_i
```

用六个 pair one-hot、无额外截距；OLS 最多 7 列，完整设计残差自由度 29。β 单位为“同句 TTS silhouette 相差 0.1 时的 Sync-C 增益差”。pair 固定效应消除各 TTS 组合的总体均值差，避免仅靠 cloud 高、Cosy 低的模型排名产生相关；句差分消除共享 natural 与句难度。

主 estimand 是被选合格来源中的**同句、同组合内部关联**。它不检验 TTS 相对 natural 的共同整体位移；若所有 TTS 都有相似增益但彼此 S 无变化，β 可能不可识别。主结果不估计未筛选全部音频或全体 LRS3 的 population effect。

推断合同：

- 完整块≥30，六种 pair 各≥4 块，矩阵满秩；对去掉 pair 均值后的 x，方差>1e−12、最大 leverage<0.5。否则主科学状态 `INSUFFICIENT_SUPPORT`，不删除高 leverage 点来让门槛通过。
- 输出 OLS β、HC3 SE 和 t(df=n−rank) 双侧 95% CI；数值诊断包含 condition number、leverage、残差与逐来源 leave-one-out β。
- 主 p 值用 null-imposed、studentized wild residual bootstrap：先拟合仅 pair 项的 H0 模型；残差按 sqrt(1−h0) 校正，每来源独立 Rademacher 权重；`y*=fitted0+w*adjusted_residual0`，重拟合完整模型并用 HC3 t 统计量，B=9999。这是依赖回归误差假设的近似推断，不是把 TTS label permutation 称作可分度的随机化检验。
- 单一主检验 α=0.05，不因看过其他结果换层/换指标。β>0 且 HC3 CI 下界>0 且 wild p<0.05 才写 `POSITIVE_WITHIN_PAIR_ASSOCIATION`；对应负方向为 `NEGATIVE_WITHIN_PAIR_ASSOCIATION`；其余 `INCONCLUSIVE`，不足为前述状态。
- 双侧检验和 CI 都保留；不能把 p≥0.05 写为“没有关系”。报告区间允许的正/负斜率及其实际量级，不将“未显著”当作等效性。

TTS−natural 增益表另算四个 TTS 的 group-equal mean/CI 和 positive fraction（每种预期 n=18），用来源 bootstrap。汇总“平均 TTS 增益”时每来源先平均其两个 ΔC，再对36来源平均，不把72配对当独立。四种模型的 mean(S) 对 mean(ΔC) 可以作描述图，不计算它们的 n=4 显著相关作为核心证据。

预先限定次要关联族 9 项，统一 BH-FDR，缺失保留为 null：① XLSR-L10 silhouette→主 C；② HuBERT-L6 viseme silhouette→主 C；③ log(Fisher)→主 C；④主 S→gain_D；⑤主 S→ΔB；⑥主 S→static C；⑦主 S→generated−static C；⑧主 S→FULL C；⑨主 S→INTERIOR C。仍使用同样的块差分和 pair 项；次要指标缺失不能改变主 cohort。报告每项支持分母。

解释性敏感性单列、不搜 p 值：在主回归加入 `log(duration_b/duration_a)` 和 `silence_fraction_b−silence_fraction_a` 两项；另在上文 duration-ratio 子集重算，以及逐 pair 留一组分析。时长/停顿可能是共同原因或中介，调整前后都呈现，不把调整模型叫因果直接效应。

CPU 功效规划在生成前完成：固定实际 x 与 pair 设计，令残差 σ∈{0.25,0.5,1.0} Sync-C，残差化 x 一个 SD 的效应 θ∈{0,0.1,0.3,0.5} Sync-C，正态与 t5 两类噪声，每格2000次模拟，以HC3双侧检验估计拒绝率及 Monte Carlo 误差；另抽固定100次做wild检验一致性检查。θ=0.3 是规划默认值，不是实测阈值/研究发现。必须注明真实 σ 未知，不能拿模拟功效承诺阳性。

直觉上，36 个独立样本的简单相关大约只能对 |r|≈0.45 的效应达到80%功效；这是 Fisher-z 正态近似，未计 pair 项/缺失/多重比较，不是本实验功效保证。弱关联需要更多独立来源，给同一42来源多生成几百视频不会自动获得100个独立样本。

不做“显著就停/不显著追加12条”。v1 固定预算完整收尾。可另外导出下一轮所需来源数建议，但不得自动扩到500条。

### 6.6 产物、接口与最终交接

CLI 为 `python -m scripts.experiments.phoneme_tfg_association.run --run-id <id> --config <yaml> --stage <stage> [--resume]`。
stage 枚举：`audit, features, plan, generate, score, analyze, validate, report, all`。`all` 遵循冻结预算执行主档；optional Ditto 由冻结配置决定。重复 `plan` 不能覆盖已有 protocol hash；改配置必须新 run-id。

```text
runs/phoneme_tfg_association_<id>/
  00_protocol/{input_audit,protocol,blocks,generation_plan,power_plan}.json
  00_protocol/{eligibility,failures}.jsonl
  01_features/{features.jsonl,feature_meta.json}
  02_reference/<sample_id>/{reference.png,crop.png,receipt.json}
  03_video/<tfg>/<sample_id>/<arm>.{mkv,receipt.json}
  04_syncnet/<tfg>/<sample_id>/<arm>/{visual,audio,distance}.npy
  04_syncnet/{scores,static_scores}.jsonl
  05_analysis/{pairs,block_differences}.csv
  05_analysis/{association,gains,sensitivity,missingness}.json
  06_review/{blind_manifest.json,ratings_template.csv}
  validation.json
  report.md
```

稳定 row key：`(protocol_id,sample_id,tfg,arm,seed,support)`；`source_group` 是不可空关联列。各 row 需要 `status`、`reason`、输入/receipt hash。缺失数值用 null，并记录 `expected/attempted/completed/eligible` 四种分母。

`association.json` 必含 `estimand,feature,endpoint,formula,n_blocks,n_source_groups,pair_counts,beta,beta_units,se_hc3,ci95_hc3,p_wild,draws,seed,design_rank,scientific_status,engineering_status,claim_boundary`。`report.md` 主图为 residualized x/y 散点（每点一来源），次图为各TTS ΔC/CI及C分解；模型排名图不得替代主图。

方法依据：本协议以来源块而非 token/视频条数推断；少量来源时普通独立误差会低估不确定性，因此使用来源差分、稳健SE及bootstrap诊断。参考 [Cameron & Miller, A Practitioner's Guide to Cluster-Robust Inference](https://cameron.econ.ucdavis.edu/research/Cameron_Miller_JHR_2015_February.pdf)。36来源、108视频和上述门槛是本项目的设计选择，不是该论文给出的样本量建议。

## 7. Expected Change Surface

### Must change

- 新增本 spec 与 `input-bindings.json`（本次交付）。
- 下游新增 `scripts/experiments/phoneme_tfg_association/{__init__,run,protocol,features,generation,scoring,analysis,check}.py`。
- 下游新增 `scripts/configs/phoneme_tfg_association_v1.yaml`。
- 下游新增 `tests/experiments/phoneme_tfg_association/` 中面向合同的测试。
- BM 新建本实验 planned 设计指针；执行完在同一实验实体记录结果/科学与工程状态。

### May change

- 只有实际底层接口无法表达变长输入时，才小幅扩展对应纯 helper 并加兼容测试；优先新适配器，不能改变旧实验默认行为。
- 运行完成后更新 BM 结果笔记和 HANDOFF；本次不改已有 HANDOFF/CONTEXT 的历史结论。

### Should not change

- 旧音频、MFA、embeddings、旧视频/评分、旧科学门槛；TTS provider 工厂；TFG/SyncNet 权重；核心00–05脚本默认配置；第三方模型网络实现。
- 不引入通用调度框架、自动大规模云资源部署、全库格式化、额外训练或无关重构。

## 8. Validation Plan

以下测试是下游验收要求；实现已完成，实验相关测试 13 passed，完整媒体/分数 checker 为 PASS/READY。

| test / command | 验证行为 | 预期 |
|---|---|---|
| `python -m compileall -q scripts/experiments/phoneme_tfg_association` | 模块可编译 | 无错误 |
| `pytest -q tests/experiments/phoneme_tfg_association/test_protocol.py` | hash绑定、36来源、各pair6、各TTS18、固定随机种子、无补样、同名不同内容缓存 | 错误资产拒绝；排序输入顺序改变不影响选择；108/144预算准确 |
| `.../test_features.py` | 层内索引、静音/无效过滤、英文送气标签、逐句而非pooled、资格边界 | 小型人工算例与既有silhouette一致；不存在伪0或复制均值 |
| `.../test_generation.py` | 不同音频长度/末尾、同句同脸、外部帧绑定、LRS3泄漏拒绝、PCM保真、失败重试/resume | 各臂帧数随mel变化；>93帧不会被截；renderer 不加载 cohort 源视频帧序列；失败不覆盖成功receipt |
| `.../test_scoring.py` | 动态T、排padding、三臂K、static对照、音轨绑定 | 不同T的小矩阵有人工可核验C/D；ΔC=gain_D+ΔB；换音频hash缓存失效 |
| `.../test_analysis.py` | 回归主estimand、provider/pair混杂、共享natural、已知斜率、null、缺失 | 仅pair均值造出的正相关被消除；natural扰动不改变x_i/y_i；斜率方向/单位正确 |
| `.../test_analysis.py` | HC3、studentized wild、df、奇异设计、p下界 | 对照手算/独立参考；弱支持返回INSUFFICIENT_SUPPORT；p不为0 |
| `.../test_check.py` | 独立复算、重复key、数组篡改、失败分母、静态embedding来源 | 故意损坏均被检测；checker不能只验证自身输出JSON |
| `pytest -q tests/test_lrs3_english_phoneme_transfer.py tests/experiments/phone_separability_mechanism/test_metrics.py` | 被复用指标的原合同 | 原测试继续通过 |
| 技术smoke：冻结计划中2来源×3臂 | 真模型、真PCM、缓存/资源、完整窗口 | 6个cell可复算；成功cell正式复用，不改分析/样本 |
| `...run --stage validate --run-id <id>` | 独立从文件与embedding重建 | 核对108 expected、实际失败/支持；float32曲线tol=1e−4，C/D tol=1e−4，代数分解tol=1e−10；统计复算tol=1e−8 |

独立 checker 使用保存的 bootstrap 随机权重/索引或其可复现seed；以直接矩阵公式重建OLS/HC3，并以独立循环抽查至少100个wild draw。抽检至少每种pair一块的token→S和embedding→curve全链路；所有cell做哈希、shape、有限值与PCM校验。测试必须包含“不同T的三臂”以及“同一natural被两个TTS共享”的反例。

最终科学支持只依据预注册主检验。bootstrap模拟校准、敏感性、静态对照不允许自动改结论门槛；若暴露严重模型失配，则明确标注推断不稳健，不能只报告最有利版本。

## 9. Risks and Edge Cases

- **模型均值/伪重复**：4个TTS均值或72配对都不是72独立来源；主分析只有36块，pair项必须保留。
- **局部口型≠音素标签**：phoneme区别未必可视觉辨别，viseme是预定次指标；同句phonetic content仍可能因漏读/错读变化。
- **MFA成功≠文本忠实**：CosyVoice2有已知疑似严重截断样本，97/100成功也不能证明其余全部读对；记录内容核验状态和duration敏感性。
- **silhouette的组成效应**：标签频次/单例/短phone比例变化可能改变指标；保留支持，次指标核验，不把它视为无噪声真值。
- **筛选后总体改变**：五臂可测、可见脸的合格来源是主目标；失败率单列，不能把该子集宣称为各模型完整质量排名。
- **生成后选择性缺失**：主回归是三臂均成功的complete-case关联；即使达到30块门槛，仍需按pair/arm报告缺失并比较预先可见的S、时长和质量。失败依赖潜在同步质量时，bootstrap不能消除选择偏差；不能补渲染别的样本后称随机设计完整。
- **source不是speaker**：同一说话人可能跨来源出现；不得声称speaker-controlled。若后续在冻结前找到可信speaker映射，需新版本定义更高层抽样单位，不能把42当已知42说话人。
- **TFG与评价器相关**：Wav2Lip采用同步相关训练，Wav2Lip+SyncNet正关联可能具有模型特异性；Ditto/人工可提供补充，但v1默认不保证跨模型或感知效度。
- **静态对照有限**：无运动的reference分布与生成视频不同；它是诊断，不是完整反事实。未见静态关联也不能排除评价器音频偏好。
- **视觉泄漏**：若把 LRS3 cohort 原视频或其动态帧送入 Wav2Lip，自然臂会携带真实口型而形成不对称基线；因此 v1 强制外部 allowlist、首帧-only、三臂同帧和 receipt/checker 双重拒绝。任何违反都使该 run 科学作废，不能仅按工程 PASS 使用。
- **短视频/长尾/全局offset**：保留支持行与offset边界比例；不能用无支持phone或padding制造可分度、用tail扩展制造同步增益。
- **速度与长度混杂**：节省的是要生成的cell；完整句可能很长，不能把同样108条的不同RTF当同成本。任何进一步裁剪另设协议并重算片段指标。
- **已看数据/少来源**：没有独立未见来源验证；历史50与当前来源重叠，不是外部确认。无论p值如何，都不能以本次单一关联证明机制必要性/充分性。

## 10. Assumptions / Unknowns

- VERIFIED：当前cohort有100条/42来源，五臂497/500对齐；粗预资格91条/41来源；父输入及缓存SHA已写入bindings。
- VERIFIED：历史Ditto/LeapTalk50条同条ID交集为0，来源交集42；本地Wav2Lip与SyncNet权重路径存在。
- VERIFIED：既有快速probe的93/88帧和N音轨硬编码不能直接复用；底层render_arm与SyncNet _forward/pairwise_distance可接受动态长度。
- VERIFIED：当前工作树含大量未提交改动；git HEAD不足以描述实验代码，必须附本次实际依赖代码hash。
- LIKELY：重用缓存与模型常驻能显著降低成本，具体GPU/CPU时间要由技术smoke量测；不能仅凭权重存在断言环境可用。
- LIKELY：至少36来源能通过最终资格；目前仅完成粗预审计，脸/标签/哈希核验可能进一步减少来源。
- UNKNOWN：实际 within-pair S方差、噪声σ、主斜率、统计功效和视觉/感知真实性；必须在规划/结果中显式呈现。
- UNKNOWN：Ditto离线环境是否现成可用，独立英语视觉模型是否校准通过，可信speaker映射是否存在；这些不阻断默认Wav2Lip关联范围。
- LIKELY（意图假设）：用户希望先得到可复算的英文多TTS关联证据，因此本设计选择36来源、两TTS不完全区组和完整utterance。更强因果问题需要后续自然时钟干预＋对照，不能借本spec静默扩大为训练实验。

## 11. Handoff Contract

下游按列出的 code anchors 实现新增封装，保持所有 invariants，仿照已命名的缓存/worker/评分模式，patch 限于 Expected Change Surface；无无关重构。

执行顺序固定：输入/资格审计 → 缓存逐条指标 → 预算/组合/主检验冻结 → 6-cell技术smoke → 剩余已计划生成 → native/static评分 → 主分析与敏感性 → 独立checker → 报告/BM。默认108条Wav2Lip，预先开启扩展档才加36条Ditto。复用成功cell，遇到资源忙用现有锁/调度，不抢占未知任务。

若仓库证据否定输入绑定、变长接口、至少36来源或实际检查点合同，停止依赖该假设的昂贵阶段并报告具体差异；仍交付已完成的审计与不可判定状态。资源/资格不足不能触发降低门槛、伪造配对、换主指标或扩到全量。

完成标准：来源与失败分母可追溯，唯一主斜率及CI/p可独立复算，计算预算与实际耗时明确，科学结论严格区分“音频侧差异”“原生SyncNet增益关联”“跨生成器支持”“感知/因果未验证”。工程通过但结果不显著，同样是完整交付。
