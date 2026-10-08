---
title: Ditto-50 VSR 与 Sync-C 增益关联验证 Spec
type: note
permalink: tts-exp/research/ditto-50-vsr-与-sync-c-增益关联验证-spec
status: implemented
question: 在同一批 Ditto-50 视频上，VSR 动态内容增益是否与重新评分的 Sync-C 增益相关？
tags:
- vsr
- ditto
- syncnet
- spec
- implementation-architect
---

## 1. Objective

在本地已有 Ditto-50 自然/TTS 配对视频上，复用冻结中文 CMLR visual-only VSR，重新计算同一批 MP4 的 SyncNet 分数，回答：

1. TTS 驱动视频是否给 VSR 提供更多可辨认的目标语句内容？
2. 在逐句层面，Sync-C 增益越大的样本，VSR 动态内容增益是否也越大？

这是已有 AVTR-1 13 对 pilot 的跨生成器扩展与预先规定的关联检验，不是 Ditto-100 的恢复，也不是独立因果试验。上轮 G 均值 +0.234、95% CI 跨 0；事后 Spearman(G,ΔC)=0.022，n=13。不能据此宣布无嘴型改善，也不能把此次 n=50 自动称为足够统计功效。

本轮只做两个原始条件 natural_raw / tts_raw；不纳入 enhanced_raw、不补跑 Wav2Lip、不生成新 TFG、不训练增强头。只新增一个 runner、一个纯统计模块和一个测试文件，复用现有 VSR 实现，不建设通用框架。本文已按下文完成实现与全量验证；结果记录见 [[Ditto-50 VSR 与 Sync-C 增益关联验证结果 2026-09-18]]。

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 初稿：Ditto-50 来源核验、VSR 复用、同视频 SyncNet 重评分与关联检验 | September 18, 2026 | user requested |
| 自审：固定同一分析集合、完整 QC、独立重建 G、缓存证据与相关退化情况 | September 18, 2026 | agent self-review |
| 二次自审：明确 numeric id 抽样顺序、复用 helper 的实际 QC 返回值与缓存恢复行为 | September 18, 2026 | agent self-review |
| 实现 Ditto-50 runner、smoke/全量运行、独立复算与结果记录；补充 sync_status | September 18, 2026 | agent |
| 自审补强缺失 cell 与 scientific status 的独立一致性检查，并记录 expected/completed/failed 数量 | September 18, 2026 | agent |

关联：
- extends [[VSR 视觉内容辨识与 TTS 配对验证实施部署 Spec]]
- relates_to [[VSR 视觉内容辨识与 TTS 配对验证结果 2026-09-18]]
- uses [[增强头下游评估 (Ditto + SyncNet)]]

## Implementation Result

- status: implemented；实现文件为 `scripts/experiments/vsr_ditto50_linkage.py`、`scripts/experiments/vsr_ditto50_metrics.py`，针对性测试为 `tests/experiments/test_vsr_ditto50_linkage.py`。
- smoke `runs/vsr_ditto50_linkage_smoke/` 已跑通且标记为 `SMOKE_TECHNICAL_ONLY`，仅作链路校验；全量 `runs/vsr_ditto50_linkage_v1/` 完成 50 个候选的来源审计，47 对 source-eligible，470 个 VSR 视图、94 个 SyncNet cell 完成，J=47。
- `validation.json` 为 `PASS`：2,820 个 CTC 候选独立复算，最大 CTC 误差 `8.88e-15`，统计重建最大误差 `6.22e-15`；JSON 完整性和针对性测试（17 passed）通过。
- 科学状态：`sync_status=EXPLORATORY_SYNC_GAIN`；VSR natural strict top1 为 25/47=`0.532`，未达到预注册 `0.60`，故 `calibration_status=INCONCLUSIVE_VSR_VALIDITY`、`visual_status=INCONCLUSIVE_VSR_VALIDITY`、`association_status=NOT_INTERPRETABLE`。不要把未经校准的正 G 均值写成嘴型准确性证据。
- 完整数值与限制见 [[Ditto-50 VSR 与 Sync-C 增益关联验证结果 2026-09-18]]；`HANDOFF.md` 已同步当前状态。

## 2. Repository Model

以下路径均相对 repo root，运行时冻结为绝对路径。简称：
- D = runs/two_stage_hubert_aishell1_20260810/ditto_videos
- A = results/rhythm_style_500/aishell1_test_400
- S = results/rhythm_style_500/source_manifests

已核实的数据流：

D/sample_mapping.json（JSON list，50 行，id/paired_key/speaker/natural/tts）
→ paired_key 的 WID 连接 S/aishell1_test_400_paired_audio.json.records
→ 同条记录 expansion_id 连接 A/audit/tts_meta.json.results.sample_id
→ 交叉核对 transcript、TTS text、A/transcripts/{expansion_id:04d}.txt
→ 当前自然/TTS WAV 的 SHA256 对照源清单
→ D/{natural_raw,tts_raw}/{id}.mp4
→ 分成两条评估链：

- 去音轨、帧/PTS 一致性检查 → 官方嘴部裁剪 → 五视图 → VSR encoder/CTC → 每句 G。
- 原始 MP4（保留原音轨）→ SyncNet 官方人脸跟踪/裁剪 → 对确定的 track 评分 → 每句 ΔC。

最后以 (cohort_id, paired_key) 一对一连接 G 与 ΔC；整数 id 仅是本 cohort 的文件定位键。

设计时核实：
- id=1..50，speaker 全部 S0765，自然/TTS WAV 各 50 个均在本地。
- 100 个 WAV 均与 paired_audio 清单中的对应 SHA256 一致；50 个 transcript 文件与该清单文本一致。
- 例：id=1 → aishell1_test_400__BAC009S0765W0122 → “楼市地市交相升温房价会不会再度暴涨”。绝不能套用 R2/AVTR-1 或 AISHELL100 的同名 1.txt。
- D/ditto_meta_natural_tts.json：mode_used=trt_online、seed=42、conditions 两臂，results 键如 natural_raw:1；部分为 ok-cached。没有逐视频生成时 SHA，当前 SHA 不能追溯证明原始生成身份。
- D/syncnet_eval/eval_meta_natural_tts.json：100/100 条成功，complete_case_ids=50；历史 ΔSync-C=+1.081，46/50 对为正。本文已看过历史均值，因此是预先固定新 VSR 分析，不是对未知 SyncNet 效应的盲确认。
- 历史实验笔记记录固定参考脸 data/data/image/1.png，该图当前存在。人像来源仅 historical_documented，不声称生成时 hash verified。
- 本地模型/环境沿用旧 VSR spec：[redacted-local-path] 用 [redacted-local-path]
- 当前根盘余量约 9.3 GB，旧 13 对五视图特征约 252 MB。先 smoke 实测再估算新产物；不自动清理用户数据。

## 3. Code Anchors

| path | symbol | current role | required change |
|---|---|---|---|
| 新 scripts/experiments/vsr_ditto50_linkage.py | audit_ditto50、verify_inputs、extract_ditto50、score_syncnet、analyze_ditto50、validate_ditto50、main | 尚不存在 | 薄 runner：来源适配、阶段编排、按句子连接、报告与独立复算 |
| 新 scripts/experiments/vsr_ditto50_metrics.py | freeze_decoys、build_pair_rows、summarize_linkage | 尚不存在 | 去重的冻结错误文本、严格完整对筛选、Spearman/置换/比例与 CI；不加载模型 |
| scripts/experiments/vsr_tts_pilot.py | normalize 以外的 sha256_file、write_json、_make_config、load_vsr、_set_determinism、extract_views、_extract_one_pair、_view_metrics、_independent_ctc_nll | 已跑通的 VSR I/O、前向、内容打分、独立 DP | import 复用；此文件默认不修改。下划线函数在本研究脚本内允许明确引用 |
| scripts/experiments/vsr_tts_pilot.py | audit_inputs、analyze_run、validate_run、_environment_payload、_verify_frozen_inputs、_write_report | 写死 AVTR-1/13、历史分数路径及部分自审逻辑 | 不直接调用为 Ditto 实验服务，不 monkeypatch SAMPLE_IDS/ROOT/PROTOCOL_ID；在新 runner 写小型对应逻辑 |
| scripts/experiments/vsr_tts_metrics.py | normalize_text、token_ids、make_views、length_normalized_nll、content_margin、paired_summary | 已测试的文本/CTC/变换/配对均值 | 复用数学定义；paired_summary 仅参考，新模块按 numeric id 顺序统一抽样并固定 min_pairs=40；不直接用写死 n≥10 的 decision_status |
| 同上 | build_decoys | 长度匹配，但不保证候选之间文本唯一 | 新 freeze_decoys 仿照排序规则并按规范化文本去重；不改旧 pilot 的候选 |
| scripts/04_eval.py | run_syncnet_pipeline、parse_syncnet_output | 两 subprocess 评分与首次正则匹配 | 只参考 subprocess 形态；不要直接 import（Py3.8 不兼容且项目级 utils）；旧 parser 会漏负 offset、多 track 歧义，不复制 |
| third_party/syncnet_python/run_pipeline.py | crop_video；CLI；tracks.pckl 写入段 | S3FD 跟踪、声画裁剪 | 原样 subprocess 使用，固定 min_track=50、frame_rate=25 等参数 |
| third_party/syncnet_python/demo_syncnet.py | CLI → SyncNetInstance.evaluate | 对一个明确视频/track 评分 | 对选定 crop 单独调用，唯一 C/D/offset 记录 |
| 新 tests/experiments/test_vsr_ditto50_linkage.py | 来源、QC、连接、相关、验证的行为测试 | 尚不存在 | 见第 8 节 |

## 4. Reference Pattern

最接近实现是 scripts/experiments/vsr_tts_pilot.py 的 extract_views → _extract_one_pair → _view_metrics。复用已经验证的去音轨、官方裁剪、五视图、native tensor 保存及 CTC 数值流程；复用 vsr_tts_metrics.py 的纯函数。

不能直接复制旧 analyze_run：它可能将 qc_eligible=False 但特征文件存在的记录纳入 pair_rows；校准 top1 与 Q 的分母也可能不同。新分析必须先建唯一 complete 集合，再在该集合内计算全部主指标。不能复制旧 validate_run 从 producer pair_rows 复算统计的做法；新 validator 必须从原始 logp/decoy loss 和新 SyncNet 日志独立重建 pair_rows。

SyncNet 参考 scripts/04_eval.py 的独立环境 subprocess、绝对路径、cwd 和日志处理；只保留必要薄封装，不接入整条 00→05 流水线。

## 5. Invariants

1. 候选固定为 mapping 的全部 50 对，不能按已有 Sync-C、VSR、CER 或“肉眼嘴型好”筛选；音频说话人和历史 valid 集身份如实报告，不冒充 heldout 新样本。
2. 当前视频只读；VSR 不读音频。新 SyncNet 只对同一 SHA256 原 MP4 的原音轨评分，不用其他 WAV 重新配音，不做交叉音频条件。
3. 来源连接必须同时核对 WID、expansion_id、文本与 WAV hash；同名整数 id 不足以证明身份。
4. natural、tts 两臂使用同一目标和五个不同的 decoy，固定模型、裁剪、参数，不因结果调整。
5. 不把 native 时长强制拉齐作为主结果；matched 只是敏感性分析。
6. 统计单位是一句配对，不是帧、视图或音素。50 对不写成 n=100。不能与 AVTR-1 13 对或 Ditto-100 账本拼接。
7. 无效/缺失单元显式记录，不能填 0、删除失败记录、无声换样本或拿历史评分补缺。
8. 缓存必须绑定 protocol、source hashes、代码/模型/配置和输出哈希；不能仅因 npz 存在就认为可续跑。
9. 保存原始精度；表格 Sync-C 三位小数，CTC/G/ρ 不先四舍五入再分析。
10. 代码错误可以修复，科学规则不能看完结果后调优。修复改变产物时新建 run，保留旧 run 与无效原因。

## 6. Implementation Plan

### 6.1 CLI 与阶段

新增 runner 支持：
--run-dir（默认 runs/vsr_ditto50_linkage_v1）
--stage audit|extract|syncnet|analyze|validate|all
--smoke、--resume、--device（默认 cuda:0）
--syncnet-python（默认 [redacted-local-path]

Python 3.8 兼容；all 顺序执行 audit→extract→syncnet→analyze→validate，VSR 用完释放 GPU 再运行 SyncNet，避免两模型争用显存。不要新增 YAML 配置层；常量放 runner 顶部，audit 冻结进 manifest。

从 repo root 执行，最终实现应支持：

    [redacted-local-path] scripts/experiments/vsr_ditto50_linkage.py --run-dir runs/vsr_ditto50_linkage_smoke --stage all --smoke
    [redacted-local-path] scripts/experiments/vsr_ditto50_linkage.py --run-dir runs/vsr_ditto50_linkage_v1 --stage audit
    [redacted-local-path] scripts/experiments/vsr_ditto50_linkage.py --run-dir runs/vsr_ditto50_linkage_v1 --stage extract --resume
    [redacted-local-path] scripts/experiments/vsr_ditto50_linkage.py --run-dir runs/vsr_ditto50_linkage_v1 --stage syncnet --resume
    [redacted-local-path] scripts/experiments/vsr_ditto50_linkage.py --run-dir runs/vsr_ditto50_linkage_v1 --stage analyze --resume
    [redacted-local-path] scripts/experiments/vsr_ditto50_linkage.py --run-dir runs/vsr_ditto50_linkage_v1 --stage validate --resume

smoke 固定处理 id=1,2,3（含短句），但候选/decoy 池仍由完整 50 条文本冻结；smoke 与 formal 用不同目录，不能扩写 smoke 冒充 formal。smoke 验收最多 30 个视图与 6 个 SyncNet cell，不生成科学支持状态。

### 6.2 audit：冻结来源和候选

1. 读取第 2 节映射及三个文本来源；源清单里的旧 /mnt/e/Documents/tts-audio/tts-exp 前缀可换为当前 repo root，必须只替换这个明确前缀并记录映射，不能任意 basename 搜索。
2. 从 paired_key 提取 WID，以 paired_audio.records.source_utterance_id 唯一匹配，核对 expansion_id 与 mapping 中自然/TTS 文件编号，以及 speaker_id。id/paired_key/WID 重复、指向不同文本或重复视频/音频 SHA 冒充独立话语，阻塞相关 pair 并报告。
3. 文本规范化沿用 normalize_text；paired_audio.transcript、natural manifest 的 transcript、对应 transcript 文件、tts_meta.results.text 必须一致。tts_meta 的 sample_id 用 expansion_id 查，不默认等于视频 id。目标 OOV/空文本排除 pair，不修改真值。
4. 对当前 WAV SHA 核验 paired_audio 和 tts_meta 的 natural_sha256_remote/tts_sha256_remote；这些 SHA 对应原驱动音频，不能拿 AAC 解码 bytes 做相等比较。
5. 保存全部 source 文档、两个 MP4、两个 WAV、参考图 SHA，generation manifest 的状态和历史来源说明。记录 audio_provenance=hash_verified、video_generation_provenance=historical_documented。生成状态冲突/缺失要报告，不用本轮 hash 强行升级历史身份。
6. ffprobe 全部视频：25 fps、视频流和音频流、帧数、PTS/start、各流时长、可解码性。原视频必须有可解码音轨；无音轨不能把外部 WAV 补进去。视频和内嵌音频时长差 >0.12s、两流 start 差 >0.04s、内嵌音频与源 WAV 时长差 >0.12s 标为 TIMING_PROVENANCE_MISMATCH，不静默裁剪/移动。
7. 图来源缺少生成时哈希属于已知历史边界，不单独使所有 pair 失败；文档明确冲突才阻塞。
8. manifest 保存 candidate_count=50、全部 records（含失败）、run_sample_ids、目标 token、decoy、eligibility/reasons、阈值、code/model/config/source hashes。每条兼容旧提取函数的字段 id/natural_video/tts_video/target_token_ids/decoys，并新增 cohort_id=ditto50_s0765、paired_key、wid、speaker、tts_audio 等。
9. 新 environment.json 记录两个环境版本、实际可执行文件、AVSR/SyncNet 权重与源码 SHA、设备。复用 _make_config，但不要把旧 _environment_payload 的 AVTR protocol 当成新记录。

freeze_decoys：从全部来源一致且目标可编码的 50 条候选文本中，先排除自己的文本，再按 normalized_text 去重（保留最小数值 id），按 (字符数差绝对值, numeric id) 取 5 条。不同文本句子不足 5 则该 pair 不可评分；不因视频 QC 改 decoy 集。不从 AVTR-1 或全 400 条中临时挑更容易的候选。

### 6.3 extract：复用 VSR 五视图

调用旧 load_vsr、_set_determinism（seed=20260918）及 _extract_one_pair。旧函数输入只需要 id 与两视频路径；冻结 manifest 的其它字段供后续打分。

五视图和公式与旧 spec 完全一致：
- native：官方标准化嘴部 x[1,T,88,88]。
- frozen：第 floor(T/2) 帧重复 T 次。
- reversed：输入视频张量时间反转，重新前向；文本不反转。
- matched：N/T 分别线性重采样至 M=floor((T_N+T_T)/2+0.5)，align_corners=True。
- matched_frozen：matched 中间帧重复 M 次。

不得通过反转 logp 或 encoder 代替重新前向。保存 native_x、五视图 encoder/logp、现有 helper 返回的原始检测比例/最长缺失/单脸与帧一致性 QC、三帧预览。现有 extract_views 不返回逐帧 raw mask；本轮无需为此修改旧 helper，不把插值后的有效性误报为原始检测率。

继承检测率 ≥0.95、最长缺失 ≤5 帧、无多脸帧、去音轨前后帧哈希和相对 PTS 一致；任一不满足则 pair 不能进入主统计。两侧 native 重复前向的 encoder/logp max_abs ≤1e-5。

续跑：先 verify_inputs，检查 cell receipt（输入/输出 SHA）。_extract_one_pair 自带的“文件存在就跳过”不够，未验证缓存必须以 resume=False 重算本 run 的该 pair；禁止借用旧 AVTR 特征。逐 pair 捕获异常并记失败，不让一个坏视频终止其余 49 对。expected=500 views，不含重复性检查的额外前向。

### 6.4 syncnet：同视频新评分

使用 third_party/syncnet_python/data/syncnet_v2.model，环境隔离为 subprocess。输入为 manifest 中原 MP4，hash 必须与 VSR 源一致。处理的是两个原始条件，没有 2×2 音频交叉矩阵。

对每个 cell，cwd=third_party/syncnet_python，传绝对路径：
1. run_pipeline.py --videofile <source.mp4> --reference <natural_1或tts_1> --data_dir <run内该cell临时目录> --min_track 50 --frame_rate 25 --facedet_scale 0.25 --crop_scale 0.40 --num_failed_det 25 --min_face_size 100 --overwrite。
2. 读取其 pywork/<reference>/tracks.pckl，保存每个 track 的 frame 索引至 JSON；按 frame 数最大选择一个 track，平局按编号最小。选择仅依赖跟踪长度，不能依赖 SyncNet 分数。其编号映射 pycrop/<reference>/{index:05d}.avi。
3. 要求该 track frame 索引严格递增且连续，覆盖原视频帧数 ≥0.95；否则标 TRACK_COVERAGE_FAILED。这样不会用局部片段的 SyncNet 解释整句 VSR。真实多脸还会被 VSR QC 排除。
4. 对该 crop 调 demo_syncnet.py --videofile <crop.avi> --initial_model <weight> --vshift 15 --batch_size 20 --tmp_dir <run内临时目录> --reference <unique>。
5. 保存完整命令、returncode、stdout/stderr、输入/model/config SHA、选定 track 覆盖率、crop SHA 及 C/D/offset。regex 支持有符号数和指数，但必须恰好一组完整 C/D/offset；0 组或多组失败，不能默认 offset=0。C/D 必须有限。
6. timeout 单阶段 600s；失败记录 reason，续跑只补失败/缺失且缓存一致的 cell。禁止用历史 JSON 补缺。
7. 保留选中 crop AVI、track JSON 和日志；只有保存证据成功后才删除本 runner 创建的中间 JPEG/未选 track/临时 wav。不能删除输入或其他 run。串行处理限制磁盘占用。

这里只改评估窗口选择和记录，不改 SyncNet 模型。该预先规定的 95% 覆盖 QC 可能导致新结果与历史不同；报告这种协议差别，不能要求新 ΔC 数字必须复现历史到小数点后一致。历史结果单列对照，不参与新主统计。

### 6.5 固定指标与主集合

令 ell(V,y)=CTC_NLL(V,y)/len(y)，M(V)=mean(ell(V,5个decoy))−ell(V,target)。
CTC 为 CPU float64、blank=0、reduction=sum 后仅除一次目标长度、zero_infinity=False。检查目标及全部 decoy 可达 T≥L+相邻重复字符数；任何非有限/不可达则该 view 失败，不移除 decoy。

每条完整 pair：
- Q_N=M(N_native)−M(N_frozen)，Q_T 同理。
- G=Q_T−Q_N：主要 VSR 动态内容增益。
- B=M(T_native)−M(N_native)：不减静态基线的内容增益。
- Gmatched=[M(T_matched)−M(T_matched_frozen)]−[M(N_matched)−M(N_matched_frozen)]。
- R_N=M(N_native)−M(N_reversed)，R_T 同理；分别报告，不混作 TTS 增益。
- ΔC=C_T−C_N；ΔD=D_N−D_T（统一正号表示改善）。
- ΔCER=CER_N−CER_T；raw target loss、strict top1、时长、blank/entropy 只作辅助诊断。

先建立集合：
- V：来源有效、两臂 QC 合格、全部 10 views/60 candidate losses 有效且重复性检查通过。
- S：来源有效、两个新 SyncNet cell 完整且 track QC 合格。
- J=V∩S：唯一主集合，按 numeric id 排序；校准、G/B/Gmatched/ΔC 主均值、主要比例、相关全部用 J。
- V 与 S 各自全集结果可附录描述，必须显示自己的 n；禁止把 V 的均值与 J 的相关混称同批结果。

最低正式集合 n_J≥40 是提前固定的可用性门槛，不是功效证明；少于 40 仍完成数据和复算，输出 INSUFFICIENT_JOINT_PAIRS，并显示原始统计，不能换 cohort/降门槛。

分别报告 50→source eligible→V→S→J、各类失败 id；校准只在 J 上计算，两侧分别须 strict top1≥0.60、Q>0 比例≥0.75、mean(Q)>0，并通过技术验证。不能丢掉 top1 错误样本以通过校准。失败状态 INCONCLUSIVE_VSR_VALIDITY，关联数字仍可显示但不能作嘴型机制证据。

### 6.6 统计、关联与比例

这是带明确主关联终点的探索性扩展，不作多指标显著性搜寻。

- 对 J 的 G、B、Gmatched、R_N/R_T、ΔC、ΔD、ΔCER 报告 count、mean、median、95% percentile bootstrap CI、positive/zero/negative 数量。以配对话语为单位，有放回抽取 J 的行，20,000 次、seed=20260918；所有均值同一行抽样索引。正比例是严格 >0，0 不计正；用未舍入值，另报分辨率边界。
- 唯一主要关联检验：Spearman(G,ΔC)。average ranks 处理 ties，不依赖 scipy 默认近似 p。
- 双侧置换：固定 G，把 ΔC 在 J 行间置换 20,000 次，seed=20260919；p=(1+sum(abs(rho_perm)≥abs(rho_obs)−1e-12))/(20000+1)。不能只对两指标同时为正的子集求相关，也不能逐帧置换。此推断依赖话语可交换性，单说话人主题相关可能限制有效性；不是随机处理的因果检验。
- 相关 CI：成对重采样 (G,ΔC) 行 20,000 次、seed=20260920，重算 ranks/ρ，再取 2.5/97.5 百分位。任一常数向量相关不可估计，写 null+reason；bootstrap 退化抽样不重抽，记录数量，若有效抽样<95% 则 CI=null+INSUFFICIENT_NONDEGENERATE_DRAWS。不能把 NaN 填 0。
- Spearman(B,ΔC)、Spearman(Gmatched,ΔC)、Pearson(G,ΔC) 仅报告系数作为敏感性分析，不追加显著性检验或以它们替代主要关联。
- 保存逐 pair scatter 数据与 G 对 ΔC 的散点图（0 轴、id 标签、n/ρ/CI/p）；只一幅主图，无需 dashboard。

必须回答用户关心的“多少样本同时占优”：
- 2×2 表行 G>0/G≤0，列 ΔC>0/ΔC≤0；每格显示 count/n_J。
- 同时报 G>0、ΔC>0、交集、仅 VSR 正、仅 Sync-C 正、两者均非正。
- P(G>0 | ΔC>0)、P(G>0 | ΔC≤0)，分母为0时 null；报告交集在独立假设下期望 count=n_J×p_G×p_C。高基准正率下交集大不等于有关联。
- 零值/近零值分开计：报告 |G|≤1e-5、|ΔC|≤0.001 的数量作为分辨率提示，不据此改主要 >0 定义。SyncNet CLI 本身三位小数。
- 保存全部原始 id，不能把 46/50 历史正率复制成此次重评分正率。

### 6.7 科学判读分开保存

analysis.json 保留独立字段，不强行用一个标签混合三件事：
- engineering_status：技术与独立复算 PASS/FAIL。
- calibration_status：J 的双臂校准。
- visual_status：依下列冻结顺序。
- sync_status：新 ΔC 的 CI 是否全正/跨0/全负。
- association_status：主要 ρ 与置换 p、CI。
- scope：single speaker S0765、fixed documented face、historical validation cohort、not causal。

visual_status 顺序：
工程失败→TECHNICAL_INVALID；n_J<40→INSUFFICIENT_JOINT_PAIRS；校准失败→INCONCLUSIVE_VSR_VALIDITY；mean(G)>0且mean(B)≤0→CONTROL_DRIVEN_DIFFERENCE；mean(G)>0且mean(Gmatched)≤0→DURATION_SENSITIVE；G CI下界>0且B/Gmatched均值>0→EXPLORATORY_VISUAL_CONTENT_SUPPORT；G CI上界<0→EXPLORATORY_REVERSE_EFFECT；G CI含0→NO_CLEAR_PAIRED_GAIN；余下→INCONCLUSIVE_PATTERN。

association_status 只有工程/人数/校准通过时才给科学判读：ρ>0、p<0.05 且相关 CI 下界>0 → EXPLORATORY_POSITIVE_ASSOCIATION；ρ<0、p<0.05 且 CI上界<0 → EXPLORATORY_NEGATIVE_ASSOCIATION；其余→NO_CLEAR_ASSOCIATION（常数输入单列 NOT_ESTIMABLE）。均值 G 的 CI 判读与唯一关联 p 是两个探索性问题，不宣称一个整体“确认性机制检验通过”。

报告解释：
- Sync-C 增益、新 VSR 增益、正关联同时出现：支持“部分样本的视觉内容变化与 Sync-C 提高伴随出现”，不能说已证明嘴型更真实或算出因果贡献比例。
- 两种均值都正但相关不清楚：两种平均变化不能证明同一原因。
- Sync-C 正、VSR 无清楚增益或无关联：没有取得视觉解释的支持，不能宣布嘴型无改善或全部是评价器偏好。
- 本 cohort 新 Sync-C 增益未复现：先如实报告协议/样本差异，本轮不能解释未复现的效应。
- 不做 mediation、variance explained、“xx% TTS 增益来自嘴型”等因果量。VSR 也有模型偏好；此实验衡量内容可辨认，不直接测毫秒同步或人类观感。

### 6.8 自审、独立复算与交付

新 validate_ditto50 必须：
1. 校验冻结源文件、代码/模型/配置、全部产物 SHA、每对两臂完整性以及 QC。
2. 从保存 logp 与 manifest 中目标/decoy，调用已独立实现的 _independent_ctc_nll 逐个重算所有 COMPLETE 视图的候选 loss。该函数返回已按字符数归一化结果，先读源码确认，不能再除一次。
3. 用独立 loss 重建 M/Q/G/B/Gmatched/R 和 V，再独立解析新 SyncNet 日志/receipt 重建 S/ΔC/J；不能信任 producer 的 pair_rows/J/analysis。
4. 使用独立的 ties rank（如生产 scipy.stats.rankdata，验证用排序分组平均 rank）、统计循环，重算相关、置换、CI、符号表和科学判读。可使用同一冻结随机索引/排列或相同 seed，但不得调用新 producer 的 build_pair_rows/summarize_linkage。
5. 注入篡改的 G、错误键连接、错误 QC、缺 cell、变更权重的 fixture 必须使 validate FAIL。
6. 容差：logp 行概率和误差≤1e-5，独立 CTC normalized loss绝对差≤1e-5，G/均值/CI/ρ数值差≤1e-8；计数、集合、状态、置换超越次数必须一致。若 CPU 计算累计误差超限先诊断，不自动放宽。
7. 给出明确 expected/completed/failed/independently_checked 数量。全量最多500视图、3000 CTC候选评分、100 SyncNet cell；有合法失败时写 completed_with_exclusions。科学无支持可以是正确实现，不能只为状态好看改规则。

产物：
manifest.json、environment.json、features/、receipts/、syncnet/{id}/{condition}/（score.json、track.json、crop.avi、日志）、records.jsonl、pair_rows.json、analysis.json、validation.json、report.md、linkage_scatter.png。
JSON 禁 NaN/Inf，用 null+reason；恢复过程中原子写完成 receipt。单个 pair 的中间证据完整后再写 COMPLETE。

完成实现、smoke、全量、自审修复与独立复算后，按 Startup Router 写 Experiments 结果笔记，链接本文；更新 HANDOFF 当前状态，并将本文 status 改 implemented。尚未全量执行不能标 implemented；测量校准失败可以完成实现但结果需明确科学不可判读。

## 7. Expected Change Surface

### Must change
- 新 scripts/experiments/vsr_ditto50_linkage.py。
- 新 scripts/experiments/vsr_ditto50_metrics.py。
- 新 tests/experiments/test_vsr_ditto50_linkage.py。
- 下游实际执行时新 runs/vsr_ditto50_linkage_smoke 和 runs/vsr_ditto50_linkage_v1。
- 完成后 BM Experiments 结果笔记、本文状态与 HANDOFF 的当前结果指针。

### May change
- 仅当已证实复用 helper 无法完成必要行为，最小修改 vsr_tts_pilot.py，并保持原 CLI/旧实验规则不变、补对应 tests/experiments/test_vsr_tts_pilot.py。
- 如发现来源证据矛盾，修订本文相应事实并写明变更，不自行重选 cohort。

### Should not change
- 原视频/音频/文本、已有 run、现有 VSR 两脚本的默认科学规则、第三方模型/权重、全局 config.yaml。
- 不升级依赖、不下载第二个模型、不部署服务器、不操作网盘/旧压缩包、不训练 head、不 commit/push、不清理其它任务文件。

## 8. Validation Plan

先静态/CPU，再 GPU smoke，再全量与独立复算；不是运行全部历史测试来代替针对性验证。

    [redacted-local-path] -m py_compile scripts/experiments/vsr_ditto50_linkage.py scripts/experiments/vsr_ditto50_metrics.py
    [redacted-local-path] -m pytest tests/experiments/test_vsr_ditto50_linkage.py tests/experiments/test_vsr_tts_pilot.py -q

新测试必须验证：
1. 两套同名整数 id 不同 WID/text 不可连；源清单重复键、TTS text冲突、源 WAV hash错均被拒绝；允许已知旧路径前缀映射但不允许 basename猜测。
2. decoy 同文本去重，长度差/数值id排序稳定；N/T/所有视图相同；视频失败不改变候选池。
3. 特征齐全但 QC失败、缺一个视图、缺一臂评分、非有限或CTC不可达都不能进 J；校准分母严格等于 J。构造 V与S不同时验证交集。
4. synthetic 正/负单调关联得到 ±1；ties 正确；打乱输入行顺序后键连接结果不变；常数向量 null，不能报ρ=0/p=1。
5. 小 n（如4）穷举置换验证双侧超越计数；实跑 Monte Carlo +1公式正确；成对 bootstrap不拆两臂，零分母/退化抽样可报告。
6. 手工已知 2×2计数、交集期望、正比例，0属于非正；ΔD/ΔCER正向定义正确。
7. parser支持负offset、拒绝零组/多组分数；最长track平局按编号、低覆盖拒绝，不按C挑track。
8. 缓存缺receipt/输入或配置或feature内容改变拒绝复用；失败项重跑不会留下旧COMPLETE。原始证据损坏只允许在本 run 中重算对应 cell；冻结源输入/协议变动必须新 run。
9. 验证器由logp重建结果：故意只改pair_rows里的G、调换某个WID、篡改status，应被发现。
10. n=39不能正式判读；n≥40但单侧校准失败不能称口型无改善；上述visual/association各状态边界有测试。

GPU smoke固定id1/2/3：
- 检查全部10视图/对及两边SyncNet，短句不被min_track=100误伤。
- id1原音轨、去音轨、零音轨三版仅copy视频流，VSR张量/encoder/logp在1e-5以内一致；原输入只读。
- 同设备两次native输出一致；选定SyncNet track和输入SHA可追溯；没有用旧评分缓存。对 id1 的两个条件各重复一次新 SyncNet 评分，C/D 三位小数与 offset 必须一致，track 选择不变；不一致先诊断数值/预处理随机性，不继续声称关联已稳定。
- 记录峰值显存、耗时、产物与临时盘占用，按50/3外推；可用空间需覆盖估算产物+最大单cell scratch+1GB余量。空间不足写具体缺口，不能擅删文件。
- smoke是技术校验，不能用于改decoy/阈值或替换三个样本。个体QC失败可如实记录并继续评估其它候选，系统性来源/预处理故障须先修复。

正式运行：覆盖全部候选，校准与统计按固定规则，validate为PASS；若排除过多或无优势，诚实输出不足/无清楚证据，也算科学上正确交付。

## 9. Risks and Edge Cases

- Ditto-50为历史增强头valid cohort；已被多次分析、单说话人和固定脸。不能宣称全新heldout、跨说话人或跨脸泛化。
- 完整生成配置 video_natural_tts.yaml 可能不在本地；有metadata和历史记录仍不是逐视频历史hash闭环。保留documented等级，不捏造验证。
- source wav正确不自动证明TTS实际逐字发音正确；目标来自合成请求/人工语料文本，本轮不增加ASR判筛。结果是对目标内容的端到端辨认。
- 高Sync-C正率会自然带来较大交集；必须报期望交集与连续ρ。
- 冻结帧本身可能偏向某种嘴形；因此同时报告B与Gmatched，不能只凭G增加宣称绝对可读性增加。
- CTC是汉字posterior，不是音素后验；发射峰不能当MFA音素边界。
- VSR数值可算但QC失败仍不可进主集合；这是旧runner最需要避免的复用陷阱。
- SyncNet有多个track时，首次regex可能错误匹配任意track；新流程显式确定唯一track后评分。
- n=50也可能得到宽CI；没有清楚支持不能写作等效于0。主题/语速等共同因素可造成相关，相关不能说明作用方向。
- decoder beam/CER只辅助；不更换LM、层或主要G指标追求阳性。
- smoke/正式输出不同目录；恢复不能混入历史13对或其它同名1..50文件。

## 10. Assumptions / Unknowns

- VERIFIED: 本地50对Ditto原视频、100个驱动WAV与对应文本存在；源WAV hash和文本已对照paired_audio清单全量核实。
- VERIFIED: mapping所有speaker为S0765，历史记录固定face=data/data/image/1.png；历史ΔC=+1.081、46/50为正。
- VERIFIED: AVTR-1 pilot代码和已安装CMLR环境可作为复用基础；本轮未跑新的VSR推理。
- VERIFIED: 旧analyze_run/decision_status/环境记录含AVTR-1或13对假设，不能直接用于本cohort。
- LIKELY: 历史generation metadata准确描述当前视频的来源与同脸条件；无生成时SHA，保持historical_documented。
- UNKNOWN: 全50对的VSR裁剪/可测性、track覆盖与同批重评分成功率；由audit/smoke/formal验证。
- UNKNOWN: 新VSR均值和与ΔC的关系是否支持视觉内容解释；不能保证阳性。
- UNKNOWN: 未来实现时磁盘/GPU是否仍可用；启动实测，不能以本文9.3GB作实时可用量。
- UNKNOWN: 本地视频的确切历史编码/生成配置是否可完整恢复；不将其等同Ditto-100。
- UNKNOWN: 跨说话人、跨人像、其它VSR模型以及人类评价是否复现；本轮不扩张验证范围。

## 11. Handoff Contract

按第3节anchors实现薄runner与纯统计模块，遵守第5节invariants。复用已有VSR预处理/前向/CTC，不调用其硬编码AVTR分析器，不新建通用框架。按audit→3对smoke→50对全量→独立复算→自审修复→BM与HANDOFF交付。

验收必须能追溯“同一WID/目标文本 → 同一原MP4 → VSR G和新SyncNet ΔC → 同一J集合 → ρ/CI/置换p/交集比例”，并能从保存logp和评分日志独立重建。只交代码、模型加载成功或历史分数分析都不算完成。

保持最小改动范围，不做无关重构。如实际来源、模型接口或协议与本文冲突，停止依赖该假设的阶段，报告具体路径/字段/差异，同时可继续不依赖它的实现和测试；不能通过换真值、降门槛、混数据或用旧分数补洞绕过。负面或不确定科学结果应如实交付。
