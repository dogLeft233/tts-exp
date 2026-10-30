---
title: TTS 嘴部动作时间校准与自然时间轴迁移实验 Spec
type: note
permalink: tts-exp/research/tts-嘴部动作时间校准与自然时间轴迁移实验-spec
status: implemented
protocol: tts_visual_timing_v1
question: TTS 原生 Sync-C 增益是否伴随独立视觉动作时间代理改善，且能否迁移到自然音频时间轴？
tags:
- tts
- visual-timing
- vsr
- implementation-architect
- spec
---

# TTS 嘴部动作时间校准与自然时间轴迁移实验 Spec

## 1. Objective

协议 `tts_visual_timing_v1`。实现一个自动分阶段、允许科学阴性的探索实验：复核既有 VSR 内容证据 → 校准独立的视觉动作时间指标 → 比较历史原始自然/TTS 驱动视频 → 若有支持，检查本地 Wav2Lip 的原生效应并测试自然时间轴候选。主问题是“高 Sync-C 是否伴随更准确的嘴部动作时间”，不训练增强头。

必须实现全部阶段、成功和失败路径、独立复算、最终报告及 BM 回写；实际数据执行由下面的门控决定。校准失败时不能强行运行依赖它的科学分析；有合法科学停止原因、完整台账和报告可以是正确交付。依赖缺失、代码错误与科学阴性必须分开。

原有 CMLR VSR 测量的是内容辨识，不是毫秒级同步。新增主指标采用 MediaPipe 嘴部开合低谷事件与真人参考的时间差；它不接收音频特征，不依赖 SyncNet。但真人参考映射到 TTS 时钟需要音频 MFA，因此仍是“给定音频对应关系的视觉时间代理”，不是 TTS 的真实视觉真值，不覆盖舌位、牙齿、全部音素或人类感知。

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 根据现有 VSR 与时间实验制定三阶段自动执行规范；尚未实现或运行 | September 20, 2026 | user request |
| 后续只读自审发现静音、连续支持和审计缺口；关联 v2 修复 Spec，保留 v1 历史结果 | September 20, 2026 | user request |
| 自审核对现有函数、模型资产、24个裁剪起点记录；补充Python环境隔离、原始坐标有效性、时钟绑定和独立复算要求 | September 20, 2026 | architect self-review |

## 2. Repository Model

两条资料链不得混合样本或语言：

- 中文历史内容链：`runs/vsr_ditto50_linkage_v1/{manifest,analysis,pair_rows,validation}.json`、`features/` → `vsr_ditto50_metrics.py` / `vsr_tts_metrics.py`。已有 47 对、五视图，VSR 校准未通过；只重算存量结果，不重新训练/下载中文模型，不在 LRS3 上运行 CMLR。
- 英文视觉时间链：`runs/tts_time_instance_20260917_v1/inputs.json` 的 IDs 151–162 → 已绑定真人视频、原始 N/T 音频、A 阶段 24 个 TextGrid → 历史 LeapTalk N/T 视频 → 独立嘴部几何。新生成阶段单独使用本地 Wav2Lip，不冒充恢复 LeapTalk。

时间输入：`runs/tts_time_instance_20260917_v1/A/A_textgrids/{id}_{N,T}_audio.TextGrid`。其对应的是 inputs.json 的 `N_audio/T_audio`，不是 B 阶段新生成的 `T1/T2`。检查 `A/A_corpus` 目录的 WAV/LAB 与对应音频和文本哈希；不能仅凭 TextGrid 文件名复用。

历史视频从 `runs/tts_native_gain_attribution_implementation_20260915_v1/00_audit/assets.json` 和 `runs/lrs3_tts_gain_mechanism_review_v15/02_curves/{manifest,media_audit}.json` 按 sample_id、condition、source_group 联结。具体视频字段为 assets.records[].sources.{N,T}.media.path/sha256；原始路径形如 `runs/leaptalk_multiset/{natural_raw,tts_raw}/{id}.mp4`。优先使用具有完整 PTS 的原始 MP4，不把 SyncNet 的滑窗 embedding 当视频帧。查不到匹配则记缺失，不按数字文件名猜。

校准用另一批 4 个真人片段，来自 `runs/lrs3_real_video_local_timing_20260905_tail_v2/protocol.json`，与 12 个主分析 source_group 不重叠：

1. `lrs3_6ul2TSvUDog_00007`
2. `lrs3_6wk4dkYSrV0_00006`
3. `lrs3_73jPh0eRPSY_00008`
4. `lrs3_6qqqVwM6bMM_00007`

4 个 ID 是按父清单顺序剔除主分析 source 后确定的，未查看新视觉得分；失败不替补。主分析仍固定全部 151–162，不能选其中 Sync-C 阳性的样本。本实验使用已研究过的材料，是新的测量探索，不称独立确认。

## 3. Code Anchors

| path | symbol | current role | required change |
|:--|:--|:--|:--|
| `scripts/experiments/vsr_ditto50_linkage.py` | `analyze_ditto50`, `validate_ditto50` | 中文成对五视图与关联结果 | 只读仿照；不得在旧run上调用会写文件的analyze/validate；新run重算并保留旧结论 |
| `scripts/experiments/vsr_tts_metrics.py` | `ctc_nll`, `length_normalized_nll`, `content_margin` | CTC 内容得分 | 复用现有公式，不以新阈值重写旧实验 |
| `scripts/experiments/lrs3_tts_visual_advantage/video_features.py` | `VisualSequence`, `create_landmarker`, `extract_video_features`, `save_visual_sequence` | 已有 478 点和有效掩码 | 复用检测与存储；新模块修正像素坐标/时间审核，不改旧产物 |
| 同上 | `mouth_features`, `canonicalize_landmarks` | 眼距归一化嘴部几何 | 不直接复制 derived validity：点 13 未包含在旧 MOUTH_INDICES 的有效性集合；新测量显式检查 13、14、61、291、33、263 |
| `scripts/experiments/tts_time_instance.py` | `parse_textgrid`, `run_mfa`, `match_occurrences` | MFA 层解析、缓存身份、词内音素匹配 | 复用解析和缓存核验思路；新建严格 ordinal-word 匹配，不盲复制贪心游标实现 |
| 同上 | `_static_render` | 静态脸 Wav2Lip 编排 | 仿照参数与 receipt，使用原始 T_audio，不使用 T1/T2，不直接调用旧硬编码三臂循环 |
| `scripts/experiments/tts_native_gain_attribution/syncnet.py` | `SyncNetEngine.extract_visual`, `extract_audio`, `distance_matrix` | 官方权重与MFCC/距离矩阵 | 复用已校验前向；剔除padding支持，不复制历史run硬编码 |
| `scripts/experiments/static_image_bridge/render_worker.py` | CLI / `main` | 冻结 Wav2Lip 渲染 | 子进程直接调用，不改模型源码；五臂同图、同 box、seed=42 |
| `scripts/experiments/lrs3_real_video_local_timing/media.py` | `build_mapping`, `encode_ffv1`, `verify_muxed_media` | 历史真实视频 timing 控制 | 借鉴帧映射和无损输出；新增本协议的 80/160ms 控制，不复制旧 SyncNet 科学判据 |
| `scripts/experiments/tts_visual_timing.py`（新） | `audit_inputs`, `audit_vsr`, `calibrate`, `compare_native`, `run_generation`, `report`, `main` | 无 | 唯一编排入口、状态机、绘图和交付 |
| `scripts/experiments/tts_visual_timing_metrics.py`（新） | `aperture_events`, `event_distance`, `audio_time_map`, `group_summary`, `decide` | 无 | numpy/标准库纯函数，无模型/网络导入 |
| `scripts/experiments/tts_visual_timing_worker.py`（新） | `extract`, `retime_video`, `stretch_audio`, `main` | 无 | 环境隔离的视觉/媒体薄 worker，兼容 Python 3.8 |
| `scripts/experiments/check_tts_visual_timing.py`（新） | `check_run`, `main` | 无 | 从原始证据独立重算，不能调用主脚本的统计/判决函数 |

如实际符号名称不同，先定位并更新本 spec 的锚点，再实现；不要为凑名称修改旧接口。CodeGraph 已尝试，部分未跟踪文件未被索引，必要时对这些文件用 rg。

## 4. Reference Pattern

- `vsr_ditto50_linkage.py`：借鉴“先固定来源、五视图缓存、再统计”的薄脚本结构。旧 60% 内容校准门槛只适用于旧 VSR，不能当作新视觉时间校准门槛。
- `tts_time_instance.py::write_json`：临时文件 + 原子替换；禁止 NaN/Infinity。缓存按输入哈希和配置哈希复用。
- `lrs3_tts_visual_control/run_visual_teacher_audit.py::_record_result`：借鉴三臂几何缓存、检测有效率和逐记录台账。不要照搬 133 条 cohort、122 条门槛、旧审查文件依赖或视觉 DTW；本轮不优化视觉对齐。
- `tts_time_instance.py::_static_render`：复用固定静态人像与独立重复渲染。不能让已有视频文件存在就算成功，必须复验 receipt 与当前音频哈希。

## 5. Invariants

1. 下游先读 AGENTS、Startup Router 和实验指令。已有脏工作树只读保留。新文件若已存在，读后续作，不覆盖其他工作。
2. 所有历史结论保持：VSR 未校准、F0 支持不足、旧 Wav2Lip timing-transfer 未确认。新实验不以 ENGINEERING_PASS 覆盖这些结果。
3. 模型看到的视频帧不含音轨、字幕文字或目标文本输入；嘴部事件检测是纯视觉。文本/MFA 仅在检测完成后建立时间映射。
4. CTC blank 不是静音；字符峰不是音素边界；VSR 整句得分不是同步误差。禁止用 CMLR 给英文生成“置信度”。
5. 禁止用 SyncNet 最优 lag、视觉 DTW、嘴部曲线互相关来校准被评估视频。这些操作会把要测的时间错误消掉。只允许核实并应用容器中记录的真实流起点差。
6. 不按 Sync-C、视觉好坏或 TTS 优势筛样。生成视频没闭嘴/没有开合动态是待测结果，不是删除理由。输入缺失、解码失败和检测失败逐个记账。
7. 使用真实 PTS；仅在 ffprobe 确认 CFR、起点和间隔后才能采用 i/25。音视频尾长只取可评分支持，不裁源音频、不循环或冻结补帧。音频多 1–2 帧的容器尾部不是自动失败。
8. 同一条原始自然 PCM 用于所有 replacement 评分；评分时不得按候选重采样/调增益/调延迟。原生 TTS 只与其自己的 T 音轨作为单独原生参照。
9. 新 Wav2Lip 实验与历史 LeapTalk 分开统计。不能用 Wav2Lip 的阴性解释全部 Ditto/LeapTalk；不能把新 Wav2Lip 标为旧 LeapTalk B 阶段完成。
10. 本轮不训练、不做参数/层搜索、不访问 sealed split、不新增大模型。未知依赖只按固定恢复策略处理。任何门槛都不能看结果再降。

## 6. Implementation Plan

### 6.1 自动入口、运行目录与状态

新增 `--stage all|audit|vsr-audit|calibrate|native|generation|report|validate`、`--run-dir`、`--resume`、`--smoke`。默认正式目录 `runs/tts_visual_timing_v1/`；smoke 使用独立 `runs/tts_visual_timing_smoke/`。默认 20,000 次 bootstrap、PCG64 seed=20260920。不得因科学阴性返回未捕获异常。

输出：`manifest.json`、`environment.json`、`cells.jsonl`、`vsr_audit.json`、`calibration.json`、`native.json`、`generation.json`、`analysis.json`、`validation.json`、`report.md`、`features/*.npz`、`events/*.json`、必要控制视频/新生成视频、`figures/`。台账字段固定含 stage/id/source_group/arm/input_hash/config_hash/status/reason/artifacts；missing/failed/skipped 也要有行。

每阶段具有两个独立状态：
- execution：`COMPLETE|PARTIAL|DEPENDENCY_BLOCKED|ERROR|SKIPPED_BY_GATE`。
- science：`SUPPORTED|NO_CLEAR_SUPPORT|MEASUREMENT_INVALID|INSUFFICIENT_SUPPORT|NOT_TESTED|EXPLORATORY_ONLY`。

`all` 顺序执行 audit → vsr-audit 与 calibrate → native → generation → report → validate。VSR 历史校准失败不阻断独立 landmark 路线；landmark 校准失败阻断 native 机制判读和 generation，仍完成旧 VSR 重算、报告、复算。任何阶段科学门未通过，所有下游项明确 SKIPPED_BY_GATE；不得用空数组冒充完成。代码必须用合成 fixture 覆盖未实际进入的成功分支，fixture 明确 `SYNTHETIC_NOT_SCIENCE`。

### 6.2 audit：锁输入、环境和模型

完整绑定第 2 节四份父清单、对应音视频、文本、TextGrid、代码文件、checkpoint、landmarker task 的绝对路径及 SHA256。区分文本文件字节哈希、规范化文本哈希、WAV 文件哈希、PCM 哈希，不能交叉比较。时间结构写入 manifest：PTS、fps、音频 sr/n_samples、流起点、每个词/phone 的起止。父ORIGINAL.wav可能来自SyncNet crop.avi，必须解析crop_selection的原视频帧起点/音频裁剪起点，建立N_audio到真人R、T_audio到G_T的时钟变换；0偏移也要有来源证明。容器start_time=0不等于原句起点=0。缺可追溯裁剪起点则对应record的native不可解释，记CLOCK_BINDING_UNRESOLVED；不得用声音互相关、SyncNet最优lag或嘴部匹配估计这个缺失偏移。

主队列固定 12 条；校准固定 4 条。检查 source_group 不重叠。主队列需要至少 8 个可测 source groups，且真人参考共至少 16 个合格事件；不足仍保存全部可测描述，science=INSUFFICIENT_SUPPORT。分母 12 不减少。不能扩样“补齐显著性”。

环境使用已存在路径：视觉 worker `[redacted-local-path]`；Wav2Lip `[redacted-local-path]`；SyncNet `[redacted-local-path]`；MFA 从父 receipt 解析并核对当前 executable/version。已确认landmarker位于 `runs/lrs3_tts_visual_advantage_20260824/01_metric_parity_retry7/snapshots/mediapipe_landmarker.task`，SHA256=`64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`；绑定来源为 `runs/lrs3_tts_visual_control_20260825/01_visual_teacher_audit_retry4/manifest.json`。Wav2Lip checkpoint为 `third_party/Wav2Lip/checkpoints/wav2lip_gan.pth`；SyncNet为 `third_party/syncnet_python/data/syncnet_v2.model`。MFA默认 `[redacted-local-path]`、dictionary=`english_us_mfa`、acoustic=`english_mfa`，均与父receipt逐项复验。不得拿AVSR的四点裁剪坐标代替478点。

实现前不需立刻安装生成阶段依赖。只有 generation 放行后才恢复 Rubber Band CLI：优先现有二进制；不存在时在新专用前缀以 conda-forge 安装 `rubberband`（不修改已有 Python 环境），记录求解后版本、可执行文件 SHA 和 `--full-help`。安装命令模式为 `[redacted-local-path] create -y -p <run-dir>/tools/rubberband -c conda-forge rubberband`，实施者传绝对路径参数列表。若包无 CLI/无 `--timemap`，记 DEPENDENCY_BLOCKED，不改成普通线性 waveform 插值。官方时变映射使用 source/target **sample frames**，并需要总体 duration/ratio，见 [作者 CLI 帮助](https://www.breakfastquay.com/rubberband/usage.txt)。采用 R3 `-3`、pitch=0，参数固定，不比较引擎挑结果。

smoke 只用第一个校准 ID 和 151，验证 I/O、重复性、映射方向；不输出科学结论，也不据此调整阈值。预算先磁盘/GPU审计；串行模型运行，复用现有 GPU lease。只清理本 run 自己创建的临时文件，不删除旧 runs、模型或数据；空间不足写出需求和剩余量。

### 6.3 vsr-audit：只读重算过去做过的实验

新runner只调用旧纯指标函数，不能对父目录执行会重写报告的analyze_ditto50/validate_ditto50。从 Ditto-50 已保存 logp/targets/decoys 重算 G、Gmatched、strict top1、Spearman 和 2×2 表；严格使用旧公式、分母与阈值。不重跑 470 视图，不把校准阈值改低。若缓存缺失就指明具体 cell，保留父结果并标 `CACHE_NOT_RECOMPUTABLE`，不声称独立复验完成。

交付解释必须说明：原生视频内容得分存在正迹象；时长匹配结果未确认；它不是动作时间指标。该支线不产生英文 VSR 分数。

### 6.4 纯视觉指标：先固定算法，再运行正式数据

每视频检测一次，保留所有帧的 478×2 normalized landmarks 和原始 valid。视觉worker运行在Python 3.8，不直接import带`X | None`类型注解的旧video_features模块；在新worker薄封装官方FaceLandmarker检测，将原始landmarks/valid/PTS落盘，主进程执行纯numpy指标。新worker使用与现有检测器相同的MediaPipe task和IMAGE模式，但num_faces=2以暴露多脸；检测出多脸的整个视频记不可测，不挑最大脸；旧num_faces=1缓存无法证明单脸，须补本次检测。新计算先做 `x*=width; y*=height`，再用眼距归一化，避免不同画幅比例造成几何差异。以下必要点必须有限且原始检测成功：13、14、61、291、33、263；不得把旧数组中失效帧的零值当闭嘴。

开口量 `a(t)=||p13-p14||/||p33-p263||`。只在连续有效的 3 帧上取中值平滑；缺失不插值。全片有效率至少 0.95、最长无效段不超过 5 帧；不满足是 detection failure。全片 P90(a)-P10(a)<0.015 时标 `LOW_MOTION`：真人参考无法校准；生成视频仍保留，以空事件列表参与惩罚。

对非 LOW_MOTION 的有效帧，`z=(a-P10)/(P90-P10)`，不 clip。事件为 z<=0.25 的连续低谷区间：长度 1–8 帧；左右 200ms 内各至少存在一个 z>=0.60 的有效帧；所需邻域不得跨缺失段或片段边缘。时间取区间内最小 a 的所有并列帧的中间时间。候选相距<200ms时保留 a 更小者，并列取更早者。按时间升序输出。

这些事件称“开合低谷”，不能自动命名为 /p,b,m/ 或完全闭嘴。相对幅度归一化意味着本指标主要评价时序，不说明动作幅度逼真。

定义 reference 事件集 R 和 prediction 集 P 的有序一对一编辑距离。匹配成本 `2*min(abs(tR-tP),0.240)` 秒；漏事件、额外事件成本各 0.240 秒。用动态规划求最小总成本，除以 `len(R)+len(P)`，得 E 秒（报告乘1000为ms），越低越好。平局先匹配、再删除 reference、再删除 prediction，固定回溯。R非空、P空时 E=240ms；两者空不能评分。禁止仅在成功匹配事件上算平均误差。另报告漏检/多检率、匹配误差分布，但不替代主指标。

### 6.5 calibrate：已知局部时间变化的纯视觉响应

4 个校准真人视频各生成 7 条无音轨控制：REAL、独立重编码 REPEAT、LOCAL_+80、LOCAL_-80、LOCAL_+160、LOCAL_-160、FROZEN。共28个cell。REAL/REPEAT 使用相同无损编码；FROZEN重复中间帧。控制均在解码 RGB 全帧上构造并重新检测，不直接平移提取好的 landmarks 冒充端到端校准。

令视频时长 L，s(t)=t+d*w(t)，w为经过 `(0,0),(0.20L,0),(0.35L,1),(0.65L,1),(0.80L,0),(L,0)` 的折线；d分别为±0.08、±0.16秒。输出帧 j 取原帧 `floor(25*s(j/25)+0.5)`；逐帧检查索引合法、单调不降、端点不动、实际位移，禁止 clip 修复错误索引。L<2s校准支持不足。正d表示读取未来画面，事件应提前，不要写反符号。

主校准区只使用 `[0.35L+0.24, 0.65L-0.24]` 内真人参考事件，各控制均用同一冻结区。由实际离散索引映射计算每事件应出现的输出时间（重复帧区取中点），并与重检测事件做有序匹配。每条至少2个参考事件。

单record通过需：REPEAT E<=20ms；四种warp的期望事件恢复率各>=0.8、已恢复事件的中位时间误差各<=40ms；符号正确；相对未warp参考的E在每个符号下 E160>E80>Erepeat；FROZEN须无有效动态，不得得到“完美同步”。至少3/4 record通过，才是 `VISUAL_TIMING_CALIBRATED`。保存所有record，失败不替补/调阈值。

这只验证“该检测器能追踪受控动作时间变化”，不证明事件等于某种音素。后续结论必须继续保留真人参考映射假设。既有 SyncNet 22/22 的校准不替代本步骤。

### 6.6 native：历史 LeapTalk 原生 TTS 是否改善视觉时间代理

对全部12条真人R、自然驱动G_N、TTS驱动G_T提取事件，共36个视频。不生成新视频。音频侧从 words tier建立规范化词序列，要求两臂词序列一致并按词序号配对；不能跨句追找重复单词。每个对应词内，对 normalized phone 标签做确定性LCS（并列优先较早N index，再较早T index），只保留正时长、非unknown、非silence的匹配。

每匹配phone的N/T起止构成分段仿射映射 f:N时间→T时间。自然和TTS两侧匹配speech时长覆盖均>=0.80。未匹配phone/静音不靠邻近同标签补齐。映射只定义在匹配phone区间；边界相邻映射一致才可连接；不一致的间隙不进入主评估。没有 words tier 则记 alignment unsupported，不使用旧 parser 的无词 fallback。

先仅依据R和音频映射固定事件支持：真人参考事件处于有效匹配phone，且其±240ms搜索邻域完整落在一个连续有效映射段内、远离无效检测帧。对G_N事件直接用自然时钟；对G_T事件用 f逆映射回自然时钟；只用相同支持区域，边界半开。不得按G_N/G_T是否存在好事件筛R。多个支持区域分别算编辑成本，最后汇总成本/事件计数，不能跨不支持的间隙匹配。

记录 `E_N,E_T,B_V=1000*(E_N-E_T)`，BV>0支持TTS。标准化到自然时钟是为了避免TTS更短导致绝对毫秒误差天然更小。同时报告各自原生时钟下的误差，明确它们不可直接混为相同尺度。

从有效父SyncNet矩阵独立重算同一完整交集的原生 ΔC（若缺则新评分，不用其他cohort均值顶替）。报告BV与ΔC的Spearman、置换p和符号交集，均为次要探索。不得在12对中先选ΔC>0再分析BV。

主门：至少8/12 source、合计至少16个reference事件；BV的98.333% bootstrap CI下界>0、均值>=20ms且至少2/3有效source为正，判 `VISUAL_NATIVE_SUPPORT`。否则 NO_CLEAR_SUPPORT/INSUFFICIENT_SUPPORT；不是“无嘴型改善”。同交集ΔC的95% CI下界>0才能称“本cohort高Sync-C伴随视觉时间优势”。若BV有支持但ΔC未复现，保留视觉发现，generation不作为TTS同步评分机制继续。

### 6.7 generation：同一来源、本地Wav2Lip的自然时间轴检验

仅在校准通过且native同时支持BV与ΔC时运行本节；必须实现本节全部代码，即使正式运行被门控跳过。固定全部12个主ID，不按native逐条结果筛选。复用 inputs 的portrait及 B/audio_manifest.json 的box，逐一核验portrait SHA；只复用box，不复用B的T1/T2或旧生成状态。

使用原始N_audio/T_audio，先制作恒等处理 N_ID/T_ID，再生成 N、T、N_ID、T_ID 四臂。T_NAT为第五臂，在下述baseline门通过后生成。每臂使用同一Wav2Lip checkpoint、同一静态脸、同一box、seed42、batch4和帧率。首ID151每个已运行臂独立重渲染一次；不能复制文件作为repeat。完整最大60个主视频+5个repeat；已有完全同源同配置视频可以哈希核验后复用，否则新生成。

恒等处理：Rubber Band整段运行，使用与实际映射数量相同的identity knots（N端点→自身；T端点→自身）、--time 1、-3、--pitch 0、--ignore-clipping。T_NAT：将下节映射的T sample knots作为输入、N sample knots作为输出，--time=N_samples/T_samples。只调用一次整段time-map，不逐音素拼接声波、不普通线性重采样波形。二进制缺参数则DEPENDENCY_BLOCKED，不换算法。

映射用已配对phone端点，加(0,0)与(T_samples,N_samples)。内部knots按秒乘16000后四舍五入half-up到整数sample；重复输入坐标仅当输出坐标也相同才能合并，其余冲突拒绝。所有相邻source与target严格递增；区间局部伸缩比须在[0.5,2.0]内。未匹配间隙仅在候选构造时线性连接，明确这部分不保证音素对齐，不进主视觉事件支持。禁止clamp伸缩比或按视频评分调knots。

音频QC：保持16k mono；原始N/T不额外loudnorm；Rubber Band日志不得出现自动降增益重启，输出不得有clipping/非有限值；若出现则该candidate失败。实际长度与目标允许至多1 sample的舍入差，只允许尾部修整这一sample并记账，禁止整句补零/截断修复。重新用同一MFA模型对N_ID、T_ID、T_NAT对齐作操纵检查：匹配speech覆盖>=0.80；ID相对原边界中位误差<=20ms、P90<=40ms；T_NAT相对目标N边界中位<=40ms、P90<=80ms。该检查只衡量同一aligner下目标是否达到，不叫独立音质真值。输出WAV若是float32，先检查原始浮点峰值，再以rint(x*32768)量化到PCM16；若CLI只输出PCM16，记录subtype，任何饱和sample或clipping警告都拒绝，不用clip()掩盖溢出。身份处理RMS变化绝对值<=0.5dB，否则科学比较不放行；不回调增益修正。T_NAT的RMS/F0/频谱改变只报告，不据此声称纯时间干预。

生成阶段分两个门：

1. **baseline门**：四臂全部完成后，分别用N评分N、N_ID，用T评分T、T_ID。新Wav2Lip的原生 ΔC=C(T,T)-C(N,N) 在同一可测交集95% CI下界>0；BV_W=E(N)-E(T) 同样95% CI下界>0；N_ID相对N、T_ID相对T的Sync-C差CI都完全落在[-0.20,+0.20]、视觉E差CI都落在[-20,+20]ms。至少8个source支持。未通过记 `WAV2LIP_BASELINE_UNCONFIRMED` 或 `IDENTITY_PROCESSING_NOT_EQUIVALENT`，不运行T_NAT；不能写TTS机制被推翻。
2. **replacement门**：通过后生成T_NAT，每条与N、N_ID统一配回未经修改的原始N PCM评分。用同一真人R、自然时钟、同一冻结支持求视觉E，不再为T_NAT重新视觉/音频对齐来消除误差。主收益为 `R_V=1000*(E(N)-E(T_NAT))`、`R_C=C(T_NAT,N)-C(N,N)`；另报告相对N_ID的两个对比以及T_ID/T原生参照。只有R_V和R_C的校正CI下界均>0，且R_V均值>=20ms、两者各至少2/3source正向，才称本协议 `REPLACEMENT_SUPPORT`；相对N_ID两个对比也须方向为正。若不满足，准确写视觉/评分各自结果，不把一项阳性当完整成功。

SyncNet始终使用官方预处理与权重；参考 `vsr_ditto50_linkage.py::_run_syncnet_cell` 调用官方run_pipeline并按固定最长track规则得到crop，之后通过 `tts_native_gain_attribution/syncnet.py::SyncNetEngine` 提取视觉/音频和矩阵。不要把static_image_bridge的另一种固定box crop协议冒充历史官方评分。保存crop的原始frame_indices、音频起点、distance matrix与完整支持，转换成原视频时钟后求共同窗；不同视频crop起点不能直接用相同行号作为同一时间。固定时间对比用N、N_ID、T_NAT的共同可评分窗，但不改变传入PCM或视频；每个新视频先提取完整数据，再求共同窗。官方全矩阵行q至少需要完整5帧视觉窗与所有±15帧音频候选有效；只保留这些行，按时间先平均距离，再C=median_lag(mean_distance)-min_lag(mean_distance)。至少50行方可作为该record主Sync-C；少于50保留但不可测，不降低门槛。官方全片C与共同窗C分别列名，主R_C使用共同窗C。原生T和N时钟不同，原生ΔC用各自完整有效内部窗，并报告数量差；不把等窗数量称为发音对齐。

### 6.8 统计、结论与自审循环

按source group先对record收益求均值，再对group等权求均值；本批预计一source一record，仍不能把帧/事件当独立样本。20,000次配对group bootstrap，保存draw索引供独立重算；所有主收益只使用各自两臂都可测的同一交集，同时报告12条固定队列的缺失原因。

三项主检验BV、R_V、R_C统一Bonferroni，双侧98.333333% percentile CI；各阶段未执行的检验仍计入3项，不事后改alpha。95% CI用于描述/预定baseline门；序贯门控和历史数据使整项研究是探索性，不能借校正区间宣称独立验证。Spearman和置换只做描述，不因相关不显著断言两机制独立。

必须生成：校准每条的预期/测得时间偏移图；12条BV与ΔC散点；每条事件时间轴（真人/N/T），低谷失配和缺失明确显示；若C运行，五臂视觉误差与Sync-C配对图。绝不只挑最好视频展示。

最终报告先回答：指标是否校准？历史增益是否伴随视觉时间代理改善？本地Wav2Lip原生效应是否复现？迁移是否通过？每项写固定分母、可测分母、效应、区间、失败原因和解释边界。声学时间调整损害原生收益，只能说明该处理未保留收益；因算法伪影仍可能存在，不能宣称“天然时间安排是唯一原因”。

审阅顺序：实现者逐项对照本spec → 独立checker从落盘数据复算 → 修复工程错误 → 仅重跑受影响缓存 → 再checker。科学阴性/支持不足不属于待修复错误。写BM结果节点并链接本spec与核心枢纽，按实际阶段记录，不写进度日志。

## 7. Expected Change Surface

### Must change

- 新增第3节四个脚本。
- 新增 `tests/experiments/test_tts_visual_timing.py` 与 `tests/experiments/test_check_tts_visual_timing.py`。
- 新run产物；BM `Experiments/`正式结果节点；当前spec状态与核心枢纽结果链接。

### May change

- `HANDOFF.md`仅追加/改写当前快照指针，避免重复整份报告。
- 若检测接口确实无法暴露必要有效性/PTS，可在新worker局部封装官方MediaPipe，不为本实验大改旧visual模块。
- 专用Rubber Band前缀，只有C运行才安装。

### Should not change

第三方模型代码/权重；原有四类实验判据；历史runs、spec与测试；主流水线与config.yaml；sealed数据；全局Python环境；旧CMLR词表/预处理。不要复制通用stage框架、添加数据库或任务调度系统。

## 8. Validation Plan

先进行针对性静态检查：

```bash
python -m py_compile scripts/experiments/tts_visual_timing.py scripts/experiments/tts_visual_timing_metrics.py scripts/experiments/check_tts_visual_timing.py
[redacted-local-path] -m py_compile scripts/experiments/tts_visual_timing_worker.py
pytest -q tests/experiments/test_tts_visual_timing.py tests/experiments/test_check_tts_visual_timing.py
```

必须覆盖有科学意义的行为：

- 同一几何在不同画幅比例/分辨率下像素校正一致；点13失效、无效零值、static/LOW_MOTION不会成为“好闭嘴”；flat生成视频算240ms惩罚。
- 已知事件集合：identity为0、+80ms为80ms、漏事件与多事件被罚、重复事件一对一、两边空拒绝；DP独立穷举小例验证。
- 局部帧映射方向、±2/±4帧平台、端点、单调、真正重编码后的重复；测试须验证已知注入位移而不仅检查数组shape。
- 重复词不能跨词匹配；silence不当phone；time-map正反方向、重复knots冲突、half-up取整、长度差1sample与2sample分支。
- 生成视频事件缺失不能降低reference分母；MFA/landmark缺失与低动作分开；不跨unsupported gap匹配。
- 固定seed下bootstrap独立复算一致；使用group而非frame；三主检验alpha不随skip变化。
- 音轨替换后帧RGB/PTS一致，视觉事件相同；同PCM、不同容器header不能误判内容不同；评分音轨必须绑定原N PCM。
- 门控fixture包含A失败、B无支持、Wav2Lip原生未复现、ID不等效、C全通过、缺cell/输入变更/非法resume。手改一个事件、score、输入hash或判决后checker必须报错。

下游最终操作命令（必须真正实现CLI后运行）：

```bash
python scripts/experiments/tts_visual_timing.py --run-dir runs/tts_visual_timing_smoke --stage all --smoke
python scripts/experiments/tts_visual_timing.py --run-dir runs/tts_visual_timing_v1 --stage all
python scripts/experiments/check_tts_visual_timing.py --run-dir runs/tts_visual_timing_v1
```

资源中断后使用同命令加 `--resume`，仅补缺失cell。成功证据包括已运行阶段的原始landmarks/PTS/事件/音频映射/矩阵以及独立复算，不只检查JSON count。checker可以复用I/O库，但不复用主指标实现。检查结束生成可供审阅的`validation.json`；错误必须非零退出，科学阴性且证据完整可零退出。

## 9. Risks and Edge Cases

- 真人嘴型只是同内容的参照，TTS可能有另一种合法协同发音；因此native是speech-phase下的代理一致性，而非准确率真值。C在原始自然时钟上证据更直接，但仍只评价开合事件。
- 已知warp恢复属于测量响应校准，不等于证明语义正确；不能在结论里跳过这一层。
- 低谷归一化可能放大小幅动作；用固定绝对动态范围门、额外幅度图和LOW_MOTION惩罚控制，仍不能评价全部视觉质量。
- 高头部旋转、遮挡、多脸可能令landmarks不可靠；每视频存首中末预览和失败统计。自动多脸检测发现任一帧多脸时记该视频不可测，不人工挑脸继续。
- 历史mp4音频可能经过AAC；驱动WAV与评分音轨分别记来源，不能要求AAC解码后逐字节等同原WAV。已有father binding不足则无法宣称密码学级历史来源证明。
- 本地只保证Wav2Lip可用；C复现失败是有效信息，不请求下游不断换模型直到阳性。
- Rubber Band处理可能改变瞬态、音质及MFA输出，ID等效不能完全排除非均匀warp伪影。不得把候选失败当时间机制的因果否证。
- small-n与历史数据：报告效应和缺失，不宣称跨说话人/跨模型泛化。门槛是本次决策规则，不是假装已有功效分析。

## 10. Assumptions / Unknowns

- VERIFIED: 中文VSR已有47对五视图存量，旧校准未通过；英文不能直接用该CMLR模型。
- VERIFIED: 151–162原始N/T音频绑定和24个A TextGrid存在；它们与B的T1/T2不同。真实视频和静态portrait在父输入有明确指针。
- VERIFIED: 既有video_features保存完整478点、时间序列和mouth_open；现有Wav2Lip render_worker支持静态图和固定box。
- VERIFIED: 4个校准source与12个主source不同；从历史22记录按原序排除后确定。
- VERIFIED: 编写时系统PATH没有rubberband，系统ffmpeg没有rubberband filter；C阶段有小型CLI依赖需要恢复，不能假设已经部署。
- LIKELY: 本地MediaPipe task及autoavsr环境可复用；必须复验asset hash及实际加载，不从历史“完成”状态推断当前文件完好。
- UNKNOWN: 新事件指标能否通过4条校准、12条有多少可支持reference事件；由A与audit回答，不能临时换指标。
- UNKNOWN: MFA可否给出足够覆盖的双向映射、时变stretch操纵是否有效；由固定QC回答。
- UNKNOWN: 历史LeapTalk视觉优势以及新Wav2Lip原生优势；不得将这些待测假设写进输入筛选。

## 11. Handoff Contract

沿第3节锚点实现四个薄脚本，遵守全部不变量与固定分支，保持第7节改动边界。先完成输入核验、纯函数测试和smoke，再自动运行能被科学门放行的全部内容，独立复算，自审修复工程错误，最后回写BM。不要只交付计划、模型加载成功或synthetic fixture。

证据与spec冲突时，停下依赖该假设的阶段，明确矛盾并继续不依赖它的工作。禁止改阈值、替换cohort、同名音频猜来源、换模型或降低校准门来把实验跑成阳性。科学阴性是结果，依赖阻塞是阻塞，未运行的C不能记完成。

### Observations

- [status] spec_ready
- [question] 原生TTS同步评分增益是否伴随独立嘴部动作时间代理的改善，且能否迁移到自然音频时间轴？
- [decision] 复用旧中文VSR内容证据；以LRS3纯视觉事件校准和有条件的Wav2Lip生成检验补充时间证据。
- [constraint] native依赖真人参考的音频时间映射，不能当TTS视觉真值；失败分支必须保留。

### Relations

- extends [[TTS 增强 TFG 的原因：证据、排除项与因果分叉]]
- follows [[Ditto-50 VSR 与 Sync-C 增益关联验证结果 2026-09-18]]
- follows [[VSR 视觉内容辨识与 TTS 配对验证实施部署 Spec]]
- relates_to [[LRS3 真实视频局部时间敏感性诊断]]
- relates_to [[LRS3 Wav2Lip 局部时间传递诊断结果]]
- relates_to [[TTS 时间辨识与同文本实例交叉实验 2026-09-17 结果]]
- relates_to [[Ditto TTS 局部时间辨识与共同语速控制结果 2026-09-20]]


## Implementation Result

- status: implemented；入口为 `scripts/experiments/tts_visual_timing.py`，纯指标为 `tts_visual_timing_metrics.py`，视觉/媒体 worker、独立 checker、SyncNet worker 和 `SYNTHETIC_NOT_SCIENCE` 门控 fixture 已落地；对应测试 11 passed。
- smoke 与正式队列均已执行。正式队列 audit 完整锁定 12 个主样本、4 个固定校准样本；28 个校准 cell、原始 478 点/valid/PTS、预览和事件文件均在 run 目录，独立 checker 为 PASS。
- 正式科学门结果：4 条校准记录仅 1/4 通过，规则要求 3/4，因此 calibration 为 `PARTIAL/NO_CLEAR_SUPPORT/NOT_CALIBRATED`；native 与 generation 按预注册规则 `SKIPPED_BY_GATE/NOT_TESTED`，没有降低阈值或把跳过写成完成。
- 结果节点：[[TTS 嘴部动作时间校准与自然时间轴迁移实验结果 2026-09-20]]。
- Changelog: September 20, 2026 — implemented, smoke/formal run and independent validation completed; calibration gate stopped downstream science.


后续修复入口（September 20, 2026）：[[TTS 视觉时间校准阻塞修复 Spec]]。只读复查发现中央支持不足之外，尚有静音伪词层缺失、逐 phone 侵蚀支持、LOW_MOTION 状态处理及校准独立复算缺口。新协议 tts_visual_timing_v2 为 planned；v1 的 implemented 只描述已有实现，不表示其全部科学契约已经正确满足。旧结果不覆盖。
