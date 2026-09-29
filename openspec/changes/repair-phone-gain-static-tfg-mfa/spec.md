# LRS3 音素增强静态 TFG 实验修复 Implementation Spec

## 1. Objective

修复 `phone_gain_static_tfg_mfa_v1` 的生成、测量、选型和验收偏差，使下游能够可靠回答：冻结 DIRECT 是否提高自然音轨上的静态 Wav2Lip 同步；其音素可分度是否达到 TTS；训练和推理均使用 MFA 的固定增强器是否有效。

本文件是修复合同，状态 `READY_FOR_IMPLEMENTATION`，2026-09-22。本次只做源码/产物审查、CPU 反例验证和 spec 编写；未修改实验代码、未启动新 GPU 实验。原实现合同见 BM《LRS3 音素增强的静态肖像 TFG 验证与 MFA 条件增强 Implementation Spec》。本合同明确的纠偏优先，其余科学定义继承原合同。

旧 run（以下称 OLD）：`runs/phone_gain_static_tfg_mfa_lrs3_phone_gain_static_tfg_mfa_v1_20260922b`。所有旧媒体、音频、模型与分数只读保留。旧 560 个官方分数确实由官方评价器产生，但其输入视频生成脸框错误；这些数值只能描述该错误生成配置，不能据此断言规则增强没有 TFG 效应。

采用两条明确隔离的执行路径，先 A 后 B：

- **A / frozen_audio_remeasurement**：保持 OLD 的 N/T/D/A/B/C 音频逐样本不变，只纠正生成/评分/统计，先恢复可解释的 TFG 测量。旧 A/B/C 标 `LEGACY_SELECTION`，不宣称恢复了原训练合同。DIRECT 的效果不依赖新训练完成。
- **B / protocol_repair**：修正支持、训练、DEV/seed、质量和缺失分支，重新训练/选型/推理，完成原协议所要求的实验。即使 A 阴性仍执行 B；B 不得以 A 的 E/TFG 分数调参。

两者使用不同 `protocol_id`、run 和结果表：`phone_gain_static_tfg_mfa_remeasure_v2`、`phone_gain_static_tfg_mfa_repair_v2`。修复不能把已看过的 E_SEEN 重新命名为盲测，也不能保证阳性。

## 2. Repository Model

令 P=`scripts/experiments/phone_gain_static_tfg_mfa/`，S=`scripts/experiments/static_image_bridge/`。

现有链路：P/run.py → assets 导入父 PCM/tokens/肖像 → support → calibrate → train → lock → infer → quality/phone → render → score → analyze/report。官方复评由 P/official_score.py 独立执行，尚未纳入主状态机。render_worker 调用 S/render_worker.py；后者将 generation box resize 到 96×96 后输入 Wav2Lip。官方 run_pipeline 自行检测生成视频中的脸，不使用 P 的固定 score_box。

审查发现如下；“已证实”指源码/实际产物或 CPU 反例支持，不表示已量化其对所有样本的影响。

| ID / 优先级 | 已证实的问题与证据 | 影响与重做范围 |
|---|---|---|
| F01 / P0 | assets::_portrait_meta 把整张 512×512 图设为 generation_box，score_box 同为 full-frame。原 spec 明确要求静态 PNG 检测框。 | OLD 全部视频及两个评分分支需重做。不是换 evaluator 就能修复。 |
| F02 / P0 | stage_calibrate 只执行首句 STFT identity 和条件词表统计；OLD/01_calibration 仅两个 JSON。无 SyncNet repeat/delay、MFA 全局/局部校准、processor parity 实际验收。 | 必须先建立仪器有效性；没有校准不能将分数/时序 gate 当科学结论。 |
| F03 / P0 | score_candidate_set 对 natural_primary 的 T 使用自然 spans，却查 tts_token_id；stage_phone 还对该视图计算 T−N。 | OLD natural_primary 的 TTS 对比无效；T 仅能在 matched_nt 用自身时间轴评分。不能把 −0.506553 解释为 TTS 音素能力崩溃。 |
| F04 / P0 | natural_primary 为两 encoder 复制同一 token 列表，20655 entries/encoder，0 个保存 frame_indices；未先冻结实际有效 core/共同 occurrence；评分也未按 pair_eligible 排除基线不合格句。 | 支持需重建并对所有臂重评分；基线无帧与候选失败不得混为一谈。当前数据 pair_eligible=false 为0，资格漏洞尚未在此批触发。 |
| F05 / P0 | stage_score 缓存只检查 sidecar/batch manifest 存在；official_score::evaluate 缓存只检查 COMPLETE、C 非空和 media 存在，随后还会重写 protocol。 | 换脸框、权重、PCM 或代码后可能复用旧分数；单阶段入口绕过 _load_protocol 检查。必须修复后再跑。 |
| F06 / P0 | checker 仅检查 JSON/manifest/肖像及两个 summary 是否存在；CPU 反例仅一条 N/N 就使 analyze_tfg_scores 返回 COMPLETE；两个 INCONCLUSIVE 经 interpret 变成 NEITHER_CLEARLY_UP。 | 已报告 checker PASS 不代表实验完整/正确；缺失不得写成阴性。需要独立复算和预定 cell 集验证。 |
| F07 / P1 | DEV 使用65句而非父 pilot24；跨两个seed一起选型。实际 A/B=20260923/175、C=20260922/300，原合同发布seed固定20260922。 | B 必须每seed独立选 checkpoint，固定发布seed；A保留旧选择作审计，不回填原协议。 |
| F08 / P1 | _select_dev_candidate 将残差0经 `or 1.0` 当成1；CPU等分反例选择step25而不是零残差step0；排序另加了coverage/seed优先项。 | 精确恢复 margin、accuracy、残差、step 的顺序；step0是合法候选。 |
| F09 / P1 | train/infer 没有FIT 24带mean/std标准化；只取 protected，再用反布尔mask，丢弃原5ms taper；词表取 probe labels，而非FIT自然labels；SIL/UNK只初始化零、没有冻结。 | B 训练输入/renderer行为不符合合同，需重训；不能仅重选旧checkpoint并声称合规。 |
| F10 / P1 | _balanced_order 仅生成一次并循环；checkpoint无CUDA RNG/epoch游标/累计墙钟；resume只找step文件且不核对训练绑定；stage_train无条件把summary写COMPLETE。 | 重试预算、恢复一致性、公平更新数和状态需修复。无有效phone目标不能以零loss计一次训练。 |
| F11 / P1 | stage_quality错误地把DIRECT当作可能异长比较输入，T0/MFA只跑A/B/C；ASR被硬编码DEPENDENCY_BLOCKED，没有真实runner；质量总状态未纳入ASR。 | D必须是N-clock并接受同等质量审计；依赖已安装也无法改变现ASR占位。 |
| F12 / P1 | monotonic_phone_match的edit_rate=1−matched/source，插入不计。CPU `[a,b]→[a,x,b]` 返回0。MFA缓存仅锁模型名称/launcher文件，不锁实际词典/声学权重/版本，存在空phone tier仍COMPLETE的路径。 | 重算T1，不沿用65/120作最终时序结论；MFA校准及有效性另判。 |
| F13 / P1 | OLD 7条T0失败均无新增饱和；ratio=0.010000281～0.010001938，略超0.01，SNR略低20。浮点投影之后没有PCM域再次约束。 | 高度疑似量化越界，须用实际PCM确认。B导出修复，A不能偷偷改旧音频或放宽门槛。 |
| F14 / P1 | phone仅accuracy/95%CI，缺D−T正式比较和98.75%CI/非劣结论；phone gate缺C−B和margin；TFG缺D_anchor，C−A仅C统计；总体取C，掩盖D可单独有效。 | 重建逐对比状态、原定多重比较与护栏；官方/固定窗两种口径分开。 |
| F15 / P1 | 只渲染肖像3；第二seed音频/稳健性TFG、肖像6/9、条件消融、ABX、盲听和播放包未接通；仍有多个阶段COMPLETE。 | 预定分支必须运行或如实PARTIAL；不能把函数存在当执行证据。 |
| F16 / P1 | render校验只检查文件存在与自述frame0；未对真实文件访问/ROI外像素做程序验收。official key/join不含portrait/seed，依赖音频路径反推身份。 | 尚未发现实际读取LRS3视频，但防泄漏证据不足；新增肖像/seed会冲突，需语义cell key和访问审计。 |
| F17 / P2 | 官方parser取首条正数匹配，无track多解拒绝；official入口无GPU资源门；静态worker累积全部帧再拼大bytes；协议只hash P/*.py，漏实际共享前端。 | 补track/参数/依赖锁、逐阶段资源门、流式编码；保存帧级一致性证据。 |

脸框单变量证据（前一轮诊断，本轮复核日志）：pair=`lrs3_6SdtkXAQq3k_00008`，肖像3，N/N，同一音频、Wav2Lip权重、官方SyncNet；整图框 C=0.521、D=11.246、offset=−1；仅生成框改为历史 `[138,90,357,387]` 后 C=7.444、D=6.921、offset=−2。旧日志在 OLD/10_official_syncnet/logs/lrs3_6SdtkXAQq3k_00008__V_N__A_N.syncnet.log；反事实日志/sidecar 在 `/tmp/sync_diag_tight.kPKDtf/`。实现首步复制并hash保存小型证据；临时文件消失时标诊断历史记录，不伪造新重放。一个样本支持脸框因果诊断，不支持全队列新效应值。

本轮CPU检查：现有 `pytest -q tests/experiments/phone_gain_static_tfg_mfa/test_core.py` 为 **7 passed**。F06/F08/F12 的反例仍可触发，说明现有测试不足。

## 3. Code Anchors

| path / symbol | 当前职责 | 必须变化 |
|---|---|---|
| P/assets.py::_portrait_meta, import_parent_assets | PCM/肖像/manifest导入 | 静态检测及冻结框；source group集合互斥、实际count/hash、tokens合法性；分层白名单与独立工序manifest |
| P/config.py::protocol_snapshot, ensure_protocol_unchanged, resource_snapshot, resource_decision | 协议/资源 | 支持新协议ID，完整依赖hash；所有入口锁验证；进程查询失败标UNKNOWN，按GPU UUID过滤；预算含剩余媒体 |
| P/run.py::run, STAGES, _set_state | 编排 | 显式DAG、预计stage/cell清单、official_score集成；A/B scope；不以文件存在/交集计数声明完整 |
| P/render.py::render_cell; P/render_worker.py::validate_request, render_request | 请求/子进程边界 | 强输入hash/box/schema/semantic key；request_hash不再冒充sidecar hash；响应重新验证 |
| S/images.py::select_detection, generation_box, score_box; S/detect_worker.py::main | 已有静态检测/几何纯函数 | 通过P薄adapter复用，不用会从视频抽帧的prepare_images；参数显式冻结 |
| P中新建static_render_worker.py | 流式静态生成 | 仿S/render_worker::main的相同前向，流式FFV1；原共享worker不改语义 |
| P/official_score.py::load_cells, safe_key, parse_score, evaluate, strict_mux | 官方评价适配 | key含portrait/seed/scope；cache绑定/track验证/完整退出码；保持官方脚本及媒体策略 |
| P/score_worker.py, batch_score_worker.py; P/sync_support.py::frozen_support_windows, score_distance_matrix | 固定crop/矩阵/W | 评分前freeze W，保存A/V embeddings及hash，计算D_anchor；只算指定14 cells |
| P/support.py::freeze_phone_support, entries_for; P/phone_eval.py::score_waveform, score_candidate_set, group_effects, summarize_pairwise | 探针支持/聚合 | 实际frame indices/自然有效集/正确T时间轴；全group token聚合，matched D−T、margin、ABX |
| P/conditioning.py::build_conditioning, PhoneVocabulary, jitter_boundaries | 条件张量 | FIT词表/归一化引用、边界/PCM支持/OOV校验；消融仅改条件，不改评估答案 |
| P/model.py::build_model | TCN | SIL/UNK输出embedding永久为0，非法ID拒绝而非clamp；参数形状/通道仍保持公平 |
| P/train.py::make_training_sample, loss_for_sample, train_arm, _save_checkpoint | 训练 | 保留soft edit mask、FIT特征统计、固定slots；epoch/RNG/预算/optimizer-boundary恢复 |
| P/run.py::_dev_samples, _aggregate_dev_scores, _select_dev_candidate, stage_lock; P/infer.py::infer_pcm | 锁定/前向 | pilot24、per-seed锁、真实残差0、只读纯前向、完整checkpoint输入合同 |
| P中新建audio_contract.py | 本实验音频导出adapter | 单一float mask/保护/量化安全PCM导出，训练/DEV/E调用一致；不改变旧parent renderer默认行为 |
| P/mfa.py::mfa_batch_attempt, parse_textgrid; P/quality.py::pcm_contract, monotonic_phone_match, audit_timing | MFA/质量 | 实际资源锁、控制校准、插入/删除/替换编辑率、identity显式状态、D质量 |
| P中新建asr.py；P/run.py::stage_quality | 内容评价 | 固定faster-whisper实际推理和group WER；缺资源自动按授权代理下载并锁定 |
| P/analyze.py::analyze_tfg_scores, classify_phone, interpret; P/report.py::write_report | 统计/结论 | 每个contrast的完整性、校准、科学/工程状态；官方与固定窗并列报告 |
| P/check.py::check_run, main | 独立验收 | 读PCM/embedding/锁/ledger独立重算、mutation测试、验证完整性不验证阳性 |
| tests/experiments/phone_gain_static_tfg_mfa/ | 当前7个基础测试 | 按第8节新增针对实际失败的回归与集成测试 |

## 4. Reference Pattern

- 复用 S/images.py 的 `select_detection`（阈值0.9，score降序、x/y定序）、`generation_box`（floor/ceil、底部pad10）、`score_box`（1.5倍正方形、越界补零）。只在3/6/9 PNG上调用一次检测；禁止调用其中从原视频读取第一帧的prepare_images。历史固定 `[138,90,357,387]` 仅作脸框诊断重放夹具，不按E Sync-C搜索/选择生产框。
- 模仿 S/render_worker.py 的 mel切块、96×96 BGR输入、mask和写回，改为本包流式编码，并验证解码帧像素相同。25fps/FFV1/无音轨master不变。
- 复用 P/official_score.py::strict_mux 的完整PCM逐字节核验、视频stream copy和官方cwd；不要复制其缓存检查和首track parser。
- 复用 `phone_separability_enhancement/audio.py::make_protected_mask` 的 **mask与protected两者**，保留guard/taper。复用 metrics 的cosine分类与label等权定义，不复制其invalid-vector仍算valid_count的漏洞；在本包adapter独立计数。
- 复用 S/frontal_probe.py 的固定窗/延迟控制思路和原spec的明确阈值，不继承硬编码cohort或共享分析函数做“独立验收”。没有可直接替代本轮全部修复的现成runner。

## 5. Invariants

1. 保留 FIT135/12组、DEV65/6组、E_SEEN40/40组；DEV选型仅父pilot24。保存具体ID，不基于新结果补样。source_group不是已确认speaker。A/B使用同一E账本。
2. A的所有驱动/评价PCM与OLD逐字节一致，B更改PCM必须新hash/新路径。N/T/D原始资产始终只读，D失败不偷偷改成新优化结果。T保持自身时长和MFA。
3. audio train/infer无LRS3视频；TFG只接收外部PNG+指定PCM，不接收MFA/TTS目标/评分音轨；官方scorer只读生成视频。C训练/推理均使用N+固定转写得到的MFA，明确text-assisted。
4. A/B/C所有原定网络、loss、400 updates上限、每seed两小时预算、学习率和失真阈值保持原spec；不借修复提高容量/调阈值。修复输入语义后新训练，不能resume旧训练状态。
5. geometry、measurement、model-selection三个身份分开版本化；改变任一依赖使其下游缓存失效。旧锁不覆写；所有partial写新attempt且原子提交。
6. 主要replacement对比固定 `V(E),A(N)−V(N),A(N)`；T/T仅native描述性。natural_primary绝不含T。音素非劣/等效只用matched_nt与两encoder。
7. 官方端到端与固定窗scorer各有名称、hash、支持/裁剪/offset约定。二者绝对C不用一致，也不允许事后选择较有利scorer作为“主结果”。
8. COMPLETE表示该scope的预定执行/验收齐全，PASS表示工程合同成立，GAIN表示科学门通过。缺失、未校准、实现错误、阴性分别建状态。
9. Sync-C展示3位小数；不存在全样本C必须≥5的验收标准。异常应由重放/对照诊断，不能按低分删样本或调脸框。
10. 每次可能占GPU的检测/SSL/训练/渲染/评分/ASR前检查进程与资源，保留用户已授权的gnome进程allowlist。不得杀别人进程或删除旧数据腾空间。

## 6. Implementation Plan

### 6.1 先建审计账本和新run身份

新增两份配置 `scripts/configs/phone_gain_static_tfg_mfa_{remeasure,repair}_v2.yaml`，共同引用本spec sha256、旧config/lock/registry/audio manifest及实际PCM hash。为每条F01–F17写 `audit_findings.json`（证据path/hash、影响阶段、remedy、验证目标、status）。复制小型诊断日志与tight_result至新run `00_protocol/diagnostic_evidence/`，不依赖/tmp长期存在，不复制数百张临时帧。

新run有 `expected_artifacts.json`：scope、每个branch的阶段依赖、完整pair/portrait/seed/arm/cell键、合格性与失败原因字段。主40句不得被音频manifest交集悄然缩小。索引join使用明确ID+hash，重复键即错误；相同PCM hash不等于同一arm。

`protocol_snapshot`显式绑定本包与实际执行的共享renderer/teacher/mask/metrics、Wav2Lip前端/权重、S3FD权重、SyncNet前端/权重、ffmpeg版本、模型revision及环境版本。不能仅依赖git commit（工作树有大量未提交文件）。所有stage，包括单独 `--stage score` 和official独立入口，先验证锁。

缓存键至少含scope、pair、source_group、portrait、seed、video_arm、audio_arm、PCM/PNG pixels、generation_box、score_geometry/evaluator、模型/代码/配置/support hashes。重用时重新hash真实输入输出及sidecar；不匹配拒绝旧cell并建新attempt。`--limit`必须标SMOKE/PARTIAL_SCOPE，不等于全队列COMPLETE。科学JSON禁止NaN被 `_json_safe` 静默转null；无定义数值显式reason+null。

### 6.2 恢复静态几何与真实媒体合同

GPU preflight后用 S/detect_worker 对外部PNG 3/6/9一次性检测，参数阈值0.9、bottom pad10冻结。检测器/输入hash、全部候选、选择规则、bbox、generation_box、fixed_score_box与预览图入锁；检测失败阻断该肖像，不fallback整图。若多个高分脸导致身份歧义则记阻断，不看Sync-C挑脸。静态PNG provenance未知字段写UNKNOWN，不从当前LRS3抽帧补充。

worker在模型加载前验证PNG magic/单帧/RGB hash、mono16k PCM16/hash、合法box、checkpoint hash、mode及白名单。生成框必须与锁中的detector计算结果一致；同肖像各臂用完全相同框。固定score_box另行计算，禁止直接复用generation_box替代评分crop。

使用P/static_render_worker流式送ffmpeg，记录输入视觉tensor hash、frame0重复、mel前端/chunk数、decoded frame hash。独立check读取每帧验证ROI外像素等于PNG、fps=25、frame数等于音频mel实际chunk数、master无音轨。smoke做文件访问审计，只允许指定PNG/WAV、模型和依赖；将目标视频路径置为不可读仍可生成。不要以request里没有video字段代替访问审计。

### 6.3 校准后执行 A：固定音频纠正测量

冻结FIT四个不同source，按原spec固定hash选取，三个肖像各生成N及独立N_REPEAT。同V(N)配实际PCM ±3200 samples移位，保持长度、零填充；W_delay排除填零及前端边界。固定窗重复C/D/D_anchor误差≤1e−4、lag一致；每肖像≥3/4检出±5帧（容差1帧），自然锚距离变差≥0.10。官方端到端也跑重复/移位，记录官方offset符号与内部k映射，不能假定两者同号。官方日志只打印三位小数时按0.001精度检查重复，不能假装已有1e−4精度。

另外固定重放一个历史官方正常评分媒体，锁定媒体/模型/脚本/CLI参数，要求C/D与已锁日志在打印精度内一致，offset一致；缺历史媒体时显式NOT_AVAILABLE，仍必须完成本轮FIT真实控制。前述tight样本重放是诊断，不作为挑选E的依据。不要求任意样本C≥5。

校准通过后，A对40句×6臂×portrait3重生成240个视频；每句14 cells，官方与固定窗各560项。保留完整交叉解释：N/N、E/N、N/E、E/E（E=D/A/B/C）以及T/T。现有接口产出36个raw组合属多余，不再把未对齐T/N组合当endpoint。

固定窗：依据PCM长度、实际mel切块前端和5帧/MFCC真实支持，在候选embedding前锁W_N和W_T，支持需所有lag有效；半开边界最大end≤L，验证off-by-one、短音频和末端重复chunk。W_N≥20为预先资格，失败不补样。N/D/A/B/C样本数必须相同；候选异常是candidate failure，不能让其改变baseline eligibility。内部query NaN为失败，不缩W。

官方：保持run_pipeline.py→run_syncnet.py完整链路、cwd和min_track=25、冻结其他有效默认参数，完整PCM无损mux；不加-shortest、不拉伸/校时。保存官方实际内部audio样本数、track帧数/边界、offset、crop和日志。官方内置`-async 1`可能改变内部PCM，需报告，不伪称预mux exact意味着内部也exact。它目前不是低C的首要原因（同mux紧脸框重放恢复分数）。official结果无法等同固定W的endpoint。

parser要求每条score绑定唯一track，C/D/offset全部finite、完整同组；零track失败、多track标MULTI_TRACK_UNRESOLVED，不取首条/最高C。支持有符号/科学记数法，不静默漏offset。边界±15全部保留并标记，不能扩大lag挑高分。

A只在本scope完成时写REMEASUREMENT_COMPLETE，并列旧/新C/D、paired delta、完整性和校准结果；不写原训练协议已完成。N/T/D可以先产出，不能因B未训完阻断D的诊断。A仍要对固定旧音频做D/A/B/C质量审计，失败不会取消TFG描述性测量。

### 6.4 重建音素测量与比较

先在固定FIT校准集验证每个SSL前端的processor parity：手工标准化相对锁定processor输入最大误差≤1e−4、cosine差≤1e−5；teacher.eval、参数无梯度，训练模式下waveform梯度可达增强器。zero-gain使用真实edit mask导出回读，PCM必须与N逐样本相同，不能用全保护mask掩盖renderer误差。校准失败阻断依赖它的phone/训练分支。相关receipt及前端源码hash入锁。

P/support在原始N/T上按真实encoder卷积几何计算core frame indices，冻结每encoder的有效N occurrence；主集合取两encoder可用occurrence交集，并以冻结label集合、≥5labels/≥10tokens/≥0.70覆盖资格判定。记录全部原tokens、frame indices、eligibility及排除原因。自然support不依赖T，matched_nt才要求N/T单调唯一匹配和两侧有效；保留legacy_matched独立重放。

接口必须拒绝 `arm=T, view=natural_primary`；调用者仅请求N/D/A/B/C。matched_nt：T只读自己的tokens与tts frame indices，其余沿N；不能仅把condition字符串换成natural规避错误。输出更换MFA仅用于质量，不改评分slots。

保存逐token预测/正确label margin、NPZ embeddings和support引用；候选缺失/无效向量accuracy=0、margin=−2留在分母，coverage按真实valid逐条计数；整arm缺失也不得从comparisons消失。先同group同label汇集跨句occurrences，再label等权，再group等权；不要平均各句label均值替代它。DEV和E同口径。

legacy重放目标仅适用于原2256 keys/39 groups及原口径：HuBERT N/T/D=0.649969/0.742155/0.848565，XLSR=0.562609/0.622082/0.644031（六位显示值只容许半个末位的舍入差；有父全精度值时比较≤1e−6）。新natural_primary不强行凑这些数。

accuracy与signed margin平行统计。主phone XLSR C−N/C−A/C−B用98.333333%CI、accuracy点差≥0.01且下界>0、margin同向；N/D也完整报告。matched D−T/C−T×两encoder用98.75%CI、ε=0.02，分别判非劣、等效、更高、INCONCLUSIVE，不以不显著充当等效。D/胡伯特优化暴露明确标注。

ABX在任何新候选前冻结N侧triplets，同label A/X必须不同occurrence，B异label；固定键用于所有N-clock臂；T只能用matched映射有效triplets，不能复制N时间。每组每label最多20，tie=0.5；缺triplets标NOT_ASSESSED。

### 6.5 B：修复训练输入、恢复和锁定

FIT自然tokens构建条件词表（SIL/UNK额外）；probe label集合保持历史锁，不要混淆两者。FIT自然帧估计24带log-power mean/std（下限1e−6）与duration stats；冻结后训练/DEV/E一致使用。tokens必须finite、不重叠、有序、位于[0,L/sr]内；超音频的STFT时刻给SIL/全0。OOV给UNK并计数，不clamp非法ID。SIL/UNK在embedding输出处强制零、梯度为零，测试optimizer步后仍为零。

Conditioning仅含model mode/身份/时间；真实监督slots与独立eval slots单独传入。保护对象同时含hard protected和float edit mask，5ms taper保持。A条件19通道零、B仅时间、C全部；三臂相同公共初始权重、网络形状、训练样本顺序和预算。无有效FIT loss目标预排除并记账，不用零loss计更新。

新audio_contract包裹renderer/PCM export：浮点训练投影照原定义，导出后在PCM域复查0.01/20dB/1dB/无新饱和/保护一致。若越界，沿原残差方向使用固定衰减网格alpha=1,0.99,...,0.01,0导出，取首个通过者；每步都量化、恢复保护并实算，alpha=0保证identity可返回，禁止根据phone/TFG挑alpha。记录投影前后PCM hash、alpha、失败原因及实际剂量。这个修复是B方法的一部分，对DEV候选也同样执行；A和冻结D不能原地修改。identity写snr=null、reason=IDENTITY；无editable区原样输出并写NO_EDITABLE_SUPPORT，不能伪造训练目标。

每epoch按seed+epoch重新生成source-group round-robin采样序列；保存epoch/order/hash、next cursor、CPU/CUDA/NumPy/Python RNG、optimizer、累计updates/microbatches/训练活跃墙钟、配置/词表/统计/support绑定。checkpoint只在optimizer边界原子提交；中断半个accumulation时恢复上一个完整边界重放，不跳过数据或丢梯度。last与step统一候选时间戳/updates校验，按最大完整update恢复；所有恢复累计同一两小时预算，不能每次重启再得两小时。COMPLETED job仍重新核验checkpoint及合同，不仅凭result.json。

父 `00_audit/pilot.json` 的24 DEV ID必须实际存在且subset DEV，缺失是输入合同冲突，不替换65句。每arm每seed在0、25、...、400及预算终止的完整last候选独立选型，要求全部DEV导出T0通过且固定support完整；排序依次margin降序、accuracy降序、DEV平均实际残差比升序、step升序，0就是0、不用truthiness。发布seed=20260922，20260923单列稳健性，禁止跨seed选赢家。step0获胜如实NO_LEARNED_IMPROVEMENT。

训练更改影响权重语义，B六个job从step0重训；旧六job仅作为A历史描述。保持每job400 updates/两小时原预算（GPU工程失败按原spec最多一次同配置重试），超时标BUDGET_LIMITED且报告真实公平程度。P/run先核验依赖校准，不盲目继续下一GPU阶段。

两个seed A/B/C全E输出封存；C_ID_PERM和C_BOUNDARY_JITTER在预定12句运行（hash选样、每occurrence含pair_id）；jitter只改真正相接speech边界，不能跨gap，真实evaluation labels/mask不变。保存条件差异和输出hash，不允许这些诊断参与选型。

### 6.6 完成质量/仪器分支

N/D/A/B/C全40句做T0；D与N不同长度直接违反DIRECT合同，保留失败、阻断该cell replacement分析，不改成native TTS。T仅原始参照。审计OLD七条越界的真实float/PCM差异；若没有float资产，只能写量化机制LIKELY，不能当已复现因果。

MFA缓存绑定实际exe版本/包、词典和声学模型文件hash、align参数/配置、PCM/转写、split/condition；解析phones tier必须非空、区间有效且覆盖合理，无spn-as-silence偷换。不同split/condition使用独立corpus，禁止旧失败result永久成为cache hit。

先按原spec固定FIT8句：N_REPEAT、真实+80ms移位、局部边界+640 samples重采样。全局repeat8/8、shift≥7/8，局部目标边界同向≥20ms至少6/8；使用唯一occurrence匹配定位目标，不按候选token数组同index猜边界。控制失败标UNCALIBRATED，保留描述数值，不给T1通过/失败科学判定。短/非相接区间不作为局部夹具。

候选T1用唯一单调匹配的边界统计；speech edit rate改为标准Levenshtein (S+D+I)/N，另报I/D/S和匹配coverage，重复label歧义显式不通过；长停顿保序匹配、新增/缺失失败。阈值继承原spec：edit≤.05、coverage≥.90、median≤20ms、p95≤40ms、长停顿IoU≥.5/edges≤20ms。每arm至少90%合格才给自动时序支持；不能把所有臂汇成一个率，也不能误要求100%。

接通固定faster-whisper base.en，先下载解析确切revision/hash（代理7890获用户授权），固定language=en/beam5/temp0/vad=False/condition_on_previous_text=False，无reference prompt。N/D/A/B/C/T逐句识别；相对同一自然参考算source内总edit/总词数再source等权WER。内容非劣：相对N点差≤.005且95%CI上界≤.01。无法下载/运行才DEPENDENCY_BLOCKED，不能硬编码。CPU可跑ASR则无需占GPU；选GPU则先资源门。

导出12句随机别名盲听文件/评分表，arm映射隔离到审计目录，不让blind manifest暴露arm或含arm的源路径；导出统一N音轨和各自音轨视频展示副本。未收集人工评分一直HUMAN_NOT_ASSESSED，自动实验可完成但不能宣称人工质量已证实。

### 6.7 B：补齐TFG、统计与报告

B复用已验证几何/仪器控制仅当完整依赖hash一致；候选视频全部新生成。主40句×6臂=240视频、14 cells/句；敏感性6/9两肖像×固定12句×6=144视频；第二seed C在固定12句/主肖像增加12视频及C2/N、N/C2、C2/C2，复用匹配N/N。所有key含seed/portrait，即使PCM相同也保持arm身份。每种scorer预期集显式列举，未完成不能宣称原协议全完成。

固定窗保存完整curve、D0、C/D/lag及N/N的k_N，候选D_anchor取其curve在同k_N的值。比较D−N、C−N、C−A均有C、D、D_anchor、lag护栏，不仅C−A算一个C差；C−A的D_anchor差为curve_A(k_N)−curve_C(k_N)，lag稳定为两cell lag差≤1的比例。所有主项分别判定，不以C主臂阴性遮住D阳性。

主TFG每项C点增益>0.05、98.333333%CI下界>0；D与D_anchor改善点≥0、95%下界≥−.05、lag稳定比例≥.90。固定窗支持完整可靠同步metric gate；官方完整评价独立报告同三项C family及D/offset，但其D_anchor/fixed-W不可由三行stdout补造。官方判定字段为OFFICIAL_C_GAIN/NO_CLEAR_GAIN/INCONCLUSIVE，不能冒称完整fixed-support gate。若两分支方向不同，报告SCORER_DEPENDENT并检查crop/support差异，不能挑一方。

每contrast在预定baseline eligible cohort上检查缺失；候选特有缺失→INCONCLUSIVE_MISSING，主有效组<30→INCONCLUSIVE_SMALL_COHORT，校准失败→INCONCLUSIVE_CALIBRATION。完整对子集估计单列条件性，不给失败值补0。任何缺失不得被interpret合并为NEITHER_CLEARLY_UP。统计按group聚合，多肖像不当独立n；同family共享已保存PCG64 seed20260922/20000 draws抽样表。native T/T和N/N固定窗描述另给自身时间轴十等分bin等权分析及有效窗/时长，官方全轨native口径单列，不称replacement。

报告分层：measurement_valid、execution_complete、protocol_compliant、phone_gain、TFG各contrast、T0/T1/ASR、human、practical_success。实用成功需T0全过/T1与ASR门过；质量失败不抹掉TFG描述。A/B分开，旧分数标INVALID_FOR_INTENDED_TFG_CONCLUSION，新结果出前不能写规则增强成功或失败。

### 6.8 独立checker和收口

checker可以复用hash/I/O，不导入runner的support/selection/metrics/analysis函数。直接从PCM/embeddings/冻结slots/curve重算抽样及主aggregate（小数据全算），验证模型选择、全部cell、媒体decode、依赖链与多重CI。验收应能抓住第8节故障注入，不能仅检查存在summary。

新增 `--require-complete`：FAIL/完整性未满足返回非零；审计旧run可生成有原因的PARTIAL/INVALID，不因缺失研究分支崩溃。顶层run根据明确必需stage集合判断，不能对“已有的stage”执行all()即完整。阴性但所有实验/校准完成可execution_complete=true；未校准即measurement_valid=false，不伪装阴性。

更新BM旧实验现状：保留旧数值/来源，撤回其针对正确生成链路的TFG阴性推论；新实验单实体链接A/B scope和本spec。不要覆写OLD的summary或改旧数字，不创建多份进度日志。

## 7. Expected Change Surface

### Must change

P中的assets/config/run/render/render_worker/official_score/score_worker/batch_score_worker/sync_support/support/phone_eval/conditioning/model/train/infer/mfa/quality/analyze/check/report；新增P/static_render_worker.py、audio_contract.py、asr.py；新增两份v2 YAML；tests/experiments/phone_gain_static_tfg_mfa下有意义的回归/集成测试；新run和BM纠错/结果指针。

### May change

新增P/calibration.py以隔离FIT仪器控制；新增P/cache.py用于通用绑定校验。确有阻断才小范围修共享纯helper，默认兼容并跑父包回归；优先本包adapter。依赖环境允许按本run锁安装缺失资源，记录版本。

### Should not change

OLD、父run、原N/T/D PCM、LRS3视频、旧checkpoint、旧v1配置、官方SyncNet脚本/权重、Wav2Lip权重、核心00–05、TTS providers、AGENTS/CONTEXT/HANDOFF及用户其他dirty changes。不得为了分数重选肖像、seed、E cohort、gain剂量或lag范围。

## 8. Validation Plan

按下面顺序执行。测试以实际行为/故障反例为准，不写“常量等于自己”的镜像测试。

1. 静态：`python -m compileall -q scripts/experiments/phone_gain_static_tfg_mfa`；`git diff --check -- scripts/experiments/phone_gain_static_tfg_mfa scripts/configs tests/experiments/phone_gain_static_tfg_mfa`。新增文件也显式检查空白和语法。
2. CPU定向：`pytest -q tests/experiments/phone_gain_static_tfg_mfa/`，至少新增如下测试组：

| 测试目标（建议文件） | 操作/反例 | 必须可观察结果 |
|---|---|---|
| test_geometry.py | 固定PNG检测返回紧框；无检测/非法框 | 推导generation/score框与冻结值一致；无检测阻断、不整图fallback；换box令render/score缓存失效 |
| test_cache.py | 同路径替换PCM/PNG/权重/前端；只保留旧COMPLETE receipt；传新参数单独stage | 拒绝旧缓存；新protocol不能覆写旧锁；损坏输出/少一sidecar不复用 |
| test_phone_support.py | 短phone无core、两个encoder不同帧、T比N短且timestamps不同 | 无基线有效帧不入主support；candidate失败留分母；T natural_primary请求拒绝；matched只读T自己slots |
| test_phone_metrics.py | 重复label不均衡跨句、cosine tie、invalid embedding、整个D缺失 | 精确group-label等权；tie按label顺序；coverage正确；D−T缺失为INCONCLUSIVE，非劣与等效CI分开 |
| test_selection.py | 同分step0残差0 vs step25残差.01；次seed更高；额外DEV样本 | 选step0；主seed始终20260922；只pilot24影响选择；checkpoint变更hash拒绝 |
| test_training_contract.py | 两epoch、梯度累积4、mid-job重启/超时、零有效phone | 每epoch顺序按锁更新；1update=4microbatch；恢复与不中断toy权重/数据序列一致；预算累计；零目标不假训练 |
| test_conditioning.py | SIL/UNK经过多个optimizer步骤、OOV、越界ID、STFT尾、置换/jitter | SIL/UNK始终零；非法ID拒绝；尾部SIL；A/B身份不影响输出；消融不变eval labels/mask |
| test_audio_contract.py | 带taper波形、±32768、全部保护、identity、量化后超.01夹具 | mask taper传递；保护PCM精确；identity显式null；PCM复投影严格过原阈值，旧A/D字节不变 |
| test_quality.py | `[a,b]→[a,x,b]`；重复label歧义；空phones；80ms/局部40ms实际波形控制 | 编辑率.5；歧义/空tier不PASS；校准缺失则T1 UNCALIBRATED；D被纳入质量 |
| test_asr.py | stub真实调用receipt、插删替、同group不同长度句、空reference | 所有六臂转写；按词数聚合group WER；无reference prompt；空参考NOT_SCORABLE；缺依赖不COMPLETE |
| test_sync_contract.py | 全lag边界、W内NaN、重复min tie、lag sign、两个scorer | W先锁不缩水；最小lag tie；D_anchor用N锚；官方与fixed分开，无法互换hash |
| test_official.py | 0/1/2 tracks、负数/科学记数、缺offset、已有同path旧media、limit模式 | 唯一完整track才score；不选最好track；cache严格；SMOKE不冒充全量 |
| test_analysis_states.py | 只有一条N/N；候选缺失；29组；C_ONLY；D阳性但C阴性；两边未评 | 不COMPLETE；INCONCLUSIVE传播；完整family护栏；按D/C分别判断，不归结“两者均无明确提升” |
| test_checker.py | 修改summary分数/support/model lock/ROI像素；删除必需分支但保留summary | 独立checker逐项失败；阴性完整结果可工程PASS；人为评分缺失不伪填 |

3. 跑相关共享回归：`pytest -q tests/experiments/static_image_natural_to_tts_bridge/test_geometry.py tests/experiments/phone_separability_enhancement/`；只有改共享代码才扩大到对应调用者测试。
4. CPU权限/媒体smoke：伪造目标视频读取即抛错，验证manifest稳定；真正GPU smoke前检查GPU进程、显存、RAM和磁盘。一个N、一个D完整生成→官方→固定窗→checker，并保存strace等真实访问证据、ROI像素、audio hash。
5. 真实FIT校准按6.3/6.6执行；通过后执行A 240视频/两个560评分及修复phone/质量审计。最低支持失败如实保留，不按观察结果换人。
6. B真实六job/选型/双seed音频/完整质量/phone/主与敏感性TFG/消融/报告；执行独立checker `--require-complete`。每个failure都进入预定ledger，不以“下游跑完了”替代验收。
7. 修复版CLI保持原 `python -m ...run --config <v2 yaml> --run-id <new-id> --stage ...` 形式；新增official_score阶段纳入all并可单跑。A配置的all不触发训练；B的all执行全部就绪分支直到完成或真实外部阻断。退出码0只表示请求scope真实完成。

本次spec审查阶段已完成的是源码核验、OLD小型产物核验、F06/F08/F12 CPU反例和原7项测试。上述新增测试和GPU验证属于下游验收，尚未执行。

## 9. Risks and Edge Cases

- 只把generation_box改紧但复用OLD video/embedding/score会让修复无效；必须更换run并做全依赖缓存验证。
- 官方动态crop可能随生成嘴形变化，固定crop则控制视觉几何；这是两个estimand的差异，不应强迫分数一致。官方重放保持历史端到端口径，固定窗提供受控机制诊断。
- TTS的文本/时长/支持与N不同，错误natural_primary T分数不是TTS退化证据；不能拿它解释phoneme→TFG机制。
- 原E已看过，补救选择仍探索性；不能声称回到从未查看结果的预注册状态。新脸框来自PNG检测/历史实现依据，不根据E分数优化。
- 字典/声学模型同名称不同内容、symlink换目标、零track成功退出、同一audio被两个seed重用，都会绕过当前弱检查；必须用真实hash和semantic key。
- 残差量化越界量很小也不等于可以篡改门槛；修复导出会改变PCM，必须只在B声明新方法。MFA时序失败也可能来自测量器，不可未经校准归因音频本身。
- B重训和官方crop产物可能耗大量空间；先profile最长FIT cell，用剩余cell预计产物×1.25+下一cell峰值+4GiB余量门，不复制36个无用cross media。不得删OLD。
- S3FD/no-face不应通过整图fallback“让流程继续”。对应肖像branch应显式blocked；其他独立分支仍可完成。
- 人工评分未提供不是自动实验假失败的理由，但“实用音质好/可懂度更高”不能仅靠SNR/探针/SyncNet宣布。

## 10. Assumptions / Unknowns

- VERIFIED：F01脸框问题、官方脚本实际执行及该样本0.521→7.444的诊断日志；F03/F05/F07/F09–F12/F14–F17源码偏差；F06/F08/F12的CPU反例。现有7测试不能检测这些缺陷。
- VERIFIED：OLD每encoder natural support无frame indices；本批pair_eligible全true。当前证据不说明存在实际LRS3视频泄漏，说明实际访问隔离验收缺失。
- LIKELY：OLD七条PCM质量失败由投影后量化造成；保存的float及再导出差异尚需重放证明。SIL embedding未冻结可能改变行为，大小需审计旧checkpoint，不能凭源码宣称它解释训练阴性。
- UNKNOWN：修复后的完整40句规则增强TFG收益、D相对TTS正式非劣/等效、B训练收益、三肖像稳健性、真实ASR及校准后T1、人类听感；不得提前写答案。
- UNKNOWN：静态肖像/预训练数据与LRS3的身份重合，独立speaker重合。只对当前禁止目标视频时序输入的合同给结论。
- ASSUMPTION FOR IMPLEMENTATION：本轮优先恢复冻结音频的正确测量，再补齐原方法，不额外搜索新增强器或扩大模型；这与用户“找修复点并写spec”的目标一致。本次交付不授权篡改原始数据。

## 11. Handoff Contract

下游按第3节锚点和第6节顺序实施，保持第5节不变量，复用第4节几何/前向/PCM参考模式，将补丁限制在第7节。先让F01–F06的错误无法再伪装为有效实验，再重测A；随后完整修复并运行B。不得只改脸框后沿用旧选型/统计/COMPLETE状态宣称原spec完成。

每个F-ID必须落到实际代码、失败前反例、通过后的测试/产物路径。交付包含审计账本、修复代码、测试、两个新scope的hash绑定产物、独立checker、分层结论和BM旧结论纠错。未完成分支如实PARTIAL/阻断；异常时继续可独立开展的安全工作，不用占位分数、调低门槛或借“无显著差异”证明无效。

若仓库新证据与本spec矛盾，暂停受影响分支并给出具体冲突/最小修订，不静默换协议。成功标准是实验可验证且解释边界正确，不是产生阳性结果。
