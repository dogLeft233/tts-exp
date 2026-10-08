---
title: MFA-linear 视频重定时 Embedding 代理搜索 v2 Implementation Spec
type: research_topic
permalink: tts-exp/research/mfa-linear-视频重定时-embedding-代理搜索-v2-implementation-spec
status: planned
implementation_status: NOT_IMPLEMENTED
scientific_status: NOT_RUN
protocol: mfa_linear_video_retiming_v2
question: 初始 embedding 代理能否以更少真实前向找到更同步的视频时间映射？
tags:
- mfa-linear
- video-retiming
- syncnet
- embedding-surrogate
- implementation-spec
cohort_size: 64
speaker_count: 8
---


## 1. Objective

实现协议 mfa_linear_video_retiming_v2：MFA-linear 音频 M 驱动 Wav2Lip 得到视频 V_M，保持配对自然音频 N 完全不变，通过视频时间重采样提高 V_R/N 的实际同步度。使用初始 SyncNet embedding 的时间插值作为廉价搜索代理；只有真实像素重采样后重新前向的分数，才能决定最终输出。

主要优化 no-offset 距离 D0；Sync-C、Sync-D、offset 和局部指标约束最终输出。C≥7.000 是描述性目标，不是工程完成条件，不允许为达到它改变评分口径。交付包括代码、测试、可复现代理回溯分析、单样本 smoke、冻结64条配对语音的大队列评估、独立检查与最终报告。

本 spec 是 v1 的明确协议修订：允许中间 C 下降；多起点、多路线搜索；代理和真实分数分离；最终先保证时间对齐和 C 非劣。首轮保持 v1 的映射族和模型，避免同时改变搜索算法、位移范围及插值器。更密 knots、超过±3帧、RIFE、独立 nearest 搜索和训练增强器不属于本轮。

### Observations

- [status] planned；implementation_status=NOT_IMPLEMENTED；scientific_status=NOT_RUN。
- [question] 初始 embedding 代理能否以更少真实前向找到更同步的 MFA-linear 视频时间映射？
- [evidence] v1 扩展 smoke：256个候选，16个 |offset|≤1，全部被中间 C/D 非劣门拒绝。global_seed_-3 的 C/D0/offset=5.374/7.642/0；identity=5.484/14.372/-3；最高真实搜索 C=6.177，但 offset=-2。
- [evidence] 本会话对同256候选做离线中心插值回溯：C Spearman=0.962018、D0 Spearman=0.978564、offset一致率=0.992188、C MAE=0.133661、C top10重合0/10。该临时分析尚无持久脚本，P1必须复现并落盘，不能把这些数当独立评估。
- [boundary] 发明人目前未提供代码；本方案落实其“初始 embedding 模拟重定时”的建议，不声称精确复现其方法或“轻松到7”。
- [cohort] 主队列为8名说话人×8条独立配对语音=64条；sample1仅用于开发和smoke，冻结后主分析63条；S0770的8条作为单独的说话人保留组。历史n25中的24条主分析语音及sample1另标记为已见，新增39条单列。
- [asset_verification] 冻结strict源清单有392对/8位说话人；按每人paired_key排序前8条选出的64对，仓库本地自然/TTS WAV 128/128存在且容器SHA与源清单source_sha256逐一一致。MFA-linear M目前只有历史strict25或clean3产物，不能宣称64条M已就绪；须先按同一来源生成并核验64/64。
- [boundary] S0770未进入历史n25复核，但不能据此声称它在整个项目从未被使用；统计结论按说话人聚类，8个说话人仍需报告不确定性。

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 根据用户要求编写 embedding 代理搜索与真实前向验收 v2 下游实施合同 | September 24, 2026 | user（编写请求）；agent（设计参数） |
| 用户指出三条样本太少；核验strict源与本地128条N/TTS音频后将正式队列改为8人×8条=64条，并补数据准备、预算及分层分析合同 | September 24, 2026 | user（样本量修订）；agent（核验与修订） |

## 2. Repository Model

P = scripts/experiments/mfa_linear_video_retiming/。

现有流程：
P/run.py::run_stage → audit/render → P/search_worker.py::calibrate_record → search_record → seal/transfer → P/official.py::score_official_cell → P/check.py::check_run → P/report.py::write_report。

真实模型边界：
P/generation.py::build_render_argv/render_baselines → static_image_bridge/render_worker.py → Wav2Lip GAN。
P/scorer.py::load_frozen_syncnet/load_audio_embeddings/forward_video/score_embeddings → SyncNet V2。
P/official.py::score_official_cell → strict_mux → SyncNet run_pipeline.py/run_syncnet.py。

v2在同包内增加 surrogate.py、search_v2.py、diagnostics.py；保留校准与媒体链，按冻结 protocol/engine 显式分派，不替换 v1 算法。

v2流程：
cohort_v2.prepare（冻结64条、生成同源64条M）→ audit → render → calibrate → [6个全局种子真实评分 → 代理搜索10k → 真实短名单25 → 反馈代理搜索10k → 真实短名单25 → 真实局部精修40] → select → seal → transfer → official → check → report。

每个候选都从原始 V_M 定义绝对映射 q；代理始终使用原始 V_M embedding。反馈只更新起点映射与搜索优先级，不把上一个候选 embedding 当新原视频。

## 3. Code Anchors

| path | symbol | current role | required change |
|:--|:--|:--|:--|
| P/config.py | validate_config/build_run_fingerprint | 仅接受v1及固定256预算、3个sample_id | 按protocol分支验证v1/v2；v2接受且只接受冻结64条cohort；新增预算、engine契约；绑定新源码与代理坐标定义 |
| P/run.py | stage_audit/_request/_protocol_for/stage_calibrate_search | 请求、协议、worker调用；audit拒绝非空run | v2 audit认可且核验预先生成的00_cohort及其预算账本，初始化protocol时承接prep已耗GPU时间；请求携带protocol/engine和64条cohort指纹；protocol_version取配置；v2 search用新worker分支 |
| P/search_worker.py | run_request/calibrate_record/score_exported_video | 校准、v1搜索、fresh评分 | 只在run_request按engine分派v2；校准和fresh模型前端保持；校准state作不可变输入 |
| P/cohort_v2.py（新） | select_cohort/prepare_audio/verify_cohort | 无 | 严格源清单选8×8；本地N/TTS逐条SHA复核；构建64条manifest/tokens/TTS元数据并用既有MFA-linear生成器生成64条M；冻结来源 |
| P/assets.py | freeze_inputs | v1读取clean3并硬编码其source | v2读取64条完整新summary及cohort manifest，逐条核验paired_key/speaker/split/transcript/N/M SHA、精确等长及三肖像；保留v1路径 |
| P/surrogate.py（新） | interpolate_visual/score_proxy | 无 | 纯NumPy代理、固定W评分；无模型调用、无媒体编码；可审计dtype与边界 |
| P/search_v2.py（新） | initialize_state/propose_maps/select_beam/select_shortlist/evaluate_true/select_final/search_record_v2 | 无 | v2候选库、多起点、两轮反馈、真实精修、恢复、真实最终门 |
| P/diagnostics.py（新） | compare_cached_candidates/main | 无 | 离线复现256候选代理/真实误差及排序，不访问GPU |
| P/run.py | stage_seal/stage_transfer/_official_cells | 封存及肖像迁移 | v2强制selected/global为真实分数；写engine/evidence类型；既有媒体与cell结构兼容 |
| P/check.py | _verify_search_independently/_fresh_metric_parity/_candidate_for_map_sha/check_run/main | v1缓存、fresh候选定位与最终选择独立重算 | 按protocol调用独立v2验证；fresh从校准state取M、从state_v2.true_candidates取R；CLI核验显式配置与run协议一致；保留v1验证 |
| P/check_v2.py（新） | verify_search_v2 | 无 | 用独立NumPy实现重算代理、真实评分和选择；不调用生产选择器 |
| P/report.py | _scientific_status/write_report/_plot_sample | v1结果与图 | v2独立解释分支；分开proxy/true/official；无固定v1阈值套用v2 |
| scripts/configs/mfa_linear_video_retiming_v2.yaml（新） | frozen protocol | 无 | v2唯一参数源，含搜索、最终门、诊断、资源预算 |
| tests/experiments/mfa_linear_video_retiming/ | 新v2测试及原集成测试 | v1测试 | 补v2有意义行为验证和版本兼容检查 |

P/retime.py 的 build_map/validate_map/global_seed/render_map/regularization，P/scorer.py 的模型前端和评分数学，P/assets.py::freeze_inputs、P/generation.py 的实际调用、P/official.py 的官方链均直接复用。若需要改这些稳定函数，限于明确兼容的元数据或接口适配，并列出理由与回归结果。

## 4. Reference Pattern

- P/search_worker.py::calibrate_record：仿照原模型加载、输入绑定、identity一致性、旧前端parity、延迟符号校准。禁止重新发明MFCC或误用Wav2Lip训练用SyncNet。
- P/search_worker.py::_record_candidate：仿照原始帧→render_map→crop→真实视觉前向→embedding哈希持久化；v2独立实现计数和存储，不继承其 NONINFERIORITY_* 中间淘汰。
- P/check.py::np_distance_matrix/np_metrics/rebuild_q_independently：仿照独立复算方式；不能复用v1最终选择规则来验证v2。
- P/run.py 的 _timed_stage/_resource_preflight/stage_seal：仿照资源、失败记账、封存和音轨审计。
- 原始候选缓存是代理诊断数据源；没有现成的同等多路线代理搜索，surrogate/search_v2为必要新增模块。

## 5. Invariants

1. generation_audio=M、optimization_audio=N、final_mux_audio=N；M/M、R/M只作封存后的诊断。N/M输入WAV、PCM、长度、音量不变。
2. Wav2Lip和SyncNet权重冻结、eval、无梯度。真实结果必须能追溯实际解释器、模块路径、参数状态、checkpoint SHA与输入哈希。
3. 主队列在任何搜索前固定64条、8名说话人，sample1只做开发；所有64条M由同一冻结strict源N/TTS/token方案生成，任何缺失或哈希不符阻断，不以同名旧文件代替。每个候选从原始像素生成；代理只插值原始视觉embedding，音频embedding不重定时、不归一化。禁止将代理embedding存进真实embedding路径。
4. score_kind取且仅取 proxy_embedding、true_fixed_crop、official_pipeline。代理字段用proxy_metrics；最终selected.metrics必须是true_fixed_crop。schema不能靠缺省值把proxy升级为true。
5. 固定同一record的W、crop、帧率、有效帧前缀、滞后范围及输入音频。候选不得通过裁掉坏片段、改变padding支持或换图提分。
6. 映射限制与v1一致：25fps；|δ|≤3帧；速度0.5..1.5；斜率变化≤0.5；最多16 knots、最小间隔12帧、δ网格0.25帧；首尾各5帧及补帧区不动；端点、帧数、PTS不变。
7. 几何非法候选直接拒绝；几何合法候选即使C、D暂时恶化，仍可作为搜索起点。最终门仅用于最终输出。
8. 全局offset从实际数据诊断；不得因为N/N和M/N均为-3而直接减去“系统偏差”。主目标固定物理零偏移D0；N/N对照单独报告。
9. v1配置、映射哈希语义、选择器及历史run只读保留；v2使用独立run目录、独立搜索state，不跨版本续跑。
10. 输出先封存再官方评分；不依据官方分数重新挑选、扩预算或改门。官方与搜索使用同一权重，不叫独立模型验证。
11. 代理模拟 C≥7.000 不能记为真实C达到7.000。真实固定裁剪和官方C也不能混用。
12. 肖像3搜索，6/9原样迁移q；三肖像不增加独立语音n。人工未评时明确 HUMAN_NOT_ASSESSED。

## 6. Implementation Plan

### P0：版本、配置与模型合同

新增 protocol=mfa_linear_video_retiming_v2，search.engine=embedding_beam_v2，output_root=runs/mfa_linear_video_retiming_v2。v1缺省engine视为原算法；不接受未知组合。配置schema可保持1；v2搜索state schema为2，旧校准state schema1只作为输入。

主队列固定为8名说话人×8条=64条，肖像仍为[3,6,9]，只在肖像3搜索，6/9迁移。来源是 runs/two_stage_hubert_aishell1_20260810/data_boundary/aishell1_400_raw_mfa_faster_qwen3_heldout.json（392对strict N/TTS）；按说话人列表[S0765,S0770,S0901,S0906,S0912,S0913,S0914,S0915]各取paired_key字典序前8对。样本ID固定为1–8、51–58、101–108、151–158、201–208、251–258、301–308、351–358；按paired_key/speaker/transcript/split以及natural/tts source_sha256核验后冻结cohort_v2.json，不按分数、长度或生成成功率换样本。

源清单内音频路径为旧/mnt/e路径，不得直接信路径。对每条记录从结果库 results/rhythm_style_500/aishell1_test_400/{natural,tts}/<四位sample_id>.wav定位，再逐文件比较源清单source_sha256。设计时已核验入选的128/128个本地N/TTS容器SHA匹配；执行时重新验证并保存每文件SHA、PCM属性。strict清单中自然/TTS tokens及门控标签用于构建64条tokens_v2.json；任何token缺失、门控失败或身份冲突不得静默删条。构建64条cohort manifest和TTS meta，TTS路径指向已验证的同源本地文件，source_audio_sha256绑定tokens侧TTS音频。

调用 scripts/pilot_generate_mfa_linear.py::generate（或仅为64条输入增加薄adapter），用同一WavLM-Large L6、mfa_linear_target、prematched HiFi-GAN与exact_natural_length生成64条M，全部保存新run；生成前后逐条检查N/TTS/token/输出身份和N/M采样数。不得把旧strict25、clean3、qwen_cloud25的M混入新64条。若64/64未完成，主评估记INPUT_INCOMPLETE并保留失败分母；不自动替换样本。

开发smoke只使用sample1/portrait3，以新64条M的sample1为输入；旧clean3 smoke仅是代理诊断历史来源。冻结代码/阈值后运行剩余63条；主分析n=63，64条总体表仍列出开发样本。S0770的8条独立列为说话人保留组；既有历史n25中入选的25条要标注prior_seen，主分析含24条，另39条为本次新增。历史分组仅用于解释，不依据分数筛选。

- Wav2Lip解释器：[redacted-local-path]
- SyncNet解释器：[redacted-local-path]
- venv调用保留bin/python路径，不能resolve后调用基础解释器。加载后检查模块__file__、参数状态与receipt；模型加载失败直接失败。
- Wav2Lip只在N/M基线生成时调用，各音频/肖像一次；候选不重新生成Wav2Lip。
- SyncNet输入：BGR、0..255、224×224、[B,3,5,H,W]；PCM16 16kHz mono→官方MFCC 13×20；窗口起点i对应MFCC列4i..4i+19；1024维原始向量、不做L2归一化。
- 每个真实候选必须重新运行视觉塔，音频塔按冻结N缓存。沿用校准parity及最终fresh M/R。
- seed=20260923，Wav2Lip batch4、SyncNet batch20。权重、前端和生成框/评分框不可为提分而变。

准备队列CLI：
python -m scripts.experiments.mfa_linear_video_retiming.cohort_v2 --source-manifest runs/two_stage_hubert_aishell1_20260810/data_boundary/aishell1_400_raw_mfa_faster_qwen3_heldout.json --output-dir runs/mfa_linear_video_retiming_v2/<new-id>/00_cohort --config scripts/configs/mfa_linear_video_retiming_v2.yaml --prepare

准备阶段输出带哈希的cohort_v2.json、tokens_v2.json、tts_meta_v2.json与mfa_summary_v2.json，后者绑定64/64 M；run.audit必须验证这些产物与配置一致。新run若已有部分M，可按每条输入指纹续生成；输入变动禁止沿用旧M。准备阶段的模型时间单列，并计入同一run总GPU预算。现有stage_audit会拒绝非空run；v2必须特判并校验唯一允许的00_cohort目录及其manifest，不得放开任意文件。prep写原子预算账本；audit初始化protocol.active_gpu_seconds从经哈希验证的prep账本继承，运行时不得清零或重复记账。准备前也执行v1资源检查；RESOURCE_WAIT保留当前产物和固定分母。

主流水线CLI沿用：
python -m scripts.experiments.mfa_linear_video_retiming.run --config scripts/configs/mfa_linear_video_retiming_v2.yaml --run-id <new-id> --stage all [--smoke]

### P1：先落地离线代理诊断

新增诊断CLI：
python -m scripts.experiments.mfa_linear_video_retiming.diagnostics --source-run runs/mfa_linear_video_retiming_v1/smoke_20260923_stageb256_rounds4 --output-dir <new-diagnostic-dir>

只读state.json及其中已绑定的原始V_M/N和候选真实embedding；核验文件哈希并由真实embedding复算缓存分数，不能只信摘要。输出diagnostics.json、per_candidate.csv、report.md，绑定源state、代码SHA、坐标锚点、支持W和环境。仅运行CPU；不要使用启动时强制GPU的通用audit入口。

主代理为window_center，另做window_start诊断。分别报告C/D/D0 MAE、Spearman、offset一致率、真实top10被代理top10/25/50覆盖的比例、代理挑选候选的真实分数和真实最优候选代理排名。并保存耗时、真实前向调用数=0。

必须报告整批256与去除6个全局种子/identity后的局部候选子集，避免容易排序的种子夸大相关性。同分排名使用平均rank；top-K并列按map_sha256固定排序；候选ID唯一，label不唯一不得据此去重。

本会话的近似数仅供核对；若差别明显，先解释算法、dtype、集合或排序定义差异，不为追平数字改变样本。此诊断只验证近似可用性，不证明搜索成功。

### P2：代理定义与计算

对原始visual embedding v_i，i表示五帧窗口起点；时间映射output帧j读取source q_j=j+δ_j。

主代理坐标：
s_i=q_(i+2)-2。
令 l=floor(s_i)、h=ceil(s_i)、α=s_i-l，则 v_tilde_i=(1-α)v_l+αv_h。
诊断start代理为s_i=q_i，不能与主代理混算。

在冻结W上，对k=-15..15：
d_tilde(i,k)=sqrt(sum((v_tilde_i-a_(i+k)+1e-6)^2))。
全部输入、插值、距离、按时间平均用float32；汇总展示可float64。C=median(curve)-min(curve)，D=min(curve)，D0=curve[k=0]，offset=15-argmin(curve)，并列取最左。使用NumPy与真实pairwise_distance的eps语义一致。

W复用v1：n=min(Fvalid,floor(L/640))-5，W=range(15,n-15)，至少25行。所有访问坐标先验合法；越界报 PROXY_SUPPORT_OUT_OF_BOUNDS，不能clip embedding坐标、pad零或删行。q的媒体几何验证仍逐候选执行。

identity代理需在1e-4内复现原始真实curve/C/D/D0、offset相同。整数全局移位在完整五帧整体平移的内部区域可检查窗口映射；非匀速或fractional映射不要求代理等于真实。

高效实现用分块向量化，并可预缓存两端embedding差向量/内积；不得每个候选调用torch模型。任何加速版先与上述直接公式对照，误差超过1e-4则使用直接公式。候选块峰值内存配置上限512MiB；不一次建立20k×W×31×1024张量。

### P3：候选、预算和多路线搜索

固定单样本预算：
- proxy_unique_maps≤20000，阶段A/B各≤10000；
- true_candidate_forwards≤96：6个全局种子、第一短名单≤25、第二短名单≤25、局部精修≤40；
- identity及N基线/校准、fresh M/R、官方评分另列计数，不能宣称整个run只有96次模型前向；
- search wall time≤1200秒，覆盖代理CPU、真实GPU、渲染和持久化；资源等待独立计账；
- full64 total active GPU预算129600秒（36小时），资源阈值沿用v1；cohort音频生成、64×3×2基线Wav2Lip、搜索、官方评分、fresh checker全部计入。单条smoke不预耗满full预算。CPU代理时间单独记录；GPU worker占用按既有保守计账。预算是上限，不是保证完成；到限时报告已完成的固定分母与未完成原因。
- 未用满某阶段预算不转移给其他阶段；不足、超时或资源等待都有明确状态。

候选map_sha256按现有map规范计算；身份候选同时保存candidate_id=identity与map_sha256。代理档案与真实候选库分开，同一map最多一次完成的代理评分、一次完成的真实评分；新真正前向尝试（含失败重试）都计真实预算。

初始化7个根：identity和global_seed(-3,-2,-1,+1,+2,+3)。6个全局种子先真实评分，无中间C/D门。每个合法根都必须至少被扩展一轮，不能因C稍低被beam提前清除。

每阶段beam宽24，step按[1.0,0.5,0.25]，每step最多5轮；每轮从各父候选分别提出：
1. 单knot ±step；
2. 相邻2个knot同向±step；
3. 连续4个knot同向±step；
4. 加/减step×global_seed(+1)的knot profile，整体平移内部区域并保留边界ramp。
所有proposal均为原始V_M上的绝对map；不用复合插值。非法记录原因，去重不耗score预算。按父候选、move类型轮流取proposal，父排序与同分均按map_sha；不让第一个父耗光预算。步长和轮次未完成时记录精确游标。

beam从已评分代理档案确定性选择，依序去重：
- 8个按(D0, |offset|, -C, R, map_sha)；
- 8个优先|offset|≤1再按(-C, D0, R, map_sha)；
- 余下8个用farthest-first覆盖map：候选间距离为mean_W(abs(δ_a-δ_b))，与已选集合的最小距离最大者优先，同分按D0、map_sha。
某一路不足由D0排序补满。根保存在单独root_archive，不受beam淘汰影响；后续stage重启仍包括全局根。终止条件为阶段预算/时限或完整step轮次用尽；代理中不执行最终门。

第一阶段代理搜索后，对未真实评分map构造25个短名单：
8个D0优先、8个aligned C优先、9个farthest-first；已选及已真实评分map去重后继续该路线取下一名，实在不足按D0补。不能只取proxy C top-K。

第二阶段起点为全部7根，以及第一短名单真实评分后D0最优4个、aligned C最优4个（去重）；仍不执行中间C/D非劣门。代理仍读取原始embedding，不从真实候选embedding插值。第二阶段最多再评10k个新map，按相同规则选25个新的真实短名单。

局部真实精修：4轮，每轮≤10个新真实候选。每轮起点取真实D0最低2个和aligned C最高2个并去重；按step=[0.25,0.5,1.0]产生上述四类邻域。用剩余代理额度评分尚未代理评分的邻域；无额度时按确定性proposal顺序保留为proxy_metrics=null，不伪造代理值。每轮选4个D0路线、3个aligned C路线、3个覆盖路线（有代理先用代理排序，无代理候选在各路线排序末尾按map_sha）；真实前向后更新起点。若所有邻域已评分则结束，不重复调用凑预算。

### P4：真实评分、最终选择与对照

evaluate_true读取冻结原始V_M → render_map → crop_frames → forward_video → score_embeddings。保存真实visual embedding、curve、C/D/D0/offset/local_windows、像素hash、耗时与实际模型指纹。不得调用v1 _record_candidate给中间候选打feasible=false后沿用旧选择器。

真实候选最终资格：
- C≥C_identity-0.050；
- D≤D_identity+0.050；
- D0≤D0_identity-0.020；
- |offset|≤1；
- v1连续25行局部窗的median D0≤baseline，|offset|的q90≤baseline；
- 全部几何、媒体、有限值与真实评分凭据通过。

这是相对v1“必须ΔC≥0.100”的显式修订：目标优先是直接播放的时间同步，保留C非劣门；不能把轻微降C的对齐收益描述成C提升。

在合格集合中找到最小D0，取D0≤最小值+0.020的候选，再按C降序、R升序、map_sha排序，第一名为R。数值比较用未舍入值。没有合格项则identity；单独保存真实 best_d0、best_c、best_aligned及每项最终拒绝原因。

标签：
- ALIGNMENT_GAIN：通过上述门；
- ALIGNMENT_AND_C_GAIN：通过门且ΔC≥0.100；
- NO_ACCEPTABLE_WARP：没有合格项；
- termination单独为COMPLETE/BUDGET_LIMITED/RESOURCE_WAIT/FAILED；不能用一个status混淆终止原因与是否找到收益。

GLOBAL对照取6个真实全局种子中D0最小者（其次|offset|、-C、R、map_sha）；它是诊断对照，不要求通过最终C门。由此即使R回退，也不会把真正校时的全局对照抹成identity。

N、M、R、GLOBAL、NEAREST、MIRROR沿用已有封存/评分矩阵；NEAREST仍只是选中q的渲染对照，不声称独立优化；MIRROR非法明确记录。官方搜索音轨固定N；M/M、R/M保持诊断角色。肖像6/9迁移同q并按既有cell合同评分。

### P5：状态、恢复和独立检查

校准写既有03_search/<sid>/state.json，v2读它并绑定SHA；v2搜索写state_v2.json，不修改校准state。result.json保留seal所需baseline、selected、global_control、best_attempt、search_result字段，另写engine、protocol、state_v2_path、selection_status、termination。

state_v2包含：
schema_version=2、协议/请求/评分器/校准state/原始embedding指纹；
proxy_candidates（map、proxy_metrics、stage、parents、moves），true_candidates（map、true metrics、真实embedding引用），beam、root_archive、短名单、阶段/step/round/move游标；
attempted/completed/failed计数、elapsed、已消耗预算、inflight、最终门逐项结果。

真实候选库的metrics是兼容字段，必须伴score_kind=true_fixed_crop；proxy库只有proxy_metrics。校准identity可在true_candidates中创建绑定原始真实embedding的只读引用，不增加前向计数；原校准state不回写。result.status保留运行状态兼容值SEARCH_COMPLETE/BUDGET_LIMITED等，selection_status单列；best_attempt定义为best_aligned（存在时），否则best_d0，全部为真实记录。保存短名单及选取依据，允许独立重放。fresh验证的M读取校准state，R必须按map_sha从state_v2.true_candidates查找；不能仍在旧state.candidates中找v2选中项。配置中的engine、代理锚点、两套预算、最终门全部纳入请求与缓存指纹；checker拒绝config.protocol与run.protocol不一致。避免每个代理候选重写全量20k档案：按固定64个候选一批原子提交chunk+manifest，推进游标与计数同批提交；孤立临时文件不得被恢复为成功记录。

启动真实前向前先持久化inflight及attempted计数；崩溃后重跑消耗新一次预算。已完成artifact通过SHA校验才命中缓存。代理已发出批次在崩溃时保守计已消耗，重算也占剩余额度；预算不得因重启清零。保存RNG状态（若无需随机则不引入随机）；幂等续跑不重评完整短名单。

check_v2独立重算：
所有map几何；全部代理分数（分块）；全部真实embedding距离/局部指标；短名单规则和最终选择；版本/指纹/预算；封存像素、PCM、PTS；fresh M/R与搜索真实值一致。不得调用surrogate.score_proxy或search_v2.select_final作为唯一验证。

### P6：执行与报告

先完成64条cohort与64/64同源M的准备及哈希审计，运行离线诊断和相关测试；再完成新cohort的sample1/portrait3 smoke并自审修复。代码或参数修改后用新run，不继续旧指纹run。smoke通过后冻结代码、阈值、64条ID、肖像、预算和报告字段，运行其余63条及预定三肖像矩阵。发生个别失败保留64条分母，不补抽别的样本；完成率分别报64总体与63非开发集。

报告必须列：
- 每样本proxy候选数、真实尝试/完成数、校准/fresh/official前向另计、CPU/GPU/总耗时；
- identity、GLOBAL、best_real_d0、best_real_c、R的真实固定crop指标；代理列单独；
- N/N、M/N、R/N官方C/D/D0/offset与共同支持配对差；7.000达标数分别统计真实固定crop/官方；
- 64条总体与63条非开发集的通过率、identity回退率、ΔC/ΔD0逐样本及分布下尾（p10、min）；另列S0770说话人保留组8条和prior_seen24/新增39；缺失项计入预定分母，不伪造0分；
- 代理短名单对真实最优的覆盖、选中候选代理误差，C升高但D0/offset未改善的候选；
- 保留v1局部门，补充真实矩阵上的25行滑窗（stride5，含最后一个完整窗）诊断，报告D0 p90及局部offset，不把诊断改成新门；
- 固定crop与官方时间范围、crop、帧数和支持差异；N/N系统性offset未查明时如实报告，禁止直接做常数补偿；统计按utterance配对、按speaker（8簇）bootstrap区间与逐speaker结果并报；只有一名S0770保留说话人，不能声称跨说话人显著泛化；
- 所有样本盲看包、画质状态与失败原因。无人工评分可交付工程报告，不能声称视觉自然度通过。

人脸/音频/模型不变的v1回溯仅作为历史参照；不同搜索预算不称公平速度胜出。要比较效率须报告实际工作量与时长，不把模拟候选数当真实前向数。

## 7. Expected Change Surface

### Must change

- P/config.py、run.py、search_worker.py、check.py、report.py：仅版本分派及所需v2集成。
- 新增P/cohort_v2.py、surrogate.py、search_v2.py、diagnostics.py、check_v2.py；P/assets.py增加v2输入解析。
- 新增scripts/configs/mfa_linear_video_retiming_v2.yaml。
- 新增tests/experiments/mfa_linear_video_retiming/test_cohort_v2.py、test_surrogate.py、test_search_v2.py、test_check_v2.py；更新已有config/resume/seal/report相关测试。
- 本Research spec实施状态及最终Experiments笔记（先读Startup Router）；原v1结论保留。

### May change

- P/common.py：仅必要的原子chunk/状态写入辅助。
- 原generation/official/helper：仅兼容元数据适配，需回归说明。

### Should not change

- v1 YAML与历史run、原v1最终门、retime映射/像素算法。
- third_party模型源码、权重、MFCC/图像前端、输入音频、portrait registry。
- TTS或增强器训练、其他实验流程、服务器配置。

## 8. Validation Plan

| 层级/命令 | 检查 | 预期 |
|:--|:--|:--|
| python -m compileall scripts/experiments/mfa_linear_video_retiming | 语法与导入静态检查 | 通过 |
| pytest -q tests/experiments/mfa_linear_video_retiming/test_cohort_v2.py | 8×8固定选样、split与paired_key联合、旧路径重定位必须SHA匹配、tokens/TTS/M严格同源、缺失与错SHA、续生成预算 | 64条固定；128源SHA及64条M完成才放行；失败留分母 |
| pytest -q tests/experiments/mfa_linear_video_retiming/test_surrogate.py | identity parity；中心/起点坐标差；整数整体shift内部窗口；eps、float32、越界；无模型调用 | identity误差≤1e-4；非法支持拒绝；不会声称非线性warp严格等价 |
| pytest -q tests/experiments/mfa_linear_video_retiming/test_search_v2.py | 低C但aligned根得到扩展；多路线保留；去重；短名单真实重排；proxy高C不能直接选中 | 可跨越中间C下降；最终仅真实分数有效 |
| 同上 | 构造代理排名与真实排名相反的候选；高C且offset=2；C轻降但D0改善；局部恶化 | 真实重排正确；offset超门拒绝；v2/v1门差异正确；局部恶化拒绝 |
| pytest -q tests/experiments/mfa_linear_video_retiming/test_resume.py | 代理批次和真实前向前/后崩溃；缓存篡改；版本/模型/音频/代码变化 | 无重复成功前向、失败计数不丢、预算不清零、指纹不符阻断 |
| pytest -q tests/experiments/mfa_linear_video_retiming/test_check_v2.py | 代理冒充true、N/M调包、修改q/分数/selected、假官方来源 | 独立checker拒绝 |
| pytest -q tests/experiments/mfa_linear_video_retiming | v1行为与v2集成 | v1原测试继续通过，v2新增行为通过 |
| P1离线诊断CLI | 256候选、哈希、真实矩阵复算、误差与排序、计时 | 完整落盘，可复算；0新增GPU/模型前向 |
| v2 CLI --stage all --smoke | 真实Wav2Lip、代理、大量搜索、真实候选、官方链、fresh检查 | 工程证据PASS；若无收益，明确回退且保留GLOBAL和拒绝原因 |
| python -m scripts.experiments.mfa_linear_video_retiming.check --run-dir <v2-run> --config scripts/configs/mfa_linear_video_retiming_v2.yaml | 核验配置与run协议一致后分派checker；fresh定位v2真实候选 | v2真实/proxy/媒体/预算独立复算通过；误传v1配置明确拒绝 |
| v2 cohort_v2 CLI --prepare | 64条strict配对、128个本地源SHA、64条M与token/长度/身份 | 64/64或INPUT_INCOMPLETE；失败不替补 |
| 冻结v2 CLI --stage all | 64样本3肖像预定矩阵及失败记录 | 完整大队列报告或明确未完成项，不删困难样本；开发样本单列 |

测试使用小型构造数据验证关键行为，不复制实现流程做形式测试。pytest用具备项目依赖的环境；真实生成、评分分别使用指定venv。若自动扩展测试失败先修复相关原因；不因已经跑过v1而跳过v2真实smoke。

## 9. Risks and Edge Cases

- 高整体相关性不能保证top-K正确；本次top10重合为0是强制多路线和真实精修的依据。
- embedding线性混合可能缩小向量范数、虚假降低距离；不擅自归一化改变SyncNet定义，靠真实前向排除代理伪优。
- s_i=q_(i+2)-2中的减2不可漏；不能把帧中心、embedding起点、音频MFCC列混为一个时钟。
- 原视频132帧、有效128帧，官方crop可有不同帧数；支持集差异必须审计，不能拿两套C直接算提升。
- ±3帧位移加首尾保护可能无法完全校时；负结果只约束本映射族。不能自动放开约束直到阳性。
- 六个全局种子也需要真实评分；已有代理分数不等于其真实凭据。GLOBAL为诊断可不满足最终门。
- 代理库不能假装具有embedding_path；v1 checker若直接遍历代理会报错或误验，必须版本分派。
- 官方C是寻找最佳lag后的置信度；D0改善不保证C增加，C增加也不保证播放同步。
- 真实精修和原代理存在系统性误差；保留真实最优，不能后续被代理优胜覆盖。
- 未完成真实验证的候选不能封存为R；预算用尽时从已完整验证的合格集合选择。
- 更改协议后继续原run会污染比较；缓存只在完整指纹相同时复用。
- 不把搜索上的过拟合收益、同权重复评、同一speaker的多条语音或开发样本解释为增强器泛化证据。

## 10. Assumptions / Unknowns

- VERIFIED: v1真实调用、固定crop校准、官方8个smoke cell和fresh检查已有凭据；此前R回退identity。
- VERIFIED: 原始V_M/N及256候选真实embedding可用于离线代理回溯；本会话已完成初步计算。
- VERIFIED: strict源清单有392对、8位说话人；所选每人前8条共64对的本地自然/TTS WAV 128/128容器SHA与source_sha256一致。历史M只有strict25或clean3等局部产物，不等于v2同源M64齐备。
- VERIFIED: 现有配置/报告/检查器有v1硬编码；不能仅新增search函数就认为v2可运行。
- VERIFIED: v1的中间非劣门拒绝所有16个达到|offset|≤1的候选。
- LIKELY: 多路线代理搜索可用更少真实前向探索更多map；真实收益与墙钟加速尚待运行。
- UNKNOWN: 发明人的具体代理、搜索、数据、评分口径；模拟/真实C能否达到7.000。
- UNKNOWN: N/N与M/N共同偏移来自生成器、评分感受野还是媒体链；本轮报告证据，不猜常数修正。
- UNKNOWN: 代理短名单覆盖在新样本上是否足够、最终画质与跨肖像迁移是否保持。
- UNKNOWN: 新64条M在既有WavLM/声码器环境能否64/64生成、全64官方检测成功率、真实总耗时及GPU资源；失败必须留分母。遵循既有资源等待，不终止其他任务。

## 11. Handoff Contract

实施者遵循以上代码锚点和顺序，复用真实模型/媒体/评分边界，保留全部不变量，改动限制在第7节；不要做无关重构。先冻结64条strict来源并生成同源M，完成可复现代理诊断及测试，再跑新cohort真实smoke、自审修复，冻结后完成剩余63条评估及三肖像矩阵。参数已在本spec固定，实施时不得自行换目标函数、评分器、模型、样本、位移上限或最终门。

交付清单：代码与配置；64条cohort及N/TTS/M来源和哈希；相关测试结果；诊断脚本及256候选回溯产物；新cohort smoke与64条run路径；模型/输入/状态/候选凭据；独立checker结果；64总体、63非开发、S0770保留组和逐speaker指标/耗时；人工评估状态；自审发现与修复摘要。工程完成不要求C≥7.000；任何收益声明必须来自真实前向，最终官方结论来自封存后官方结果。

仓库事实与本spec冲突时，报告具体路径、SHA、形状或调用差异并停止依赖该事实的分支；继续独立可做的工作。v1结论和历史产物保持可追溯；将v2作为后续协议，不覆盖v1结果。

### Relations

- follows_from [[MFA-linear 视频重定时与自然音频同步 Implementation Spec]]
- informed_by [[MFA-linear 视频重定时 Sync-C 约束工程 smoke 结果 2026-09-23]]
