---
title: 音素可分度与 TTS-TFG 增益低成本配对关联实验 Spec 2026-09-21
type: experiment
permalink: tts-exp/experiments/音素可分度与-tts-tfg-增益低成本配对关联实验-spec-2026-09-21
status: concluded
protocol: phoneme_tfg_association_v1
date: '2026-09-21'
tags:
- lrs3
- phoneme-separability
- tfg
- paired-association
- implementation-spec
---

# 音素可分度与 TTS-TFG 增益低成本配对关联实验 Spec 2026-09-21

本设计回应“探究音素可分度与TTS对TFG增强的关联，但避免全量视频推理”的需求，按 implementation-architect 完成仓库锚点、分析协议、输入绑定、失败处理与下游验收合同。首次候选 run 因把 LRS3 源视频动态帧送入 renderer 造成视觉泄漏，已明确作废；随后按外部非 LRS3 首帧协议完成 corrected 主 Wav2Lip/SyncNet 实验，科学结果为 INCONCLUSIVE；人工盲评与可选 Ditto 扩展仍属于补充工作。

## Observations

- [status] concluded
- [progress] 主协议 phoneme_tfg_association_v1 的 corrected 主实验已完成，补充盲评/Ditto 不改变主检验状态；旧视觉泄漏 run 仅保留作审计。
- [hypothesis] 同一句话内，扣除TTS组合平均差异后，HuBERT L6逐句phone silhouette更高的TTS是否有更大的原生Sync-C增益。
- [design] 主方案36个不同source_group，每组1句、natural与随机分配的2种TTS；四种TTS六种pair各6句，每种TTS18句；主TFG为Wav2Lip，共108个video cells，72个TTS-natural配对但仅36个来源块。
- [budget] 对同一TFG的100×5=500全量方案减少78.4%生成条数，非实测耗时比；严格匹配历史视频可进一步减少新增推理。可在评分前开启12来源×3臂=36条Ditto扩展，总144条；不按p值追加。
- [reuse] 复用英文n100的canonical音频、独立MFA和HuBERT/XLS-R pooled vectors；CPU重建逐句silhouette，禁止把pooled Fisher复制给每条样本；13个父输入/缓存文件SHA已核验并写入input-bindings。
- [inventory] 当前100条来自42来源；MFA497/500，CosyVoice2缺3条；五臂时长≥3s且valid tokens≥20的粗资格91条/41来源，最终标签/脸/哈希资格仍需下游审计。
- [reuse_boundary] 历史Ditto/LeapTalk50条与本次n100 exact sample_id交集0、source_group交集42，不能直接拼评分，也不能当来源独立验证集。其他历史视频需另按PCM/参考脸/checkpoint/seed/代码合同检查，零交集结论仅针对该50条。
- [statistics] 主分析36行TTS-TTS同句差分，六个pair固定截距加唯一silhouette斜率；HC3 CI与null-imposed studentized wild bootstrap，主检验双侧0.05；来源块bootstrap用于TTS-natural均值，9项预定次要关联统一BH。
- [evaluation] 完整utterance、各视频自身音轨；SyncNet动态T距离矩阵，去padding后三臂等窗口数为主支持；并列Sync-D、背景B、offset与静态reference视觉负对照。静态对照复用音频embedding，不生成新TFG视频。
- [visual_leakage_fix] 正式 run `runs/phoneme_tfg_association_external_visual_cpu_audit_v1_20260922/` 的 36 个 source blocks 全部使用 MEAD/VFHQ/TalkVid/GRID allowlist 中的外部视频首帧；LRS3 源视频仅保留为音频/文本 provenance，未进入 renderer。每臂同一首帧重复到 mel chunks，评分再按冻结 box 裁成 224×224；旧 `impl_cpu_preflight_v2` GPU 结果不纳入科学结论。
- [implementation] 旧wav2lip_probe_gpu硬编码93帧，runtime评分使用88行/固定U_ROWS，score封装硬编码N音轨；下游新增变长封装，复用generation_worker与SyncNetScorer底层，不直接套旧probe主入口。
- [limitation] 关联不等于可分度干预的因果收益；source_group不是已确认speaker；MFA成功不代表无漏读；Wav2Lip/SyncNet可能存在模型特异性；36来源对弱效应功效有限。未获得独立感知数据时perceptual_status=NOT_MEASURED。
- [claim_boundary] 不显著、支持不足、工程PASS均不能写成“没有关联”或“嘴型得到因果改善”；中文旧104项FDR无阳性只是背景，不迁移数值或方向。
- [report] 完整下游spec：openspec/changes/probe-phoneme-tfg-association/spec.md；严格11节，包括code anchors、invariants、实施步骤、测试与handoff。
- [inputs] 输入绑定：openspec/changes/probe-phoneme-tfg-association/input-bindings.json；父run为runs/lrs3_english_phoneme_transfer_n100_20260921/。
- [outputs] 下游目标runs/phoneme_tfg_association_<id>/；包含协议、逐条特征、108-cell计划、native/static评分、斜率/CI、缺失分母、独立checker与盲评包。

- [implementation_status] 已实现 `scripts/experiments/phoneme_tfg_association/` 下的审计、特征复用、协议冻结、外部视觉首帧绑定、Wav2Lip/可选Ditto生成适配、动态 SyncNet 评分、块级关联分析、解释性敏感性、独立校验与 CLI；receipt/checker 已绑定 protocol hash、失败分母、外部视觉 hash、固定 box 与 C 分解，Ditto 扩展按每个 pair 取 2 个 block；实验相关测试 13 个通过。
- [cpu_preflight] `runs/phoneme_tfg_association_impl_cpu_preflight_v2_20260921/` 已完成输入绑定 13/13 hash 校验、100 条样本审计、97 条可生成清单、HuBERT/XLS-R 特征复用（91 条样本/41 个来源组满足主筛选），并写入 peak/RMS/silence_fraction；36 个来源块和 108 个 Wav2Lip cells 确定性冻结，CPU 级校验与空分数分析通过。
- [tests] `pytest -q tests/experiments/phoneme_tfg_association`：13 passed；`compileall` 通过；全仓 pytest 1457 passed、17 个既有 fixture/protocol 失败，与本修正无关，未将其冒充为全仓通过。
- [gpu_guard] 2026-09-22 复核时 GPU 无其他计算进程；generation/score 门禁均通过。6-cell smoke 先行成功，随后完整主实验安全运行。
- [next_step] 主 Wav2Lip 关联已完成；可选后续仅为收集 12 组/36 clip 人工盲评和按显式配置运行 36-cell Ditto 扩展，不得据此改写主结论或按 p 值追加样本。
- [scientific_status] corrected 主结果 INCONCLUSIVE：β=-0.091052 Sync-C/0.1 silhouette，HC3 CI=[-0.412220,+0.230115]，wild p=0.4975；这不是“没有关联”的证明。

- [resource_status] 主实验启动前 GPU 为空闲；完整运行期间仅本实验进程占用 GPU，结束后已释放。历史误用 run 目录已移到可恢复的 `/tmp/tts-exp-accidental.b0bg31/`。

- [review] `06_review/blind_manifest.json` 与 `ratings_template.csv` 已就绪：12 个 blind groups、36 个 clip，状态 `READY`；尚未填入人工评分，因此感知状态仍为 `NOT_MEASURED`。

- [result] corrected 主 Wav2Lip/SyncNet 完成 36/36 source blocks、108/108 media cells、36/36 native score rows、36/36 static-control rows；强制媒体/分数与外部视觉 hash checker 为 PASS/READY，缺失分母为 0。
- [tts_natural_gain] corrected EQUAL_COUNT Sync-C 的 TTS−natural 描述性增益：qwen_cloud +0.073（95% CI [-0.091,+0.216]）、qwen_local -0.004（[-0.162,+0.150]）、index_tts2 +0.086（[-0.016,+0.209]）、cosyvoice2 +0.056（[-0.130,+0.232]），每项 n=18；各模型的区间均跨 0，positive fraction 分别为 11/18、9/18、13/18、11/18。这是当前筛选 LRS3 英文音频来源、外部静态视觉源与 Wav2Lip 条件下的描述，不作因果或全体模型排名。
- [arm_summary] corrected EQUAL_COUNT 平均 Sync-C：natural 1.064、qwen_cloud 1.162、qwen_local 1.072、index_tts2 1.146、cosyvoice2 1.089；C=gain_D+delta_B 分解、动态长度/PCM、固定外部视觉 box 裁剪检查通过。
- [sensitivity_result] duration/silence 调整后 silhouette β_x=-0.077，HC3 CI [-0.356,+0.202]，36 blocks；duration-ratio [0.5,1.5] 子集 35 blocks，β=-0.081，CI [-0.405,+0.242]，均未改变“INCONCLUSIVE”边界。
- [secondary_result] 预注册次要族中，主 silhouette→gain_D 与主 silhouette→ΔB 分别为负/正方向信号（BH q=0.0468/0.0072），但主 Sync-C 关联不显著；它们应解释为 C 分解诊断，不替代主检验。

## Relations

- follows [[LRS3 English TTS Phoneme Separability Transfer n100 2026-09-21]]
- relates_to [[15-tfg-link]]
- relates_to [[TTS 音素可分性机制 LRS3 严格 Atlas 与自然时钟构造 2026-09-21]]

## Changelog
| Notes | Date | By |
| --- | --- | --- |
| 核验同句区组设计及官方关联数值；澄清natural抵消、官方full-track与equal-count标签不符、pooled p未做聚类修正；补充偏相关和时长/静音敏感性，仍INCONCLUSIVE | September 22, 2026 | agent CPU audit; user requested validity review |
| 应用户请求完成低成本关联设计、仓库锚点spec和13文件输入绑定；未运行视频实验 | September 21, 2026 | user requested design; agent-authored protocol |
| 完成下游实现与CPU预审；GPU生成/评分因外部进程 PID 504080 占用而暂停，未产生科学关联结论 | September 21, 2026 | agent implementation; resource gate |
| v2 CPU预审补齐音频波形统计与解释性敏感性输出；测试 9 passed；GPU仍由 PID 504080 占用，未生成/评分视频 | September 21, 2026 | agent implementation; resource gate |
| 完成 receipt/checker protocol-hash 与 C 分解校验、Ditto pair 配额修正；测试 10 passed；仍未运行GPU科学阶段 | September 21, 2026 | agent implementation; contract hardening |
| 报告阶段补齐 12 来源盲评包接口；当前无视频，review 状态 NOT_RUN；GPU仍由 PID 504080 占用 | September 21, 2026 | agent implementation; handoff artifact |
| 首次 GPU 结果发现 renderer 读取 LRS3 cohort 源视频动态帧；该 run 的视频/分数保留作审计，但科学结论撤回 | September 22, 2026 | agent audit; leakage invalidation |
| 按外部非 LRS3 首帧、固定 box、SyncNet 224 crop 修正后完成 6-cell smoke、108-cell Wav2Lip 生成、36-block native/static SyncNet 与主统计；β=-0.091052，CI [-0.412220,+0.230115]，wild p=0.4975，科学状态 INCONCLUSIVE | September 22, 2026 | agent execution; corrected primary protocol |


## 2026-09-22 诊断补充：历史明显 LRS3 增益与当前弱增益不可直接比较

- [diagnosis] 历史 LRS3 50条的 Ditto 官方 SyncNet 结果为 natural C=4.740、TTS C=5.864、ΔC=+1.124（45/50为正）；当前 corrected run 的 EQUAL_COUNT Wav2Lip/自定义 SyncNet 端点，四个 TTS 合计平均增益仅 +0.053。当前弱增益首先是协议改变后的结果，不足以推出 TTS 原生优势消失。
- [tfg_mismatch] 历史主结果使用 Ditto（另有 LeapTalk +1.389 的历史结果），当前 corrected 主实验只使用 Wav2Lip。Wav2Lip 的局部时间传递诊断也未通过；因此 TFG 生成器差异是当前排查中的最高优先级因素。
- [metric_mismatch] 历史 `04_eval.py` 走官方 SyncNet 检测/裁剪与 Confidence/Min dist；当前 run 使用固定外部视觉 box、224 crop 和动态距离矩阵的自定义 C。当前 C≈1.06 与历史 C≈4.74/5.86 不在同一测量尺度，不能直接比较绝对值或增益大小。
- [visual_boundary] 历史 multiset 的 Ditto 输入是 `data/data/image/{i}.png`，两臂共享一张静态图；LRS3 子集的图像来自 LRS3 片段。它不是本次已撤回的“把 LRS3 动态源视频送进 renderer”泄漏，但属于同源、同域的 LRS3 参考脸。当前 corrected protocol 按要求改用 MEAD/VFHQ/TalkVid/GRID 的非 LRS3 首帧并重复到 mel chunks，因此是更严格的无泄漏估计，也会改变 TFG 的人脸/口型匹配条件。
- [audio_text_mismatch] 历史 multiset 的 TTS 使用单一 faster_qwen3 0.6B ICL，文本是对每个 clip 调用 qwen3-asr-flash 得到的片段转写；当前 run 使用 LRS3 官方带词时序文本，并混合 qwen_cloud/qwen_local/index_tts2/cosyvoice2。当前部分输出相对自然音频明显缩短，可能改变音频-视频时钟与 SyncNet 支持。
- [estimand] 历史是 n=50 的记录级官方分数；当前是36个来源块、EQUAL_COUNT、固定 box、块级配对/稳健统计。样本量和估计对象都变了，功效下降可以解释“不显著”，但不能单独解释 +1.124 到 +0.053 的点估计变化。
- [boundary] 旧的 LRS3 动态视频输入候选 run 仅作审计、不得用于结论；历史 Ditto 静态图结果也不能作为当前无 LRS3 视觉协议的直接基线。当前主结果应表述为“外部非 LRS3 视觉 + Wav2Lip + 自定义 SyncNet 端点下增益弱”，而不是“所有 TTS 都没有 TFG 增益”。
- [next_control] 最小定位实验：固定同一批 LRS3 音频/文本、同一批外部非 LRS3 首帧、同一评分器，做 Ditto vs Wav2Lip 两臂；再在同一外部视觉下只切换“clip ASR 文本 vs LRS3 官方文本”。先做12–18个来源块即可区分 TFG、文本/时长和纯功效因素；正式实验仍不使用 LRS3 视频输入。


## 2026-09-22 官方 SyncNet 复核结果

- [official_score] 对 corrected run 的全部108个 Wav2Lip media cells 使用仓库官方 `run_pipeline.py → run_syncnet.py` 重评分，`min_track=25`，S3FD 人脸检测、官方 SyncNet V2 权重；完成108/108，失败0。GPU启动前无其他 ML 计算进程，评分严格串行；结束后 GPU 已释放。
- [official_gain] 官方 Confidence（越高越好）的同 source_group 配对 TTS−natural 增益：qwen_cloud +0.736（95% bootstrap CI [+0.365,+1.144]，14/18为正）；qwen_local +0.408（[-0.035,+0.857]，12/18为正）；index_tts2 +0.400（[+0.187,+0.599]，15/18为正）；cosyvoice2 +0.613（[+0.097,+1.052]，14/18为正）。72个 TTS-natural 配对总体 +0.539（[+0.335,+0.741]，55/72为正）。
- [official_distance] 官方 Min dist（越低越好）变化：qwen_cloud -0.237、qwen_local -0.228、index_tts2 +0.059、cosyvoice2 -0.258；总体 -0.166。Confidence 的结果比原自定义端点更一致地支持当前存在 TTS 增益，但 IndexTTS 的 Min dist 未改善。
- [metric_diagnosis] 同一批、同一视频的原自定义 EQUAL_COUNT C 增益为 +0.073/-0.004/+0.086/+0.056，而官方为 +0.736/+0.408/+0.400/+0.613；因此“当前增益不明显”主要由评分端点/裁剪流程造成，不能再把原自定义 C 结果作为当前 TTS 增益结论。
- [remaining_boundary] 官方复核后的当前无泄漏 Wav2Lip 增益已明显恢复，但仍低于历史 LRS3-Ditto 的 +1.124；剩余差异仍可能来自 TFG（Wav2Lip vs Ditto）、非 LRS3 外部视觉、官方文本 vs clip-ASR 文本以及 TTS 时长差异，不能归因于评分器单一因素。
- [artifact] 官方逐条分数、日志和中间裁剪结果位于 `runs/phoneme_tfg_association_external_visual_cpu_audit_v1_20260922/07_official_syncnet/full/`；汇总为 `official_gain_summary.json`，执行器为 `scripts/experiments/phoneme_tfg_association/official_syncnet_eval.py`。


## 2026-09-22 官方 Sync-C 下音素可分度关联复核
- [primary_official_association] 复用 corrected run 的36个独立source blocks、HuBERT layer6逐句phone silhouette与108条官方 SyncNet V2 分数；沿用原协议的“pair固定截距 + 同句TTS-TTS silhouette差分”统计模型。官方 Confidence 下 β=+0.452806 Sync-C/0.1 silhouette，HC3 95% CI=[-0.130703,+1.036315]，HC3 t p=0.123333，null-imposed studentized wild p=0.0741，状态仍为 INCONCLUSIVE。按3位小数展示应为 β=+0.453。官方评分让点估计从自定义端点的 -0.091 转为正，但尚不能称为统计确证。
- [audit] 2026-09-22 从 blocks.json、features.jsonl、官方 summary.json 重新按 sample_id×arm 关联，复算主模型及9999次wild bootstrap一致；另外用独立NumPy最小二乘/HC3公式得到相同β和CI。108条分数各有一条Confidence日志，36个source_group唯一；特征音频SHA/PCM SHA与生成receipt一致，同句三臂参考帧/crop/视觉源/checkpoint/seed一致，未发现配错。检查使用CPU，无新视频或GPU评分。
- [design_validity] 四模型分别18句且子集不同，是预定平衡不完全区组设计：每句natural+两种TTS，六种pair各6句。主分析 x=(S_b−S_a)/0.1、y=(C_b−C_a)；共享natural在两次差分中严格抵消。因而natural子集不同不使该主关联失效，也不要求四模型有共同交集。句内差分控制共享句子/视觉因素，pair截距控制组合平均差异；它不能排除语速、发音忠实度等随TTS变化的混杂。
- [estimand_boundary] 主模型回答“同句两种TTS的可分度差是否伴随Sync-C差，扣除组合均值后是否仍有关联”；不直接等于natural与所有TTS总体的绝对S-C关系，也不识别自然→TTS增益的因果机制。各模型原始均值/平均增益排名不是完全同样本四模型比较。
- [endpoint_correction] 官方适配器 official_syncnet_eval.py 直接执行 run_pipeline.py→run_syncnet.py 并读取各视频自身完整检测轨迹的Confidence，没有原自定义评分的INTERIOR_EQUAL_COUNT窗口选择。official_association.json 中 endpoint=official_syncnet_v2_equal_count_sync_c 的标签不准确；实际应描述为official full-track Confidence。原始JSON本次保留未改。官方结果属于用户要求切换评分器后的复核，沿用原统计模型，但不能声称完整沿用了预注册评分端点/支持。时长/轨迹支持差异仍需敏感性检查。
- [descriptive_official_association] 72个TTS cell的简单描述性相关：绝对silhouette与官方Sync-C Pearson r=-0.156、Spearman rho=-0.083；TTS相对natural的silhouette差与Sync-C gain Pearson r=-0.112、Spearman rho=-0.087；36个同句pair差分不加pair固定效应时Pearson r=+0.149。72行共享来源和natural，原输出普通Pearson/Spearman p值不能作为来源聚类校正的推断；JSON末尾“source-group clustered”不能理解为这些描述性p值已经聚类修正。
- [partial_association] CPU核查：将36行x、y分别扣除六类pair均值后，残差Pearson r=+0.242926（约+0.243），仅作与主回归一致的效应量描述，显著性仍使用主模型HC3/wild结果。
- [exploratory_sensitivity] 本次额外CPU探索性复算官方端点，加入同句log(duration_b/duration_a)及silence_fraction_b−silence_fraction_a：β=+0.401260，HC3 CI=[-0.166301,+0.968821]，依然跨零。这不是重新预注册的主检验，也不能证明消除了全部混杂。
- [interpretation_boundary] 当前数据支持“官方scorer修正了原自定义端点对TTS增益的低估”，但对“可分度更高伴随更高Sync-C”仅有不确定正向趋势。不能称为确证正相关，也不能称为没有关联；不能把中文旧实验结论直接迁移到本批LRS3英文/Wav2Lip/外部静态视觉条件。
- [artifact] 官方关联输出：runs/phoneme_tfg_association_external_visual_cpu_audit_v1_20260922/07_official_syncnet/full/official_association.json；逐条分数在同目录summary.json与scores/。主统计实现 scripts/experiments/phoneme_tfg_association/analysis.py::build_block_differences, fit_primary。