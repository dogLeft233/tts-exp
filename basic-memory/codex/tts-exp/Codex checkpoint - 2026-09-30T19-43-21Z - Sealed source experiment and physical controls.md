---
title: Codex checkpoint - 2026-09-30T19-43-21Z - Sealed source experiment and physical
  controls
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-09-30-t19-43-21-z-sealed-source-experiment-and-physical-controls
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/tts-mechanism-20260930
git_sha: 5f2941a485a0834aee72d399e680d1912340620d
started: 2026-09-30 19:43:21+00:00
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
trigger: compact
model: gpt-6.1-sol
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-09-30T19-43-21Z - Sealed source experiment and physical controls

## Summary
新的 Qwen 自克隆主实验已完成两张脸的全部 2,560 行并封存，独立复核正在接入服务器 CPU；物理声源/谱形移植和女性外部复核继续准备。

## Story
用户要求从本地代码、Git、BM 和百度网盘找回 TTS 增强 TFG 的研究路线，并在可用服务器上继续因果实验，直到形成明确结论。用户授权 Qwen TTS、下载数据和重跑实验；root 负责理论与解释，三个 gpt-6-luna/max 子代理负责代码、实验、独立审阅。项目内容保存在本地 tts-exp BM。

研究把生成端变化与 SyncNet 评分端变化分开。新 AISHELL 男声 cohort 为 CAL 10 speakers×2 clips、EVAL 40×2，Qwen 同文本自克隆100条全部完成。来源方向干预、响度正交干预、随机等剂量、raw/unit、固定lag、置换和两张脸均已提前冻结。主实验 retry3 已完成 primary/secondary 各80 clips×16 rows；root 尚未读取正式效应摘要，先核验封印、剂量和缓存矩阵。短 T clip 的 guard20 空支持保留为 too_short；没有改抽样或提升次要端点。

物理确认通过同音素映射，在 recipient 原时间轴上分别移植 WORLD excitation、谱形和局部能量，双向、各9臂。CAL 的完整20对 QA、条件 AP 与能量残差还在获取中，root 冻结 QA 后才生成 EVAL 音频。女性40条独立复核沿用原男声 d/u/k；官方 HF 子集仅有男声 WAV，故按预设女性 IDs 下载完整 OpenSLR 官方包并验证公布 MD5，未更改端点。

## Working State
Durable: 当前本地 HEAD 为5f2941a，未推送；无 PR（GitHub REST直接查询为[]）。主实验 protocol SHA 8ba27fcb395dccbb493ad4831e8b68ac18c485b61e39d6e3b7b3018fd668cf39，rows SHA 19f805f6935c85475ad28319e17d2694a333a85d055d3e19a0eb93bda7390dab，evaluate SHA e50a31eaa2493b83e87cb4426b742e8f28e73836cbb9374b2319e7b60c940880。正式运行位于 /root/autodl-tmp/tts-research-20260930/portable_source_axis_20260930/evaluation_20260930_retry3，80+80 clips、2560 rows、两阶段 exit0。继承链最终校验3033/3033，parent mismatch0、非detached child mismatch0、missing0。

Core 分析包 manifest 位于 runs/recovered_20260930/remote/evaluation_20260930_retry3_analysis_20261001/analysis_transfer_manifest.json，SHA 98350b737fcb40434da7fb653b9560869adaf4c9d85cc070f6c55939df47dd73；301片、6311成员、1.587GB。results JSONL SHA fd1f9ce1b23861de5eea7d87470f74e52451f1cc8bb050cbe85141a62975d3fa，14组字节分片按封印顺序重组。下载暂停于62/301片；独立 consumer 可在服务器 CPU 先运行，受全归档/成员/延续链验证门控。最新 sealed runner SHA b4efe233bbb8b770f184c2731ea041dbc59626a32009874eeaf5433c8dedf79f。

CAL visual tone v2 已在正式效应读取前冻结：只去 IPA 调号 U+02E5–U+02E9，不去 aspiration/palatalization。旧 freeze 原始字节 SHA45cc2ee35980177b126a05b79fb19caa06e43624c24c1c7f85c41575c677b287 已恢复；绘图是独立 sidecar。双类 joint14/20 clips、双clip完整5/10speakers；原事件 matcher −5frame 控制失败，event_metric_calibrated=false，事件时间只探索。Phone aperture separation 与 scale-invariant AUC 分开；full80 primary支持集保持不变，secondary availability 由原N音素/共同有效track定，禁止按效应筛选。

WORLD 主 spec 保持旧冻结 SHA f085e291…e62ca；execution-bindings.md 新 annex SHA cde66fa11f140866fc87f07b7a39ca68ddefc403ff7839b239987549e9f7e47d 明确 phone-wise envelope 查询、能量目标、原k3来源、文本/波形/版本绑定及方向。producer SHA7a6661afced892037d5f32fa3760e8d0906b77f40bb06f5f3ce53a66a04729b2。成功 CAL one-pair smoke 各9/9臂、长度/RMS/RT bitexact合格；完整20对仍需检查旧前台进程与输出，不能当作已完成。

Female 官方 data_aishell.tgz 为15,582,913,665B，公布 MD5 2f494334227864a8a8fec932999db9d8。最新子代理报告230/233 blocks已校验；完整哈希尚未读回。resource 包 MD5 957d480a0fcac85fc18e550756f624e5 已核验，speaker.info 与原 pinned SHA569cef88f7fb1c3173cf3a45b00b4659b427d90af120af91513a2c6f151f7e08 相同。20F S0124–S0143各前两条 eligible WAV，旧盘4条已知自然语音只排除精确utterance，保留 source selection audit。root 后续运行 Qwen key 只走运行时环境。

Machine-local/fragile: runs/、BM、.codex/scratch/忽略/未提交；各 agent 已有未tracked脚本和测试，需完成 QA 后选择性提交。远端 host root@connect.weste.seetacloud.com:12349 已在server-access记忆持久保存。旧 loopback SSH bridge 已停止，新的短期会话已连接，凭据/令牌只在运行内存，不进入笔记或文件。CPU长任务后续使用 nohup 与 start/exit sidecars，避免桥接重启影响。MFA CAL40 grids合格，EVAL160尚待完整 qualification/bundle；GPU队列为 CAL runtime→sham80→WORLD→female。

## Changed Files
- scripts/experiments/portable_source_axis_20260930/independent_consumer.py 与 independent_validator.py：按缓存 A/V SHA 去重，独立重建所有 unique raw/unit cross 距离及边界零距离；重复行 cross hash一致，旧抽样 aligned-lag parity额外保留。
- scripts/experiments/portable_source_axis_20260930/independent_sealed_run.py：全301 tar/6311 regular成员、14组重组、原run及恢复元数据、runtime/CAL axes封印门控，再执行CPU consumer。
- openspec/changes/confirm-world-source-filter-20261001/execution-bindings.md：物理实施补充边界，主spec未改。
- openspec/changes/tts-sex-replication-20261001/spec.md：正式全OpenSLR下载/覆盖与官方校验路径修订。
- scripts/experiments/tts_mechanism_visual_tone_20261001.py、对应test/spec：CAL前冻结 tone adapter 与可用性/事件端点分离。
- 未tracked CAL benchmark true/false runner、resampling_sham.py、female selector、WORLD producer/readonly AP与能量sidecar；尚未跑新GPU对照。

## Verification
root直接读取 Git状态/HEAD/branch/remote、项目配置/router/schema及上个checkpoint身份，查询 GitHub canonical repo、已推送 baseline 和 PR[]。新桥接连接成功。子代理报告独立 seal/consumer/validator targeted tests20通过，visual wrapper suite41通过，WORLD producer8通过；root未另跑测试。正式主实验exit0和3033文件校验来自封印侧car，效应尚未解读。

重要记录更正：calibration_summary.json 在 forward后报告 cudnn_benchmark=true；eval只在人脸检测前设置false，外部 detector 开启true，无post-detection reset或per-forward标志记录。不能将旧正式运行记为实际false。新不可变 correction sidecar保留原封印，将先做前两CAL样本true/false工程敏感性；新sham各四格，包括q00基线，统一post-detection false并记录实际forward。WORLD缓存桥接只有在实际runtime相容时使用，否则重生成。

MFA identity更正：qualification raw SHA dbebb40ce39f1a8f4a71493c0ae19871e97948d186908d2f1545e0015a301650，embedded fingerprint556d8577…e10832；alignment protocol raw aeb753c937a8c1dfd8d6d837f4ee79d6b653a1aa6cfdb6479b2a86e154dc6f18，embedded17a61ff1…3bc05f。不要把 embedded 值写为原始文件 SHA。旧visual snapshot5a43…4d0f和新metadata-only core3a28…ea15不同，不能覆写原对齐封印。

## References
- [Canonical repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed baseline c4a1998](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- 当前 HEAD `5f2941a485a0834aee72d399e680d1912340620d` 为本地未推送提交。
- 主 spec：openspec/changes/confirm-tts-source-mechanism-20260930/spec.md；physical spec/annex、female spec、visual tone spec 为后续入口。

## Observations
- [result] 新主实验已封存两张脸2560行；正式效应待独立消费，不能将完整执行等同于机制成立。
- [decision] 统一评分协议、原cohort、原lag3、raw/unit/guard域；短样本缺失与事件QC失败单列，不提升次要端点。
- [decision] 后续重采样和物理干预统一记录实际cuDNN状态，重采样基线重新生成，原封印保持不变。
- [decision] FULL-track WORLD AP受UV混合影响，解释依赖双方voiced/recipient-active同音素统计；能量目标有限迭代/饱和残差必须报告，不能声称精确匹配。
- [blocker] 本地分析归档未完整、WORLD全CAL QA和MFA全EVAL qualification待取得；服务器CPU可先推进主效应统计。
- [next_step] 在新桥接上首先上传/启动独立CPU全封印复核与consumer，读取合格后的四主门槛、随机剂量、unit/null/anchor和完整支持集统计，并据此推进物理确认。

## Relations
- continues [[Codex checkpoint - 2026-09-30T18-24-22Z - Source mechanism continuation and physical transfer]]
- relates_to [[解释 TTS 的 TFG 与 SyncNet 优势]]
- relates_to [[TTS 来源方向与声学机制确认]]
- relates_to [[TTS 机制确认分离生成端与评分端]]
- relates_to [[WORLD 声源谱形与局部能量的双向移植]]
- relates_to [[女性说话人上的 TTS 来源方向独立复核]]