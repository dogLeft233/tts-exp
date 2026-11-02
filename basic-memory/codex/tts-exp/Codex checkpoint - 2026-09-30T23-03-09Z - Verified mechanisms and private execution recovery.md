---
title: Codex checkpoint - 2026-09-30T23-03-09Z - Verified mechanisms and private execution
  recovery
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-09-30-t23-03-09-z-verified-mechanisms-and-private-execution-recovery
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/tts-mechanism-20260930
git_sha: 39523b2a4e8a4abe428301df9ab75fcef576119a
started: '2026-09-30T23:03:09Z'
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

# Codex checkpoint - 2026-09-30T23-03-09Z - Verified mechanisms and private execution recovery

## Summary

已独立核验完整来源实验，确认音量经生成端改变 SyncNet 响应，并封存 WORLD 的逐音素物理干预；磁盘故障后已恢复完整方向/范数实验，继续随机对照、女性与原生视觉诊断。

## Story

用户最初要在旧电脑无法启动、记录未同步时从本地代码、Git、BM 和百度盘恢复研究路线，现在要求自主完成“为什么 TTS 增强 TFG”的研究，不能停在排除项。用户授权服务器、Qwen TTS API、历史数据和下载。Root 维护理论与主线，三位 gpt-6-luna/max 代理负责代码、数据与实验，项目记忆留在本地 tts-exp；Qwen 男性100条与女性40条均已完成。

完整主实验的独立消费已成功。原自然音频评分下，固定 RMS 的生成端效应在40男性×2 clips上为 ΔC=+0.192，99% CI [+0.111,+0.276]；第二脸 +0.135 [+0.071,+0.207]，单位化特征也分别保持 +0.027/+0.016。因此存在实际生成响应，不能全部归为评价器向量范数缩放。统一 CAL 拟合的 TTS 来源方向在自然80条主支持集上，orth+−baseline C=-0.019，99%[-0.048,+0.011]；四项确认门槛均未通过。这个静态方向不代表全部逐音素声学机制。

原生 T−N 在 guard20 只有39个完整说话人：原生 ΔC=+0.421，固定 RMS后 +0.213，属于控制后的差值缩小，不能写成中介比例，也不能把39人的原生结果与40人的方向结果混用。格式重采样四格80/40的 ΔC=-0.015，99%[-0.043,+0.010]。来源干预的可见口型复核640轨迹全部有效，但音素可用性只有16个完整说话人，静态来源方向在该子集的音素分离效应近零；这不是原生 TTS 视频的视觉检验。新原生320视频诊断已冻结，使用各自原始 N/T MFA 时钟，无物理 overlay。

发现旧消费者的次要 search_gain 字段错误地等于 C；定义应为 D0−D。保留原08字节，单独重算补充：attempt03 140537检查/0错误，2560 NPZ哈希与20464跨距离矩阵重建通过。新的方向 search gain orth+−baseline=-0.01646，99%[-0.04271,+0.01373]；观测减100次时间置换均值=-0.01556，99%[-0.04055,+0.01303]。主 B/D/C/anchor 数值未改变。

WORLD 的 CAL 独立参数/波形9060检查与能量迭代3810检查均无错误，120条能量波形 float32 位一致；没有按残差筛选或调参数。Root 已冻结 v2 授权80 pairs×两方向×九臂=1440波形，随后必须做真实 Wav2Lip/SyncNet GPU 四格和独立消费，不能停在波形生成。EX含 UV/F0/AP，EX减F0/AP不是纯清浊干预。原S0064“侵蚀”分词修复仅适用于物理移植，原视觉79/80资格与其余158 grids不改。

## Working State

权威输入及主run仍在 /root/autodl-tmp/tts-research-20260930/portable_source_axis_20260930/run/evaluation_20260930_retry3/。protocol SHA8ba27fcb395dccbb493ad4831e8b68ac18c485b61e39d6e3b7b3018fd668cf39；row plan19f805f6935c85475ad28319e17d2694a333a85d055d3e19a0eb93bda7390dab；producer e50a31eaa2493b83e87cb4426b742e8f28e73836cbb9374b2319e7b60c940880。Core transfer manifest98350b737fcb40434da7fb653b9560869adaf4c9d85cc070f6c55939df47dd73，301片/6311成员，最新本地150/301；继续传输且不当作全部本地已恢复。

服务器准确入口 root@connect.weste.seetacloud.com:12349；密码/API key只在用户消息和运行时内存，不入文件、BM或argv。loopback桥接器是易失状态，不记录token，恢复时由原用户凭据重连。

旧50G数据盘满导致 norm06在28完整clips、116评分行后中断，不能读取部分效应当结论。已SHA/权限核验搬迁2.6G纯下载缓存，旧缓存路径改为symlink，未改环境或科学产物；新私有输出根 /root/tts-research-system-20261001/，0700，系统盘约18G可用。不要写 /autodl-pub 公共NFS，不要误用 /root/autodl-tmp/tts-research-system-20261001/。迁移凭据在私有根 cache-migration/verify_copy.json 与 cache_migration_complete.json。

norm source23ee82bc421ad386765454e36222f78edffb8ace47923d1c94f9a7159ede3ecc，spec631e76042a523f598e9135de7ff9bf328f3656eabc800a0aa85f38f66bc01252。06实际两EVAL工程样本8评分PASS，之后盘满。07启动路径拼错，无forward且保留失败；08完整重跑已实际启动 PID751800，私有 source_axis/norm_angle_radius_20261001_attempt08_formal_80x4，首clip四格完成、0失败。全frame factorization q10=F32(z1*r0/r1),q01=F32(z0*r1/r0)，不要另加inactive identity要求；q00/q11 exact且报告继承activity mask。独立消费者正在编写，等待完整320行与seal后才读效应。

旧随机剂量51/22428活跃帧、37 clips的实际F32误差略超冻结1e-6公差，最大1.022595779e-6。保留旧产物、公差及随机失败，不删除有效正负方向主行。新640行辅助复核spec679b11c8a67b18ac397d63b95900d55865b6dbf1ad7e8662bf3a5b2e915c8c79，对每个搜索候选在cast F32后验证实际剂量，单独操作hash，不改旧ops。女性640首forward也必须用此修正。女性40 N/T封存完整；MFA80 grids完成，38/40合格，两OOV不重抽样；潜变量主复核仍用全部40。

WORLD root freeze v2 runs/recovered_20260930/independent_validation/world_cal_qc_freeze_20261001_v2.json，SHA b69cabb26a12407c144ceb20e325451517103d1f15d91d60819d9c64968a593c，继承v1 CAL QA，仅绑定实际执行 overlay source de324dfdb0418cd545c1a8ec918f17a5235e1a68d09a5d3d3f43941991a365dc；相对dba旧source只有最终stdout统计字段修复，diff SHA3e228f8903ae97f8419b941415c445d53cd4fbf9c798bc9762f4a159994cc205。EVAL view保留原160 EVAL grids/producer/资格/协议，搭配精确v2 CAL report/plan/QC/phone table，v2 producer与旧phone table另留provenance。local_recovery正在上传v2 freeze并启动1440 CPU，无EVAL波形效应已读。

## Changed Files

本地未推送协议：openspec/changes/confirm-latent-angle-radius-20261001/spec.md、confirm-random-specificity-20261001/spec.md、confirm-world-source-filter-20261001/{spec.md,execution-bindings.md,alignment-overlay.md}、diagnose-native-geometry-visual-20261001/spec.md（SHA b11c48d185beb30a3ae3f0036ca7ba02eb4d54ef24fdc13f5f96dfd3d71810bd）。Norm runner已选择性提交3c63ff4，原生诊断协议c5ff671，其他实际执行代码尚有dirty/untracked，不整体stage。

机器本地目录 .codex/scratch/、runs/、basic-memory/ 仍不是已推送科学归档。Dirty主consumer/validator/writer与旧test，untracked WORLD/MFA/女性/重采样/剂量脚本需选择性收尾；不新增或运行pytest。BM当前主实验、任务、ADR、WORLD与norm已修订；norm运行状态在盘满后有暂时过时文本，下一次更新需记录08而不是继续称06运行中。

## Verification

Root直接核验本地08主summary SHA55988932b41c0410f10acdb87db43ac30d21315d9ef1a5e06f423462c5403f1c、完整2560/160/0fail支持及四门/生成端/raw-unit效应。次要补充本地SHA768934c1ab011bf12dfb53d0ec49592802895d74018c594b1f4f088e39ccc86a已核hash；其指标详情由独立代理报告，root后续再直接读数。WORLD audit c3c2865a2d2fdade5c8aae280bbed4a0fcbcc785fada32caf3e42cd578aad8d1与能量audit f05b23eda4e8d6ff213cffa1bf441979ec21c2ba8fe1009a232e094cfba4e590均直接读回核验；freeze v1/v2及source/diff hash吻合。

GitHub只读API已确认canonical repo及已推送c4a1998，当前分支无PR；gh不存在。先查同agent/session的coding_session和codex_session共7条，无更多页，legacy搜索0条；已直接读上一checkpoint并核对精确agent/session。启动Router与实验指令以及checkpoint/writing skill已读。新阶段以真实runner产物、静态审阅与独立数组重建为核验，没有另跑pytest。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed baseline](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- runs/recovered_20260930/independent_validation/source_axis_consumer_attempt08/independent_consumer_summary.json
- runs/recovered_20260930/independent_validation/secondary_reanalysis_20261001_attempt03/independent_secondary_reanalysis.json
- runs/recovered_20260930/independent_validation/world_cal_qc_freeze_20261001_v2.json

## Observations

- [result] 固定RMS改变生成端，主脸和第二脸raw/unit响应均正；统一CAL来源方向在完整自然80/40四主门未通过。
- [decision] 保持完整计划分母、独立arm资格和raw/unit/anchor分层；次要指标错误另立补充，不改原封存主结果。
- [decision] 盘满后的完整实验在私有系统根从80全量重跑，保存旧失败；公共NFS不作为项目存储。
- [decision] 全部WORLD九臂、两个recipient方向、RAW reference与within-WORLD bridges均执行并报告，再综合物理机制、几何与视觉。
- [blocker] WORLD GPU评分、norm完整320独立消费、修正随机640、女性640及原生320可见口型仍未完成；完整历史9/27统计未恢复。
- [next_step] 完成正在重新执行的norm80四格与WORLD1440波形并封存，继续GPU物理评分及独立消费形成机制结论。

## Relations

- continues [[Codex checkpoint - 2026-09-30T22-02-00Z - Qualified main analysis and mechanism followups]]
- relates_to [[解释 TTS 的 TFG 与 SyncNet 优势]]
- relates_to [[TTS 机制确认分离生成端与评分端]]
- relates_to [[WORLD 声源谱形与局部能量的双向移植]]
- relates_to [[原生 TTS 优势的时间配对与可见口型诊断]]