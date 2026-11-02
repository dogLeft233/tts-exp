---
title: Codex checkpoint - 2026-10-08T05-28-23Z - Actual Dev binding and plan awaiting
  QC06 consumer correction
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-08-t05-28-23-z-actual-dev-binding-and-plan-awaiting-qc06-consumer-correction
agent: codex
model: gpt-6.1-sol
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
repo_root: E:/Documents/tts-exp
capture: deliberate
branch: codex/cross-tfg-audio-head-20261003
repository: dogLeft233/tts-exp
username: dogle
hostname: LAPTOP-E133GSLE
cwd: E:/Documents/tts-exp
trigger: compact
status: open
project: tts-exp
started: '2026-10-08T05:28:23.7322528+00:00'
git_sha: 6126c7d1aeba5498f1f61042a49afe8f1e9ff59f
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-08T05-28-23Z - Actual Dev binding and plan awaiting QC06 consumer correction

## Summary
真实CPU收集与计划构建已完成，Dev GPU启动前最后的预检遇到QC06收据字段选择错误，正在一次性核对完整producer/consumer接口。

## Story
项目从解释中文TTS增强TFG，推进到开发跨模型自然音频增强头。用户要求持续迭代到可用方法，增强PCM只用于视频驱动，交付换回完整原自然F32音频和原源时钟；允许重跑实验、下载历史数据、Qwen ASR/TTS和网盘。根代理维护研究主线、独立复核及私有SSH，GPT-6 Luna/max子代理实现与实验。服务器已迁移到root@connect.westd.seetacloud.com:18625，密码与API key仅存在根代理私有运行环境。

冻结Fit40/20说话人Adam120三遍候选未变。Dev20/10说话人×三个种子×Wav2Lip/MuseTalk/Ditto固定为360个identity/RMS对照和180候选；三模型各自为正、等权超过RMS +0.050以及D0/Canchor、内容、视觉、GUARD20、完整失败分母共同决定是否接受。Confirm80/40未见说话人与LatentSync仍封存。Fit结果支持时变门控改善零延迟距离，Sync-C还受频谱和背景距离影响；本阶段需要真实TFG生成及独立视觉检查才能定案。

上一交接之后，根代理实际确认四个MuseTalk/Ditto原native回执已在服务器且整文件身份精确匹配，因此没有执行多余恢复。新版本移除了原producer不存在的内部自哈希要求，保留外层SHA、字节数、完整解析对象相等和schema/status/row/model/seed。随后真实CPU运行依次暴露三个接口错误：controls把repeat_row_ids读在source_pins.lineage而非entry.lineage；原Wav2Lip回执计数字段为wav2lip_generator_forward_count；collector给19个已整文件验证的backend依赖留下bytes:null。修复采用不可变新源码版本，依赖大小由同一真实verified_dependency记录回填，计划构建的严格整数大小门槛保留。候选guard同步修正实际字段，rebinder修复了旧guard自pin。

最新实际recovery06正确封包requests_attempt02的q00至q06全部PASS，q07 collector09和builder15均完成：runtime_binding_attempt07及controls_plan_attempt07的plan/manifest/360行JSONL已经真实生成。controls16预检随后失败：已正确核对顶层QC06 source与determinism policy SHA，却又把全部五个QC06字段都要求出现在qc06_receipts；collector按契约只在该对象保存三个原始收据SHA，另外两个位于顶层。当前没有preflight07、summary06或Dev GPU结果。已安排dev和独立signal子代理同时核对整个剩余validate_bindings与真实binding07/plan07，避免继续逐字段盲修。

## Working State
研究持久入口是Basic Memory任务“开发跨 TFG 的自然音频增强头”、实验“跨 TFG 自然音频增强头的训练与原音换轨验证”、机制结论“冻结增强头的门控时序与频谱贡献”及ADR017。任务snapshot101/实验100本次实际更新并读回，但还未纳入recovery05/06的新接口诊断，下一次更新需纠正Task101中手抄的65字符probe SHA，原文件实际64字符值见下方。CONTEXT.md仍93行，HANDOFF.md被Git忽略且最新恢复段落偏旧，均需在真实启动里程碑后更新。

当前本地HEAD是未发布6126c7d1aeba5498f1f61042a49afe8f1e9ff59f，原19文件同机执行链与snapshot01已提交；新collector07→09、builder13→15、controls14→16、candidate09→11、rebinder10及新封包未提交。五个无关历史源码/测试脏文件必须保留。不可覆盖任何已消耗源码、封包或旧响应。

私有PTY57588、bridge07端口12249 PID10892是易失运行状态。远程根/root/tts-research-system-20261001，namespace cross_tfg_audio_head_20261003，GPU UUID GPU-d09622d0-3535-e662-bb45-3df2b049fcea。机器本地scratch、SSH、下载和B盘CloudFS放置不是云端同步证明。容量协议用真实pilot加显式两倍余量，540行场景需求22,972,800,792字节，执行仍检查GPU≥18GiB、RAM≥16GiB、SHM/输出/cgroup并逐行继续检查。不要重跑已有资格、重封seed、删除整个SHM或改科学冻结。

## Changed Files
- 已安装且真实消费：`scripts/experiments/cross_tfg_audio_head_20261003/collect_v3_samehost_runtime_binding_attempt09.py`、`build_v3_samehost_dev360_controls_plan_attempt15.py`、`run_v3_samehost_dev360_controls_attempt16.py`、`run_v3_step120_candidate_samehost_attempt11.py`、`rebind_v3_step120_candidate_plan_samehost_attempt10.py`。根代理完整核对五个最小差异与fresh worker差异。
- 最新源图指针：`.codex/scratch/dev360_samehost_source_snapshot_attempt04/source_snapshot_attempt04.json`，15753字节，raw SHA7c7db8de68e625f2e1f590347ea635a182fcf5f8db8d5387233af8898f2325cb，尚待完整根复核及最终有效源图提交。
- 最新实际封包：`.codex/scratch/dev360_phase2_native_receipt_fix_attempt01/recovery_attempt06/requests_attempt02/request_index_attempt02.json`，7812字节，raw SHA897f55967bd2b0f59bd505d3be49e289de857f9657b7ddc3ba6fde3286e66a26。旧requests_attempt01错误probe digest在根代理与独立审核时发现，未执行。
- 四个真实产物只读下载封包：`recovery_attempt06/failed_output_readback_attempt03/requests_attempt01/request_index_attempt01.json`，根代理已完整读index和四个download请求，正在私有PTY依序下载。
- launcher/status05已由根代理完整继承审查并核对新retarget差异，仍为source-only、尚无GPU启动；应等预检有效后按最新实际summary重定向。
- postcandidate视觉joiner attempt02增加完整540行分母和未知状态拒绝；子代理已审查，根代理尚未读完整7424字节差异。ASR仍未调用API。

## Verification
公共源码installer02、全部22项输入恢复probe02以及native4 exact-existing probe实际PASS，原恢复与资格不变。inputprobe02原12117字节SHA5f162dc906d614b72037592ae4707a3808b7c88d94f3e1489d21ebf8fccb91de；native probe响应5130字节SHA66c814d7a40a804fced1e5c80bc519a0a97ad5c4e2e3116ff38d64365dbcd9d9。

recovery05实际binding06下载566494字节raw SHA59ff6719eb15e6ebb9bd99eae287b30af8c1f2d08cc87a253d1d74100b379a94。根代理独立核对raw/canonical self hash、63真实依赖身份、6,827,187,137字节唯一依赖合计、全部19个null bytes与同一实测记录匹配、20 contexts及三个backend lineage PASS。独立子代理仅内存回填大小并翻译本地路径后对真实binding执行完整builder gate PASS；这不是远端preflight PASS。

recovery06实际q00至q06 PASS；q07 collector与builder输出成功后，controls16.py:589因QC06三收据与五字段选择错误FAIL。没有运行通用测试、没有启动Dev GPU、没有新的跨模型分数。读过所有新小源码差异、封包、Bash UTF8 stdin -n语法及整文件pins；未把工程失败写成科学阴性。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)，本次直接只读HTTP200核验。
- [已推送历史提交c4a1998](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)，直接解析HTTP200。
- 当前HEAD `6126c7d1aeba5498f1f61042a49afe8f1e9ff59f`为本地未发布提交；分支`codex/cross-tfg-audio-head-20261003`的直接GitHub搜索没有匹配PR。

## Observations
- [result] 真实CPU collector09与builder15已生成binding07及plan07/manifest07/360 planned rows；尚无Dev GPU结果。
- [decision] 新版修复真实producer/consumer契约，保留全部科学冻结、原文件身份和失败分母；不为通过预检虚构或放宽证据。
- [blocker] controls16把顶层已核对的QC06 source/policy SHA重复要求进入仅含三个原始收据的qc06_receipts，阻止最后预检完成。
- [next_step] 完整核对真实binding07/plan07与所有剩余预检consumer，发布最小一致新版并完成fresh CPU preflight，再实际启动360行Dev对照。

## Relations
- continues [[Codex checkpoint - 2026-10-08T04-54-35Z - Restored Dev inputs and native adapter schema gate repair]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[冻结增强头的门控时序与频谱贡献]]