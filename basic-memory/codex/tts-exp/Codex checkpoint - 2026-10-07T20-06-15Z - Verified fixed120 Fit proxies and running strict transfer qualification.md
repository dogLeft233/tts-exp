---
title: Codex checkpoint - 2026-10-07T20-06-15Z - Verified fixed120 Fit proxies and
  running strict transfer qualification
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-07-t20-06-15-z-verified-fixed120-fit-proxies-and-running-strict-transfer-qualification
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
started: '2026-10-07T20:06:15.0368322+00:00'
cwd: E:/Documents/tts-exp
model: gpt-6.1-sol
repo_root: E:/Documents/tts-exp
status: open
project: tts-exp
hostname: LAPTOP-E133GSLE
username: dogle
agent: codex
branch: codex/cross-tfg-audio-head-20261003
trigger: compact
git_sha: 6a9220efc74f89c7b87ed4235da125ab98504d4b
repository: dogLeft233/tts-exp
capture: deliberate
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-07T20-06-15Z - Verified fixed120 Fit proxies and running strict transfer qualification

## Summary
固定 Adam120 增强头的40条 Fit代理评分和完整原音/分析ROI读回已独立核验；统一三模型严格重复已启动，正准备未训练样本上的正式换轨迁移评估。

## Story
原目标是解释中文 TTS 对 TFG Sync-C 的增强机制，用户将目标扩展为适配较多模型的自然音频增强头，要求持续失败分析与方法迭代直到形成可用方法。增强音频只驱动视频，最终成片与评分换回精确完整原自然 PCM 和源时间轴。根 agent 维护理论、冻结协议、Basic Memory、私有 SSH 与独立核验；GPT-6 Luna/max 子 agent 完成实验与代码。

固定方法用32ms/8ms STFT、四频带静态项与正负谱通量门控的相加掩码、逐帧去公共频谱偏移、全局 RMS 归一化和3dB有界约束。Fit40/20说话人 Adam120/3遍训练已完整核验。新 rescore13仅修复 NumPy int32 诊断字段序列化，不改变权重或评分数学，实际完成40条。根 agent 下载全部七个原始数组/metadata、逐文件与分块SHA，独立重算31点距离曲线、端点、embedding、样本时钟与50k说话人聚类99%区间。ΔCbest=+0.057816875（28/40条、15/20说话人正；Fit99%CI[0.019059472,0.101961554]），ΔCanchor=+0.148420191（37/40、20/20正），Δfar=+0.141070795，ΔD0=-0.046865845（28改善、12变差）。这是冻结 Wav2Lip 神经前向及原生裁剪几何的代理评分，尚不是最终编码视频的官方评分或跨模型验证。

生产归档4979成员未改。rawdecode04实际完成40条完整自然PCM、40条精确源ROI、120个NPZ/AVI与80个包时钟核验；根 agent 独立重建ROI区间与原PCM SHA。旧status04错误地期待v1审计schema而失败，新只读status05期待v2并实际通过，不重跑worker或改产物。ADR016区分分析切片与完整交付PCM；完整成片仍保留全部原样本。

旧repeat05/Muse02原生控制失败保留。新的bootstrap03/Muse07/Ditto06统一原生子进程的sitecustomize路径，Wav2Lip仅增加历史librosa位置参数到关键字兼容，12源码闭包、17请求与Bash语法已逐一核对。strict09预检实际PASS并启动全部3模型6条；PID73073/start_ticks325802260，启动收据SHA c0ca2d780f341d5e85f8339f72fd8772bb51e4bcc6ad94aaaf3c0c465db8edab，GPU独占。尚未读取终端结果。

Dev180 metadata01/02因缺legacy builder/历史base plan在产物前失败。子 agent 已封存一次恢复完整20依赖的metadata03包，根 agent 尚待完整审核/执行。当前runtime binder旧版也会拒绝最终闭包；新binder02已准备，输入不可变template04，输出新template05并重算canonical seed hash。template04旧seed_sha256不参与strict09门槛，不能带入新Dev binding。Fit-only静态/正负门控错位诊断和小型训练/评分持久备份在准备，尚未运行。

## Working State
持久任务 [[开发跨 TFG 的自然音频增强头]] 为Task90 active；[[跨 TFG 自然音频增强头的训练与原音换轨验证]] 为Exp89 implemented。冻结Dev20/10说话人×3种子×Wav2Lip/MuseTalk/Ditto=360 identity/RMS对照+180候选；要求各模型正增益、等模型额外+0.050 over RMS，并核验D0/Canchor/content/visual/GUARD20和完整失败分母。Confirm80/40未见说话人与LatentSync仍封存。Fit诊断不触发权重选择或变更冻结Dev设计。

服务器 root@connect.westd.seetacloud.com:18625；GPU GPU-d09622d0-3535-e662-bb45-3df2b049fcea。地址保存在 basic-memory/docs/reference/server-access.md，密码/API key只在根agent私有运行时变量。远端持久namespace /root/tts-research-system-20261001/cross_tfg_audio_head_20261003，易失产物 /dev/shm/cross_tfg_audio_head_20261003。PTY57588 bridge07/12249为机器局部连接；训练11和评分13仍需小型持久备份。E盘约0.9GiB空闲，不复制263MB生产归档。GPU优先strict09，metadata与备份可CPU执行。

## Changed Files
CONTEXT.md的93行状态已单文件提交为本地未发布 `6a9220efc74f89c7b87ed4235da125ab98504d4b`；被忽略HANDOFF.md为20行快照。五个既有portable_source_axis/overlay/tests修改未动；大量既有未跟踪实验源码与忽略runs/scratch是机器局部状态，不作为已发布产物。

权威本地核验入口：`.codex/scratch/root_rescore13_raw_readback_attempt01/root_independent_final120_fit40_qc_attempt01.json`（SHA a128c293773e94c836ad03f2b608135545c4c0f1f0ddb61f55104ef5142faa41）；`.codex/scratch/fit40_native_rms_geometry_attempt05_rawdecode_terminal_readback_attempt01/root_independent_terminal_qc_attempt05.json`（SHA 15f0686ca5f6c1a92d07e95a93ec3f6ff8d4f4b722e964ec4d2e1e32873a76a3）。严格重复读取索引 `.codex/scratch/root_strict09_status_index_attempt01.json`；metadata03索引在 `runs/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/materialized_candidate_dev180_plan_attempt03/requests_attempt01/request_index_attempt01.json`。

正式执行链为 strict09全部6条实际通过 → binder02 → builder09/controls08实际360 → rebinder05/candidate04实际180 → joiner04/rawmedia03/readback02。旧direct-StageA包只保留源码/历史provenance，不作为最终执行路线。只读审计已确认原音完整PCM、每行前端/seed/native控制闭包；controls collector缺每cell相同context集合的提前断言，但下游adapter/joiner会拒绝重复或缺失context，根独立检查仍覆盖完整集合。

## Verification
实际PASS：完整Adam120账本/末步权重；七个原始rescore文件SHA与独立40行全端点；原生产归档全4979成员；rawdecode04全40完整PCM/40ROI/120NPZ/120AVI/80时钟及根独立核验；新只读status05；strict09源码/请求/资源预检与实际detached启动。保留失败：rescore09–12零持久行、旧repeat05/Muse02、rawdecode03、旧reader04 schema和metadata01/02依赖。strict09终端、metadata03、新binder、Dev540和Confirm尚未执行或核验；未运行通用pytest。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
本次新HTTP200页面确认当前分支无匹配PR；当前HEAD未发布。

## Observations
- [result] 冻结step120的Fit40代理ΔCbest=+0.058，40条原始数组已独立重算；正式视频/跨模型收益仍待Dev评估。
- [result] 全40完整原PCM和分析ROI实际通过独立读回，保持原始归档与失败证据。
- [decision] 保留冻结末步、同GPU controls、完整原音/源时钟、完整失败分母；Fit代理诊断不替代正式迁移验收。
- [blocker] 尚需strict09实际原生6行重复通过、完整metadata恢复和新runtime binder闭包，才能执行Dev540。
- [next_step] 读取strict09实际状态与原生重复证据，同时审核执行CPU metadata03和训练/评分持久备份，再用新binder闭包推进Dev540。

## Relations
- continues [[Codex checkpoint - 2026-10-07T19-37-14Z - Running fixed120 scores and repairing native render policy]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[016-source-clock-roi-and-full-delivery-pcm]]
- relates_to [[015-same-gpu-deterministic-controls]]