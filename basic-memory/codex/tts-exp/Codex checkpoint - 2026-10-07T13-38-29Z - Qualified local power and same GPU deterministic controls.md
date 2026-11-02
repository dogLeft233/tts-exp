---
title: Codex checkpoint - 2026-10-07T13-38-29Z - Qualified local power and same GPU
  deterministic controls
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-07-t13-38-29-z-qualified-local-power-and-same-gpu-deterministic-controls
cwd: E:/Documents/tts-exp
repository: dogLeft233/tts-exp
capture: deliberate
git_sha: b5bcd8b1759a902d4e154fe183dfb9effbd7ca5d
repo_root: E:/Documents/tts-exp
project: tts-exp
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
username: dogle
status: open
trigger: compact
hostname: LAPTOP-E133GSLE
started: '2026-10-07T13:38:29.0912192+00:00'
model: gpt-6.1-sol
branch: codex/cross-tfg-audio-head-20261003
agent: codex
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-07T13-38-29Z - Qualified local power and same GPU deterministic controls

## Summary
新服务器的十一参数局部声学资格和确定性重复控制均已独立复核通过，后续训练与三模型对照将统一绑定新 GPU 的确定性执行产物。

## Story
用户要求持续做成可用于多种 TFG 的自然音频增强头：增强 PCM 只驱动视频，评分与交付恢复精确原自然 PCM 和原时钟。根代理负责机制推理、协议、BM、Git 与独立复核，代码和实验交给 GPT-6 Luna/max。仍采用 Fit40/20spk 固定120步、Dev20/10spk×三种子×Wav2Lip/MuseTalk/Ditto；以三模型均正且等模型平均比 RMS 再增0.050为接受门槛，保留 D0、Canchor、内容、视觉、完整失败分母和 GUARD20。Confirm80/40未见说话人与 LatentSync 在方法冻结前继续封存。

迁移后服务器为 root@connect.westd.seetacloud.com:18625，GPU UUID 为 GPU-d09622d0-3535-e662-bb45-3df2b049fcea。旧 SHM 全部消失，40条原始 WAV 和 Wav2Lip/SyncNet 持久模型可用。十一参数 CPU attempt10 实际完成160检查；根代理不导入 worker/head 的 NumPy verifier04 重新计算480项物理40ms统计、所有约束、原始PCM和梯度证据，PASS_INDEPENDENT_NUMERIC_READBACK。实际测量梯度8次、另32点复用零点梯度，已明确记录，不误报40次独立梯度测量。这是声学构造资格，尚无 TFG 增益结论。

新 GPU 的 QC05 三臂真实重复失败后保留全部失败源和产物。QC06 固定随机种子、单线程、严格确定性算法、关闭 TF32/cuDNN benchmark，实际108.797秒返回0；同输入两次视频码流、裁剪、轨迹、bbox和完整评分矩阵精确相同。根代理下载27项产物并独立核对全字节和SHA。identity C=6.339759826660156，RMS C=6.397294998168945，此 pilot 仅作工程控制。其 RMS 收益与旧执行设置不同，因此 ADR015 明确要求新 GPU 重跑 Dev360 自然/RMS对照，再比较180候选，旧900/540归档只能提供历史来源。

已审阅并提交 Dev 晚绑定执行源01/02，含 cgroup 内存限额；尚未运行候选。Exporter04 新源码恢复精确91520样本原音PCM SHA门槛，并在 face_detect 前采用 QC06 确定性设置，尚未执行。下一步实际导出新 context、绑定新 Fit40 producer05、通过zero-head40后执行新training09固定120步。旧08部署不可变。当前仍无优化器更新、学习权重、终态120步重评分、新 Dev 或 Confirm 结果。

## Working State
HEAD `b5bcd8b1759a902d4e154fe183dfb9effbd7ca5d` 为本地未发布提交，分支 codex/cross-tfg-audio-head-20261003。Task80 active、Experiment79 implemented 当前快照应在新context产物完成后更新；ADR015已创建并直接读回。CONTEXT当前93行已记录CPU资格及QC05，尚待更新QC06/ADR015。既有五个dirty文件保留，当前完整状态见`.codex/scratch/checkpoint_EU_git_status.txt`；ignored HANDOFF、runs、scratch、PTY57588和bridge07均属本地或易失状态，凭证不入文件或BM。

CPU资格的本地持久副本为 `runs/cross_tfg_audio_head_20261003/local_power_signal_qualification_attempt10/`，17项全字节核对复制，manifest为 `root_evidence_copy_manifest_attempt01.json`；verifier04 SHA `2d394f0ee260b9ccb6bebc00285bbf9c72451cdca8c3e2c2b2deeef0f97bc056`，独立结果为该目录 `root_independent_numeric_readback_attempt04.json`。

QC06权威终态响应 `.codex/scratch/qc06_packet_attempt01/05_root_read_terminal_status_attempt02.json.response.json` SHA `0b81ea4667243cdce94050d59058423ac0e3aea6850db1385c05c2892eb05136`；manifest SHA `f09cc4f339c43e96ea532cdaac4f9818fc42caec6ba7b212dcfe7ae4dc2c5436`；27项下载入口 `.codex/scratch/qc06_readback_root_attempt01/artifacts/` 与 `root_whole_file_readback_qc_attempt01.json`。远端控制位于持久 NS/job_control，媒体/context运行目录在SHM，需要及时原字节备份。

实际 cgroup 可用64.91GB、SHM33.25GB、overlay7.66GB、autodl-tmp仅188.76MB可用；不能按宿主477GB可用内存或autodl-tmp大盘容量排任务。本地 B: 是已挂载CloudDrive，约3.758TB可用；远端未发现可用云盘CLI或mount。Ditto HuBERT和部分Muse模型旧路径在已丢失SHM，需从持久来源/本地挂载恢复或下载并校验固定SHA；Wav2Lip训练可独立推进。

## Changed Files
- `CONTEXT.md`：CPU资格快照，提交0f3359b。
- `scripts/experiments/cross_tfg_audio_head_20261003/v3_dev180_newhost_latebind_packet_attempt01.py` 与 `...attempt02.py`：实际配置/120步receipt晚绑定、执行和收集、cgroup限额，提交d8bf639，仍未运行。
- `scripts/experiments/cross_tfg_audio_head_20261003/engineering_qc_oldpilot_attempt05.py` 与 `...attempt06.py`：保留真实失败并实现严格确定性重复，提交b5bcd8b。
- Exporter04未提交，32028字节 SHA `cd5103349f8b0bd40ea455464065eaf4bc6202b5a568582b844b1188a4d12e1a`，尚待根代理请求审阅与实际导出。

## Verification
根代理实际运行 NumPy verifier04 返回0、读取完整独立结果、核对17项资格副本。QC06五请求与终态实际完整无截断/错误，3臂PASS、5重复门槛精确相同；27项下载全字节SHA独立PASS。原声PCM和时钟门槛保留。Dev01/02与QC05/06源完整审阅，Exporter03→04差异已读。未执行通用测试、未调用新增API、未开启Confirm；方案和未运行源码不写成实验成功。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
GitHub当前分支无匹配PR；b5bcd8b及本轮0f3359b/d8bf639均本地未发布。

## Observations
- [result] 十一参数局部声学构造已实际完成并通过根代理独立数值复核，尚未用于TFG效果检验。
- [result] QC06在新GPU实现相同输入完整视频/轨迹/评分矩阵精确重复；QC05失败保留。
- [decision] ADR015要求新自然/RMS对照360行与候选180行共享同一新物理GPU和确定性设置，旧归档不充当新候选的配对对照。
- [decision] 科学cohort、种子、接受门槛、完整失败分母、内容/视觉保护和Confirm封存不变。
- [blocker] 新context、Fit40缓存、zero-head40和固定120步尚未完成；Ditto/Muse的SHM权重需恢复。
- [next_step] 执行并读回Exporter04新context，以实际SHA晚绑定producer05和training09，先通过zero-head40再运行固定120步与终态重评分。

## Relations
- continues [[Codex checkpoint - 2026-10-07T13-02-41Z - Completed local power signal run and isolating migrated GPU repeat control]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[015-same-gpu-deterministic-controls]]