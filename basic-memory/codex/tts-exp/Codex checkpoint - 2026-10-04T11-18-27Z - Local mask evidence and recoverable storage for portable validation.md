---
title: Codex checkpoint - 2026-10-04T11-18-27Z - Local mask evidence and recoverable
  storage for portable validation
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-04-t11-18-27-z-local-mask-evidence-and-recoverable-storage-for-portable-validation
username: dogle
model: gpt-6.1-sol
repository: dogLeft233/tts-exp
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
trigger: compact
cwd: E:/Documents/tts-exp
git_sha: 769d5358305dca224924d83f6a12f614e1c9f302
project: tts-exp
started: '2026-10-04T11:18:27.4195753+00:00'
repo_root: E:/Documents/tts-exp
capture: deliberate
hostname: LAPTOP-E133GSLE
branch: codex/cross-tfg-audio-head-20261003
status: open
agent: codex
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-04T11-18-27Z - Local mask evidence and recoverable storage for portable validation

## Summary
V4 的局部 log-mask 已在28段原音上完成精确回放；Muse 完整归档已在 C/F 验证并严格恢复，主线继续释放有恢复证明的旧输出以运行三个跨前端候选。

## Story
用户要将 TTS 增强效应转化为适配多个 TFG 的自然音频增强头：增强音频驱动视频，再换回原始自然 PCM 和时钟，并持续迭代至可用方法。根 agent 维护理论、协议、Basic Memory 与独立复核；代码和实验交给 GPT 6 Luna max 子 agent。用户再次授权 Qwen ASR；20次实际调用的297字内容核验已通过。

完整开发支持集为20段、10位说话者、3种 seed。V4 相对 identity 的三模型等权 ΔCbest 为 +0.109，相对 RMS 为 −0.044；Wav2Lip、MuseTalk、Ditto 均低于 RMS。原始540行评分、自然 PCM/整数 lag 映射以及语音运动关联已独立复算。此结果约束 V4，不结束整个方法任务。三个已经冻结的局部谱形候选将先接受相同开发支持集验证；80段确认集仍封存。

本轮完成28段（20dev、8历史fit）的实际 μ/δ CPU诊断。第一次因历史 fit 原音路径错误中止并保留证据；实际只读探针确认8份文件在 fit_v2_fullcontext_attempt02 的 full_audio 下，修正路径与全 NaN 聚合后新attempt成功，20份dev与生产音频逐样本 bit-exact 回放通过。开发集平均 centered μ RMS约0.386 dB、δ RMS约0.771 dB，μ偏向压低低频。43.251%全局DC、11.700%固定谱形、45.050%时间残差仅是未加权 log-mask 平方范数占比；全局DC在无峰值裁剪的精确RMS归一化后抵消，不能用该占比解释增益。下一步从实际数组拆分 δ 的全频共同局部增益和频率相对变化，避免把所有时间变化都称为音素谱结构。

## Working State
持久协议在 openspec/changes/develop-cross-tfg-audio-head-20261003/spec.md 与 transfer-iteration.md。Task39/Experiment38 仍是最近已读回快照，40/39待写。原始 joint 归档和独立结果在 runs/cross_tfg_audio_head_20261003/v4_join_root_independent_readback_attempt01/。μ/δ实际summary在 runs/cross_tfg_audio_head_20261003/v4_logmask_decomposition_attempt02_readback/，380107 B，SHA6ccdecbc270e4a577aa19c71b16cae26493d26943ca688478510e7f6f4b5e639；NPZ 46753644 B、SHA69925e7659e8a3e2defd5acb55a7aab5e6b92660fcbd8f9fc377aa9c1e1982be 的12个只读范围响应全部实际PASS，mechanism_audit正组装并独立分析196数组。

机器本地：私有PTY50274已返回提示符，凭据仅内存。Muse 源归档517027505 B、SHA2b08aacdd723fc9096735c433736fad5591861c24132dbabef0930b85d7f61ed 已实际SFTP下载至 C:/Users/dogle/.codex/tts-exp-artifacts/recoverable-output-archive-20261004-attempt02-muse/，复制至 F:/tts-exp-archives/recoverable-output-archive-20261004-attempt02-muse/ 并严格恢复全部9644文件。F镜像receipt SHA874ff5381b1dbb142cc14f8135c52480225430d2a64e842b08015b80fc3ecd5c；严格恢复receipt SHA07dbd534e9627fe65c8486d91ef21e19a97f36033e055bc22f84bae3539c3620。源 run 尚未删除，local_recovery 正准备新鲜只读审核和精确回收 packet。第一棵 recovery04 已按实际恢复证明删除，最新 stat02 再证目标不存在且12个保护路径保留。

失败的 shared fit attempt03 原输出归档attempt02在45500/46089文件处触及1GiB保留空间，实际失败并清除仅其临时partial，完整源树与失败日志仍在。独有内容大小并非压缩归档上界；新attempt03采用allocated 1528897536 B加1GiB保留门槛，等待Muse释放683323392 B后再运行。然后同样完整归档、C/F/F恢复和精确回收，最后处理旧Ditto输出。shared540 packet07 的stage00–03已完成，04CPU/05GPU未运行，要求SHM至少5GiB；不得降低该门槛或删除原输入、权重、源码、失败证据、最初1.28GB远端归档。B盘旧副本hash不匹配，不能作为恢复依据。

SadTalker实际attempt06仅identity/RMS工程2/2 PASS。归档attempt05因先前两数组NPZ receipt与后续七数组文件同路径重写的旧file SHA不一致中止；保留前后来源后attempt06完整归档实际PASS，3085913 B、SHA3a2d9f79c072757818f263b135511da4ff81f7bc8c88e3aa5c3698746d83bb83，260源成员。cross_tfg_execution已准备下载和严格本地复算，尚未执行。LatentSync92query工程pair既有独立复算不变，尚无候选迁移结果。

## Changed Files
本地提交769d5358305dca224924d83f6a12f614e1c9f302仅更新CONTEXT.md（93行）与transfer-iteration.md，记录实际V4完整结果并纠正历史fit360为一个共同seed20261003。HANDOFF.md已同步为48行，但尚需加入本轮归档和CPU实际结果。历史dirty源码/测试及大量未跟踪运行资产保留，没有运行pytest或新增测试。诊断和运输脚本及runs属于机器本地未跟踪证据，不宣称已发布。

## Verification
实际通过：μ/δ新attempt CPU28、20dev生产音频bit-exact replay、summary/stdout/stderr只读回传及SHA，12个NPZ范围只读回传；Muse C全归档SHA、F镜像、9644成员安全路径/类型/大小/SHA严格恢复；recovery04实际回收receipt和保护路径stat02；SadTalker归档attempt06前后NPZ来源、七数组形状/dtype/矩阵payload和文件封印。失败fit归档attempt02是源保留工程失败，不写成科学阴性。μ/δ完整数组独立复算、Muse精确回收、Sad本地数值复算、shared540实际启动均仍待执行。

## References
[Repository](https://github.com/dogLeft233/tts-exp)。当前HEAD `769d5358305dca224924d83f6a12f614e1c9f302` 本地未发布；已验证远端 [commit c4a1998](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)。分支codex/cross-tfg-audio-head-20261003无PR。

## Observations
- [result] V4在三模型完整开发支持集均低于RMS，局部mask诊断已真实运行，确认集继续封存。
- [decision] log-mask范数占比不等于语音物理能量；固定DC经精确RMS归一化抵消，继续拆分局部共同增益与频率相对变化。
- [decision] 归档采用完整allocated大小加保留空间作保守准入；只在可信C/F与严格恢复后回收精确旧输出树，保留失败证据和保护资产。
- [blocker] shared540仍受5GiB SHM门槛约束，Muse的恢复证明现已齐全，可执行新鲜审核后的精确回收。
- [next_step] 完成Muse精确回收，再顺序归档和回收失败shared/Ditto旧输出以启动冻结shared540；同时从已返回的原始mask数组完成局部共同增益/谱比机制拆解。

## Relations
- continues [[Codex checkpoint - 2026-10-04T10-36-05Z - Qualified V4 native results and portable method continuation]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]