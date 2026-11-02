---
title: Codex checkpoint - 2026-10-04T09-56-54Z - Qualified raw matrices and common
  visual support before transfer effects
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-04-t09-56-54-z-qualified-raw-matrices-and-common-visual-support-before-transfer-effects
model: gpt-6.1-sol
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
repository: dogLeft233/tts-exp
cwd: E:/Documents/tts-exp
trigger: compact
repo_root: E:/Documents/tts-exp
hostname: LAPTOP-E133GSLE
branch: codex/cross-tfg-audio-head-20261003
username: dogle
git_sha: 7088c1e1a563a9b627946f313cbc324b135736ad
project: tts-exp
capture: deliberate
agent: codex
status: open
started: '2026-10-04T09:56:54.1539090+00:00'
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-04T09-56-54Z - Qualified raw matrices and common visual support before transfer effects

## Summary
V4 的180次原生生成、1260个原始评分数组和540条共同支持视觉轨迹已独立复核；实际跨前端效应汇总正在修正路径和索引映射后执行，目标仍为可迁移自然音频增强头。

## Story
用户从恢复TTS增强TFG的机制路线，进一步要求产出适配多种TFG的自然音频增强头，并明确授权失败后继续迭代、Qwen ASR/TTS、历史网盘及下载数据。用h(x)生成视频，播放及主SyncNet评分始终使用原始自然PCM和原时钟。root维护理论、冻结协议、BM、Git及独立证据；实验和模型代码由gpt-6-luna/max子代理执行。开发20条/10位说话人/3配对种子，覆盖Wav2Lip mel、MuseTalk Whisper、Ditto HuBERT。确认80条/40说话人仍封存，不能由开发集筛选替代。

本次接续后，V4-only原始小包已经实际完成并取回：7,601,327字节档案、641个安全TAR成员、634个科学资产、180行和1260个NPZ数组，逐项散列、官方矩阵、原音PCM及时钟核验通过。独立视觉档案已取回，V4 180条与三模型identity/RMS基线360条连接成540条，180上下文全部完成。root重算8640个共同支持几何数值和完整种子→片段→说话人→模型均值，最大残差0。V4平均开口.072927，RMS .071474；开口速度RMS分别.474874和.474698，仅为描述量。子代理随后完成原音40ms log-RMS关联：V4与RMS零时滞开口相关均值.348833/.349786，待root独立重算；不能将更大开口称为更好同步。

完整SyncNet效应尚未取得。执行前发现join包的基线目录、result_manifest/repeat文件名和GUARD20不完整说话人均值问题，已要求新鲜包修正，保留原失败/未执行尝试。packet07修正上述问题，仍未执行；又发现过滤valid_query_mask后enumerate键并不一定等于原NPZ行号，子代理正以显式key→raw index修正新包。V4所有掩码独立检查均为有效前缀，但不能据此假定基线相同。FULL科学汇总规则不变，GUARD20缺失保持完整计划分母、全cohort均值null，另报告共同完整说话人子集。修正发生在读取效应前。

共享谱三候选540次开发执行已经实际只完成00–03准备，仍待5GiB SHM门槛。首个旧fit run档案C/F全包散列及C严格恢复47,105文件通过，B后来全包散列不一致，未作为备份。首个只读退休审计实际检查源树后因保护路径指向不存在的project namespace而失败，未删除任何源；新鲜诊断证明实际任务namespace在SHM，新的只读audit02绑定该实际路径和code/control/ditto_control，正在root审阅。其余三旧输出共3.840GB，去重仅15%，采用顺序标准档案/C-F副本/F严格恢复。Sad实际06原音/RMS工程两臂已完成，正确取回尚待主线后处理。LatentSync历史fit两臂92×31原始矩阵已独立合格，ΔCbest=-.051154仅为n=1工程基线。

## Working State
权威验收为openspec/changes/develop-cross-tfg-audio-head-20261003/spec.md：至少三模型正收益、均衡增益相对RMS达到+.05实践门槛、内容及独立视觉、冻结后的确认族和未参与选型模型。当前方法仍迭代，报告26未完成；不能在排除项或工程PASS结束目标。下一机制动作是只用现有音频/冻权重分解V4 log-mask时间均值μ(f)与动态δ(t,f)，先取得描述诊断；任何视频四格保持ADR011共同κ和完整原时钟，不另行改变科学配方。

已实际更新CONTEXT.md（93行）、transfer-iteration.md和本地HANDOFF.md（32行）。任务snapshot38和实验snapshot37仍active/implemented，部分内容需要随后跟随新实际矩阵更新。当前branch codex/cross-tfg-audio-head-20261003，HEAD 7088c1e1a563a9b627946f313cbc324b135736ad仅本地，无PR；历史dirty完整保留、无staged、未添加/运行pytest。runs、.codex/scratch、BM及HANDOFF为机器本地或忽略产物。

易失私有连接在PowerShell PTY50274，桥接端口2133及SSH/API SecureStrings只在内存；服务器root@connect.weste.seetacloud.com:12349，hostpin SHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0。NS /root/tts-research-system-20261001/cross_tfg_audio_head_20261003，SHM /dev/shm/cross_tfg_audio_head_20261003。秘密不写文件/BM/argv/output，公共NFS不用作私有归档。桥接单exec上限3600秒；Git bash位于F:/Apps/Git/bin/bash.exe。

## Changed Files
root新增独立核验.codex/scratch/root_v4_raw_archive_qc_attempt01.py、root_visual_common_support_qc_attempt01.py及对应PASS回执。子代理新鲜源码scripts/audio_enhancement_eval/build_visual_speech_motion_join_attempt02.py、summarize_visual_speech_motion_common_support_attempt02.py、speech_motion_association_attempt03.py和关联层级汇总。join包当前路径.codex/scratch/cross_tfg_audio_head_20261003/v4_cross_frontend_join_attempt02/packet_attempt07/未调度，正在新索引修复。只读退休包.codex/scratch/recovery04_output_retirement_attempt05/request_index_attempt02.json未调度；旧审计失败回执保留。

## Verification
V4原始档案SHA9641f272f5e77b3608945a5cf1020e6cd90f829d31206a65a18e78986bdb553e，local runs/cross_tfg_audio_head_20261003/v4_candidate_score_evidence_readback_attempt01/，641项/1260数组及原音PASS。V4视觉档案SHAc8ed6434aa15c2435ff472b5e5064c0e1f38f4fe66fa2d0ddda26a4b25730a01，189TAR项/180NPZ/2520数组/19035帧独立PASS。540视觉join receipt168c0edd61f19ea3ff79c0af70614f80ff1bd02eb30ef0e4afed0d3862bfab88，common summary raw fbb4cac3dee560f2b2b5fc47d616aa12ff0165f78191e9e13c872a888f56aa18，root全部8640数值最大残差0。20条原音本地映射子代理独立读回receipt0fbbd17ed38bc3ce89e3c82ebe727ccbda0894f0238f53bd39d6fd2a22972f78，speech-motion summary raw60994eae17f7a6684c21c4a2b581497230613d7d08a956fa47d2901c5b0d0bbb待root重算。

本次checkpoint先加载StartupRouter和实验指令，schema第一次读取误用schema/而非schemas/，已改实际路径；same-chat/legacy搜索分页证据、predecessor直接读取、GitHub仓库/PR/已发布提交及本地Git实际核对后创建。所有未执行包不记成PASS。

## References
[仓库](https://github.com/dogLeft233/tts-exp)，[已验证发布提交](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)。当前HEAD为未发布本地提交。

## Observations
- [result] V4原始评分档案和540条共同支持视觉证据实际合格；尚未取得V4三前端主效应数字。
- [decision] 效应读取前修复路径、严格完整说话人分母和显式查询原始行映射，保留旧包，不据无效mask假定索引安全。
- [decision] C/F及严格恢复作为ADR012退休前提；B盘全包散列冲突保持未解决，未删除任何源。
- [blocker] 共享540受SHM 5GiB门槛限制；只读退休审计需实际namespace修复，完整V4效应join待新鲜包。
- [next_step] 审查显式key→raw index的新鲜join包并实际运行FULL/GUARD20完整三前端效应汇总，按结果继续训练/谱机制迭代。

## Relations
- continues [[Codex checkpoint - 2026-10-04T09-23-34Z - Completed candidate visual extraction and portable score assembly]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[011-common-scale-mask-and-envelope-ablation]]
- relates_to [[012-verified-local-mirror-for-artifact-retirement]]