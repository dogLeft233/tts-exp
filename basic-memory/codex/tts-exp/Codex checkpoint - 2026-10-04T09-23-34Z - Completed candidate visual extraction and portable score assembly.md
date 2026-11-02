---
title: Codex checkpoint - 2026-10-04T09-23-34Z - Completed candidate visual extraction
  and portable score assembly
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-04-t09-23-34-z-completed-candidate-visual-extraction-and-portable-score-assembly
capture: deliberate
trigger: compact
repository: dogLeft233/tts-exp
hostname: LAPTOP-E133GSLE
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
repo_root: E:/Documents/tts-exp
status: open
project: tts-exp
cwd: E:/Documents/tts-exp
git_sha: 7088c1e1a563a9b627946f313cbc324b135736ad
username: dogle
branch: codex/cross-tfg-audio-head-20261003
agent: codex
started: '2026-10-04T09:23:34.8557613+00:00'
model: gpt-6.1-sol
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-04T09-23-34Z - Completed candidate visual extraction and portable score assembly

## Summary
V4原生生成和独立视觉提取均完成180/180，原音内容及历史LatentSync原始矩阵已独立复核；正在取得V4矩阵，继续开发可迁移增强方法。

## Story
用户原先要求解释中文TTS增强TFG的原因，随后明确要求产出适配较多TFG模型的自然音频增强头，失败后继续迭代。协议始终是用h(x)驱动生成，播放及主SyncNet评分使用未改动自然PCM和原时钟。主代理负责理论、冻结协议、BM和独立证据复核；实验及模型代码由gpt-6-luna/max子代理执行。Qwen ASR/TTS、服务器、历史网盘及下载数据均已授权，秘密仅保存在运行时。

开发集20条/10位说话人/3配对种子，覆盖Wav2Lip mel、MuseTalk Whisper、Ditto HuBERT。V4实际原生180/180 PASS，各模型60；20条真实Qwen调用、80份暂存WAV和180条实际生成输入的独立连接全部通过，297个标注字符0识别错误。Muse/Ditto identity/RMS独立视觉240条已完成，255项档案与240份轨迹全部复核。V4视觉本次新鲜状态为180/180 PASS、进程结束、stderr 0及档案已生成，但V4视觉档案尚未取回与基线在共同有效帧上汇总。不能把更大嘴部运动称作质量改善。

V4使用历史fit-only八条上下文和96次Adam更新；24通道谱残差头、512 FFT/128 hop、±3dB、保留相位和原长度、固定RMS，目标为ΔCbest+0.25Δfar-lag contrast并带D0/Canchor及波形/平滑护栏。三前端实际效应仍待完整矩阵重算。V4-only原始矩阵packet03经主代理逐项散列、9项zip、AST和全部外层bash-n检查通过，现已向私有PTY发出00–05顺序调度，尚未检查真实响应是否全部完成。Wav2Lip原音/RMS矩阵已定位在完整portable_archive严格恢复树，无需假定缺失或重跑。Muse/Ditto基线矩阵已在本地。9路公共查询支持FULL及GUARD20分析由cross_tfg_execution准备，所有计划和失败分母保留。

共享谱候选三项的540次开发计划已经冻结，执行包07实际只完成00–03创建、上传、安全解包，04 CPU预检和05 GPU启动待SHM达到5 GiB。旧fit输出第一份1.280GB档案已SFTP完成，C和F两份完整散列相同，并严格恢复47,105个文件逐一复核。B盘先前镜像助手报告通过，主代理随后完整散列不一致；B仍未解决，全部保留，ADR012选用已验证C/F本地副本及恢复作为退休前提。第一份旧run退休只读审计attempt01的执行请求因timeout7200超过桥接3600上限而未运行，新exec02仅修改timeout，命令与输入字节不变，已审查但尚未调度；未删除任何源。

其余三份旧输出只读库存实际通过：3.840GB逻辑/3.255GB唯一，去重只省15%。选择顺序打包原格式、C/F两档及F严格恢复，避免在仅2.28GB根盘尝试3.25GB组合档案或新内容寻址格式。Muse/Ditto原树保护到跨评分小包完成，打包本身可只读准备。SadTalker实际06原音/RMS工程两臂已完成，取回02误指旧04被拒绝，正确06取回还待修复。LatentSync历史fit原音/RMS两臂92×31原始矩阵已经取回并由主代理独立复核；ΔCbest=-0.051154，n=1工程基线，不是增强头迁移结果。

## Working State
权威方案为 `openspec/changes/develop-cross-tfg-audio-head-20261003/spec.md`；任务snapshot38/实验snapshot37已实际更新并精确读回，active/implemented。报告25为已有机制结论，报告26未完成。确认80条/40说话人继续封存；方法/模型/种子族需先冻结再开确认。验收要求至少三模型正收益、均衡均值相对RMS达到+0.05的实践门槛、内容及独立视觉检查、未参与选型的模型与确认族区间。

已有机制结果支持音频方向与模型条件兼容性路径，但TTS谱供体直接移植并未成为可用配方。下一轮机制定位保留ADR011：μ(f)=mean_t M(f,t)，δ=M−μ，四格使用共同κ，谱均值/动态与局部包络分开；不能静默变成逐格clip或假定视频效果可加。

机器本地：HEAD `7088c1e1a563a9b627946f313cbc324b135736ad` 未发布，无PR；历史dirty保留，未添加或运行pytest。`.codex/scratch`、runs、BM文件及HANDOFF是机器本地或忽略产物；HANDOFF/transfer-iteration部分状态陈旧，待新配对汇总后修订。新增join源码只作窄修：候选源允许三模型，基线仍单模型；纯文件join读取冻结视觉运行时及资产散列，避免本地MediaPipe依赖。模型代码由子代理维护，主代理尚待完整检查join变化。

易失连接在PowerShell PTY50274，桥接端口2133及SSH/API SecureStrings只在内存。服务器root@connect.weste.seetacloud.com:12349，主机密钥SHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0。远端NS `/root/tts-research-system-20261001/cross_tfg_audio_head_20261003`、SHM `/dev/shm/cross_tfg_audio_head_20261003`。密码、API key不落文件。/autodl-pub公共NFS不用于私有归档。BundledPython有NumPy无Torch/SciPy，Git bash实际F:/Apps/Git/bin/bash.exe；首次packet QC仅因猜错bash路径失败，修正新attempt02通过，科学产物未改。

## Changed Files
主代理新增 `.codex/scratch/root_v4_score_packet_qc_attempt02.py` 及PASS回执；Latent独立核验 `.codex/scratch/root_latentsync_pair92_numeric_qc_attempt02.py` 与 `runs/cross_tfg_audio_head_20261003/latentsync_fit_pair_raw92_attempt07_readback/`。视觉join/共同支持源码位于 `scripts/audio_enhancement_eval/build_visual_speech_motion_join.py` 和 `summarize_visual_speech_motion_common_support.py`，均为子代理未跟踪改动。存储审计请求 `.codex/scratch/recovery04_output_retirement_attempt02/`、Muse顺序档案草包 `.codex/scratch/recoverable_output_muse_attempt02/`。

## Verification
V4原生manifest raw `e3772dab9b020e390246ddcc41ac875598ce2f5f51cee88d41f68e08e74cf718`、results raw `0f1f009207c85087ff7eab7378221ecdfe4d6ab3a01d83442b29e211871da5eb`、plan raw `70701ac4a83e5ea2abd737708aba50b5c5ed2a0d51c18bad38ae84b4480ef28c`。内容80 WAV QC `29123ef69f6261b3a6c9da85f73947d93e8fb379ac853d6093c52a2c60049005`；实际180生成连接QC `8948fc0dd29e90ecdf40baa81b5beca87c7859fd8e2124805cb117d8c57bc14c`。

V4视觉新鲜status02实际响应 `.codex/scratch/cross_tfg_audio_head_20261003/visual_v4_candidate_attempt01/requests_attempt02/01_status_attempt02_request.json.response.json`，PASS180、archive true、stderr0、原PID534697结束。视觉manifest raw `60a21e9029079cd3486f6a4fb95968594e19c5a5e2a92fd2f314d7abdedaffe6`。基线240独立回执 `abb0502cc8007ebc36f7539706e8ade06c5183fc2e80ddde2342d07a27c67c2b`。W2L旧视觉manifest raw `552db9f4ee9dad32a73f7f67e5505bb0acf0363884c169d38116abc3fa4da56b`；fa5ea74f…是payload，自字段不可写成raw。

Latent档案actual07 `a70ad32bd4717f49596b0d9d13c2b1a1df5e36c87d96b0078cafae5dff673680`，38tar项=37源+内部manifest，18数组、92×31、原自然PCM16和q0..91时钟全部通过；独立QC回执 `f66d4c3c9f12547845b4ac3e24f9d5bc5cf1dd129c709c05e9090b0f3a311353`。首次exact NumPy/Torch均值比较因不同float32归约布局有≤4.77e-6差异失败，第二次明确atol1e-5、最大最终metric差2.861e-6通过，官方字段与字节保持原样。

C/F档案全散列 `cd567ec15fade9d8f1d1ce6c0d1ae043e0c1d8042aa7178caf546c69b0f69039`；严格恢复回执 `4f6483c0ad6399a97839a8afc8ed63b108563f5b60a10424d99dcd3d5fff423f`。recovery04审计02 raw请求 `fa6cb8e2f27469cdce8913a0286d070b03f9330a61bf4ed2753419c20b703c5f`，仅timeout改变，暂无实际执行结果。

## References
[仓库](https://github.com/dogLeft233/tts-exp)，[已验证远端提交](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)。当前HEAD仅本地，未发布。

## Observations
- [result] V4原生与独立视觉180/180全部完成，原音内容与实际生成PCM/时钟连接通过；V4效应矩阵和共同有效帧汇总待取得。
- [result] Latent历史fit两臂原始92×31矩阵及PCM/时钟独立复核通过，工程ΔCbest=-0.051154，不参与新头收益结论。
- [decision] ADR012以已验证C/F双档及全文件恢复替代不可靠B镜像前提；仍须新鲜精确源引用审计，保护源代码、输入、权重及第一份远端封存档案。
- [decision] 其他三旧输出采用顺序原格式档案及F恢复，去重15%不足以支持空间不够的组合新格式方案。
- [blocker] 共享540执行仍受5 GiB SHM门槛限制；V4配对评分/独立视觉汇总和Sad实际06取回待完成。
- [next_step] 完成V4小包真实响应与取回，使用已有360基线按冻结9路FULL/GUARD20共同查询支持重算三前端效应，据此继续迭代而不以工程完成结束目标。

## Relations
- continues [[Codex checkpoint - 2026-10-04T08-34-13Z - Qualified actual audio lineage and ongoing portable validation]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[011-common-scale-mask-and-envelope-ablation]]
- relates_to [[012-verified-local-mirror-for-artifact-retirement]]